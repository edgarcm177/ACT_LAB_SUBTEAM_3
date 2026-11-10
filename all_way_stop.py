"""
Simple MPC + Trajectory-Conflict Controller for an Intersection
===============================================================

There is no mandatory stop. Cars drive through at the desired speed and only
slow down or stop before the stop line when they must yield: an obstacle
blocks their path, or their time in the intersection would overlap with a
car whose path conflicts with theirs.

Each car is a 2-D point mass in its own road frame: x is the distance along
its lane, y the lateral offset from the lane centre. The MPC chooses both
the longitudinal (ax) and lateral (ay) accelerations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches 
from scipy.optimize import LinearConstraint, minimize


# =============================================================================
# 1. PARAMETERS
# =============================================================================

DT = 0.20                 # sampling time [s]
SIM_TIME = 35.0           # maximum simulation time [s]
N = 20                    # prediction horizon
V_DES = 8.0               # desired speed in GO [m/s]
A_MIN = -5.0              # maximum braking [m/s^2]
A_MAX = 3.0               # maximum acceleration [m/s^2]
AY_MAX = 2.0              # maximum lateral acceleration [m/s^2]

STOP_LINE = 0.0           # stop line position [m]
INTERSECTION_EXIT = 12.0  # intersection is clear after this point [m]
DELETE_POINT = 25.0       # remove vehicle after this point [m]

STOP_POS_TOL = 0.50       # stopping position tolerance [m]
STOP_SPEED_TOL = 0.20     # stopping speed tolerance [m/s]

# Objective-function weights from the report
W_POSITION = 100.0
W_SPEED = 150.0
W_ACCEL = 0.20
W_CROSS = 2000.0
W_COLLISION = 1.0e5      # penalty for predicted car-to-car overlap
W_ACCEL_Y = 0.20         # lateral acceleration, W_ACCEL_Y * ay^2
W_LAT_SPEED = 150.0      # lateral speed, W_LAT_SPEED * vy^2 (stops sideways drift)
W_ROAD = 1.0e4           # soft road boundary, W_ROAD * (distance past the edge)^2

MAXITER = 60
# Bounds on the MPC decision vector [ax(0..N-1), ay(0..N-1)]
ACCEL_BOUNDS = [(A_MIN, A_MAX)] * N + [(-AY_MAX, AY_MAX)] * N

LANE_OFFSET = 3.0         # lane centre offset from the road edge [m]
CONFLICT_CLEARANCE = 2.5  # paths closer than this are in conflict [m]
NEARBY_RADIUS = 40.0      # cars within this distance are tracked [m]
CAR_HALF_WIDTH = 1.0      # used for obstacle clearance [m]
CAR_LENGTH = 4.5          # [m]

# Road boundaries for the car's centre in its road frame (y = 0 is its lane
# centre). The road is its own lane (-3 to 3 m) plus the oncoming lane
# (3 to 9 m); the bounds keep the whole car body on the road.
ROAD_Y_MIN = -LANE_OFFSET + CAR_HALF_WIDTH        # -2 m
ROAD_Y_MAX = 3 * LANE_OFFSET - CAR_HALF_WIDTH     #  8 m
ROAD_MODE = "soft"        # "soft" (penalty), "hard" (constraint) or "none"

# Each car body is approximated by circles placed from the front bumper
# (position p) back to the rear bumper.
CIRCLE_OFFSETS = np.linspace(0.0, CAR_LENGTH, 3)
COLLISION_DIST = 2.0 * CAR_HALF_WIDTH  # circle centres closer than this = collision [m]
CAR_BUFFER = 1.0                        # clear space kept between car bodies [m]
SAFE_DIST = COLLISION_DIST + CAR_BUFFER # circle centres closer than this are penalised [m]

# Right of way
A_GO = 2.0                # acceleration assumed when predicting crossing times [m/s^2]
A_BRAKE = 4.0             # a car that cannot stop within this braking is committed [m/s^2]
T_MARGIN = 0.5            # extra time gap between conflicting cars in the intersection [s]
T_LOOKAHEAD = 20.0        # how far ahead crossing times are predicted [s]

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


# =============================================================================
# GEOMETRY
# =============================================================================

def position_to_xy(origin, x, y=0.0):
    """
    Map a car's road-frame position to intersection (x, y) coordinates.
    x is the distance along its lane, y the lateral offset from the lane
    centre (positive to the driver's left).
    """
    if origin == "W": return (x, LANE_OFFSET + y)
    if origin == "E": return (INTERSECTION_EXIT - x, INTERSECTION_EXIT - LANE_OFFSET - y)
    if origin == "S": return (INTERSECTION_EXIT - LANE_OFFSET - y, x)
    if origin == "N": return (LANE_OFFSET + y, INTERSECTION_EXIT - x)
    return (0.0, 0.0)


def body_points(origin, x, y=0.0):
    """
    Circle centres of a car body for one or more positions -> shape (K, C, 2).
    The body stays aligned with the lane.
    """
    x = np.atleast_1d(x)[:, None] - CIRCLE_OFFSETS[None, :]
    y = np.broadcast_to(np.atleast_1d(y)[:, None], x.shape)
    gx, gy = position_to_xy(origin, x, y)
    return np.stack(np.broadcast_arrays(gx, gy), axis=-1)


def overlap_penalty(points1, points2, safe_dist):
    """Sum of squared circle overlaps between two bodies over the horizon."""
    d = np.linalg.norm(points1[:, :, None, :] - points2[:, None, :, :], axis=-1)
    return np.sum(np.maximum(safe_dist - d, 0.0)**2)


def point_segment_distance(pt, a, b):
    pt, a, b = np.asarray(pt), np.asarray(a), np.asarray(b)
    ab = b - a
    t = np.clip(np.dot(pt - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(pt - (a + t * ab))


def segments_intersect(a, b, c, d):
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    return d1 * d2 < 0 and d3 * d4 < 0


def segment_distance(a, b, c, d):
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def windows_overlap(w1, w2):
    """True if two (t_enter, t_exit) time windows overlap, including T_MARGIN."""
    return w1[0] < w2[1] + T_MARGIN and w2[0] < w1[1] + T_MARGIN


def is_stopped_at_line(car):
    return abs(car.x - STOP_LINE) <= STOP_POS_TOL and car.vx <= STOP_SPEED_TOL


def path_distance(path1, path2):
    """Minimum distance between two polyline paths."""
    return min(
        segment_distance(path1[i], path1[i + 1], path2[j], path2[j + 1])
        for i in range(len(path1) - 1)
        for j in range(len(path2) - 1)
    )


# =============================================================================
# OBSTACLE
# =============================================================================

class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def blocks(self, path):
        clearance = self.radius + CAR_HALF_WIDTH
        return any(
            point_segment_distance((self.x, self.y), path[i], path[i + 1]) < clearance
            for i in range(len(path) - 1)
        )


# =============================================================================
# 2. CAR
# =============================================================================

class Car:
    def __init__(self, car_id, origin, position, speed, destination=None):
        self.id = car_id

        # Route information
        self.origin = origin
        self.destination = destination or OPPOSITE[origin]
        if self.destination != OPPOSITE[origin]:
            raise ValueError("Only straight-through routes are supported by the 1-D model")
        self.direction = self.destination           # heading of travel
        self.path = [
            position_to_xy(origin, STOP_LINE),
            position_to_xy(origin, INTERSECTION_EXIT),
        ]

        # Awareness of the surroundings, refreshed every step
        self.nearby_cars = {}                       # id -> path of nearby cars
        self.conflicts = []                         # ids of cars whose paths conflict
        self.obstacles = []                         # obstacles on or near the path

        # State in the car's own road frame: x along its lane, y lateral
        # offset from the lane centre (positive to the driver's left).
        self.x = position
        self.y = 0.0
        self.vx = speed
        self.vy = 0.0

        self.state = "GO"

        self.stop_time = None       # when the car came to rest at the line, if it did
        self.enter_time = None      # when the car crossed the stop line

        # Initial guesses for the two optimization problems.
        # Each guess contains N values of ax followed by N values of ay.
        self.stop_guess = np.zeros(2 * N)
        self.go_guess = np.zeros(2 * N)

        # Predicted positions over the horizon, shared with other cars so
        # they can penalise collisions (constant-speed guess to start).
        self.predicted_x = position + speed * DT * np.arange(N + 1)
        self.predicted_y = np.zeros(N + 1)

        # Data saved for plots.
        self.time_history = []
        self.x_history = []
        self.y_history = []
        self.vx_history = []
        self.vy_history = []
        self.ax_history = []
        self.ay_history = []
        self.state_history = []


# =============================================================================
# 3. CONTROLLER
# =============================================================================

class Controller:
    def __init__(self):
        self.collisions = []      # (time, id1, id2) of detected collisions

    # -------------------------------------------------------------------------
    # Vehicle prediction model
    # -------------------------------------------------------------------------
    def predict(self, car, acceleration_sequence):
        """
        2-D point-mass dynamics from the car's current state. The sequence
        holds N values of ax followed by N values of ay.

            x(k+1)  = x(k)  + vx(k) * DT + 0.5 * ax(k) * DT^2
            y(k+1)  = y(k)  + vy(k) * DT + 0.5 * ay(k) * DT^2
            vx(k+1) = vx(k) + ax(k) * DT
            vy(k+1) = vy(k) + ay(k) * DT

        Returns x, y, vx, vy over the horizon (N + 1 values each).
        """

        ax, ay = acceleration_sequence[:N], acceleration_sequence[N:]

        x = np.zeros(N + 1)
        y = np.zeros(N + 1)
        vx = np.zeros(N + 1)
        vy = np.zeros(N + 1)

        x[0], y[0], vx[0], vy[0] = car.x, car.y, car.vx, car.vy

        for k in range(N):
            x[k + 1] = x[k] + vx[k] * DT + 0.5 * ax[k] * DT**2
            y[k + 1] = y[k] + vy[k] * DT + 0.5 * ay[k] * DT**2
            vx[k + 1] = vx[k] + ax[k] * DT
            vy[k + 1] = vy[k] + ay[k] * DT

        return x, y, vx, vy

    # -------------------------------------------------------------------------
    # Problem 1: STOP
    # -------------------------------------------------------------------------
    def collision_cost(self, car, x, y, others):
        """
        J_collision = W_COLLISION * sum over conflicting cars, horizon steps
                      and body circles of max(SAFE_DIST - distance, 0)^2
        """

        if not others:
            return 0.0

        mine = body_points(car.origin, x[1:], y[1:])
        cost = 0.0
        for other in others:
            theirs = body_points(other.origin, other.predicted_x[1:], other.predicted_y[1:])
            cost += W_COLLISION * overlap_penalty(mine, theirs, SAFE_DIST)

        return cost

    def lateral_cost(self, acceleration_sequence, vy):
        """
        J_lateral = sum[ W_ACCEL_Y * ay(k)^2 + W_LAT_SPEED * vy(k+1)^2 ]
        """

        ay = acceleration_sequence[N:]
        return W_ACCEL_Y * np.sum(ay**2) + W_LAT_SPEED * np.sum(vy[1:]**2)

    # -------------------------------------------------------------------------
    # Road boundaries
    # -------------------------------------------------------------------------
    def road_cost(self, y):
        """
        Soft road boundaries (ROAD_MODE = "soft"):

        J_road = W_ROAD * sum[ max(y(k+1) - ROAD_Y_MAX, 0)^2
                             + max(ROAD_Y_MIN - y(k+1), 0)^2 ]

        Zero while the car stays on the road, growing with how far it goes
        past an edge.
        """

        if ROAD_MODE != "soft":
            return 0.0
        over = np.maximum(y[1:] - ROAD_Y_MAX, 0.0)
        under = np.maximum(ROAD_Y_MIN - y[1:], 0.0)
        return W_ROAD * np.sum(over**2 + under**2)

    def road_constraint(self, car):
        """
        Hard road boundaries (ROAD_MODE = "hard"): ROAD_Y_MIN <= y(k) <= ROAD_Y_MAX
        for k = 1..N. From the dynamics, y(k) is linear in the ay sequence,

            y(k) = y(0) + k * vy(0) * DT + DT^2 * sum_{j<k} (k - j - 1/2) * ay(j)

        so the bounds are linear constraints on the decision vector [ax, ay].
        """

        k = np.arange(1, N + 1)[:, None]
        j = np.arange(N)[None, :]
        matrix = np.zeros((N, 2 * N))
        matrix[:, N:] = np.where(j < k, DT**2 * (k - j - 0.5), 0.0)
        free_motion = car.y + k[:, 0] * car.vy * DT
        return LinearConstraint(matrix, ROAD_Y_MIN - free_motion, ROAD_Y_MAX - free_motion)

    def optimize(self, cost, guess, car, args=()):
        """Minimise a cost over [ax, ay] with the acceleration bounds and the road boundaries."""

        if ROAD_MODE == "hard":
            # L-BFGS-B only accepts bounds on ax and ay, not on y, so the hard
            # road boundaries need a solver that takes constraints.
            return minimize(cost, guess, args=args, method="SLSQP", bounds=ACCEL_BOUNDS,
                            constraints=[self.road_constraint(car)],
                            options={"maxiter": MAXITER, "disp": False})
        return minimize(cost, guess, args=args, method="L-BFGS-B", bounds=ACCEL_BOUNDS,
                        options={"maxiter": MAXITER, "disp": False})

    def conflicting_cars(self, car, cars):
        return [other for other in cars if other.id in car.conflicts]

    def stop_cost(self, acceleration_sequence, car, others):
        """
        J_stop = W_POSITION * x(N)^2
               + W_SPEED    * vx(N)^2
               + sum[ W_ACCEL * ax(k)^2
                    + W_CROSS * max(x(k+1), 0)^2 ]
               + J_lateral
               + J_road
               + J_collision
        """

        x, y, vx, vy = self.predict(car, acceleration_sequence)

        cost = W_POSITION * x[N]**2
        cost += W_SPEED * vx[N]**2

        for k in range(N):
            cost += W_ACCEL * acceleration_sequence[k]**2
            cost += W_CROSS * max(x[k + 1], 0.0)**2

        cost += self.lateral_cost(acceleration_sequence, vy)
        cost += self.road_cost(y)
        cost += self.collision_cost(car, x, y, others)

        return cost

    def solve_stop_problem(self, car, cars):
        """Solve the STOP problem and return the first optimal (ax, ay)."""

        result = self.optimize(self.stop_cost, car.stop_guess, car,
                               args=(car, self.conflicting_cars(car, cars)))

        optimal_sequence = result.x
        car.stop_guess = optimal_sequence

        # MPC: apply only the first input.
        return optimal_sequence[0], optimal_sequence[N]

    # -------------------------------------------------------------------------
    # Problem 2: GO
    # -------------------------------------------------------------------------
    def go_cost(self, acceleration_sequence, car, others):
        """
        J_go = sum[ W_SPEED * (vx(k+1) - V_DES)^2
                  + W_ACCEL * ax(k)^2 ]
             + J_lateral
             + J_road
             + J_collision
        """

        x, y, vx, vy = self.predict(car, acceleration_sequence)

        cost = 0.0

        for k in range(N):
            cost += W_SPEED * (vx[k + 1] - V_DES)**2
            cost += W_ACCEL * acceleration_sequence[k]**2

        cost += self.lateral_cost(acceleration_sequence, vy)
        cost += self.road_cost(y)
        cost += self.collision_cost(car, x, y, others)

        return cost

    def solve_go_problem(self, car, cars):
        """Solve the GO problem and return the first optimal (ax, ay)."""

        result = self.optimize(self.go_cost, car.go_guess, car,
                               args=(car, self.conflicting_cars(car, cars)))

        optimal_sequence = result.x
        car.go_guess = optimal_sequence

        # MPC: apply only the first input.
        return optimal_sequence[0], optimal_sequence[N]

    # -------------------------------------------------------------------------
    # Trajectory-conflict logic
    # -------------------------------------------------------------------------
    def update_awareness(self, cars, obstacles):
        """Refresh each car's nearby cars, trajectory conflicts and obstacles."""

        live = [car for car in cars if car.state != "DONE"]

        for car in live:
            here = np.array(position_to_xy(car.origin, car.x, car.y))

            car.nearby_cars = {}
            car.conflicts = []
            for other in live:
                if other.id == car.id:
                    continue
                there = np.array(position_to_xy(other.origin, other.x, other.y))
                if np.linalg.norm(here - there) > NEARBY_RADIUS:
                    continue

                car.nearby_cars[other.id] = other.path
                if path_distance(car.path, other.path) < CONFLICT_CLEARANCE:
                    car.conflicts.append(other.id)

            car.obstacles = [obs for obs in obstacles if obs.blocks(car.path)]

    def crossing_window(self, car):
        """
        Predict when the car would be inside the intersection if it kept
        going: from its front bumper reaching the stop line until its rear
        bumper is CAR_BUFFER past the exit. Returns (t_enter, t_exit) in
        seconds from now, or None once the car has cleared the intersection.
        """

        clear_point = INTERSECTION_EXIT + CAR_LENGTH + CAR_BUFFER
        if car.x >= clear_point:
            return None

        p, v, t = car.x, car.vx, 0.0
        t_enter = 0.0 if p >= STOP_LINE else None

        while t < T_LOOKAHEAD:
            a = A_GO if v < V_DES else 0.0
            p += v * DT + 0.5 * a * DT**2
            v = min(v + a * DT, V_DES) if a else v
            t += DT

            if t_enter is None and p >= STOP_LINE:
                t_enter = t
            if p >= clear_point:
                return (t_enter, t)

        return (t_enter if t_enter is not None else T_LOOKAHEAD, T_LOOKAHEAD)

    def is_committed(self, car):
        """A car past the stop line, or too close to stop before it, must keep going."""

        if car.x > STOP_LINE:
            return True
        if car.state != "GO":
            return False
        return STOP_LINE - car.x < car.vx**2 / (2.0 * A_BRAKE)

    def assign_right_of_way(self, cars):
        """
        Decide which cars keep going (GO) and which must yield (YIELD).

        A car yields only when an obstacle blocks its path, or when its
        predicted time in the intersection overlaps with a conflicting car
        that goes first. Committed cars go first, then the others in order
        of predicted entry time (lower ID on a tie).
        """

        live = [car for car in cars if car.state != "DONE"]
        windows = {car.id: self.crossing_window(car) for car in live}

        def order(car):
            window = windows[car.id]
            return (not self.is_committed(car), window[0] if window else -1.0, car.id)

        going = []
        for car in sorted(live, key=order):
            window = windows[car.id]

            if window is None or self.is_committed(car):
                car.state = "GO"
            else:
                blocked = bool(car.obstacles) or any(
                    other.id in car.conflicts and windows_overlap(window, windows[other.id])
                    for other in going
                )
                car.state = "YIELD" if blocked else "GO"

            if car.state == "GO" and window is not None:
                going.append(car)

    # -------------------------------------------------------------------------
    # Collision handling
    # -------------------------------------------------------------------------
    def update_prediction(self, car):
        """Shift the car's latest plan forward one step for the other cars to use."""

        if car.state == "YIELD" and is_stopped_at_line(car):
            car.predicted_x = np.full(N + 1, STOP_LINE)
            car.predicted_y = np.full(N + 1, car.y)
            return

        plan = car.go_guess if car.state == "GO" else car.stop_guess
        ax, ay = plan[:N], plan[N:]
        shifted = np.concatenate([np.append(ax[1:], ax[-1]), np.append(ay[1:], ay[-1])])
        car.predicted_x, car.predicted_y, _, _ = self.predict(car, shifted)

    def detect_collisions(self, cars, time):
        """Record any pair of cars whose bodies actually overlap."""

        live = [car for car in cars if car.state != "DONE"]
        for i, car in enumerate(live):
            for other in live[i + 1:]:
                d = np.linalg.norm(
                    body_points(car.origin, car.x, car.y)[0][:, None, :]
                    - body_points(other.origin, other.x, other.y)[0][None, :, :],
                    axis=-1,
                )
                if np.min(d) < COLLISION_DIST:
                    self.collisions.append((time, car.id, other.id))


# =============================================================================
# 4. DEMONSTRATION VEHICLES
# =============================================================================

def create_cars():
    return [
        Car(1, "W", -24.0, 7.0),
        Car(2, "E", -30.0, 7.5),
        Car(3, "S", -36.0, 8.0),
        Car(4, "N", -42.0, 8.0),
    ]


# =============================================================================
# 5. SIMULATION
# =============================================================================

def run_simulation(obstacles=None):
    cars = create_cars()
    controller = Controller()
    obstacles = obstacles or []

    times = np.arange(0.0, SIM_TIME + DT, DT)

    for time in times:

        # ---------------------------------------------------------------------
        # A. Update trajectory conflicts/obstacles and decide which cars
        #    keep going and which must yield.
        # ---------------------------------------------------------------------
        controller.update_awareness(cars, obstacles)
        controller.assign_right_of_way(cars)

        # ---------------------------------------------------------------------
        # B. Compute one (ax, ay) to apply to each car.
        # ---------------------------------------------------------------------
        accelerations = {}

        for car in cars:
            if car.state == "GO":
                accelerations[car.id] = controller.solve_go_problem(car, cars)

            elif car.state == "YIELD" and is_stopped_at_line(car):
                accelerations[car.id] = (0.0, 0.0)

            elif car.state == "YIELD":
                accelerations[car.id] = controller.solve_stop_problem(car, cars)

            else:  # DONE
                accelerations[car.id] = (0.0, 0.0)

        # ---------------------------------------------------------------------
        # C. Save the current state and move each car one simulation step.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state == "DONE":
                continue

            ax, ay = accelerations[car.id]

            car.time_history.append(time)
            car.x_history.append(car.x)
            car.y_history.append(car.y)
            car.vx_history.append(car.vx)
            car.vy_history.append(car.vy)
            car.ax_history.append(ax)
            car.ay_history.append(ay)
            car.state_history.append(car.state)

            if car.state == "YIELD" and is_stopped_at_line(car):
                car.x = STOP_LINE
                car.vx = 0.0
                car.vy = 0.0
                if car.stop_time is None:
                    car.stop_time = time

            else:
                car.x = car.x + car.vx * DT + 0.5 * ax * DT**2
                car.y = car.y + car.vy * DT + 0.5 * ay * DT**2
                car.vx = car.vx + ax * DT
                car.vy = car.vy + ay * DT

            if car.enter_time is None and car.x > STOP_LINE:
                car.enter_time = time + DT

            controller.update_prediction(car)

        controller.detect_collisions(cars, time)

        # ---------------------------------------------------------------------
        # D. Remove cars after the downstream deletion point.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state == "GO" and car.x >= DELETE_POINT:
                car.state = "DONE"

        if all(car.state == "DONE" for car in cars):
            break

    return cars, controller


# =============================================================================
# 6. RESULTS
# =============================================================================

def print_summary(cars, controller):
    print("\nCrossing summary")
    print("----------------")

    ordered_cars = sorted(
        cars, key=lambda car: car.enter_time if car.enter_time is not None else float("inf")
    )

    for car in ordered_cars:
        entered = f"{car.enter_time:.1f} s" if car.enter_time is not None else "never"
        yielded = "yes" if "YIELD" in car.state_history else "no"
        stopped = f"at {car.stop_time:.1f} s" if car.stop_time is not None else "no"
        min_speed = min(car.vx_history) if car.vx_history else 0.0
        max_offset = max(np.abs(car.y_history)) if car.y_history else 0.0
        print(
            f"Vehicle {car.id} ({car.origin} -> {car.destination}): "
            f"entered = {entered}, yielded = {yielded}, stopped = {stopped}, "
            f"min speed = {min_speed:.1f} m/s, max lateral offset = {max_offset:.2f} m"
        )

    pairs = sorted({(a, b) for _, a, b in controller.collisions})
    if pairs:
        print(f"Collisions: {', '.join(f'V{a}-V{b}' for a, b in pairs)}")
    else:
        print("Collisions: none")


def plot_results(cars):
    # Position plot
    plt.figure(figsize=(9, 5))

    for car in cars:
        plt.plot(
            car.time_history,
            car.x_history,
            label=f"Vehicle {car.id} ({car.origin})",
        )

    plt.axhline(STOP_LINE, linestyle="--", linewidth=1, label="Stop line")
    plt.axhline(
        INTERSECTION_EXIT,
        linestyle=":",
        linewidth=1,
        label="Intersection exit",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Vehicle Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("position_vs_time.png", dpi=200)
    plt.close()

    # Speed plot
    plt.figure(figsize=(9, 5))

    for car in cars:
        plt.plot(
            car.time_history,
            car.vx_history,
            label=f"Vehicle {car.id} ({car.origin})",
        )

    plt.axhline(V_DES, linestyle="--", linewidth=1, label="Desired speed")

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Vehicle Speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speed_vs_time.png", dpi=200)
    plt.close()



#Animation Function
def animate_results(cars):
    max_frames = max(len(car.time_history) for car in cars)
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect('equal')
    ax.set_facecolor('#2e2e2e')
    ax.set_title("Intersection MPC Controller", color='white', pad=20, fontsize=16)

    road_width = 12
    intersection = patches.Rectangle((0, 0), road_width, road_width, color='#555555', zorder=1)
    ax.add_patch(intersection)

    ax.add_patch(patches.Rectangle((-50, 0), 50, road_width, color='#444444', zorder=0))  
    ax.add_patch(patches.Rectangle((12, 0), 50, road_width, color='#444444', zorder=0))   
    ax.add_patch(patches.Rectangle((0, -50), road_width, 50, color='#444444', zorder=0))  
    ax.add_patch(patches.Rectangle((0, 12), road_width, 50, color='#444444', zorder=0))   

    ax.plot([-50, 0], [6, 6], color='#e6c800', linestyle='--', zorder=2)
    ax.plot([12, 50], [6, 6], color='#e6c800', linestyle='--', zorder=2)
    ax.plot([6, 6], [-50, 0], color='#e6c800', linestyle='--', zorder=2)
    ax.plot([6, 6], [12, 50], color='#e6c800', linestyle='--', zorder=2)

    ax.plot([0, 0], [0, 6], color='white', linewidth=4, zorder=2)       
    ax.plot([12, 12], [6, 12], color='white', linewidth=4, zorder=2)    
    ax.plot([6, 12], [0, 0], color='white', linewidth=4, zorder=2)      
    ax.plot([0, 6], [12, 12], color='white', linewidth=4, zorder=2)     

    STATE_COLORS = {
        "YIELD": "#e74c3c",
        "GO": "#2ecc71"
    }

    car_length = 4.5
    car_width = 2.0
    car_patches = {}
    car_texts = {}

    for car in cars:
        rect = patches.Rectangle((0, 0), car_length, car_width, color='white', zorder=3)
        ax.add_patch(rect)
        car_patches[car.id] = rect
        
        text = ax.text(0, 0, f"V{car.id}", color='black', weight='bold', fontsize=8,
                       ha='center', va='center', zorder=4)
        car_texts[car.id] = text

    # Telemetry coordinates mapped exactly to top-left quadrant
    time_text = ax.text(-40, 40, '', color='white', fontsize=14, fontfamily='monospace')
    telem_text = ax.text(-40, 32, '', color='white', fontsize=10, fontfamily='monospace', va='top')

    # Fetch DT from the global scope of the file
    global DT

    def update(frame_idx):
        current_time = frame_idx * DT
        time_text.set_text(f"Simulation Time: {current_time:05.2f} s")
        telem_lines = ["TELEMETRY:"]
        
        for car in sorted(cars, key=lambda c: c.id):
            vid = car.id
            approach = car.origin
            
            if frame_idx < len(car.x_history):
                x = car.x_history[frame_idx]
                y = car.y_history[frame_idx]
                v = car.vx_history[frame_idx]
                state = car.state_history[frame_idx]
            else:
                state = "DONE"
                x = car.x_history[-1] if car.x_history else 0
                y = car.y_history[-1] if car.y_history else 0
                v = 0.0

            cx, cy = position_to_xy(approach, x, y)
            rect = car_patches[vid]

            if state != "DONE":
                telem_lines.append(f"V{vid}({approach}): {state:<8} | x={x:>5.1f} | y={y:>4.1f} | v={v:>4.1f}")
            
            if state == "DONE" or cx < -45 or cx > 45 or cy < -45 or cy > 45:
                rect.set_alpha(0.0)
                car_texts[vid].set_alpha(0.0)
            else:
                rect.set_alpha(1.0)
                car_texts[vid].set_alpha(1.0)
                rect.set_facecolor(STATE_COLORS.get(state, "white"))
                
                if approach == "W":
                    rect.set_width(car_length)
                    rect.set_height(car_width)
                    rect.set_xy((cx - car_length, cy - car_width/2))
                    car_texts[vid].set_position((cx - car_length/2, cy))
                elif approach == "E":
                    rect.set_width(car_length)
                    rect.set_height(car_width)
                    rect.set_xy((cx, cy - car_width/2))
                    car_texts[vid].set_position((cx + car_length/2, cy))
                elif approach == "S":
                    rect.set_width(car_width)
                    rect.set_height(car_length)
                    rect.set_xy((cx - car_width/2, cy - car_length))
                    car_texts[vid].set_position((cx, cy - car_length/2))
                elif approach == "N":
                    rect.set_width(car_width)
                    rect.set_height(car_length)
                    rect.set_xy((cx - car_width/2, cy))
                    car_texts[vid].set_position((cx, cy + car_length/2))

        telem_text.set_text("\n".join(telem_lines))
        return list(car_patches.values()) + list(car_texts.values()) + [time_text, telem_text]

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=200, blit=True)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. MAIN
# =============================================================================

if __name__ == "__main__":
    cars, controller = run_simulation()
    print_summary(cars, controller)
    plot_results(cars)
    animate_results(cars)
