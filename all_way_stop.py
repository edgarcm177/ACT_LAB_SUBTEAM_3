"""
Simple MPC + Trajectory-Conflict Controller for an All-Way-Stop Intersection
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches 
from scipy.optimize import minimize


# =============================================================================
# 1. PARAMETERS
# =============================================================================

DT = 0.20                 # sampling time [s]
SIM_TIME = 35.0           # maximum simulation time [s]
N = 20                    # prediction horizon
V_DES = 8.0               # desired speed in GO [m/s]

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

MAXITER = 60

LANE_OFFSET = 3.0         # lane centre offset from the road edge [m]
CONFLICT_CLEARANCE = 2.5  # paths closer than this are in conflict [m]
NEARBY_RADIUS = 40.0      # cars within this distance are tracked [m]
CAR_HALF_WIDTH = 1.0      # used for obstacle clearance [m]
CAR_LENGTH = 4.5          # [m]

# Each car body is approximated by circles placed from the front bumper
# (position p) back to the rear bumper.
CIRCLE_OFFSETS = np.linspace(0.0, CAR_LENGTH, 3)
COLLISION_DIST = 2.0 * CAR_HALF_WIDTH  # circle centres closer than this = collision [m]
CAR_BUFFER = 1.0                        # clear space kept between car bodies [m]
SAFE_DIST = COLLISION_DIST + CAR_BUFFER # circle centres closer than this are penalised [m]

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


# =============================================================================
# GEOMETRY
# =============================================================================

def position_to_xy(origin, p):
    """Map a 1-D position along a car's lane to intersection (x, y) coordinates."""
    if origin == "W": return (p, LANE_OFFSET)
    if origin == "E": return (INTERSECTION_EXIT - p, INTERSECTION_EXIT - LANE_OFFSET)
    if origin == "S": return (INTERSECTION_EXIT - LANE_OFFSET, p)
    if origin == "N": return (LANE_OFFSET, INTERSECTION_EXIT - p)
    return (0.0, 0.0)


def body_points(origin, p):
    """Circle centres of a car body for one or more positions -> shape (K, C, 2)."""
    p = np.atleast_1d(p)[:, None] - CIRCLE_OFFSETS[None, :]
    x, y = position_to_xy(origin, p)
    return np.stack(np.broadcast_arrays(x, y), axis=-1)


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

        self.p = position
        self.v = speed

        self.state = "APPROACH"

        self.stop_time = None
        self.go_time = None

        # Initial guesses for the two optimization problems.
        # Each guess contains N acceleration values.
        self.stop_guess = np.zeros(N)
        self.go_guess = np.zeros(N)

        # Predicted positions over the horizon, shared with other cars so
        # they can penalise collisions (constant-speed guess to start).
        self.predicted_p = position + speed * DT * np.arange(N + 1)

        # Data saved for plots.
        self.time_history = []
        self.position_history = []
        self.speed_history = []
        self.acceleration_history = []
        self.state_history = []


# =============================================================================
# 3. CONTROLLER
# =============================================================================

class Controller:
    def __init__(self):
        self.active_cars = []
        self.collisions = []      # (time, id1, id2) of detected collisions

    # -------------------------------------------------------------------------
    # Vehicle prediction model
    # -------------------------------------------------------------------------
    def predict(self, p0, v0, acceleration_sequence):

        p = np.zeros(N + 1)
        v = np.zeros(N + 1)

        p[0] = p0
        v[0] = v0

        for k in range(N):
            a = acceleration_sequence[k]

            p[k + 1] = p[k] + v[k] * DT + 0.5 * a * DT**2
            v[k + 1] = v[k] + a * DT

        return p, v

    # -------------------------------------------------------------------------
    # Problem 1: STOP
    # -------------------------------------------------------------------------
    def collision_cost(self, car, p, others):
        """
        J_collision = W_COLLISION * sum over conflicting cars, horizon steps
                      and body circles of max(SAFE_DIST - distance, 0)^2
        """

        if not others:
            return 0.0

        mine = body_points(car.origin, p[1:])
        cost = 0.0
        for other in others:
            theirs = body_points(other.origin, other.predicted_p[1:])
            cost += W_COLLISION * overlap_penalty(mine, theirs, SAFE_DIST)

        return cost

    def conflicting_cars(self, car, cars):
        return [other for other in cars if other.id in car.conflicts]

    def stop_cost(self, acceleration_sequence, car, others):
        """
        J_stop = W_POSITION * p(N)^2
               + W_SPEED    * v(N)^2
               + sum[ W_ACCEL * a(k)^2
                    + W_CROSS * max(p(k+1), 0)^2 ]
               + J_collision
        """

        p, v = self.predict(car.p, car.v, acceleration_sequence)

        cost = W_POSITION * p[N]**2
        cost += W_SPEED * v[N]**2

        for k in range(N):
            cost += W_ACCEL * acceleration_sequence[k]**2
            cost += W_CROSS * max(p[k + 1], 0.0)**2

        cost += self.collision_cost(car, p, others)

        return cost

    def solve_stop_problem(self, car, cars):
        """Solve the STOP problem and return the first optimal acceleration."""

        result = minimize(
            self.stop_cost,
            car.stop_guess,
            args=(car, self.conflicting_cars(car, cars)),
            method="BFGS",
            options={"maxiter": MAXITER, "disp": False},
        )

        optimal_sequence = result.x
        car.stop_guess = optimal_sequence

        # MPC: apply only the first input.
        return optimal_sequence[0]

    # -------------------------------------------------------------------------
    # Problem 2: GO
    # -------------------------------------------------------------------------
    def go_cost(self, acceleration_sequence, car, others):
        """
        J_go = sum[ W_SPEED * (v(k+1) - V_DES)^2
                  + W_ACCEL * a(k)^2 ]
             + J_collision
        """

        p, v = self.predict(car.p, car.v, acceleration_sequence)

        cost = 0.0

        for k in range(N):
            cost += W_SPEED * (v[k + 1] - V_DES)**2
            cost += W_ACCEL * acceleration_sequence[k]**2

        cost += self.collision_cost(car, p, others)

        return cost

    def solve_go_problem(self, car, cars):
        """Solve the GO problem and return the first optimal acceleration."""

        result = minimize(
            self.go_cost,
            car.go_guess,
            args=(car, self.conflicting_cars(car, cars)),
            method="BFGS",
            options={"maxiter": MAXITER, "disp": False},
        )

        optimal_sequence = result.x
        car.go_guess = optimal_sequence

        # MPC: apply only the first input.
        return optimal_sequence[0]

    # -------------------------------------------------------------------------
    # Trajectory-conflict logic
    # -------------------------------------------------------------------------
    def update_awareness(self, cars, obstacles):
        """Refresh each car's nearby cars, trajectory conflicts and obstacles."""

        live = [car for car in cars if car.state != "DONE"]

        for car in live:
            here = np.array(position_to_xy(car.origin, car.p))

            car.nearby_cars = {}
            car.conflicts = []
            for other in live:
                if other.id == car.id:
                    continue
                there = np.array(position_to_xy(other.origin, other.p))
                if np.linalg.norm(here - there) > NEARBY_RADIUS:
                    continue

                car.nearby_cars[other.id] = other.path
                if path_distance(car.path, other.path) < CONFLICT_CLEARANCE:
                    car.conflicts.append(other.id)

            car.obstacles = [obs for obs in obstacles if obs.blocks(car.path)]

    def register_stopped_car(self, car, time):
        """Mark a car that has come to rest at the stop line as waiting."""

        car.stop_time = time
        car.state = "WAIT"

    def release_cars(self, cars, time):
        """
        Let a waiting car go only when the constraint does not apply:
        its path is not blocked by an obstacle and it does not conflict
        with any car that is already crossing.
        """

        for car in cars:
            if car.state != "WAIT":
                continue

            if car.obstacles:
                continue

            if any(other_id in self.active_cars for other_id in car.conflicts):
                continue

            car.state = "GO"
            car.go_time = time
            self.active_cars.append(car.id)

    # -------------------------------------------------------------------------
    # Collision handling
    # -------------------------------------------------------------------------
    def update_prediction(self, car):
        """Shift the car's latest plan forward one step for the other cars to use."""

        if car.state == "WAIT":
            car.predicted_p = np.full(N + 1, STOP_LINE)
            return

        plan = car.go_guess if car.state == "GO" else car.stop_guess
        shifted = np.append(plan[1:], plan[-1])
        car.predicted_p, _ = self.predict(car.p, car.v, shifted)

    def detect_collisions(self, cars, time):
        """Record any pair of cars whose bodies actually overlap."""

        live = [car for car in cars if car.state != "DONE"]
        for i, car in enumerate(live):
            for other in live[i + 1:]:
                d = np.linalg.norm(
                    body_points(car.origin, car.p)[0][:, None, :]
                    - body_points(other.origin, other.p)[0][None, :, :],
                    axis=-1,
                )
                if np.min(d) < COLLISION_DIST:
                    self.collisions.append((time, car.id, other.id))

    def check_intersection(self, cars):
        """
        Free the intersection once the active car's rear bumper is past the
        exit point by at least CAR_BUFFER.
        """

        if len(self.active_cars) == 0:
            return

        # Iterate backwards to safely remove items from the list during the loop
        for i in range(len(self.active_cars) - 1, -1, -1):
            current_id = self.active_cars[i]
            
            for car in cars:
                if car.id == current_id:
                    if car.p - CAR_LENGTH >= INTERSECTION_EXIT + CAR_BUFFER:
                        self.active_cars.pop(i)
                    break


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
        # A. Check whether APPROACH cars have stopped.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state != "APPROACH":
                continue

            close_to_stop_line = abs(car.p - STOP_LINE) <= STOP_POS_TOL
            almost_stopped = car.v <= STOP_SPEED_TOL

            if close_to_stop_line and almost_stopped:
                car.p = STOP_LINE
                car.v = 0.0
                controller.register_stopped_car(car, time)

        # ---------------------------------------------------------------------
        # B. Update trajectory conflicts/obstacles and release any waiting
        #    car whose path is clear.
        # ---------------------------------------------------------------------
        controller.update_awareness(cars, obstacles)
        controller.release_cars(cars, time)

        # ---------------------------------------------------------------------
        # C. Compute one acceleration to apply to each car.
        # ---------------------------------------------------------------------
        accelerations = {}

        for car in cars:
            if car.state == "APPROACH":
                accelerations[car.id] = controller.solve_stop_problem(car, cars)

            elif car.state == "WAIT":
                accelerations[car.id] = 0.0

            elif car.state == "GO":
                accelerations[car.id] = controller.solve_go_problem(car, cars)

            else:  # DONE
                accelerations[car.id] = 0.0

        # ---------------------------------------------------------------------
        # D. Save the current state and move each car one simulation step.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state == "DONE":
                continue

            a = accelerations[car.id]

            car.time_history.append(time)
            car.position_history.append(car.p)
            car.speed_history.append(car.v)
            car.acceleration_history.append(a)
            car.state_history.append(car.state)

            if car.state == "WAIT":
                car.p = STOP_LINE
                car.v = 0.0

            else:
                car.p = car.p + car.v * DT + 0.5 * a * DT**2
                car.v = car.v + a * DT

            controller.update_prediction(car)

        controller.detect_collisions(cars, time)

        # ---------------------------------------------------------------------
        # E. Check whether the active car has cleared the intersection.
        # ---------------------------------------------------------------------
        controller.check_intersection(cars)

        # ---------------------------------------------------------------------
        # F. Remove cars after the downstream deletion point.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state == "GO" and car.p >= DELETE_POINT:
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
        cars, key=lambda car: car.go_time if car.go_time is not None else float("inf")
    )

    for car in ordered_cars:
        stop = f"{car.stop_time:.1f} s" if car.stop_time is not None else "-"
        go = f"{car.go_time:.1f} s" if car.go_time is not None else "never"
        conflicts = ", ".join(f"V{i}" for i in car.conflicts) or "none"
        print(
            f"Vehicle {car.id} ({car.origin} -> {car.destination}): "
            f"stop time = {stop}, go time = {go}, conflicts with {conflicts}"
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
            car.position_history,
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
            car.speed_history,
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
    ax.set_title("All-Way Stop MPC Controller", color='white', pad=20, fontsize=16)

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
        "APPROACH": "#f39c12",  
        "WAIT": "#e74c3c",      
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
            
            if frame_idx < len(car.position_history):
                p = car.position_history[frame_idx]
                v = car.speed_history[frame_idx]
                state = car.state_history[frame_idx]
            else:
                state = "DONE"
                p = car.position_history[-1] if car.position_history else 0
                v = 0.0
                
            cx, cy = position_to_xy(approach, p)
            rect = car_patches[vid]
            
            if state != "DONE":
                telem_lines.append(f"V{vid}({approach}): {state:<8} | p={p:>5.1f} | v={v:>4.1f}")
            
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
