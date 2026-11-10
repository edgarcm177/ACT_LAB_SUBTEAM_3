"""
Two-axis MPC + Trajectory-Conflict Controller for an All-Way-Stop Intersection
============================================================================
State: forward position/speed plus lateral position/speed. MPC chooses two
accelerations each step, which rotate into world x/y according to origin.
This is a translating-body double-integrator simulation, not a steering/bicycle
model. Route progress follows straight lines or quarter-circle turns, with
lateral motion relative to the route. Ellipses rotate with the route tangent.
This is a route-following kinematic model, not a steering/tire model.
Collision avoidance is a soft penalty, not a safety guarantee.
Use car.request_lane_change(target_lateral) to request a lane change.
"""

from functools import lru_cache
import warnings
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches 
from scipy.optimize import minimize


# =============================================================================
# 1. PARAMETERS
# =============================================================================

DT = 0.20                 # sampling time [s]
SIM_TIME = 180.0           # maximum simulation time [s]
N = 20                    # prediction horizon
V_DES = 8.0               # desired speed in GO [m/s]

STOP_LINE = 0.0           # stop line position [m]
INTERSECTION_EXIT = 12.0  # road width / straight-route length [m]
DELETE_POINT = 25.0       # remove vehicle after this point [m]

STOP_POS_TOL = 0.50       # stopping position tolerance [m]
STOP_SPEED_TOL = 0.20     # stopping speed tolerance [m/s]

# Objective-function weights from the report
W_POSITION = 100.0
W_SPEED = 150.0
W_ACCEL = 0.20
W_CROSS = 2000.0
W_COLLISION = 1.0e5      # penalty for predicted car-to-car overlap
W_CONFLICT_POINT = 3.0e5  # penalty for simultaneous use of a crossing point
CONFLICT_POINT_HALF_LENGTH = 2.25 + 0.75
CONFLICT_POINT_SAMPLE = 0.15

MAXITER = 150
RETRY_MAXITER = 400
SOLVER_FTOL = 1.0e-4
RANDOM_SEED = 7          # change for another repeatable sequence; None = fresh each run
TURN_CHOICES = ("left", "straight", "right")
TURN_PROBABILITIES = (1/3, 1/3, 1/3)
TURN_OFFSET_LIMIT = 2.0  # match the road-bounded GO offset; route conflict checks include this allowance
ROUTE_SAMPLE_STEP = 0.35
TOTAL_CARS = 8
SPAWN_INTERVAL = 2.5      # second wave: W/E/S/N [s]
SPAWN_START = 6.0         # arrivals after the initial four cars [s]
SPAWN_POSITION = -42.0
SPAWN_SPEED = 4.0
QUEUE_GAP = 2.0           # bumper-to-bumper gap to a same-approach leader [m]
W_LANE = 100.0            # lateral target tracking
W_LATERAL_SPEED = 80.0    # settle sideways velocity after changing lanes
MAX_ACCEL = 3.0
MAX_BRAKE = 5.0
MAX_LATERAL_ACCEL = 2.0
# Each direction has 6 m of road width. Keep the 2 m body inside it.
MAX_LATERAL_OFFSET = 2.0
LANE_CENTRES = (-1.5, 1.5) # two 3 m lanes within each directional half-road

LANE_OFFSET = 3.0         # reference centre of each 6 m directional road half [m]
CONFLICT_CLEARANCE = 2.5  # paths closer than this are in conflict [m]
NEARBY_RADIUS = 40.0      # cars within this distance are tracked [m]
CAR_HALF_WIDTH = 1.0      # used for obstacle clearance [m]
CAR_LENGTH = 4.5          # [m]

# Physical ellipse: length 4.5 m, width 2.0 m. p remains the front tip,
# so the ellipse centre is half a car length behind p.
CAR_SEMI_MAJOR = CAR_LENGTH / 2.0
CAR_SEMI_MINOR = CAR_HALF_WIDTH
CAR_BUFFER = 1.0          # pairwise axis padding for predictive avoidance [m]
# Each predictive ellipse gains CAR_BUFFER / 2 on each semi-axis. This
# is an axis-based safety envelope, not an exact constant-distance offset.
ELLIPSE_CONTACT_ITERATIONS = 28
ELLIPSE_CONTACT_TOL = 1.0e-8

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


# =============================================================================
# GEOMETRY
# =============================================================================

def position_to_xy(origin, p, lateral=0.0):
    """Map forward progress and leftward offset to world (x, y)."""
    if origin == "W": return (p, LANE_OFFSET + lateral)
    if origin == "E": return (INTERSECTION_EXIT - p, INTERSECTION_EXIT - LANE_OFFSET - lateral)
    if origin == "S": return (INTERSECTION_EXIT - LANE_OFFSET - lateral, p)
    if origin == "N": return (LANE_OFFSET + lateral, INTERSECTION_EXIT - p)
    return (0.0, 0.0)


LEFT = {"W": "N", "N": "E", "E": "S", "S": "W"}
RIGHT = {"W": "S", "S": "E", "E": "N", "N": "W"}


def select_destination(origin, rng):
    """Choose once per arrival. Supplying an RNG makes runs reproducible."""
    turn = str(rng.choice(TURN_CHOICES, p=TURN_PROBABILITIES))
    return {"left": LEFT, "straight": OPPOSITE, "right": RIGHT}[turn][origin]


class Route:
    """Arc-length path for a front-tip reference, p=0 at the stop line.

    Build the W approach, then rotate it about (6, 6) for other approaches.
    Right turns use a 3 m radius, left turns a 9 m radius. Both join the
    correct outgoing lane tangentially. Curvature changes at arc boundaries.
    """
    def __init__(self, origin, destination):
        if destination not in (LEFT[origin], OPPOSITE[origin], RIGHT[origin]):
            raise ValueError("Destination must be left, straight or right; no U-turns")
        self.origin, self.destination = origin, destination
        self.turn = "left" if destination == LEFT[origin] else "right" if destination == RIGHT[origin] else "straight"
        self.radius = 9.0 if self.turn == "left" else 3.0
        self.exit_p = INTERSECTION_EXIT if self.turn == "straight" else self.radius*np.pi/2
        self.rotation = {"W": 0, "S": np.pi/2, "E": np.pi, "N": -np.pi/2}[origin]
        c, sn = np.cos(self.rotation), np.sin(self.rotation)
        self.matrix = np.array([[c, -sn], [sn, c]])
        self.desired_speed = {"straight": V_DES, "left": 4.0, "right": 2.0}[self.turn]

    def frame(self, progress):
        p = np.asarray(progress, dtype=float)
        if self.turn == "straight":
            point = np.stack((p, np.full_like(p, 3.0)), axis=-1)
            heading = np.zeros_like(p)
            curvature = np.zeros_like(p)
        else:
            sign = 1.0 if self.turn == "left" else -1.0
            arc = np.clip(p, 0, self.exit_p)
            angle = arc/self.radius
            x = self.radius*np.sin(angle)
            y = 3.0 + sign*self.radius*(1-np.cos(angle))
            x = np.where(p < 0, p, x)
            y = y + sign*np.maximum(p-self.exit_p, 0)
            point = np.stack((x, y), axis=-1)
            heading = sign*angle
            curvature = np.where((p > 0) & (p < self.exit_p), sign/self.radius, 0.0)
        point = (point - 6) @ self.matrix.T + 6
        heading = heading + self.rotation
        tangent = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
        normal = np.stack((-np.sin(heading), np.cos(heading)), axis=-1)
        return point, tangent, normal, heading, curvature

    def xy(self, progress, lateral=0.0):
        point, _, normal, _, _ = self.frame(progress)
        return point + np.asarray(lateral)[..., None]*normal

    def body(self, progress, lateral=0.0, padding=0.0, derivatives=False):
        point, tangent, normal, heading, curvature = self.frame(progress)
        lateral = np.asarray(lateral)
        centre = point + lateral[..., None]*normal - CAR_SEMI_MAJOR*tangent
        a2, b2 = (CAR_SEMI_MAJOR+padding)**2, (CAR_SEMI_MINOR+padding)**2
        tt = tangent[..., :, None]*tangent[..., None, :]
        nn = normal[..., :, None]*normal[..., None, :]
        shape = a2*tt + b2*nn
        if not derivatives:
            return centre, shape
        centre_p = (1-curvature*lateral)[..., None]*tangent - (CAR_SEMI_MAJOR*curvature)[..., None]*normal
        shape_p = (a2-b2)*curvature[..., None, None]*(
            normal[..., :, None]*tangent[..., None, :] + tangent[..., :, None]*normal[..., None, :])
        return centre, shape, centre_p, normal, shape_p


def ellipse_geometry(origin, p, lateral=0.0, padding=0.0):
    """Compatibility helper for a straight route; returns centre and shape matrix."""
    return Route(origin, OPPOSITE[origin]).body(p, lateral, padding)


def _solve_shape(shape, vector):
    """Vectorized inverse of a positive definite 2x2 shape matrix."""
    xx, xy, yy = shape[..., 0, 0], shape[..., 0, 1], shape[..., 1, 1]
    determinant = xx*yy-xy*xy
    return np.stack((yy*vector[..., 0]-xy*vector[..., 1],
                     xx*vector[..., 1]-xy*vector[..., 0]), axis=-1)/determinant[..., None]


def ellipse_contact_value(centre1, shape1, centre2, shape2, with_gradient=False):
    """Rotated ellipse contact: <1 overlap, =1 touch, >1 separated.

    Shape matrices are R @ diag(a*a,b*b) @ R.T. Gradients are with respect
    to delta=centre2-centre1 and the first shape matrix, using the envelope
    derivative at the maximizing interpolation parameter.
    """
    delta = np.asarray(centre2)-np.asarray(centre1)
    shape1, shape2 = np.asarray(shape1), np.asarray(shape2)
    change = shape2-shape1
    lower = np.zeros(delta.shape[:-1]); upper = np.ones_like(lower)
    for _ in range(ELLIPSE_CONTACT_ITERATIONS):
        t = (lower+upper)*0.5
        q = shape1 + t[..., None, None]*change
        z = _solve_shape(q, delta)
        value = np.sum(delta*z, axis=-1)
        slope = np.einsum('...i,...ij,...j->...', z, change, z)
        derivative = (1-2*t)*value-t*(1-t)*slope
        lower = np.where(derivative > 0, t, lower)
        upper = np.where(derivative > 0, upper, t)
    t = (lower+upper)*0.5
    z = _solve_shape(shape1+t[..., None, None]*change, delta)
    weight = t*(1-t)
    value = weight*np.sum(delta*z, axis=-1)
    if with_gradient:
        grad_delta = 2*weight[..., None]*z
        grad_shape = -(weight*(1-t))[..., None, None]*z[..., :, None]*z[..., None, :]
        return value, grad_delta, grad_shape
    return value


def ellipse_overlap_penalty(centre1, shape1, centre2, shape2, with_gradient=False):
    first, second = np.broadcast_arrays(np.asarray(centre1), np.asarray(centre2))
    q1 = np.broadcast_to(shape1, first.shape[:-1]+(2, 2))
    q2 = np.broadcast_to(shape2, first.shape[:-1]+(2, 2))
    # Trace is an inexpensive conservative bound on the largest eigenvalue.
    radius = np.sqrt(np.trace(q1, axis1=-2, axis2=-1))+np.sqrt(np.trace(q2, axis1=-2, axis2=-1))
    possible = np.sum((second-first)**2, axis=-1) < radius**2
    grad_c, grad_q = np.zeros_like(first), np.zeros_like(q1)
    if not np.any(possible):
        return (0.0, grad_c, grad_q) if with_gradient else 0.0
    contact, dc, dq = ellipse_contact_value(first[possible], q1[possible], second[possible], q2[possible], True)
    intrusion = np.maximum(1-contact, 0)
    grad_c[possible] = 2*intrusion[..., None]*dc
    grad_q[possible] = -2*intrusion[..., None, None]*dq
    value = np.sum(intrusion**2)
    return (value, grad_c, grad_q) if with_gradient else value


@lru_cache(maxsize=144)
def routes_conflict(origin1, destination1, origin2, destination2):
    """Reserve the complete swept route, including its rear-clearance zone.

    Shared approaches/exits conflict. For other routes, compare sampled
    rotated, padded ellipses. Extra padding covers the sample spacing and
    the small allowed GO lateral deviation. This is conservative route
    reservation, not time-slot scheduling at individual conflict points.
    """
    if origin1 == origin2 or destination1 == destination2:
        return True
    route1, route2 = Route(origin1, destination1), Route(origin2, destination2)
    padding = CAR_BUFFER/2 + TURN_OFFSET_LIMIT + ROUTE_SAMPLE_STEP
    samples = []
    for route in (route1, route2):
        end = route.exit_p + CAR_LENGTH + CAR_BUFFER
        progress = np.linspace(0, end, int(np.ceil(end/ROUTE_SAMPLE_STEP))+1)
        samples.append(route.body(progress, padding=padding))
    c1, q1 = samples[0]; c2, q2 = samples[1]
    return bool(ellipse_overlap_penalty(c1[:, None, :], q1[:, None, :, :],
                                      c2[None, :, :], q2[None, :, :, :]) > 0)



ROUTE_KEYS = tuple((origin, destination) for origin in "WESN"
                   for destination in (LEFT[origin], OPPOSITE[origin], RIGHT[origin]))


def segment_intersection_point(a, b, c, d):
    """Return a proper 2-D segment intersection, or None for parallel lines."""
    r, s = np.asarray(b)-a, np.asarray(d)-c
    denominator = r[0]*s[1]-r[1]*s[0]
    if abs(denominator) < 1e-10:
        return None
    delta = np.asarray(c)-a
    t = (delta[0]*s[1]-delta[1]*s[0])/denominator
    u = (delta[0]*r[1]-delta[1]*r[0])/denominator
    if 1e-6 < t < 1-1e-6 and 1e-6 < u < 1-1e-6:
        return np.asarray(a)+t*r, t, u
    return None


@lru_cache(maxsize=1)
def build_conflict_points():
    """Find route crossings and progress coordinates using vectorized samples."""
    routes = {key: Route(*key) for key in ROUTE_KEYS}
    samples = {}
    for key, route in routes.items():
        p = np.arange(-3.0, route.exit_p + CAR_LENGTH + 3.0, CONFLICT_POINT_SAMPLE)
        xy, tangent, _, _, _ = route.frame(p)
        samples[key] = (p, xy, tangent)
    points = []
    for i,key1 in enumerate(ROUTE_KEYS):
        p1,xy1,t1=samples[key1]
        for key2 in ROUTE_KEYS[i+1:]:
            if key1[0] == key2[0]:
                continue
            p2,xy2,t2=samples[key2]
            delta=xy1[:,None,:]-xy2[None,:,:]
            distance=np.linalg.norm(delta,axis=-1)
            crossing_angle=np.abs(t1[:,None,0]*t2[None,:,1]-t1[:,None,1]*t2[None,:,0])
            rows,cols=np.where((distance<0.22)&(crossing_angle>0.2))
            for a,b in zip(rows,cols):
                point=(xy1[a]+xy2[b])*0.5
                if not (0 <= point[0] <= 12 and 0 <= point[1] <= 12):
                    continue
                existing=next((cp for cp in points if np.linalg.norm(cp["xy"]-point)<1.5),None)
                if existing is None:
                    existing={"id":f"CP{len(points)+1}","xy":point,"routes":{}}
                    points.append(existing)
                existing["routes"][key1]=float(p1[a])
                existing["routes"][key2]=float(p2[b])
    return tuple(points)


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
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Obstacle radius must be positive and finite")
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
    def __init__(self, car_id, origin, position, speed, destination=None,
                 lateral=0.0, target_lateral=None):
        self.id = car_id

        # Route information
        self.origin = origin
        self.destination = destination or OPPOSITE[origin]
        self.route = Route(origin, self.destination)
        self.turn = self.route.turn
        self.direction = self.destination
        self.path = self.route.xy(np.linspace(0, self.route.exit_p, 41))
        self.delete_p = self.route.exit_p + (DELETE_POINT-INTERSECTION_EXIT)
        self.conflict_points = []
        self.conflict_point_progress = {}
        self.leader_progress = None
        self.wait_reason = ""
        self.wait_reason_history = []

        # Awareness of the surroundings, refreshed every step
        self.nearby_cars = {}                       # id -> path of nearby cars
        self.conflicts = []                         # ids of cars whose paths conflict
        self.obstacles = []                         # obstacles on or near the path

        self.p = position
        self.v = speed
        self.lateral = float(lateral)
        self.lateral_speed = 0.0
        self.target_lateral = self.lateral
        self.request_lane_change(lateral if target_lateral is None else target_lateral)
        if not np.isfinite(lateral) or abs(lateral) > MAX_LATERAL_OFFSET:
            raise ValueError("Initial lateral position is outside the road")

        self.state = "APPROACH"
        self.stop_target = STOP_LINE
        self.spawn_time = 0.0

        self.stop_time = None
        self.go_time = None

        # Initial guesses for the two optimization problems.
        # Each guess contains N pairs: [forward acceleration, leftward acceleration].
        self.stop_guess = np.zeros((N, 2))
        self.go_guess = np.zeros((N, 2))

        # Predicted positions over the horizon, shared with other cars so
        # they can penalise collisions (constant-speed guess to start).
        self.predicted_p = position + speed * DT * np.arange(N + 1)

        self.predicted_lateral = np.full(N + 1, self.lateral)

        # Data saved for plots.
        self.time_history = []
        self.position_history = []
        self.speed_history = []
        self.acceleration_history = []
        self.state_history = []
        self.lateral_history = []
        self.xy_history = []
        self.xy_acceleration_history = []

    def request_lane_change(self, target_lateral):
        """Request a leftward offset in metres, relative to the original centreline.

        Use either value in LANE_CENTRES for the two same-direction lanes.
        Requests are tracked upstream of p=-10 m. Cars then align with the
        reserved centreline before stopping; GO holds that route.
        This specifies a target, not an automatic gap-selection policy.
        """
        if not np.isfinite(target_lateral) or abs(target_lateral) > MAX_LATERAL_OFFSET:
            raise ValueError("Target would put the car outside its half of the road")
        self.target_lateral = float(target_lateral)

    def vector_to_xy(self, forward, leftward):
        """Rotate components into the instantaneous route frame."""
        _, tangent, normal, _, _ = self.route.frame(self.p)
        return forward*tangent + leftward*normal

    def world_acceleration(self, acceleration):
        """Front-reference acceleration on constant-curvature route segments.

        Curvature changes discontinuously at turn entry/exit; this model does
        not impose steering-rate or tire-force dynamics at those joins.
        """
        _, tangent, normal, _, k = self.route.frame(self.p)
        forward = (1-k*self.lateral)*acceleration[0]-2*k*self.v*self.lateral_speed
        sideways = acceleration[1]+k*(1-k*self.lateral)*self.v**2
        return forward*tangent+sideways*normal



# =============================================================================
# 3. CONTROLLER
# =============================================================================

class Controller:
    def __init__(self):
        self.active_cars = []
        self.collisions = []      # (time, id1, id2) of detected collisions
        self.solver_events = []   # retries / feasible but nonconverged plans
        self.obstacle_collisions = []
        self.pending_arrivals = 0
        self.finished_time = 0.0
        self.release_events = []
        self.conflict_point_events = []
        self.conflict_points = build_conflict_points()

    # -------------------------------------------------------------------------
    # Vehicle prediction model
    # -------------------------------------------------------------------------
    def predict(self, p0, v0, acceleration_sequence):
        """Double-integrator prediction for one axis, used identically on both axes."""
        acceleration_sequence = np.asarray(acceleration_sequence)
        v = np.r_[v0, v0 + DT * np.cumsum(acceleration_sequence)]
        p = np.r_[p0, p0 + np.cumsum(v[:-1] * DT +
                                     0.5 * acceleration_sequence * DT**2)]
        return p, v

    def predict_motion(self, car, sequence):
        controls = np.asarray(sequence).reshape(N, 2)
        p, v = self.predict(car.p, car.v, controls[:, 0])
        lateral, lateral_speed = self.predict(
            car.lateral, car.lateral_speed, controls[:, 1])
        return p, v, lateral, lateral_speed

    def collision_cost(self, car, p, lateral, others, with_gradient=False):
        # Buffered ellipses discourage near misses; physical, unbuffered
        # ellipses are used separately by detect_collisions.
        padding = CAR_BUFFER / 2.0
        mine, my_shape, centre_p, centre_lateral, shape_p = car.route.body(
            p[1:], lateral[1:], padding, derivatives=True)
        cost = 0.0
        grad_c, grad_q = np.zeros_like(mine), np.zeros_like(my_shape)
        for other in others:
            theirs, their_shape = other.route.body(other.predicted_p[1:], other.predicted_lateral[1:], padding)
            value, dc, dq = ellipse_overlap_penalty(mine, my_shape, theirs, their_shape, True)
            cost += W_COLLISION*value
            grad_c += W_COLLISION*dc; grad_q += W_COLLISION*dq
        for obstacle in car.obstacles:
            shape = np.eye(2)*(obstacle.radius+padding)**2
            value, dc, dq = ellipse_overlap_penalty(mine, my_shape, (obstacle.x, obstacle.y), shape, True)
            cost += W_COLLISION*value
            grad_c += W_COLLISION*dc; grad_q += W_COLLISION*dq
        if with_gradient:
            gp = np.sum(grad_c*centre_p, axis=-1)+np.sum(grad_q*shape_p, axis=(-2, -1))
            gd = np.sum(grad_c*centre_lateral, axis=-1)
            return cost, gp, gd
        return cost

    def conflicting_cars(self, car, cars):
        # Check every nearby vehicle: a lane change can create a new conflict.
        return [other for other in cars if other.id in car.nearby_cars]

    def motion_cost(self, sequence, car, others, stopping, with_gradient=False):
        p, v, lateral, lateral_speed = self.predict_motion(car, sequence)
        controls = np.asarray(sequence).reshape(N, 2)
        # Lane-change requests apply upstream; align with the reserved route
        # for the stop and crossing, including user-created cars.
        lane_target = car.target_lateral if car.state == "APPROACH" and car.p < -10 else 0.0
        cost = W_ACCEL * np.sum(controls**2)
        cost += W_LANE * np.sum((lateral[1:] - lane_target)**2)
        cost += W_LATERAL_SPEED * np.sum(lateral_speed[1:]**2)
        if stopping:
            cost += W_POSITION * (p[-1] - car.stop_target)**2 + W_SPEED * v[-1]**2
            cost += W_CROSS * np.sum(np.maximum(p[1:] - STOP_LINE, 0)**2)
        else:
            cost += W_SPEED * np.sum((v[1:] - car.route.desired_speed)**2)
        collision, grad_p, grad_lateral = self.collision_cost(car, p, lateral, others, True)
        cost += collision
        cp_grad = np.zeros(N)
        for other in others:
            for point in car.conflict_points:
                if point["id"] not in other.conflict_point_progress:
                    continue
                my_cp = point["progress"]
                their_cp = other.conflict_point_progress[point["id"]]
                mine_occ = np.exp(-0.5*((p[1:]-my_cp)/CONFLICT_POINT_HALF_LENGTH)**4)
                their_occ = np.exp(-0.5*((other.predicted_p[1:]-their_cp)/CONFLICT_POINT_HALF_LENGTH)**4)
                simultaneous = mine_occ*their_occ
                cost += W_CONFLICT_POINT*np.sum(simultaneous)
                cp_grad += (W_CONFLICT_POINT*simultaneous*(-2*(p[1:]-my_cp)**3 /
                           CONFLICT_POINT_HALF_LENGTH**4))
        if not with_gradient:
            return cost / W_SPEED
        grad_p += cp_grad
        grad_lateral += 2*W_LANE*(lateral[1:] - lane_target)
        grad_lateral_speed = 2*W_LATERAL_SPEED*lateral_speed[1:]
        grad_v = np.zeros(N)
        if stopping:
            grad_p[-1] += 2*W_POSITION*(p[-1]-car.stop_target)
            grad_p += 2*W_CROSS*np.maximum(p[1:]-STOP_LINE, 0)
            grad_v[-1] = 2*W_SPEED*v[-1]
        else:
            grad_v = 2*W_SPEED*(v[1:]-car.route.desired_speed)
        # Linear prediction sensitivities: dp[k+1]/da[j] and dv[k+1]/da[j].
        index = np.arange(N)
        position_jac = DT**2*np.maximum(index[:, None]-index[None, :]+0.5, 0)
        velocity_jac = DT*np.tri(N)
        gradient = 2*W_ACCEL*controls
        gradient[:, 0] += position_jac.T @ grad_p + velocity_jac.T @ grad_v
        gradient[:, 1] += position_jac.T @ grad_lateral + velocity_jac.T @ grad_lateral_speed
        return cost / W_SPEED, gradient.ravel() / W_SPEED

    def stop_cost(self, sequence, car, others):
        return self.motion_cost(sequence, car, others, stopping=True)

    def go_cost(self, sequence, car, others):
        return self.motion_cost(sequence, car, others, stopping=False)

    def motion_constraints(self, sequence, car):
        p, v, lateral, _ = self.predict_motion(car, sequence)
        offset_limit = MAX_LATERAL_OFFSET  # same road boundary in APPROACH and GO; avoids sudden infeasibility
        limits = np.r_[v[1:], offset_limit - lateral[1:], offset_limit + lateral[1:]]
        if car.state == "APPROACH":
            # Keep a safe following distance while allowing uninterrupted approach.
            leaders = getattr(car, "leader_progress", None)
            if leaders is not None:
                limits = np.r_[limits, leaders - p[1:]]
        return limits

    def motion_constraint_jacobian(self, sequence, car):
        index = np.arange(N)
        position_jac = DT**2*np.maximum(index[:, None]-index[None, :]+0.5, 0)
        jac = np.zeros((4*N if car.state == "APPROACH" and getattr(car,"leader_progress",None) is not None else 3*N, 2*N))
        jac[:N, 0::2] = DT*np.tri(N)
        jac[N:2*N, 1::2] = -position_jac
        jac[2*N:3*N, 1::2] = position_jac
        if car.state == "APPROACH" and getattr(car,"leader_progress",None) is not None:
            jac[3*N:, 0::2] = -position_jac
        return jac

    def solve_problem(self, car, cars, stopping):
        guess = car.stop_guess if stopping else car.go_guess
        cost = self.stop_cost if stopping else self.go_cost
        others = self.conflicting_cars(car, cars)
        bounds = [(-MAX_BRAKE, MAX_ACCEL),
                  (-MAX_LATERAL_ACCEL, MAX_LATERAL_ACCEL)] * N

        def feasible(sequence):
            sequence = np.asarray(sequence)
            if sequence.size != 2*N or not np.all(np.isfinite(sequence)):
                return False
            controls = sequence.reshape(N, 2)
            return (np.all(controls[:, 0] >= -MAX_BRAKE - 1e-6)
                    and np.all(controls[:, 0] <= MAX_ACCEL + 1e-6)
                    and np.all(np.abs(controls[:, 1]) <= MAX_LATERAL_ACCEL + 1e-6)
                    and np.min(self.motion_constraints(sequence, car)) >= -1e-6
                    and np.isfinite(cost(sequence, car, others)))

        candidates = [guess.ravel().copy()] if feasible(guess) else []
        start = guess.ravel()
        for budget in (MAXITER, RETRY_MAXITER):
            result = minimize(
                lambda seq: self.motion_cost(seq, car, others, stopping, True),
                start, jac=True, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": self.motion_constraints,
                             "jac": self.motion_constraint_jacobian, "args": (car,)},
                options={"maxiter": budget, "ftol": SOLVER_FTOL if budget == MAXITER else 1e-9, "disp": False})
            valid = feasible(result.x)
            if valid:
                candidates.append(result.x.copy())
            if result.success and valid:
                break
            self.solver_events.append((car.id, str(result.message)))
            # Resume an unfinished finite iterate rather than restarting at zero.
            if np.all(np.isfinite(result.x)):
                start = result.x
        else:
            if not candidates:
                raise RuntimeError(
                    f"MPC could not find a feasible plan for vehicle {car.id}: "
                    f"{result.message}; p={car.p:.6f}, v={car.v:.6f}, "
                    f"stop target={car.stop_target:.6f}, "
                    f"min constraint={np.min(self.motion_constraints(result.x, car)):.3g}. "
                    "Increasing SIM_TIME cannot fix this.")
            warnings.warn(
                f"Vehicle {car.id}: no converged plan passed validation; using the best "
                "constraint-feasible candidate. Collision costs remain soft.",
                RuntimeWarning)
        # Include the prior feasible plan so a failed retry cannot replace it
        # with a worse candidate. Feasible does not imply collision-free.
        best = min(candidates, key=lambda seq: cost(seq, car, others))
        plan = best.reshape(N, 2)
        if stopping:
            car.stop_guess = plan
        else:
            car.go_guess = plan
        return plan[0].copy()

    def solve_stop_problem(self, car, cars):
        return self.solve_problem(car, cars, stopping=True)

    def solve_go_problem(self, car, cars):
        return self.solve_problem(car, cars, stopping=False)

    # -------------------------------------------------------------------------
    # Trajectory-conflict logic
    # -------------------------------------------------------------------------
    def update_awareness(self, cars, obstacles):
        """Refresh each car's nearby cars, trajectory conflicts and obstacles."""

        live = [car for car in cars if car.state != "DONE"]

        for car in live:
            leaders = [other for other in live if other.origin == car.origin
                       and other.id != car.id and other.p > car.p]
            car.stop_target = STOP_LINE
            car.leader_progress = (min((other.p-CAR_LENGTH-QUEUE_GAP for other in leaders),default=None))
            here = np.array(car.route.xy(car.p, car.lateral))

            car.nearby_cars = {}
            car.conflicts = []
            for other in live:
                if other.id == car.id:
                    continue
                there = np.array(other.route.xy(other.p, other.lateral))
                if np.linalg.norm(here - there) > NEARBY_RADIUS:
                    continue

                car.nearby_cars[other.id] = other.path
                if routes_conflict(car.origin, car.destination, other.origin, other.destination):
                    car.conflicts.append(other.id)
            car.conflict_points = [dict(id=cp["id"], progress=cp["routes"][
                (car.origin,car.destination)]) for cp in self.conflict_points
                if (car.origin,car.destination) in cp["routes"]]
            car.conflict_point_progress = {cp["id"]:cp["routes"][(car.origin,car.destination)]
                for cp in self.conflict_points if (car.origin,car.destination) in cp["routes"]}

            # All obstacles contribute to the 2-D MPC cost. Release checks below
            # retain the original stop-and-wait policy for a blocked route.
            car.obstacles = list(obstacles)

    def register_stopped_car(self, car, time):
        """Mark a car that has come to rest at the stop line as waiting."""

        car.stop_time = time
        car.state = "WAIT"

    def release_cars(self, cars, time):
        """Resume any exceptional WAIT car; normal coordination is predictive."""
        for car in cars:
            if car.state == "WAIT" and not any(obs.blocks(car.path) for obs in car.obstacles):
                car.state = "GO"
                car.go_time = time

    # -------------------------------------------------------------------------
    # Collision handling
    # -------------------------------------------------------------------------
    def update_prediction(self, car):
        """Shift the car's latest plan forward one step for the other cars to use."""

        if car.state == "WAIT":
            car.predicted_p = np.full(N + 1, car.p)
            car.predicted_lateral = np.full(N + 1, car.lateral)
            return

        plan = car.go_guess
        shifted = np.vstack((plan[1:], plan[-1]))
        car.predicted_p, _, car.predicted_lateral, _ = self.predict_motion(car, shifted)
        if car.state == "GO":
            car.go_guess = shifted
        else:
            car.stop_guess = shifted

    def detect_collisions(self, cars, time):
        """Record any pair of cars whose bodies actually overlap."""

        live = [car for car in cars if car.state != "DONE"]
        for i, car in enumerate(live):
            for other in live[i + 1:]:
                mine, my_axes = car.route.body(car.p, car.lateral)
                theirs, their_axes = other.route.body(other.p, other.lateral)
                contact = ellipse_contact_value(mine, my_axes, theirs, their_axes)
                if contact < 1.0 - ELLIPSE_CONTACT_TOL:
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
                    if car.p - CAR_LENGTH >= car.route.exit_p + CAR_BUFFER:
                        self.active_cars.pop(i)
                    break


# =============================================================================
# 4. DEMONSTRATION VEHICLES
# =============================================================================

def create_cars(rng=None):
    rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng
    return [Car(i+1, origin, position, speed,
                destination=select_destination(origin, rng),
                lateral=-1.5 if origin == "W" else 0.0, target_lateral=0.0)
            for i, (origin, position, speed) in enumerate(
                [("W", -24.0, 7.0), ("E", -30.0, 7.5),
                 ("S", -36.0, 8.0), ("N", -42.0, 8.0)])]


# =============================================================================
# 5. SIMULATION
# =============================================================================

def create_obstacles():
    # In vehicle 1's initial lane, before the stop line. Passing above it
    # requires positive world-y motion, followed by a return to its lane.
    return [Obstacle(-12.0, 1.3, 0.65)]


def run_simulation(obstacles=None, cars=None, spawn_traffic=None, progress=True, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    custom_cars = cars is not None
    cars = create_cars(rng) if cars is None else list(cars)
    if spawn_traffic is None:
        spawn_traffic = not custom_cars
    next_id = max((car.id for car in cars), default=0) + 1
    arrivals = [(SPAWN_START + i*SPAWN_INTERVAL, "WESN"[i % 4], select_destination("WESN"[i % 4], rng))
                for i in range(max(0, TOTAL_CARS-len(cars)))] if spawn_traffic else []
    controller = Controller()
    obstacles = create_obstacles() if obstacles is None else obstacles

    times = np.arange(0.0, SIM_TIME + DT, DT)

    for step, time in enumerate(times):
        # Defer a spawn until its entry corridor is clear; never insert a car
        # into an existing queue. Keep future arrivals even if the road empties.
        remaining = []
        for due, origin, destination in arrivals:
            clear = all(other.state == "DONE" or other.origin != origin or
                        other.p - SPAWN_POSITION >= CAR_LENGTH + QUEUE_GAP +
                        SPAWN_SPEED**2 / (2*MAX_BRAKE) + SPAWN_SPEED*DT
                        for other in cars)
            if due <= time and clear:
                lateral = LANE_CENTRES[0] if origin == "W" else 0.0
                car = Car(next_id, origin, SPAWN_POSITION, SPAWN_SPEED, destination=destination, lateral=lateral, target_lateral=0.0)
                car.spawn_time = time
                cars.append(car)
                next_id += 1
            else:
                remaining.append((due, origin, destination))
        arrivals = remaining
        controller.pending_arrivals = len(arrivals)
        controller.finished_time = time
        if progress and step % max(1, round(10.0 / DT)) == 0:
            live_count = sum(car.state != "DONE" for car in cars)
            print(f"Simulation {time:.0f}/{SIM_TIME:.0f}s: {live_count} active, "
                  f"{len(cars)} spawned, {len(arrivals)} pending", flush=True)

        # Cars keep moving through the stop line. The MPC coordinates crossing
        # point occupancy by comparing the communicated predicted trajectories.
        # ---------------------------------------------------------------------
        # B. Update trajectory conflicts/obstacles and release any waiting
        #    car whose path is clear.
        # ---------------------------------------------------------------------
        controller.update_awareness(cars, obstacles)
        controller.release_cars(cars, time)

        # ---------------------------------------------------------------------
        # C. Compute forward/lateral acceleration pairs for each car.
        # ---------------------------------------------------------------------
        accelerations = {}

        for car in cars:
            car.conflict_point_progress = {cp["id"]:cp["routes"][(car.origin,car.destination)]
                for cp in controller.conflict_points if (car.origin,car.destination) in cp["routes"]}
            car.conflict_points = [dict(id=k,progress=v) for k,v in car.conflict_point_progress.items()]
            if car.state == "APPROACH":
                accelerations[car.id] = controller.solve_go_problem(car, cars)

            elif car.state == "WAIT":
                accelerations[car.id] = np.zeros(2)

            elif car.state == "GO":
                accelerations[car.id] = controller.solve_go_problem(car, cars)

            else:  # DONE
                accelerations[car.id] = np.zeros(2)

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
            car.acceleration_history.append(a.copy())
            car.lateral_history.append(car.lateral)
            car.xy_history.append(car.route.xy(car.p, car.lateral))
            car.xy_acceleration_history.append(car.world_acceleration(a))
            car.state_history.append(car.state)
            car.wait_reason_history.append(car.wait_reason)

            if car.state == "WAIT":
                car.p = STOP_LINE
                car.v = 0.0
                car.lateral_speed = 0.0

            else:
                car.p = car.p + car.v * DT + 0.5 * a[0] * DT**2
                car.v = car.v + a[0] * DT
                car.lateral += car.lateral_speed * DT + 0.5 * a[1] * DT**2
                car.lateral_speed += a[1] * DT

            controller.update_prediction(car)

        controller.detect_collisions(cars, time + DT)
        for cp in controller.conflict_points:
            occupants = [car.id for car in cars if car.state != "DONE" and
                cp["id"] in car.conflict_point_progress and
                abs(car.p-car.conflict_point_progress[cp["id"]]) <= CONFLICT_POINT_HALF_LENGTH]
            if len(occupants) > 1:
                controller.conflict_point_events.append({"time": float(time+DT),
                    "point": cp["id"], "vehicles": tuple(occupants)})

        for car in cars:
            if car.state == "DONE":
                continue
            centre, axes = car.route.body(car.p, car.lateral)
            for index, obstacle in enumerate(obstacles):
                contact = ellipse_contact_value(centre, axes, (obstacle.x, obstacle.y),
                                                np.eye(2)*obstacle.radius**2)
                if contact < 1.0 - ELLIPSE_CONTACT_TOL:
                    controller.obstacle_collisions.append((time + DT, car.id, index))

        # Register vehicles as crossing when their front reaches the stop line.
        for car in cars:
            if car.state == "APPROACH" and car.p >= STOP_LINE:
                car.state = "GO"
                car.go_time = time + DT
                controller.active_cars.append(car.id)
                controller.release_events.append({"time": float(time+DT), "id": car.id,
                    "turn": car.turn, "stop_time": None,
                    "already_active": tuple(x for x in controller.active_cars if x != car.id)})

        # ---------------------------------------------------------------------
        # E. Retire vehicles once the rear clears their exit.
        # ---------------------------------------------------------------------
        controller.check_intersection(cars)

        # ---------------------------------------------------------------------
        # F. Remove cars after the downstream deletion point.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state == "GO" and car.p >= car.delete_p:
                car.state = "DONE"

        if not arrivals and all(car.state == "DONE" for car in cars):
            break

    if arrivals or any(car.state != "DONE" for car in cars):
        warnings.warn("Simulation time limit reached with unfinished traffic.")
    return cars, controller


# =============================================================================
# 6. RESULTS
# =============================================================================

def print_summary(cars, controller):
    print("\nCrossing summary")
    print("----------------")
    print(f"Spawned: {len(cars)}; completed: {sum(c.state == 'DONE' for c in cars)}; "
          f"pending arrivals: {controller.pending_arrivals}")
    print(f"Optimizer retry/failure events: {len(controller.solver_events)}")
    print(f"Obstacle overlap detections: {len(controller.obstacle_collisions)}")

    ordered_cars = sorted(
        cars, key=lambda car: car.go_time if car.go_time is not None else float("inf")
    )

    for car in ordered_cars:
        stop = f"{car.stop_time:.1f} s" if car.stop_time is not None else "-"
        go = f"{car.go_time:.1f} s" if car.go_time is not None else "never"
        conflicts = ", ".join(f"V{i}" for i in car.conflicts) or "none"
        print(
            f"Vehicle {car.id} ({car.origin} -> {car.destination}, {car.turn}): "
            f"stop time = {stop}, go time = {go}, conflicts with {conflicts}"
        )

    print(f"Conflict-point simultaneous-occupancy samples: {len(controller.conflict_point_events)}")
    for event in controller.conflict_point_events[:20]:
        print(f"  {event['time']:.1f}s {event['point']}: {event['vehicles']}")
    pairs = sorted({(a, b) for _, a, b in controller.collisions})
    if pairs:
        print(f"Collisions: {', '.join(f'V{a}-V{b}' for a, b in pairs)}")
    else:
        print("Collisions: none")


def plot_results(cars, output_dir="."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for car in cars:
        plt.plot(car.time_history, car.lateral_history, label=f"Vehicle {car.id}")
    plt.xlabel("Time [s]")
    plt.ylabel("Leftward offset from route [m]")
    plt.title("Lane Changes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lateral_vs_time.png", dpi=200)
    plt.close()

    # Position plot
    plt.figure(figsize=(9, 5))

    for car in cars:
        plt.plot(
            car.time_history,
            car.position_history,
            label=f"Vehicle {car.id} ({car.origin})",
        )

    plt.axhline(STOP_LINE, linestyle="--", linewidth=1, label="Stop line")
    # Exit progress differs for left, right and straight routes.

    plt.xlabel("Time [s]")
    plt.ylabel("Distance along assigned route [m]")
    plt.title("Vehicle Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_vs_time.png", dpi=200)
    plt.close()

    # Speed plot
    plt.figure(figsize=(9, 5))

    for car in cars:
        plt.plot(
            car.time_history,
            car.speed_history,
            label=f"Vehicle {car.id} ({car.origin})",
        )

    for speed, label in ((V_DES, "Straight target"), (4.0, "Left target"), (2.0, "Right target")):
        plt.axhline(speed, linestyle="--", linewidth=1, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Vehicle Speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_time.png", dpi=200)
    plt.close()



#Animation Function
def animate_results(cars, obstacles=None):
    obstacles = create_obstacles() if obstacles is None else obstacles
    max_frames = 1 + round(max(car.time_history[-1] for car in cars if car.time_history) / DT)
    
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

    # Two same-direction lanes per half-road (centres 1.5 and 4.5 m).
    for lane_line in (3, 9):
        for low, high in ((-50, 0), (12, 50)):
            ax.plot([low, high], [lane_line, lane_line], 'w--', linewidth=0.7)
            ax.plot([lane_line, lane_line], [low, high], 'w--', linewidth=0.7)

    for obstacle in obstacles:
        ax.add_patch(patches.Circle((obstacle.x, obstacle.y), obstacle.radius,
                                   facecolor="crimson", edgecolor="white", zorder=5))
        ax.annotate("Obstacle", (obstacle.x, obstacle.y), xytext=(-20, -20),
                    textcoords="offset points", color="white", fontsize=9)

    STATE_COLORS = {
        "APPROACH": "#f39c12",  
        "WAIT": "#e74c3c",      
        "GO": "#2ecc71"         
    }

    car_length = CAR_LENGTH
    car_width = 2.0 * CAR_HALF_WIDTH
    car_patches = {}
    car_texts = {}

    drawn_routes = set()
    for car in cars:
        key = (car.origin, car.destination)
        if key not in drawn_routes:
            ax.plot(car.path[:, 0], car.path[:, 1], color="white", alpha=0.18, linewidth=1)
            drawn_routes.add(key)
        oval = patches.Ellipse(
            (0, 0), car_length, car_width,
            angle=0 if car.origin in ("W", "E") else 90,
            color='white', zorder=3)
        ax.add_patch(oval)
        car_patches[car.id] = oval
        
        text = ax.text(0, 0, f"V{car.id}{car.turn[0].upper()}", color='black', weight='bold', fontsize=8,
                       ha='center', va='center', zorder=4)
        car_texts[car.id] = text

    # Telemetry coordinates mapped exactly to top-left quadrant
    time_text = ax.text(-40, 40, '', color='white', fontsize=14, fontfamily='monospace')
    telem_text = ax.text(-40, 32, '', color='white', fontsize=10, fontfamily='monospace', va='top')


    def update(frame_idx):
        current_time = frame_idx * DT
        time_text.set_text(f"Simulation Time: {current_time:05.2f} s")
        telem_lines = ["TELEMETRY:"]
        
        for car in sorted(cars, key=lambda c: c.id):
            vid = car.id
            approach = car.origin
            
            local_frame = frame_idx - round(car.spawn_time / DT)
            if 0 <= local_frame < len(car.position_history):
                p = car.position_history[local_frame]
                v = car.speed_history[local_frame]
                state = car.state_history[local_frame]
            else:
                state = "DONE"
                p = car.position_history[-1] if car.position_history else 0
                v = 0.0
                
            cx, cy = car.xy_history[max(0, min(local_frame, len(car.xy_history) - 1))]
            oval = car_patches[vid]
            
            if state != "DONE":
                reason = car.wait_reason_history[local_frame]
                telem_lines.append(f"V{vid} {approach}>{car.destination} {car.turn:<8} "
                                   f"{state:<8} p={p:5.1f}m v={v:4.1f}m/s")

            if state == "DONE" or cx < -45 or cx > 45 or cy < -45 or cy > 45:
                oval.set_alpha(0.0)
                car_texts[vid].set_alpha(0.0)
            else:
                oval.set_alpha(1.0)
                car_texts[vid].set_alpha(1.0)
                oval.set_facecolor(STATE_COLORS.get(state, "white"))
                
                # Use the same centre and physical axes as collision detection.
                lateral = car.lateral_history[local_frame]
                centre, _ = car.route.body(p, lateral)
                oval.set_angle(np.degrees(car.route.frame(p)[3]))
                oval.set_center(centre)
                car_texts[vid].set_position(centre)

        telem_text.set_text("\n".join(telem_lines[:11]) + ("\n..." if len(telem_lines) > 11 else ""))
        return list(car_patches.values()) + list(car_texts.values()) + [time_text, telem_text]

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=200, blit=True)
    plt.tight_layout()
    plt.show()
    return ani



def save_experiment_results(cars, controller, output_dir="."):
    """Save route-offset/acceleration plots and per-car metrics for reproducible comparisons.

    Note: lateral=0 denotes the centre of the assigned route. Initial westbound
    vehicles have lateral=-1.5, so their initial path error is nonzero.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    west_cars = [car for car in cars if car.origin == "W"]
    if west_cars:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for car in west_cars:
            t = np.asarray(car.time_history)
            lateral = np.asarray(car.lateral_history)
            accel = np.asarray(car.acceleration_history)
            ax1.plot(t, lateral, label=f"V{car.id} (spawn {car.spawn_time:.1f}s)")
            ax2.plot(t, accel[:, 1], label=f"V{car.id}")
        ax1.axhline(0, color="black", linestyle="--", alpha=0.5, label="Route center")
        ax1.set_ylabel("Offset from route [m]")
        ax1.set_title("Westbound vehicles: obstacle avoidance and path tracking")
        ax1.legend()
        ax2.set_ylabel("Lateral acceleration [m/s²]")
        ax2.set_xlabel("Time [s]")
        ax2.legend()
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "westbound_path_tracking.png", dpi=200)
        plt.close(fig)

    with (output_dir / "vehicle_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["vehicle", "origin", "destination", "spawn_time_s", "completed",
                         "travel_time_s", "mean_abs_route_offset_m", "final_abs_route_offset_m",
                         "lateral_control_effort", "max_abs_lateral_accel_m_s2"])
        for car in cars:
            lat = np.asarray(car.lateral_history)
            acc = np.asarray(car.acceleration_history)
            writer.writerow([car.id, car.origin, car.destination, car.spawn_time,
                             car.state == "DONE",
                             round(car.time_history[-1] - car.spawn_time + DT, 3) if car.time_history else "",
                             float(np.mean(np.abs(lat))) if lat.size else "",
                             float(abs(lat[-1])) if lat.size else "",
                             float(DT * np.sum(acc[:, 1] ** 2)) if acc.size else "",
                             float(np.max(np.abs(acc[:, 1]))) if acc.size else ""])
    with (output_dir / "run_summary.txt").open("w") as handle:
        handle.write(f"Cars: {len(cars)}, completed: {sum(c.state == 'DONE' for c in cars)}\n")
        handle.write(f"W_LANE: {W_LANE}, W_LATERAL_SPEED: {W_LATERAL_SPEED}, seed: {RANDOM_SEED}\n")
        handle.write(f"Obstacle overlap samples: {len(controller.obstacle_collisions)}\n")
        handle.write(f"Car overlap samples: {len(controller.collisions)}\n")
        handle.write(f"Optimizer retry/failure events: {len(controller.solver_events)}\n")
        handle.write(f"Conflict-point occupancy events: {len(controller.conflict_point_events)}\n")
        handle.write(f"Unfinished arrivals: {controller.pending_arrivals}\n")

# =============================================================================
# 7. MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8-car MPC path-tracking experiment (consistent road bounds)")
    parser.add_argument("--lane-weight", type=float, default=W_LANE,
                        help="Penalty on offset from desired route (default 100)")
    parser.add_argument("--lateral-speed-weight", type=float, default=W_LATERAL_SPEED,
                        help="Penalty on lateral velocity (default 80)")
    parser.add_argument("--output-dir", default="results/baseline",
                        help="Directory for graphs and CSV")
    parser.add_argument("--no-animation", action="store_true",
                        help="Save graphs and metrics without opening the animation")
    args = parser.parse_args()
    if args.lane_weight < 0 or args.lateral_speed_weight < 0:
        parser.error("Cost weights must be non-negative")
    W_LANE = args.lane_weight
    W_LATERAL_SPEED = args.lateral_speed_weight
    obstacles = create_obstacles()
    cars, controller = run_simulation(obstacles=obstacles)
    print_summary(cars, controller)
    plot_results(cars, args.output_dir)
    save_experiment_results(cars, controller, args.output_dir)
    print(f"Saved graphs and metrics in {args.output_dir}")
    if not args.no_animation:
        animate_results(cars, obstacles=obstacles)
