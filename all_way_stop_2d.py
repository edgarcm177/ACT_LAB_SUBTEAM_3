"""
2-D MPC + Trajectory-Conflict Controller for an All-Way-Stop Intersection
=========================================================================

What changed from the 1-D version
---------------------------------
* State is now 4 terms per car:  (x, y, vx, vy)
      x  = longitudinal position along the car's route (front bumper, the old p)
      y  = lateral offset from the car's lane centre (+ = towards the left /
           opposing lane), which is what lets a car change lanes or swerve
      vx, vy = the matching velocities
  The controller optimises 2N decision variables: N longitudinal accelerations
  ax(k) and N lateral accelerations ay(k).

* Vehicles are no longer points / circle chains.  Every other car and every
  obstacle is surrounded by a soft ELLIPTICAL POTENTIAL.  For a body centred at
  C with heading h and left-normal n:

        u = (p - C) . h ,   w = (p - C) . n
        q = (u / A)^2 + (w / B)^2          q < 1  ->  inside the safety ellipse

  The semi-axes A, B circumscribe the Minkowski sum of the two car rectangles
  (factor sqrt(2)), plus a buffer, and A is stretched by the other car's speed
  (A += HEADWAY_TIME * |v|) - this is the "dynamic" part of the ellipse: a fast
  car claims more road in front of and behind itself.

  The penalty phi(q) can be swapped (COLLISION_MODEL) so the trade-offs can be
  compared:
      "hinge"   : max(1 - q, 0)^2       zero outside, smooth at the boundary
      "gauss"   : exp(-K q)             smooth everywhere, never exactly zero
      "barrier" : (ln q)^2 for q < 1    grows without bound near the centre
      "circles" : old 3-circle chain    baseline from the 1-D version
      "hard"    : q >= 1 as a HARD constraint (SLSQP) instead of a penalty

* New objective terms:  lane keeping  W_LANE*(y - y_ref)^2,
  lateral speed W_VY*vy^2 (stops left/right oscillation), lateral
  acceleration W_AY*ay^2, a jerk term W_JERK*(a(k)-a(k-1))^2 for smoothness,
  and a soft road-edge term W_ROAD.

* Gradients of the cost are computed analytically, so the optimiser moves the
  input sequence as  U(i+1) = U(i) + alpha * s, with s the search direction:
      "GD"      : s = -grad J               (plain steepest descent)
      "BFGS"    : s = -H^-1 grad J          (quasi-Newton, default)
      "L-BFGS-B": BFGS with hard box limits on the accelerations

Run:
    python all_way_stop_2d.py                    # 4 cars, hinge ellipse
    python all_way_stop_2d.py --obstacle         # adds a stalled car to swerve around
    python all_way_stop_2d.py --model gauss      # try another potential
    python all_way_stop_2d.py --compare          # table of all variants
"""

import argparse
import time as wallclock

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

# Longitudinal weights (from the report)
W_POSITION = 100.0
W_SPEED = 150.0
W_ACCEL = 0.20
W_CROSS = 2000.0

# Lateral weights (new)
W_LANE = 30.0             # stay near the reference lateral offset y_ref
W_VY = 200.0              # lateral speed -> no left/right weaving
W_AY = 2.0                # lateral acceleration effort
W_JERK = 0.5              # change of acceleration between steps (both axes)
W_ROAD = 5000.0           # soft road-edge limit
W_REVERSE = 5000.0        # no driving backwards (vx < 0)

# Interaction weights
W_COLLISION = 1.0e5       # car-to-car potential
W_OBSTACLE = 1.0e5        # car-to-obstacle potential

MAXITER = 60

LANE_OFFSET = 3.0         # lane centre offset from the road edge [m]
ROAD_WIDTH = 12.0
CONFLICT_CLEARANCE = 2.5  # paths closer than this are in conflict [m]
NEARBY_RADIUS = 40.0      # cars within this distance are tracked [m]
CAR_HALF_WIDTH = 1.0
CAR_WIDTH = 2.0 * CAR_HALF_WIDTH
CAR_LENGTH = 4.5

# Lateral limits for the car centre (own lane edge ... far edge of opposing lane)
Y_MIN = -LANE_OFFSET + CAR_HALF_WIDTH
Y_MAX = (ROAD_WIDTH - LANE_OFFSET) - CAR_HALF_WIDTH

# Elliptical potential
COLLISION_MODEL = "hinge"   # hinge | gauss | barrier | circles | hard
SOLVER = "BFGS"             # BFGS | GD | L-BFGS-B   (ignored for "hard")
SAFETY_BUFFER = 0.5         # extra clearance added to every ellipse [m]
HEADWAY_TIME = 0.4          # ellipse stretch per m/s of the other car [s]
GAUSS_K = 2.0               # decay rate of the Gaussian potential
BARRIER_DELTA = 0.1         # below this q the log barrier is extended linearly
LANE_CHANGE_LOGIC = True    # high-level layer that moves y_ref around blocked lanes
LANE_CHANGE_LOOKAHEAD = 25.0  # obstacles this far ahead trigger a lane change [m]
AX_LIMIT = 4.0              # box limits used by L-BFGS-B [m/s^2]
AY_LIMIT = 2.0

# Old circle model (kept as the comparison baseline)
CIRCLE_OFFSETS = np.linspace(0.0, CAR_LENGTH, 3)
COLLISION_DIST = 2.0 * CAR_HALF_WIDTH
CAR_BUFFER = 1.0
SAFE_DIST = COLLISION_DIST + CAR_BUFFER

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


# =============================================================================
# GEOMETRY
# =============================================================================

# Route frame of each approach: world = origin + x * heading + y * left
ROUTES = {
    "W": (np.array([0.0, LANE_OFFSET]), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
    "E": (np.array([INTERSECTION_EXIT, INTERSECTION_EXIT - LANE_OFFSET]),
          np.array([-1.0, 0.0]), np.array([0.0, -1.0])),
    "S": (np.array([INTERSECTION_EXIT - LANE_OFFSET, 0.0]),
          np.array([0.0, 1.0]), np.array([-1.0, 0.0])),
    "N": (np.array([LANE_OFFSET, INTERSECTION_EXIT]),
          np.array([0.0, -1.0]), np.array([1.0, 0.0])),
}


def route_to_world(origin, x, y=0.0):
    """Route-frame (x, y) -> world (X, Y).  Broadcasts; returns shape (..., 2)."""
    o, h, n = ROUTES[origin]
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    return o + x[..., None] * h + y[..., None] * n


def position_to_xy(origin, p):
    """Lane-centre world point for 1-D position p (kept for compatibility)."""
    X = route_to_world(origin, p, 0.0)
    return float(X[0]), float(X[1])


def body_center(origin, x, y):
    """World centre of the car body (x is the front bumper)."""
    return route_to_world(origin, np.asarray(x) - CAR_LENGTH / 2.0, y)


def ellipse_axes(my_origin, their_heading, their_half_len, their_half_wid):
    """
    Semi-axes (A0, B0) of the safety ellipse around a body with the given
    heading and half-size, as seen by a car coming from my_origin.  The
    rectangle half-extents of both bodies are summed along the other body's
    axes (Minkowski sum) and scaled by sqrt(2) so the ellipse fully contains
    that rectangle, then the buffer is added.
    """
    _, h, _ = ROUTES[my_origin]
    c = abs(float(h @ their_heading))
    s = np.sqrt(max(0.0, 1.0 - c * c))
    half_long = their_half_len + c * CAR_LENGTH / 2.0 + s * CAR_HALF_WIDTH
    half_lat = their_half_wid + s * CAR_LENGTH / 2.0 + c * CAR_HALF_WIDTH
    return np.sqrt(2.0) * half_long + SAFETY_BUFFER, np.sqrt(2.0) * half_lat + SAFETY_BUFFER


def potential(q, model):
    """Penalty phi(q) and its derivative dphi/dq for the chosen ellipse model."""
    if model == "hinge":
        s = np.maximum(1.0 - q, 0.0)
        return s**2, -2.0 * s
    if model == "gauss":
        e = np.exp(-GAUSS_K * q)
        return e, -GAUSS_K * e
    if model == "barrier":
        phi = np.zeros_like(q)
        dphi = np.zeros_like(q)
        mid = (q < 1.0) & (q >= BARRIER_DELTA)
        lq = np.log(q[mid])
        phi[mid] = lq**2
        dphi[mid] = 2.0 * lq / q[mid]
        low = q < BARRIER_DELTA
        ld = np.log(BARRIER_DELTA)
        slope = 2.0 * ld / BARRIER_DELTA
        phi[low] = ld**2 + slope * (q[low] - BARRIER_DELTA)
        dphi[low] = slope
        return phi, dphi
    raise ValueError(f"unknown potential model {model}")


def heading_angle(origin, vx, vy):
    """World heading of a car including its lateral motion [rad]."""
    _, h, n = ROUTES[origin]
    d = h * max(vx, 0.0) + n * vy
    if np.hypot(*d) < 0.3:
        d = h
    return np.arctan2(d[1], d[0])


def rectangle_corners(center, yaw, length=CAR_LENGTH, width=CAR_WIDTH):
    c, s = np.cos(yaw), np.sin(yaw)
    f = np.array([c, s]) * length / 2.0
    l = np.array([-s, c]) * width / 2.0
    return np.array([center + f + l, center + f - l, center - f - l, center - f + l])


def rectangles_overlap(r1, r2):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (r1, r2):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            p1, p2 = r1 @ axis, r2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


def point_segment_distance(pt, a, b):
    pt, a, b = np.asarray(pt), np.asarray(a), np.asarray(b)
    ab = b - a
    t = np.clip(np.dot(pt - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(pt - (a + t * ab))


def rectangle_gap(r1, r2):
    """Smallest distance between two rectangles (0 if they overlap)."""
    if rectangles_overlap(r1, r2):
        return 0.0
    return min(
        point_segment_distance(p, r2[i], r2[(i + 1) % 4])
        for p in r1 for i in range(4)
    )


def rectangle_gap_sym(r1, r2):
    return min(rectangle_gap(r1, r2), rectangle_gap(r2, r1))


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
        """True if the obstacle sits on a car's path through the intersection."""
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
                 lateral=0.0, y_ref=0.0):
        self.id = car_id

        # Route information
        self.origin = origin
        self.destination = destination or OPPOSITE[origin]
        if self.destination != OPPOSITE[origin]:
            raise ValueError("Only straight-through routes are supported")
        self.direction = self.destination
        self.heading = ROUTES[origin][1]
        self.path = [
            position_to_xy(origin, STOP_LINE),
            position_to_xy(origin, INTERSECTION_EXIT),
        ]

        # Awareness of the surroundings, refreshed every step
        self.nearby_cars = {}
        self.conflicts = []
        self.obstacles = []

        # 4-term state in the route frame
        self.x = position
        self.y = lateral
        self.vx = speed
        self.vy = 0.0
        self.y_ref = y_ref          # lateral target: 0 = own lane, change it to switch lanes
        self.last_ax = 0.0
        self.last_ay = 0.0

        self.state = "APPROACH"
        self.stop_time = None
        self.go_time = None

        # Warm starts: [ax(0..N-1), ay(0..N-1)]
        self.stop_guess = np.zeros(2 * N)
        self.go_guess = np.zeros(2 * N)

        # Predicted trajectory shared with the other cars
        t = DT * np.arange(N + 1)
        self.predicted_x = position + speed * t
        self.predicted_y = np.full(N + 1, lateral)
        self.predicted_vx = np.full(N + 1, speed)

        # Histories
        self.time_history = []
        self.position_history = []      # x
        self.lateral_history = []       # y
        self.speed_history = []         # vx
        self.lateral_speed_history = [] # vy
        self.acceleration_history = []  # ax
        self.lateral_accel_history = [] # ay
        self.state_history = []
        self.solve_times = []

    # Backwards-compatible names
    @property
    def p(self):
        return self.x

    @property
    def v(self):
        return self.vx

    def world_center(self):
        return body_center(self.origin, self.x, self.y)

    def corners(self, x=None, y=None, vx=None, vy=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        vx = self.vx if vx is None else vx
        vy = self.vy if vy is None else vy
        return rectangle_corners(body_center(self.origin, x, y),
                                 heading_angle(self.origin, vx, vy))


# =============================================================================
# 3. CONTROLLER
# =============================================================================

class Controller:
    def __init__(self, model=None, solver=None):
        self.model = model or COLLISION_MODEL
        self.solver = solver or SOLVER
        self.active_cars = []
        self.collisions = []
        self.obstacle_hits = []
        self.obstacles = []

        # Linear prediction matrices:  x(k+1) = x0 + (k+1) DT vx0 + Mp @ a
        #                              vx(k+1) = vx0 + Mv @ a
        k = np.arange(N)
        diff = k[:, None] - k[None, :]
        self.Mp = np.where(diff >= 0, DT**2 * (diff + 0.5), 0.0)
        self.Mv = np.where(diff >= 0, DT, 0.0)
        self.T = DT * np.arange(1, N + 1)

    # -------------------------------------------------------------------------
    # Vehicle prediction model (4 states, 2 inputs)
    # -------------------------------------------------------------------------
    def predict(self, car, u):
        """Return x, y, vx, vy at steps 1..N for the input sequence u."""
        ax, ay = u[:N], u[N:]
        x = car.x + car.vx * self.T + self.Mp @ ax
        y = car.y + car.vy * self.T + self.Mp @ ay
        vx = car.vx + self.Mv @ ax
        vy = car.vy + self.Mv @ ay
        return x, y, vx, vy

    def predict_full(self, car, u):
        """Same as predict, but including step 0 (shape N+1)."""
        x, y, vx, vy = self.predict(car, u)
        return (np.append(car.x, x), np.append(car.y, y),
                np.append(car.vx, vx), np.append(car.vy, vy))

    # -------------------------------------------------------------------------
    # Elliptical interaction terms
    # -------------------------------------------------------------------------
    def ellipse_terms(self, car, x, y, others, obstacles):
        """
        For every other car / obstacle return the per-step ellipse value q(k)
        and its partial derivatives w.r.t. my x(k) and y(k).
        """
        _, h, n = ROUTES[car.origin]
        C = body_center(car.origin, x, y)                          # (N, 2)
        terms = []

        for other in others:
            _, hj, nj = ROUTES[other.origin]
            Cj = body_center(other.origin, other.predicted_x[1:], other.predicted_y[1:])
            A0, B0 = ellipse_axes(car.origin, hj, CAR_LENGTH / 2.0, CAR_HALF_WIDTH)
            A = A0 + HEADWAY_TIME * np.abs(other.predicted_vx[1:])
            B = np.full(N, B0)
            terms.append(("car", other.id, *self._q(C, Cj, hj, nj, A, B, h, n)))

        for i, obs in enumerate(obstacles):
            centre = np.array([obs.x, obs.y])
            A0, B0 = ellipse_axes(car.origin, h, obs.radius, obs.radius)
            terms.append(("obstacle", i, *self._q(C, centre, h, n,
                                                  np.full(N, A0), np.full(N, B0), h, n)))
        return terms

    @staticmethod
    def _q(C, Cj, hj, nj, A, B, h, n):
        rel = C - Cj
        u = rel @ hj
        w = rel @ nj
        q = (u / A)**2 + (w / B)**2
        dq_dC = (2.0 * u / A**2)[:, None] * hj + (2.0 * w / B**2)[:, None] * nj
        return q, dq_dC @ h, dq_dC @ n

    def circle_cost(self, car, x, y, others):
        """Old multi-circle collision term, generalised to (x, y)."""
        _, h, n = ROUTES[car.origin]
        J, gx, gy = 0.0, np.zeros(N), np.zeros(N)
        mine = route_to_world(car.origin, x[:, None] - CIRCLE_OFFSETS, y[:, None])
        for other in others:
            theirs = route_to_world(other.origin,
                                    other.predicted_x[1:, None] - CIRCLE_OFFSETS,
                                    other.predicted_y[1:, None])
            rel = mine[:, :, None, :] - theirs[:, None, :, :]
            d = np.maximum(np.linalg.norm(rel, axis=-1), 1e-9)
            pen = np.maximum(SAFE_DIST - d, 0.0)
            J += W_COLLISION * np.sum(pen**2)
            dJ = (-2.0 * W_COLLISION * pen / d)[..., None] * rel     # (N,3,3,2)
            dJ_dmine = dJ.sum(axis=2).sum(axis=1)                    # (N,2)
            gx += dJ_dmine @ h
            gy += dJ_dmine @ n
        return J, gx, gy

    # -------------------------------------------------------------------------
    # Objective (STOP and GO share everything except the longitudinal part)
    # -------------------------------------------------------------------------
    def cost(self, u, car, mode, others, obstacles, include_interaction=True):
        """
        J = J_longitudinal (STOP or GO, unchanged from the report)
          + sum[ W_LANE (y - y_ref)^2 + W_VY vy^2 + W_AY ay^2 ]       lateral
          + sum[ W_JERK (a(k) - a(k-1))^2 ]                         smoothness
          + W_ROAD * road-edge violation^2 + W_REVERSE * min(vx, 0)^2
          + sum W_COLLISION * phi(q_car) + W_OBSTACLE * phi(q_obs)    potentials

        Returns J and dJ/du (analytic).
        """
        ax, ay = u[:N], u[N:]
        x, y, vx, vy = self.predict(car, u)

        gx, gy, gvx, gvy = (np.zeros(N) for _ in range(4))
        gax = 2.0 * W_ACCEL * ax
        gay = 2.0 * W_AY * ay
        J = W_ACCEL * ax @ ax + W_AY * ay @ ay

        # Longitudinal part
        if mode == "STOP":
            J += W_POSITION * x[-1]**2 + W_SPEED * vx[-1]**2
            gx[-1] += 2.0 * W_POSITION * x[-1]
            gvx[-1] += 2.0 * W_SPEED * vx[-1]
            over = np.maximum(x, 0.0)
            J += W_CROSS * over @ over
            gx += 2.0 * W_CROSS * over
        else:
            e = vx - V_DES
            J += W_SPEED * e @ e
            gvx += 2.0 * W_SPEED * e

        back = np.minimum(vx, 0.0)
        J += W_REVERSE * back @ back
        gvx += 2.0 * W_REVERSE * back

        # Lateral part
        ey = y - car.y_ref
        J += W_LANE * ey @ ey + W_VY * vy @ vy
        gy += 2.0 * W_LANE * ey
        gvy += 2.0 * W_VY * vy

        hi = np.maximum(y - Y_MAX, 0.0)
        lo = np.maximum(Y_MIN - y, 0.0)
        J += W_ROAD * (hi @ hi + lo @ lo)
        gy += 2.0 * W_ROAD * (hi - lo)

        # Jerk (smoothness) on both axes
        for a, prev, g in ((ax, car.last_ax, gax), (ay, car.last_ay, gay)):
            da = np.diff(np.concatenate(([prev], a)))
            J += W_JERK * da @ da
            gd = 2.0 * W_JERK * da
            g += gd - np.append(gd[1:], 0.0)

        # Interaction potentials
        if include_interaction:
            if self.model == "circles":
                Jc, cx, cy = self.circle_cost(car, x, y, others)
                J += Jc
                gx += cx
                gy += cy
                obs_terms = self.ellipse_terms(car, x, y, [], obstacles)
                phi_model = "hinge"
            else:
                obs_terms = self.ellipse_terms(car, x, y, others, obstacles)
                phi_model = self.model
            for kind, _, q, dqx, dqy in obs_terms:
                W = W_COLLISION if kind == "car" else W_OBSTACLE
                phi, dphi = potential(q, phi_model)
                J += W * phi.sum()
                gx += W * dphi * dqx
                gy += W * dphi * dqy

        grad = np.concatenate((
            self.Mp.T @ gx + self.Mv.T @ gvx + gax,
            self.Mp.T @ gy + self.Mv.T @ gvy + gay,
        ))
        return J, grad

    # -------------------------------------------------------------------------
    # Solvers:  U(i+1) = U(i) + alpha * s
    # -------------------------------------------------------------------------
    @staticmethod
    def gradient_descent(fun, u0, maxiter=MAXITER, c1=1e-4, tol=1e-6):
        """Steepest descent with Armijo backtracking: s = -grad J."""
        u = u0.copy()
        J, g = fun(u)
        alpha = 1.0 / max(np.linalg.norm(g), 1.0)
        for _ in range(maxiter):
            gnorm2 = g @ g
            if gnorm2 < tol**2:
                break
            s = -g
            alpha *= 2.0
            while True:
                J_new, g_new = fun(u + alpha * s)
                if J_new <= J - c1 * alpha * gnorm2 or alpha < 1e-14:
                    break
                alpha *= 0.5
            u, J, g = u + alpha * s, J_new, g_new
        return u

    def hard_constraints(self, car, others, obstacles):
        """q(k) - 1 >= 0 for every car / obstacle and horizon step."""
        def fun(u):
            x, y, _, _ = self.predict(car, u)
            terms = self.ellipse_terms(car, x, y, others, obstacles)
            if not terms:
                return np.ones(1)
            return np.concatenate([q - 1.0 for _, _, q, _, _ in terms])

        def jac(u):
            x, y, _, _ = self.predict(car, u)
            terms = self.ellipse_terms(car, x, y, others, obstacles)
            if not terms:
                return np.zeros((1, 2 * N))
            rows = [np.hstack((dqx[:, None] * self.Mp, dqy[:, None] * self.Mp))
                    for _, _, _, dqx, dqy in terms]
            return np.vstack(rows)

        return [{"type": "ineq", "fun": fun, "jac": jac}]

    def solve(self, car, mode, others, obstacles):
        guess_attr = "stop_guess" if mode == "STOP" else "go_guess"
        u0 = getattr(car, guess_attr)
        t0 = wallclock.perf_counter()

        if self.model == "hard":
            res = minimize(
                self.cost, u0, jac=True, method="SLSQP",
                args=(car, mode, others, obstacles, False),
                constraints=self.hard_constraints(car, others, obstacles),
                options={"maxiter": MAXITER, "disp": False},
            )
            u = res.x
        elif self.solver == "GD":
            u = self.gradient_descent(
                lambda z: self.cost(z, car, mode, others, obstacles), u0)
        elif self.solver == "L-BFGS-B":
            bounds = [(-AX_LIMIT, AX_LIMIT)] * N + [(-AY_LIMIT, AY_LIMIT)] * N
            res = minimize(self.cost, np.clip(u0, -AY_LIMIT, AX_LIMIT), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           args=(car, mode, others, obstacles),
                           options={"maxiter": MAXITER})
            u = res.x
        else:
            res = minimize(self.cost, u0, jac=True, method="BFGS",
                           args=(car, mode, others, obstacles),
                           options={"maxiter": MAXITER, "disp": False})
            u = res.x

        car.solve_times.append(wallclock.perf_counter() - t0)
        setattr(car, guess_attr, u)
        # MPC: apply only the first input of each axis.
        return u[0], u[N]

    def interacting_cars(self, car, cars):
        """Every nearby live car (ellipses make far-away ones cost nothing)."""
        return [c for c in cars if c.id in car.nearby_cars and c.state != "DONE"]

    def solve_stop_problem(self, car, cars):
        return self.solve(car, "STOP", self.interacting_cars(car, cars), self.obstacles)

    def solve_go_problem(self, car, cars):
        return self.solve(car, "GO", self.interacting_cars(car, cars), self.obstacles)

    # -------------------------------------------------------------------------
    # Trajectory-conflict logic (unchanged idea, now uses world centres)
    # -------------------------------------------------------------------------
    def update_awareness(self, cars, obstacles):
        live = [car for car in cars if car.state != "DONE"]
        for car in live:
            here = car.world_center()
            car.nearby_cars = {}
            car.conflicts = []
            for other in live:
                if other.id == car.id:
                    continue
                if np.linalg.norm(here - other.world_center()) > NEARBY_RADIUS:
                    continue
                car.nearby_cars[other.id] = other.path
                if path_distance(car.path, other.path) < CONFLICT_CLEARANCE:
                    car.conflicts.append(other.id)
            car.obstacles = [obs for obs in obstacles if obs.blocks(car.path)]

    def update_lane_references(self, cars, obstacles):
        """
        Lane-switching layer.  The potential alone is a LOCAL method: a car
        heading straight at an obstacle can sit in a local minimum where
        braking looks cheaper than steering.  This layer makes the discrete
        decision (which side to pass on) and hands the MPC a new lateral
        target y_ref; the MPC then executes the lane change smoothly while the
        potentials keep it clear of everything else.
        """
        for car in cars:
            if car.state not in ("APPROACH", "GO"):
                continue
            car.y_ref = 0.0
            if not LANE_CHANGE_LOGIC:
                continue
            o, h, n = ROUTES[car.origin]
            x_centre = car.x - CAR_LENGTH / 2.0
            for obs in obstacles:
                rel = np.array([obs.x, obs.y]) - o
                s_obs, d_obs = rel @ h, rel @ n
                A0, B0 = ellipse_axes(car.origin, h, obs.radius, obs.radius)
                ahead = -A0 < s_obs - x_centre < LANE_CHANGE_LOOKAHEAD
                if not ahead or abs(d_obs) >= B0:
                    continue
                left, right = d_obs + B0 + 0.3, d_obs - B0 - 0.3
                if left <= Y_MAX:
                    car.y_ref = left
                elif right >= Y_MIN:
                    car.y_ref = right

    def register_stopped_car(self, car, time):
        car.stop_time = time
        car.state = "WAIT"

    def release_cars(self, cars, time):
        """
        Release a waiting car when no obstacle sits inside the intersection on
        its path and no conflicting car is crossing.  Obstacles elsewhere are
        handled by the elliptical potential (the car steers around them).
        """
        for car in cars:
            if car.state != "WAIT" or car.obstacles:
                continue
            if any(other_id in self.active_cars for other_id in car.conflicts):
                continue
            car.state = "GO"
            car.go_time = time
            self.active_cars.append(car.id)

    # -------------------------------------------------------------------------
    # Prediction sharing and collision detection
    # -------------------------------------------------------------------------
    def update_prediction(self, car):
        if car.state == "WAIT":
            car.predicted_x = np.full(N + 1, STOP_LINE)
            car.predicted_y = np.full(N + 1, car.y)
            car.predicted_vx = np.zeros(N + 1)
            return
        plan = car.go_guess if car.state == "GO" else car.stop_guess
        ax, ay = plan[:N], plan[N:]
        shifted = np.concatenate((np.append(ax[1:], ax[-1]), np.append(ay[1:], ay[-1])))
        car.predicted_x, car.predicted_y, car.predicted_vx, _ = self.predict_full(car, shifted)

    def detect_collisions(self, cars, time):
        """Exact rectangle overlap test (cars can now be rotated / offset)."""
        live = [car for car in cars if car.state != "DONE"]
        boxes = {car.id: car.corners() for car in live}
        for i, car in enumerate(live):
            for other in live[i + 1:]:
                if rectangles_overlap(boxes[car.id], boxes[other.id]):
                    self.collisions.append((time, car.id, other.id))
            for k, obs in enumerate(self.obstacles):
                pts = boxes[car.id]
                centre = np.array([obs.x, obs.y])
                inside = all(
                    ((pts[(j + 1) % 4] - pts[j])[0] * (centre - pts[j])[1]
                     - (pts[(j + 1) % 4] - pts[j])[1] * (centre - pts[j])[0]) <= 0
                    for j in range(4)
                )
                dist = min(point_segment_distance(centre, pts[j], pts[(j + 1) % 4])
                           for j in range(4))
                if inside or dist < obs.radius:
                    self.obstacle_hits.append((time, car.id, k))

    def check_intersection(self, cars):
        for i in range(len(self.active_cars) - 1, -1, -1):
            for car in cars:
                if car.id == self.active_cars[i]:
                    if car.x - CAR_LENGTH >= INTERSECTION_EXIT + CAR_BUFFER:
                        self.active_cars.pop(i)
                    break


# =============================================================================
# 4. DEMONSTRATION VEHICLES / OBSTACLES
# =============================================================================

def create_cars():
    return [
        Car(1, "W", -24.0, 7.0),
        Car(2, "E", -30.0, 7.5),
        Car(3, "S", -36.0, 8.0),
        Car(4, "N", -42.0, 8.0),
    ]


def create_obstacles():
    """
    A stalled vehicle partly in V1's lane, upstream of the stop line.  It is
    offset 0.5 m towards the curb: with a perfectly centred obstacle the
    lateral gradient is exactly zero and the car would only brake (see notes).
    """
    centre = route_to_world("W", -14.0, -0.5)
    return [Obstacle(float(centre[0]), float(centre[1]), 1.0)]


# =============================================================================
# 5. SIMULATION
# =============================================================================

def run_simulation(obstacles=None, model=None, solver=None, cars=None):
    cars = cars or create_cars()
    controller = Controller(model, solver)
    obstacles = obstacles or []
    controller.obstacles = obstacles

    times = np.arange(0.0, SIM_TIME + DT, DT)

    for time in times:
        # A. APPROACH cars that have stopped become WAIT
        for car in cars:
            if car.state != "APPROACH":
                continue
            if abs(car.x - STOP_LINE) <= STOP_POS_TOL and car.vx <= STOP_SPEED_TOL:
                car.x, car.vx, car.vy = STOP_LINE, 0.0, 0.0
                car.last_ax = car.last_ay = 0.0
                controller.register_stopped_car(car, time)

        # B. Awareness and release
        controller.update_awareness(cars, obstacles)
        controller.update_lane_references(cars, obstacles)
        controller.release_cars(cars, time)

        # C. One (ax, ay) per car
        inputs = {}
        for car in cars:
            if car.state == "APPROACH":
                inputs[car.id] = controller.solve_stop_problem(car, cars)
            elif car.state == "GO":
                inputs[car.id] = controller.solve_go_problem(car, cars)
            else:
                inputs[car.id] = (0.0, 0.0)

        # D. Log and integrate the 4-term state
        for car in cars:
            if car.state == "DONE":
                continue
            ax, ay = inputs[car.id]

            car.time_history.append(time)
            car.position_history.append(car.x)
            car.lateral_history.append(car.y)
            car.speed_history.append(car.vx)
            car.lateral_speed_history.append(car.vy)
            car.acceleration_history.append(ax)
            car.lateral_accel_history.append(ay)
            car.state_history.append(car.state)

            if car.state == "WAIT":
                car.x, car.vx, car.vy = STOP_LINE, 0.0, 0.0
            else:
                car.x += car.vx * DT + 0.5 * ax * DT**2
                car.y += car.vy * DT + 0.5 * ay * DT**2
                car.vx += ax * DT
                car.vy += ay * DT
                car.last_ax, car.last_ay = ax, ay

            controller.update_prediction(car)

        controller.detect_collisions(cars, time)

        # E. Free the intersection
        controller.check_intersection(cars)

        # F. Remove finished cars
        for car in cars:
            if car.state == "GO" and car.x >= DELETE_POINT:
                car.state = "DONE"

        if all(car.state == "DONE" for car in cars):
            break

    return cars, controller


# =============================================================================
# 6. RESULTS
# =============================================================================

def min_gap(cars):
    """Smallest body-to-body gap between any two cars over the run [m]."""
    best = np.inf
    for i, a in enumerate(cars):
        for b in cars[i + 1:]:
            ta = {round(t, 6): k for k, t in enumerate(a.time_history)}
            for kb, t in enumerate(b.time_history):
                ka = ta.get(round(t, 6))
                if ka is None or a.state_history[ka] == "DONE":
                    continue
                ra = a.corners(a.position_history[ka], a.lateral_history[ka],
                               a.speed_history[ka], a.lateral_speed_history[ka])
                rb = b.corners(b.position_history[kb], b.lateral_history[kb],
                               b.speed_history[kb], b.lateral_speed_history[kb])
                ca, cb = ra.mean(axis=0), rb.mean(axis=0)
                if np.linalg.norm(ca - cb) > 12.0:
                    continue
                best = min(best, rectangle_gap_sym(ra, rb))
    return best


def summarize(cars, controller):
    finish = max(c.time_history[-1] for c in cars if c.time_history)
    return {
        "collisions": len({(a, b) for _, a, b in controller.collisions}),
        "obstacle_hits": len({c for _, c, _ in controller.obstacle_hits}),
        "min_gap": min_gap(cars),
        "finish": finish if all(c.state == "DONE" for c in cars) else np.inf,
        "max_y": max(max(np.abs(c.lateral_history)) for c in cars),
        "max_vy": max(max(np.abs(c.lateral_speed_history)) for c in cars),
        "solve_ms": 1000 * np.mean([t for c in cars for t in c.solve_times]),
    }


def print_summary(cars, controller):
    print(f"\nCrossing summary  (model={controller.model}, solver={controller.solver})")
    print("-" * 60)
    for car in sorted(cars, key=lambda c: c.go_time if c.go_time is not None else np.inf):
        stop = f"{car.stop_time:.1f} s" if car.stop_time is not None else "-"
        go = f"{car.go_time:.1f} s" if car.go_time is not None else "never"
        conflicts = ", ".join(f"V{i}" for i in car.conflicts) or "none"
        print(f"Vehicle {car.id} ({car.origin} -> {car.destination}): stop = {stop}, "
              f"go = {go}, max |y| = {max(np.abs(car.lateral_history)):.2f} m, "
              f"conflicts with {conflicts}")
    pairs = sorted({(a, b) for _, a, b in controller.collisions})
    print("Collisions:", ", ".join(f"V{a}-V{b}" for a, b in pairs) or "none")
    hits = sorted({c for _, c, _ in controller.obstacle_hits})
    print("Obstacle hits:", ", ".join(f"V{c}" for c in hits) or "none")
    print(f"Minimum car-to-car gap: {min_gap(cars):.2f} m")


def plot_results(cars):
    panels = [
        ("position_history", "Longitudinal position x [m]", "position_vs_time.png"),
        ("speed_history", "Longitudinal speed vx [m/s]", "speed_vs_time.png"),
        ("lateral_history", "Lateral offset y [m]", "lateral_vs_time.png"),
        ("lateral_speed_history", "Lateral speed vy [m/s]", "lateral_speed_vs_time.png"),
    ]
    for attr, label, fname in panels:
        plt.figure(figsize=(9, 5))
        for car in cars:
            plt.plot(car.time_history, getattr(car, attr), label=f"Vehicle {car.id} ({car.origin})")
        if attr == "position_history":
            plt.axhline(STOP_LINE, ls="--", lw=1, label="Stop line")
            plt.axhline(INTERSECTION_EXIT, ls=":", lw=1, label="Intersection exit")
        if attr == "speed_history":
            plt.axhline(V_DES, ls="--", lw=1, label="Desired speed")
        plt.xlabel("Time [s]")
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()


def animate_results(cars, obstacles=(), save_path=None, show=True, show_ellipses=True):
    max_frames = max(len(car.time_history) for car in cars)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor("#2e2e2e")
    ax.tick_params(colors="#aaaaaa")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.set_facecolor("#2e2e2e")
    ax.set_title("All-Way Stop 2-D MPC (elliptical potentials)", color="white", pad=20, fontsize=15)

    w = ROAD_WIDTH
    ax.add_patch(patches.Rectangle((0, 0), w, w, color="#555555", zorder=1))
    ax.add_patch(patches.Rectangle((-50, 0), 50, w, color="#444444", zorder=0))
    ax.add_patch(patches.Rectangle((12, 0), 50, w, color="#444444", zorder=0))
    ax.add_patch(patches.Rectangle((0, -50), w, 50, color="#444444", zorder=0))
    ax.add_patch(patches.Rectangle((0, 12), w, 50, color="#444444", zorder=0))
    for xs, ys in (([-50, 0], [6, 6]), ([12, 50], [6, 6]), ([6, 6], [-50, 0]), ([6, 6], [12, 50])):
        ax.plot(xs, ys, color="#e6c800", ls="--", zorder=2)
    for xs, ys in (([0, 0], [0, 6]), ([12, 12], [6, 12]), ([6, 12], [0, 0]), ([0, 6], [12, 12])):
        ax.plot(xs, ys, color="white", lw=4, zorder=2)

    for obs in obstacles:
        ax.add_patch(patches.Circle((obs.x, obs.y), obs.radius, color="#8e44ad", zorder=3))

    colors = {"APPROACH": "#f39c12", "WAIT": "#e74c3c", "GO": "#2ecc71"}
    rects, texts, ellipses = {}, {}, {}
    for car in cars:
        rects[car.id] = patches.Rectangle((0, 0), CAR_LENGTH, CAR_WIDTH, color="white",
                                          zorder=4, rotation_point="center")
        ax.add_patch(rects[car.id])
        texts[car.id] = ax.text(0, 0, f"V{car.id}", color="black", weight="bold",
                                fontsize=8, ha="center", va="center", zorder=5)
        ellipses[car.id] = patches.Ellipse((0, 0), 1, 1, fill=False, ls=":", lw=0.8,
                                           ec="#9ad0ff", zorder=3)
        ax.add_patch(ellipses[car.id])

    time_text = ax.text(-42, 41, "", color="white", fontsize=13, fontfamily="monospace")
    telem_text = ax.text(-42, 36, "", color="white", fontsize=9, fontfamily="monospace", va="top")

    def update(frame):
        time_text.set_text(f"Simulation Time: {frame * DT:05.2f} s")
        lines = ["TELEMETRY:"]
        for car in sorted(cars, key=lambda c: c.id):
            if frame < len(car.position_history):
                x, y = car.position_history[frame], car.lateral_history[frame]
                vx, vy = car.speed_history[frame], car.lateral_speed_history[frame]
                state = car.state_history[frame]
            else:
                state, x, y, vx, vy = "DONE", 0, 0, 0, 0

            r, t, e = rects[car.id], texts[car.id], ellipses[car.id]
            if state == "DONE":
                for art in (r, t, e):
                    art.set_alpha(0.0)
                continue

            lines.append(f"V{car.id}({car.origin}): {state:<8} x={x:>5.1f} y={y:>5.2f} "
                         f"vx={vx:>4.1f} vy={vy:>5.2f}")
            centre = body_center(car.origin, x, y)
            yaw = heading_angle(car.origin, vx, vy)
            r.set_xy((centre[0] - CAR_LENGTH / 2, centre[1] - CAR_WIDTH / 2))
            r.set_angle(np.degrees(yaw))
            r.set_facecolor(colors.get(state, "white"))
            r.set_alpha(1.0)
            t.set_position(centre)
            t.set_alpha(1.0)

            # Safety ellipse this car projects onto a perpendicular car
            A0, B0 = ellipse_axes("S" if car.origin in "WE" else "W", car.heading,
                                  CAR_LENGTH / 2, CAR_HALF_WIDTH)
            e.set_center(centre)
            e.width = 2 * (A0 + HEADWAY_TIME * abs(vx))
            e.height = 2 * B0
            e.angle = np.degrees(np.arctan2(car.heading[1], car.heading[0]))
            e.set_alpha(0.8 if show_ellipses else 0.0)

        telem_text.set_text("\n".join(lines))
        return list(rects.values()) + list(texts.values()) + list(ellipses.values()) + [time_text, telem_text]

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=int(DT * 1000), blit=True)
    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer=animation.PillowWriter(fps=int(1 / DT)))
    if show:
        plt.show()
    plt.close(fig)
    return ani


# =============================================================================
# 7. COMPARISON OF APPROACHES
# =============================================================================

def compare_variants():
    variants = [
        ("circles", "BFGS"), ("hinge", "BFGS"), ("gauss", "BFGS"),
        ("barrier", "BFGS"), ("hinge", "GD"), ("hinge", "L-BFGS-B"), ("hard", "SLSQP"),
    ]
    header = (f"{'scenario':<10}{'model':<9}{'solver':<10}{'coll':>5}{'obst':>5}"
              f"{'min gap':>9}{'finish':>8}{'max|y|':>8}{'max|vy|':>8}{'ms/solve':>10}")
    print(header)
    print("-" * len(header))
    rows = []
    for scenario in ("4-car", "obstacle"):
        for model, solver in variants:
            obstacles = create_obstacles() if scenario == "obstacle" else []
            cars, ctrl = run_simulation(obstacles, model, solver)
            s = summarize(cars, ctrl)
            rows.append((scenario, model, solver, s))
            print(f"{scenario:<10}{model:<9}{solver:<10}{s['collisions']:>5}{s['obstacle_hits']:>5}"
                  f"{s['min_gap']:>9.2f}{s['finish']:>8.1f}{s['max_y']:>8.2f}"
                  f"{s['max_vy']:>8.2f}{s['solve_ms']:>10.1f}", flush=True)
    return rows


# =============================================================================
# 8. MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=COLLISION_MODEL,
                        choices=["hinge", "gauss", "barrier", "circles", "hard"])
    parser.add_argument("--solver", default=SOLVER, choices=["BFGS", "GD", "L-BFGS-B"])
    parser.add_argument("--obstacle", action="store_true", help="add a stalled car on V1's approach")
    parser.add_argument("--compare", action="store_true", help="run every variant and print a table")
    parser.add_argument("--no-anim", action="store_true")
    parser.add_argument("--gif", default=None, help="save the animation to this .gif path")
    args = parser.parse_args()

    if args.compare:
        compare_variants()
    else:
        obstacles = create_obstacles() if args.obstacle else []
        cars, controller = run_simulation(obstacles, args.model, args.solver)
        print_summary(cars, controller)
        plot_results(cars)
        if not args.no_anim or args.gif:
            animate_results(cars, obstacles, save_path=args.gif, show=not args.no_anim)
