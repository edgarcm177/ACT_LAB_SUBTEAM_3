"""
Elliptical obstacle term: one car, one parked obstacle.

The car drives on the west approach and a parked car partly blocks its lane.
The obstacle gets an ellipse that no part of the car should enter. It is
checked at points around the car's outline (turned with its heading):

    d^2 = ((p_x - x_o) / a)^2 + ((p_y - y_o) / b)^2      (inside when d^2 < 1)

    p_x, p_y one point on the car's outline
    x_o, y_o centre of the obstacle
    a, b     ellipse semi-axes: half the obstacle + a safety margin

The MPC cost is the GO cost from all_way_stop.py plus

    W_OBS * sum_k sum_points c(d^2)      (+ W_LANE * sum_k y(k)^2, optional)

for one of three candidate shapes c:

    hinge     max(0, 1 - d^2)^2    zero outside the ellipse
    gaussian  exp(-d^2)            smooth bump that reaches a bit outside it
    inverse   1 / (d^2 + 1)        slide 9; never zero, so it reaches far

Run:
    python3 obstacle_test.py                     # hinge shape, animation
    python3 obstacle_test.py --shape gaussian    # another shape
    python3 obstacle_test.py --compare           # all three shapes, one road each
    python3 obstacle_test.py --lane-weight 20    # also pull the car back to its lane centre
    python3 obstacle_test.py --save-video        # write an .mp4 instead of opening a window
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

import all_way_stop as sim

STEPS = 45                      # 9 s of simulation
X_START = -40.0                 # car's front bumper at the start [m]

# Parked car against the right edge of the lane, so the way around is to the
# left. Dead centre, the ellipse gives no sideways push (left and right look
# the same) and the car does not steer.
OBSTACLE_X, OBSTACLE_Y = -10.0, -1.5
OBSTACLE_LENGTH, OBSTACLE_WIDTH = sim.CAR_LENGTH, 2 * sim.CAR_HALF_WIDTH
MARGIN_X, MARGIN_Y = 1.5, 0.5
ELLIPSE_A = OBSTACLE_LENGTH / 2 + MARGIN_X          # 3.75 m
ELLIPSE_B = OBSTACLE_WIDTH / 2 + MARGIN_Y           # 1.5 m

# Points on the car's outline, relative to its centre before turning it by
# its heading: 7 along each long side, plus the middle of each end.
_SIDE_X = np.linspace(-sim.CAR_LENGTH / 2, sim.CAR_LENGTH / 2, 7)
BODY_POINTS = np.array(
    [(dx, side * sim.CAR_HALF_WIDTH) for side in (-1, 1) for dx in _SIDE_X]
    + [(end * sim.CAR_LENGTH / 2, 0.0) for end in (-1, 1)]
)

W_OBS = 1.0e5                   # default weight on the obstacle term (keeps the outline outside)

SHAPES = {
    "hinge": ("max(0, 1 − d²)²", lambda d2: np.maximum(0.0, 1.0 - d2) ** 2),
    "gaussian": ("exp(−d²)", lambda d2: np.exp(-d2)),
    "inverse": ("1 / (d² + 1)", lambda d2: 1.0 / (d2 + 1.0)),
}

# Animation
FPS = 30
DPI = 120
SLOWDOWN = 2.0                  # 2 = half speed
SCENE_X = (-45.0, 25.0)         # stretch of road shown [m]
LANE_HALF = sim.LANE_OFFSET     # own lane: y from -3 to 3 m; oncoming lane: 3 to 9 m
BAND = 4 * LANE_HALF + 5.0      # height of one road plus the gap above it [m]
GRASS, ROAD, YELLOW = "#6E9F5E", "#3B4046", "#C9A227"
CAR_COLOR, OBSTACLE_COLOR, ELLIPSE_COLOR = "#008C95", "#C0504D", "#F28B82"
TRAIL_COLOR, PLAN_COLOR = "#4FC3F7", "#FFFFFF"


def car_centre(x):
    """x is the front bumper; the body is drawn and checked around its centre."""
    return x - 0.5 * sim.CAR_LENGTH


def body_d2(x, y, vx, vy):
    """d^2 of every outline point of the car, for one or more states -> shape (K, P)."""

    x, y, vx, vy = map(np.atleast_1d, (x, y, vx, vy))
    heading = np.arctan2(vy, vx)[:, None]
    cos, sin = np.cos(heading), np.sin(heading)
    px = car_centre(x)[:, None] + BODY_POINTS[:, 0] * cos - BODY_POINTS[:, 1] * sin
    py = y[:, None] + BODY_POINTS[:, 0] * sin + BODY_POINTS[:, 1] * cos
    return ((px - OBSTACLE_X) / ELLIPSE_A) ** 2 + ((py - OBSTACLE_Y) / ELLIPSE_B) ** 2


def closest_d2(x, y, vx, vy):
    """Smallest d^2 over the car's outline: below 1 means part of the car is inside."""
    return body_d2(x, y, vx, vy).min(axis=1)


def simulate(shape, w_obs, w_lane):
    """Drive one car past the obstacle with the MPC; return its history and plans."""

    controller = sim.Controller()
    car = sim.Car(1, "W", X_START, sim.V_DES)
    c = SHAPES[shape][1]

    def cost(u):
        x, y, vx, vy = controller.predict(car, u)
        J = controller.go_cost(u, car, [])                               # speed, ax^2, ay^2, vy^2 terms
        J += w_obs * np.sum(c(body_d2(x[1:], y[1:], vx[1:], vy[1:])))   # obstacle ellipse
        J += w_lane * np.sum(y[1:] ** 2)                                 # optional lane centring
        return J

    hist = {key: [] for key in ("t", "x", "y", "vx", "vy", "ay", "d2", "plan_x", "plan_y")}
    for k in range(STEPS):
        result = controller.optimize(cost, car.go_guess, car)   # applies sim.ROAD_MODE
        car.go_guess = result.x
        plan_x, plan_y, _, _ = controller.predict(car, result.x)
        ax_k, ay_k = result.x[0], result.x[sim.N]

        hist["t"].append(k * sim.DT)
        hist["x"].append(car.x)
        hist["y"].append(car.y)
        hist["vx"].append(car.vx)
        hist["vy"].append(car.vy)
        hist["ay"].append(ay_k)
        hist["d2"].append(closest_d2(car.x, car.y, car.vx, car.vy)[0])
        hist["plan_x"].append(car_centre(plan_x))
        hist["plan_y"].append(plan_y)

        car.x += car.vx * sim.DT + 0.5 * ax_k * sim.DT**2
        car.y += car.vy * sim.DT + 0.5 * ay_k * sim.DT**2
        car.vx += ax_k * sim.DT
        car.vy += ay_k * sim.DT

    return {key: np.array(values) for key, values in hist.items()}


def summarize(title, h):
    print(f"{title:<46} min d² {h['d2'].min():5.2f} | max offset {np.max(np.abs(h['y'])):5.2f} m | "
          f"final offset {h['y'][-1]:5.2f} m | min speed {h['vx'].min():4.1f} m/s | "
          f"peak |ay| {np.max(np.abs(h['ay'])):4.2f} m/s²")


def draw_band(ax, y_off, title):
    """One straight road (own lane + oncoming lane) with the obstacle and its ellipse."""

    x0, x1 = SCENE_X
    ax.add_patch(patches.Rectangle((x0, y_off - LANE_HALF), x1 - x0, 4 * LANE_HALF, color=ROAD, zorder=0))
    for edge in (-LANE_HALF, 3 * LANE_HALF):
        ax.plot([x0, x1], [y_off + edge] * 2, color="white", linewidth=2.5, zorder=1)
    ax.plot([x0, x1], [y_off + LANE_HALF] * 2, color=YELLOW, linewidth=1.5, linestyle=(0, (10, 10)), zorder=1)
    ax.plot([x0, x1], [y_off, y_off], color="white", linewidth=0.8, linestyle=(0, (1, 4)), alpha=0.6, zorder=1)

    ax.add_patch(patches.Ellipse((OBSTACLE_X, y_off + OBSTACLE_Y), 2 * ELLIPSE_A, 2 * ELLIPSE_B,
                                 facecolor=ELLIPSE_COLOR, alpha=0.25, edgecolor="none", zorder=2))
    ax.add_patch(patches.Ellipse((OBSTACLE_X, y_off + OBSTACLE_Y), 2 * ELLIPSE_A, 2 * ELLIPSE_B,
                                 facecolor="none", edgecolor=ELLIPSE_COLOR, linewidth=2, linestyle="--", zorder=2))
    ax.add_patch(patches.Rectangle((OBSTACLE_X - OBSTACLE_LENGTH / 2, y_off + OBSTACLE_Y - OBSTACLE_WIDTH / 2),
                                   OBSTACLE_LENGTH, OBSTACLE_WIDTH, facecolor=OBSTACLE_COLOR,
                                   edgecolor="white", linewidth=1.5, zorder=3))
    ax.text(OBSTACLE_X, y_off + OBSTACLE_Y, "obstacle", color="white", fontsize=9, fontweight="bold",
            ha="center", va="center", zorder=4)

    box = dict(boxstyle="round,pad=0.3", facecolor="#1F2933", alpha=0.75, edgecolor="none")
    ax.text(x0 + 0.8, y_off + 3 * LANE_HALF + 2.0, title, color="white", fontsize=12, fontweight="bold",
            va="center", bbox=box)
    return ax.text(x1 - 0.8, y_off + 3 * LANE_HALF + 2.0, "", color="white", fontsize=11, family="monospace",
                   ha="right", va="center", bbox=box)


def animate(runs, save_path=None):
    """runs = [(title, history), ...]; one road per run, top to bottom."""

    x0, x1 = SCENE_X
    n = len(runs)
    y_min = -LANE_HALF - 3.5
    y_max = (n - 1) * BAND + 3 * LANE_HALF + 7.0
    width_in = 14.0
    height_in = 2 * round(width_in * (y_max - y_min) / (x1 - x0) * DPI / 2) / DPI

    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((x0, y_min), x1 - x0, y_max - y_min, color=GRASS, zorder=-1))

    box = dict(boxstyle="round,pad=0.35", facecolor="#1F2933", alpha=0.75, edgecolor="none")
    ax.text(x0 + 0.8, y_max - 2.2, "Car going around a parked obstacle   ·   red ellipse: no part of "
            "the car may enter", color="white", fontsize=12, fontweight="bold", va="center", bbox=box)
    clock = ax.text(x1 - 0.8, y_max - 2.2, "", color="white", fontsize=13, fontweight="bold",
                    family="monospace", ha="right", va="center", bbox=box)
    ax.text(x0 + 0.8, y_min + 1.6, "white dashed: MPC plan for the next 4 s   ·   blue: path so far   ·   "
            f"closest d² < 1 means part of the car is inside   ·   playback {1 / SLOWDOWN:g}× real time",
            color="white", fontsize=10, va="center")

    scenes = []
    for i, (title, h) in enumerate(runs):
        y_off = (n - 1 - i) * BAND
        readout = draw_band(ax, y_off, title)
        plan, = ax.plot([], [], color=PLAN_COLOR, linewidth=1.2, linestyle=(0, (3, 3)), alpha=0.85, zorder=3)
        trail, = ax.plot([], [], color=TRAIL_COLOR, linewidth=2.5, zorder=3)
        body = patches.Rectangle((0, 0), sim.CAR_LENGTH, 2 * sim.CAR_HALF_WIDTH, rotation_point="center",
                                 facecolor=CAR_COLOR, edgecolor="white", linewidth=1.5, zorder=4)
        ax.add_patch(body)
        scenes.append((h, y_off, plan, trail, body, readout))

    # Stop shortly after the last car has left the picture
    t_last = (STEPS - 1) * sim.DT
    exit_times = [np.interp(x1 + sim.CAR_LENGTH, h["x"], h["t"]) for h, *_ in scenes]
    t_end = min(t_last, max(exit_times) + 0.3)

    def update(frame):
        time = min(frame / (SLOWDOWN * FPS), t_end)
        k = min(int(round(time / sim.DT)), STEPS - 1)
        for h, y_off, plan, trail, body, readout in scenes:
            x, y = np.interp(time, h["t"], h["x"]), np.interp(time, h["t"], h["y"])
            vx, vy = np.interp(time, h["t"], h["vx"]), np.interp(time, h["t"], h["vy"])

            past = h["t"] <= time
            trail.set_data(np.append(car_centre(h["x"][past]), car_centre(x)), y_off + np.append(h["y"][past], y))
            plan.set_data(h["plan_x"][k], y_off + h["plan_y"][k])
            body.set_xy((x - sim.CAR_LENGTH, y_off + y - sim.CAR_HALF_WIDTH))   # x is the front bumper
            body.set_angle(np.degrees(np.arctan2(vy, vx)))
            readout.set_text(f"speed {vx:4.1f} m/s   offset {y:5.2f} m   closest d² {closest_d2(x, y, vx, vy)[0]:5.2f}")
        clock.set_text(f"t = {time:3.1f} s")
        return []

    n_frames = int(t_end * SLOWDOWN * FPS) + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)

    if save_path:
        writer = animation.FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        anim.save(save_path, writer=writer, dpi=DPI)
        plt.close(fig)
        print(f"Saved {save_path}")
    return anim


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=list(SHAPES), default="hinge", help="obstacle cost shape")
    parser.add_argument("--compare", action="store_true", help="run all three shapes, one road each")
    parser.add_argument("--w-obs", type=float, default=W_OBS, help="weight on the obstacle term")
    parser.add_argument("--lane-weight", type=float, default=0.0, help="weight on y^2 (pull back to lane centre)")
    parser.add_argument("--road", choices=["soft", "hard", "none"], default=sim.ROAD_MODE,
                        help="road boundaries: soft penalty, hard constraint, or none")
    parser.add_argument("--save-video", action="store_true", help="write an .mp4 instead of opening a window")
    args = parser.parse_args()
    sim.ROAD_MODE = args.road

    shapes = list(SHAPES) if args.compare else [args.shape]
    runs = []
    for shape in shapes:
        title = f"{shape}  {SHAPES[shape][0]}   ·   w_obs = {args.w_obs:g}"
        if args.lane_weight:
            title += f"   ·   w_lane = {args.lane_weight:g}"
        h = simulate(shape, args.w_obs, args.lane_weight)
        summarize(title, h)
        runs.append((title, h))

    name = "compare" if args.compare else args.shape
    save_path = f"obstacle_test_{name}.mp4" if args.save_video else None
    anim = animate(runs, save_path)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
