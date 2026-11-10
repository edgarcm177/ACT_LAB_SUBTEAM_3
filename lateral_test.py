"""
Test the lateral (sideways) part of the 2-D controller.

One car drives alone on the west approach and starts with a sideways drift
of 1 m/s. The GO MPC has to stop the drift. The test is repeated for several
values of W_ACCEL_Y (w2, cost on ay^2) and W_LAT_SPEED (w4, cost on vy^2).

Run:
    python3 lateral_test.py                  # animation of the w2 sweep
    python3 lateral_test.py --sweep w4       # animation of the w4 sweep
    python3 lateral_test.py --save-video     # write lateral_test_w2.mp4 instead of opening a window
    python3 lateral_test.py --plots          # also open the y / vy / ay plots

The plots are always saved to lateral_test.png.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

import all_way_stop as sim

STEPS = 25          # 5 s of simulation
VY0 = 1.0           # initial sideways drift [m/s]

W2_VALUES = [0.2, 5.0, 50.0]      # W_ACCEL_Y sweep, with W_LAT_SPEED = 150
W4_VALUES = [15.0, 150.0, 1500.0] # W_LAT_SPEED sweep, with W_ACCEL_Y = 0.2
COLORS = ["#9ECAE1", "#4292C6", "#08519C"]   # light -> dark = small -> large weight

# Animation
FPS = 30
DPI = 120
SLOWDOWN = 2.0                  # 2 = half speed, so the first second of drift is easy to follow
SCENE_X = (-46.0, -12.0)        # stretch of road shown [m]; the cars start at x = -40 m
LANE_W = 2 * sim.LANE_OFFSET    # lane width in the intersection model [m]
GRASS, ROAD = "#6E9F5E", "#3B4046"
CAR_COLOR, TRAIL_COLOR = "#008C95", "#4FC3F7"


def run(w_accel_y, w_lat_speed):
    """Drive one car with an initial drift and return its history."""

    sim.W_ACCEL_Y = w_accel_y
    sim.W_LAT_SPEED = w_lat_speed

    controller = sim.Controller()
    car = sim.Car(1, "W", -40.0, sim.V_DES)
    car.vy = VY0

    hist = {"t": [0.0], "x": [car.x], "y": [car.y], "vx": [car.vx], "vy": [car.vy], "ay": []}
    for k in range(STEPS):
        ax_k, ay_k = controller.solve_go_problem(car, [])

        car.x += car.vx * sim.DT + 0.5 * ax_k * sim.DT**2
        car.y += car.vy * sim.DT + 0.5 * ay_k * sim.DT**2
        car.vx += ax_k * sim.DT
        car.vy += ay_k * sim.DT

        hist["t"].append((k + 1) * sim.DT)
        hist["x"].append(car.x)
        hist["y"].append(car.y)
        hist["vx"].append(car.vx)
        hist["vy"].append(car.vy)
        hist["ay"].append(ay_k)

    return {key: np.array(values) for key, values in hist.items()}


def summarize(label, r):
    still = np.flatnonzero(np.abs(r["vy"]) < 0.05)
    settle = f"{r['t'][still[0]]:.1f} s" if still.size else "never"
    print(f"{label:<22} drift stopped at {settle:>7} | final offset {r['y'][-1]:5.2f} m | "
          f"peak |ay| {np.max(np.abs(r['ay'])):4.2f} m/s^2")


def run_sweeps():
    """Return {"w2": (title, runs), "w4": (title, runs)}, runs = [(label, color, history), ...]."""

    sweeps = {
        "w2": ("Sweep of w2 (W_ACCEL_Y), w4 = 150", [(w2, 150.0) for w2 in W2_VALUES], W2_VALUES),
        "w4": ("Sweep of w4 (W_LAT_SPEED), w2 = 0.2", [(0.2, w4) for w4 in W4_VALUES], W4_VALUES),
    }
    results = {}
    for key, (title, cases, values) in sweeps.items():
        print(f"\n{title}")
        runs = []
        for (w2, w4), value, color in zip(cases, values, COLORS):
            label = f"{key} = {value:g}"
            r = run(w2, w4)
            summarize(label, r)
            runs.append((label, color, r))
        results[key] = (title, runs)
    return results


def plot_runs(results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)

    for row, (title, runs) in enumerate(results.values()):
        for label, color, r in runs:
            axes[row, 0].plot(r["t"], r["y"], color=color, linewidth=2, label=label)
            axes[row, 1].plot(r["t"], r["vy"], color=color, linewidth=2, label=label)
            axes[row, 2].step(r["t"][:-1], r["ay"], where="post", color=color, linewidth=2, label=label)

        axes[row, 0].set_ylabel(f"{title.split(',')[0]}\n\nLateral offset y [m]")
        axes[row, 1].set_ylabel("Lateral speed vy [m/s]")
        axes[row, 2].set_ylabel("Lateral acceleration ay [m/s²]")
        axes[row, 0].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(True, color="#E6E8EB")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(f"One car starting with {VY0:g} m/s of sideways drift", fontweight="bold")
    fig.tight_layout()
    fig.savefig("lateral_test.png", dpi=150)
    print("\nSaved lateral_test.png")
    return fig


def draw_road(ax, n_lanes):
    """Straight road seen from above, one lane per run. Returns the lane-centre heights, top first."""

    x0, x1 = SCENE_X
    top = n_lanes * LANE_W
    ax.add_patch(patches.Rectangle((x0, 0.0), x1 - x0, top, color=ROAD, zorder=0))
    for edge in (0.0, top):
        ax.plot([x0, x1], [edge, edge], color="white", linewidth=2.5, zorder=1)
    for i in range(1, n_lanes):
        ax.plot([x0, x1], [i * LANE_W] * 2, color="white", linewidth=1.5, linestyle=(0, (10, 10)), zorder=1)

    centres = [top - (i + 0.5) * LANE_W for i in range(n_lanes)]
    for yc in centres:
        ax.plot([x0, x1], [yc, yc], color="white", linewidth=0.8, linestyle=(0, (1, 4)), alpha=0.6, zorder=1)
    return centres


def animate_runs(title, runs, save_path=None):
    """Cars driving across a fixed stretch of road, one lane per run."""

    x0, x1 = SCENE_X
    road_h = len(runs) * LANE_W
    y_min, y_max = -2.5, road_h + 5.0
    width_in = 12.0
    height_in = 2 * round(width_in * (y_max - y_min) / (x1 - x0) * DPI / 2) / DPI

    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((x0, y_min), x1 - x0, y_max - y_min, color=GRASS, zorder=-1))
    centres = draw_road(ax, len(runs))

    box = dict(boxstyle="round,pad=0.35", facecolor="#1F2933", alpha=0.75, edgecolor="none")
    ax.text(x0 + 0.8, road_h + 2.5, f"{title}   ·   start: {VY0:g} m/s sideways drift",
            color="white", fontsize=13, fontweight="bold", va="center", bbox=box)
    clock = ax.text(x1 - 0.8, road_h + 2.5, "", color="white", fontsize=13, fontweight="bold",
                    family="monospace", ha="right", va="center", bbox=box)
    ax.text(x0 + 0.8, -1.3, f"dotted line: lane centre   ·   blue line: path   ·   "
            f"playback {1 / SLOWDOWN:g}× real time", color="white", fontsize=10, va="center")

    cars = []
    for (label, _, r), yc in zip(runs, centres):
        ax.text(x0 + 0.6, yc + 2.2, label, color="white", fontsize=12, fontweight="bold", va="center", zorder=5)
        readout = ax.text(x1 - 0.6, yc + 2.2, "", color="white", fontsize=11, family="monospace",
                          ha="right", va="center", zorder=5)
        trail, = ax.plot([], [], color=TRAIL_COLOR, linewidth=2.5, zorder=3)
        body = patches.Rectangle((0, 0), sim.CAR_LENGTH, 2 * sim.CAR_HALF_WIDTH, rotation_point="center",
                                 facecolor=CAR_COLOR, edgecolor="white", linewidth=1.5, zorder=4)
        ax.add_patch(body)
        cars.append((r, yc, trail, body, readout))

    # Stop shortly after the last car has left the picture
    exit_times = [np.interp(x1 + sim.CAR_LENGTH, r["x"], r["t"]) for r, *_ in cars]
    t_end = min(STEPS * sim.DT, max(exit_times) + 0.5)

    def update(frame):
        time = min(frame / (SLOWDOWN * FPS), t_end)
        k = min(int(time / sim.DT), STEPS - 1)
        for r, yc, trail, body, readout in cars:
            x, y = np.interp(time, r["t"], r["x"]), np.interp(time, r["t"], r["y"])
            vx, vy = np.interp(time, r["t"], r["vx"]), np.interp(time, r["t"], r["vy"])

            past = r["t"] <= time
            trail.set_data(np.append(r["x"][past], x), yc + np.append(r["y"][past], y))
            body.set_xy((x - sim.CAR_LENGTH, yc + y - sim.CAR_HALF_WIDTH))   # x is the front bumper
            body.set_angle(np.degrees(np.arctan2(vy, vx)))
            readout.set_text(f"offset {y:4.2f} m   vy {vy:4.2f} m/s   ay {r['ay'][k]:5.2f} m/s²")
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
    parser.add_argument("--sweep", choices=["w2", "w4"], default="w2", help="which weight sweep to animate")
    parser.add_argument("--save-video", action="store_true",
                        help="write the animation to lateral_test_<sweep>.mp4 instead of opening a window")
    parser.add_argument("--plots", action="store_true", help="also open the y / vy / ay plots")
    args = parser.parse_args()

    results = run_sweeps()
    fig = plot_runs(results)
    if not args.plots:
        plt.close(fig)

    title, runs = results[args.sweep]
    save_path = f"lateral_test_{args.sweep}.mp4" if args.save_video else None
    anim = animate_runs(title, runs, save_path)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
