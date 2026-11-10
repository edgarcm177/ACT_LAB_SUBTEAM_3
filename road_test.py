"""
Road boundaries: soft penalty vs hard constraint.

Same test as obstacle_test.py, but the parked car is nearer the lane centre
(0.75 m right of it). Without road boundaries the car squeezes past on the
right and leaves the road. The test runs the three ROAD_MODE settings of the
controller and plots the paths, the lateral offset, the speed and the
lateral acceleration.

    none   no road boundaries
    soft   W_ROAD * (distance past the edge)^2 added to the cost
    hard   ROAD_Y_MIN <= y(k) <= ROAD_Y_MAX as constraints (SLSQP solver)

Run:
    python3 road_test.py                 # plots (also saved to road_test.png)
    python3 road_test.py --animate       # top-down animation of the three runs
    python3 road_test.py --save-video    # write road_test.mp4
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import all_way_stop as sim
import obstacle_test as ot

OBSTACLE_Y = -0.75      # parked car 0.75 m right of the lane centre

MODES = [
    ("none", "no road boundary", "#8A949E"),
    ("soft", "soft penalty", "#008C95"),
    ("hard", "hard constraint (SLSQP)", "#003057"),
]
INK, MUTED, GRID = "#1F2933", "#5B6770", "#E6E8EB"
EDGE_COLOR, OBSTACLE_COLOR = "#1F2933", "#C0504D"


def run_modes():
    ot.OBSTACLE_Y = OBSTACLE_Y
    runs = []
    print(f"Parked car {abs(OBSTACLE_Y):g} m right of the lane centre, hinge ellipse, w_obs = {ot.W_OBS:g}\n")
    for mode, label, color in MODES:
        sim.ROAD_MODE = mode
        h = ot.simulate("hinge", ot.W_OBS, 0.0)
        print(f"{label:<26} y from {h['y'].min():5.2f} to {h['y'].max():5.2f} m | min speed {h['vx'].min():4.1f} m/s | "
              f"peak |ay| {np.max(np.abs(h['ay'])):4.2f} m/s² | closest d² {h['d2'].min():.3f}")
        runs.append((mode, label, color, h))
    return runs


def outcome(h):
    """One-line description of what the car did."""
    if h["y"].min() < sim.ROAD_Y_MIN - 0.05:
        return f"leaves the road (y = {h['y'].min():.2f} m)"
    if h["vx"].min() < 0.9 * sim.V_DES:
        return f"passes left, but slows to {h['vx'].min():.1f} m/s"
    return f"passes left at full speed ({h['vx'].min():.1f} m/s)"


def plot(runs):
    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
    top = fig.add_subplot(grid[0, :])
    panels = [fig.add_subplot(grid[1, i]) for i in range(3)]

    # Top: the road seen from above (sideways scale exaggerated)
    x0, x1 = ot.SCENE_X
    top.axhspan(-ot.LANE_HALF, 3 * ot.LANE_HALF, color="#F1F3F5", zorder=0)
    top.axhline(-ot.LANE_HALF, color=EDGE_COLOR, linewidth=2, zorder=1)
    top.axhline(ot.LANE_HALF, color="#C9A227", linewidth=1.5, linestyle=(0, (8, 6)), zorder=1)
    top.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (1, 3)), zorder=1)
    top.axhline(sim.ROAD_Y_MIN, color=EDGE_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    top.text(x0 + 0.5, -ot.LANE_HALF + 0.12, "road edge", color=INK, fontsize=9.5, va="bottom")
    top.text(x0 + 0.5, sim.ROAD_Y_MIN + 0.12, f"boundary for the car's centre (y = {sim.ROAD_Y_MIN:g} m)",
             color=INK, fontsize=9.5, va="bottom")
    top.text(x0 + 0.5, 0.12, "lane centre", color=MUTED, fontsize=9.5, va="bottom")
    top.text(x0 + 0.5, ot.LANE_HALF - 0.12, "lane line (oncoming lane above)", color=MUTED, fontsize=9.5, va="top")

    top.add_patch(patches.Ellipse((ot.OBSTACLE_X, OBSTACLE_Y), 2 * ot.ELLIPSE_A, 2 * ot.ELLIPSE_B,
                                  facecolor=OBSTACLE_COLOR, alpha=0.15, edgecolor=OBSTACLE_COLOR,
                                  linewidth=1.5, linestyle="--", zorder=2))
    top.add_patch(patches.Rectangle((ot.OBSTACLE_X - ot.OBSTACLE_LENGTH / 2, OBSTACLE_Y - ot.OBSTACLE_WIDTH / 2),
                                    ot.OBSTACLE_LENGTH, ot.OBSTACLE_WIDTH, facecolor=OBSTACLE_COLOR,
                                    edgecolor="white", linewidth=1.2, zorder=3))
    top.text(ot.OBSTACLE_X, OBSTACLE_Y + ot.ELLIPSE_B + 0.1, "parked car + ellipse", color=OBSTACLE_COLOR,
             fontsize=9.5, fontweight="bold", ha="center", va="bottom", zorder=4)

    for mode, label, color, h in runs:
        cx = ot.car_centre(h["x"])
        top.plot(cx, h["y"], color=color, linewidth=2.5, label=f"{label}  ·  {outcome(h)}", zorder=5)

    top.set_xlim(x0, x1)
    top.set_ylim(-ot.LANE_HALF - 1.0, ot.LANE_HALF + 0.6)
    top.set_xlabel("distance along the road x [m]")
    top.set_ylabel("sideways position y [m]")
    top.set_title("Path of the car's centre (sideways scale exaggerated)", loc="left", fontsize=12,
                  color=INK, fontweight="bold")
    top.legend(loc="center right", frameon=True, framealpha=0.95, fontsize=10)

    # Bottom: lateral offset, speed and lateral acceleration over time
    for mode, label, color, h in runs:
        panels[0].plot(h["t"], h["y"], color=color, linewidth=2, label=label)
        panels[1].plot(h["t"], h["vx"], color=color, linewidth=2, label=label)
        panels[2].step(h["t"], h["ay"], where="post", color=color, linewidth=2, label=label)

    panels[0].axhline(sim.ROAD_Y_MIN, color=EDGE_COLOR, linewidth=1.2, linestyle=(0, (4, 3)))
    panels[0].text(0.1, sim.ROAD_Y_MIN + 0.1, "road boundary", color=INK, fontsize=9, va="bottom")
    panels[0].set_ylabel("sideways position y [m]")
    panels[1].axhline(sim.V_DES, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    panels[1].set_ylabel("speed vx [m/s]")
    panels[1].set_ylim(0, sim.V_DES + 1)
    for limit in (-sim.AY_MAX, sim.AY_MAX):
        panels[2].axhline(limit, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    panels[2].set_ylabel("sideways acceleration ay [m/s²]")

    for ax, title in zip(panels, ("Sideways position", "Speed", "Sideways acceleration")):
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold")
        ax.set_xlabel("time [s]")
        ax.grid(True, color=GRID)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Road boundaries: passing a parked car {abs(OBSTACLE_Y):g} m right of the lane centre",
                 fontsize=14, fontweight="bold", color=INK, x=0.06, ha="left")
    fig.savefig("road_test.png", dpi=150, bbox_inches="tight")
    print("\nSaved road_test.png")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animate", action="store_true", help="also open a top-down animation of the three runs")
    parser.add_argument("--save-video", action="store_true", help="write the animation to road_test.mp4")
    args = parser.parse_args()

    runs = run_modes()
    plot(runs)

    anim = None
    if args.animate or args.save_video:
        titles = [(f"{label}  ·  {outcome(h)}", h) for _, label, _, h in runs]
        anim = ot.animate(titles, save_path="road_test.mp4" if args.save_video else None)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
