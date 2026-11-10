"""Trace-driven visual evidence for fair_mpc; no independent simulated motion."""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Rectangle
import numpy as np

from fair_mpc import (A_MAX, A_MIN, CAR_LENGTH, CAR_WIDTH, DT, MAX_WAIT,
                      REAR_CLEAR, V_DES, body_clearance, write_csv, xy)

POLICIES = ("fifo", "fair")
POLICY_NAMES = {"fifo": "FIFO", "fair": "Wait-aware"}
POLICY_COLORS = {"fifo": "#64748b", "fair": "#0f766e"}
STATE_COLORS = {"APPROACH": "#3b82f6", "WAIT": "#ea580c", "GO": "#0f766e", "DONE": "#94a3b8"}
TITLES = {"blocked_pair": "Queue blocking", "simultaneous": "Simultaneous arrivals",
          "staggered": "Staggered arrivals", "slow_approach": "Slow approach · tied result"}


def save_figure(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def events_for(cars):
    return [{"vehicle": c.id, "approach": c.approach, "stop_s": c.stop_time,
             "release_s": c.release_time, "rear_clear_s": c.exit_time,
             "wait_s": c.release_time - c.stop_time} for c in cars]


def event_audit(result):
    """Check chart events against the sampled trace and recorded summary."""
    cars, rows, metrics, batches = result
    events = events_for(cars)
    for c, e in zip(cars, events):
        data = [r for r in rows if r["vehicle"] == c.id]
        # A zero-wait car transitions directly from APPROACH to GO in the trace.
        self_stop = next(r["time_s"] for r in data if r["state"] != "APPROACH")
        self_release = next(r["time_s"] for r in data if r["state"] == "GO")
        assert abs(self_stop - e["stop_s"]) < 1e-7
        assert abs(self_release - e["release_s"]) < 1e-7
        first_clear = next(r["time_s"] + DT for r in data
                           if r["position_m"] + r["speed_mps"] * DT +
                           .5 * r["accel_mps2"] * DT**2 >= REAR_CLEAR)
        assert abs(first_clear - e["rear_clear_s"]) < 1e-7
    assert abs(np.mean([e["wait_s"] for e in events]) - metrics["mean_wait_s"]) < 1e-7
    assert abs(max(e["rear_clear_s"] for e in events) - metrics["last_rear_clear_s"]) < 1e-7
    for t, ids in batches:
        assert all(abs(next(c.release_time for c in cars if c.id == i) - t) < 1e-7 for i in ids)


def timeline(ax, cars, xmax, title=None):
    for i, e in enumerate(events_for(cars)):
        ax.barh(i, e["stop_s"], color=STATE_COLORS["APPROACH"], height=.58)
        ax.barh(i, e["wait_s"], left=e["stop_s"], color=STATE_COLORS["WAIT"], height=.58)
        ax.barh(i, e["rear_clear_s"] - e["release_s"], left=e["release_s"],
                color=STATE_COLORS["GO"], height=.58)
        ax.plot(e["rear_clear_s"], i, "|", color="#0f172a", markersize=13)
        if e["wait_s"] > .8:
            ax.text(e["stop_s"] + e["wait_s"] / 2, i, f"{e['wait_s']:.1f}s",
                    ha="center", va="center", fontsize=8, color="white")
    ax.set(yticks=range(len(cars)), yticklabels=[f"V{c.id} · {c.approach}" for c in cars],
           xlim=(0, xmax), ylim=(len(cars) - .4, -.6), xlabel="Simulation time [s]")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.15)


def plot_timelines(output, results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for r, name in enumerate(("simultaneous", "blocked_pair")):
        xmax = max(results[name][p][2]["last_rear_clear_s"] for p in POLICIES) + 1
        for col, p in enumerate(POLICIES):
            cars = results[name][p][0]
            timeline(axes[r, col], cars, xmax, f"{TITLES[name]} / {POLICY_NAMES[p]}")
    fig.legend(handles=[Patch(color=STATE_COLORS[s], label=label) for s, label in
                        (("APPROACH", "Approaching"), ("WAIT", "Waiting (duration labeled)"),
                         ("GO", "Released → rear clear"))], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Who crosses together — and who waits?", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, .06, 1, .95))
    save_figure(fig, output, "crossing_timeline")


def plot_vehicle_waits(output, results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for ax, name in zip(axes.flat, TITLES):
        for p, offset in (("fifo", -.19), ("fair", .19)):
            events = events_for(results[name][p][0])
            bars = ax.bar(np.arange(len(events)) + offset, [e["wait_s"] for e in events],
                          width=.38, color=POLICY_COLORS[p], label=POLICY_NAMES[p])
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.axhline(MAX_WAIT, color="#9a3412", linestyle=":", linewidth=1)
        ax.set(xticks=range(len(events)), xticklabels=[f"V{e['vehicle']} ({e['approach']})" for e in events],
               ylabel="Stopped waiting [s]", title=TITLES[name], ylim=(0, 14.5))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Individual waiting times — averages can hide who loses", fontsize=16, fontweight="bold")
    fig.text(.5, .025, "Dotted line: 6 s oldest-car eligibility threshold at next dispatch; not a maximum-wait guarantee.",
             ha="center", fontsize=10, color="#9a3412")
    fig.tight_layout(rect=(0, .06, 1, .95))
    save_figure(fig, output, "vehicle_waits")


def clearance_trace(result):
    cars, rows, _, _ = result
    by_time = {}
    for row in rows:
        by_time.setdefault(row["time_s"], {})[row["vehicle"]] = row
    times, gaps = [], []
    for t, frame in by_time.items():
        for tau in np.linspace(0, DT, 11):
            positions = {c.id: frame[c.id]["position_m"] + frame[c.id]["speed_mps"] * tau +
                         .5 * frame[c.id]["accel_mps2"] * tau**2 for c in cars}
            times.append(t + tau)
            gaps.append(min(body_clearance(c1, positions[c1.id], c2, positions[c2.id])
                            for c1, c2 in combinations(cars, 2)))
    assert abs(min(gaps) - result[2]["min_sampled_body_gap_m"]) < 1e-7
    return np.array(times), np.array(gaps)


def plot_safety(output, results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    episode = results["simultaneous"]
    for p in POLICIES:
        rows = episode[p][1]
        # Envelope across vehicles at each control sample, not a confidence band.
        frames = {}
        for r in rows:
            if r["state"] != "DONE":
                frames.setdefault(r["time_s"], []).append(r)
        t = list(frames)
        for ax, key in ((axes[0, 0], "speed_mps"), (axes[0, 1], "accel_mps2")):
            lo = [min(r[key] for r in frames[s]) for s in t]
            hi = [max(r[key] for r in frames[s]) for s in t]
            ax.fill_between(t, lo, hi, color=POLICY_COLORS[p], alpha=.12)
            ax.plot(t, hi, color=POLICY_COLORS[p], label=POLICY_NAMES[p], linewidth=1.4)
            ax.plot(t, lo, color=POLICY_COLORS[p], linewidth=1.4)
        pre_release = {}
        for r in rows:
            if r["state"] in ("APPROACH", "WAIT"):
                pre_release.setdefault(r["time_s"], []).append(r["position_m"])
        axes[1, 0].plot(list(pre_release), [max(v) for v in pre_release.values()],
                         label=POLICY_NAMES[p], color=POLICY_COLORS[p])
        times, gaps = clearance_trace(episode[p])
        axes[1, 1].plot(times, gaps, label=POLICY_NAMES[p], color=POLICY_COLORS[p])
    for bound in (0, V_DES):
        axes[0, 0].axhline(bound, color="#9a3412", linestyle="--", linewidth=.9)
    for bound in (A_MIN, A_MAX):
        axes[0, 1].axhline(bound, color="#9a3412", linestyle="--", linewidth=.9)
    axes[1, 0].axhline(0, color="#9a3412", linestyle="--", linewidth=.9)
    axes[1, 1].axhline(0, color="#9a3412", linestyle="--", linewidth=.9)
    for ax, title, ylabel in zip(axes.flat,
            ("Speed range across unfinished vehicles", "Acceleration range across unfinished vehicles",
             "Pre-release stop-line close-up", "Minimum body gap · close-up"),
            ("Speed [m/s] · bounds 0 to 8", "Acceleration [m/s²] · bounds −4 to 2.5",
             "Front position [m] · must stay ≤ 0", "Body gap [m] · overlap if negative")):
        ax.set(title=title, ylabel=ylabel, xlabel="Simulation time [s]")
        ax.grid(alpha=.15)
    axes[0, 0].set_ylim(-.4, 8.7)
    axes[0, 1].set_ylim(-4.5, 3.0)
    axes[1, 0].set_ylim(-.11, .02)
    axes[1, 1].set_ylim(-.5, 15)
    axes[1, 1].text(.98, .95, f"Observed min: {min(episode[p][2]['min_sampled_body_gap_m'] for p in POLICIES):.2f} m",
                      transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=10)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Constraint checks / simultaneous-arrival scenario", fontsize=16, fontweight="bold")
    fig.text(.5, .018, "Shading shows min–max across vehicles. Gap sampled every 20 ms; it is not a continuous-time safety proof.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save_figure(fig, output, "safety_controls")


def plot_tradeoffs(output, results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, key, title, unit in zip(axes,
            ("max_wait_s", "last_rear_clear_s", "rms_jerk_mps3"),
            ("Worst individual wait", "Last rear-clearance time", "Jerk RMS · comfort proxy"),
            ("Seconds", "Seconds", "m/s³")):
        for name, episode in results.items():
            a, b = (episode[p][2][key] for p in POLICIES)
            color = "#b45309" if b > a + 1e-7 else "#cbd5e1"
            ax.plot([0, 1], [a, b], color=color, alpha=.8, linewidth=1)
            ax.scatter([0, 1], [a, b], color=[POLICY_COLORS[p] for p in POLICIES], s=20, zorder=3)
        means = [np.mean([e[p][2][key] for e in results.values()]) for p in POLICIES]
        ax.plot([0, 1], means, color="#0f172a", marker="D", linewidth=2.8, markersize=7, label="Scenario mean")
        for x, value in enumerate(means):
            ax.annotate(f"{value:.2f}", (x, value), xytext=(8, 8), textcoords="offset points", fontsize=10)
        ax.set(xticks=[0, 1], xticklabels=[POLICY_NAMES[p] for p in POLICIES],
               xlim=(-.2, 1.4), title=title, ylabel=unit)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False)
    fig.suptitle("Benefits and costs across all scenarios · lower is better", fontsize=16, fontweight="bold")
    fig.text(.5, .025, "Each thin line is one paired scenario. Amber marks a higher (worse) value; black shows the mean, not a confidence interval.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .94))
    save_figure(fig, output, "tradeoffs")


def animate_episode(output, episode, name, filename):
    """Both panels use the same clock; playback is 2× simulation speed."""
    fig = plt.figure(figsize=(12, 8), dpi=95, facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=(4.5, 1.25, .85), hspace=.32, wspace=.23)
    artists, traces, clocks, cursors, tables = {}, {}, {}, {}, {}
    longest = max(round(max(r["time_s"] for r in episode[p][1]) / DT) + 1 for p in POLICIES)
    end = longest * DT
    for col, p in enumerate(POLICIES):
        ax = fig.add_subplot(grid[0, col])
        ax.set(xlim=(-44, 56), ylim=(-44, 56), aspect="equal")
        ax.set_title(POLICY_NAMES[p], loc="left", fontweight="bold", fontsize=14, color=POLICY_COLORS[p])
        ax.set_facecolor("#f8fafc")
        ax.add_patch(Rectangle((-44, 0), 100, 12, color="#cbd5e1"))
        ax.add_patch(Rectangle((0, -44), 12, 100, color="#cbd5e1"))
        ax.add_patch(Rectangle((0, 0), 12, 12, facecolor="#e2e8f0", edgecolor="#64748b", linewidth=1))
        for xs, ys in (((-44, 0), (6, 6)), ((12, 56), (6, 6)), ((6, 6), (-44, 0)), ((6, 6), (12, 56))):
            ax.plot(xs, ys, "--", color="white", linewidth=1)
        for xs, ys in (((0, 0), (0, 6)), ((12, 12), (6, 12)), ((6, 12), (0, 0)), ((0, 6), (12, 12))):
            ax.plot(xs, ys, color="#475569", linewidth=2)
        for x, y, direction in ((6, 51, "N"), (6, -39, "S"), (-39, 6, "W"), (51, 6, "E")):
            ax.text(x, y, direction, ha="center", va="center", color="#334155", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        cars, rows, metrics, _ = episode[p]
        traces[p] = {c.id: [r for r in rows if r["vehicle"] == c.id] for c in cars}
        for c in cars:
            length, width = (CAR_LENGTH, CAR_WIDTH) if c.approach in "WE" else (CAR_WIDTH, CAR_LENGTH)
            rect = Rectangle((0, 0), length, width, edgecolor="white", linewidth=.6, zorder=4)
            ax.add_patch(rect)
            label = ax.text(0, 0, str(c.id), color="white", fontsize=7, ha="center", va="center", zorder=5)
            artists[p, c.id] = (c, rect, label)
        clocks[p] = ax.text(.025, .96, "", transform=ax.transAxes, va="top", fontsize=10,
                            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        line_ax = fig.add_subplot(grid[1, col])
        timeline(line_ax, cars, end)
        line_ax.set_title("Recorded state timeline · cursor = now", loc="left", fontsize=9, pad=6)
        line_ax.set_xlabel("")
        cursors[p] = line_ax.axvline(0, color="#0f172a", linewidth=1.5)
        status_ax = fig.add_subplot(grid[2, col])
        status_ax.axis("off")
        tables[p] = status_ax.text(0, .95, "", transform=status_ax.transAxes, va="top", fontsize=10,
                                   fontfamily="monospace")
    fig.suptitle(f"{TITLES[name]} / FIFO vs wait-aware scheduling", fontsize=16, fontweight="bold", y=.98)
    fig.text(.5, .935, "Blue: approach   Orange: wait   Teal: released → rear clear   |   Same clock · playback 2×", ha="center", fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.04, left=.06, right=.98)

    def update(frame):
        now = frame * DT
        for p in POLICIES:
            cars, _, metrics, _ = episode[p]
            waiting, cleared, messages = 0, 0, []
            for c in cars:
                records = traces[p][c.id]
                row = records[min(frame, len(records) - 1)]
                _, rect, label = artists[p, c.id]
                is_clear = now >= c.exit_time - 1e-8
                cleared += int(is_clear)
                visible = frame < len(records) and row["state"] != "DONE"
                rect.set_visible(visible)
                label.set_visible(visible)
                if visible:
                    x, y = xy(c, row["position_m"])
                    rect.set_xy((x - rect.get_width() / 2, y - rect.get_height() / 2))
                    rect.set_facecolor(STATE_COLORS[row["state"]])
                    label.set_position((x, y))
                if is_clear:
                    message = "clear"
                elif row["state"] == "WAIT":
                    waiting += 1
                    message = f"wait {now - c.stop_time:.1f}s"
                else:
                    message = "go" if row["state"] == "GO" else "approach"
                messages.append(f"V{c.id}({c.approach}) {message:<11}")
            clocks[p].set_text(f"t = {now:4.1f} s\nWaiting: {waiting} · Rear clear: {cleared}/{len(cars)}")
            cursors[p].set_xdata([now, now])
            tables[p].set_text("  ".join(messages[:2]) + "\n" + "  ".join(messages[2:]) +
                               f"\nFinal mean wait {metrics['mean_wait_s']:.2f}s · last rear clear {metrics['last_rear_clear_s']:.1f}s")
    # Include final integrated state and hold it for one second before looping.
    frames = list(range(longest + 1)) + [longest] * 10
    animation = FuncAnimation(fig, update, frames=frames, interval=100, repeat=True)
    animation.save(output / filename, writer=PillowWriter(fps=10))
    paired = [t for t, ids in episode["fair"][3] if len(ids) == 2]
    preview_time = (paired[0] if paired else episode["fair"][3][0][0]) + 2.0
    update(min(longest, round(preview_time / DT)))
    fig.savefig(output / f"{Path(filename).stem}_snapshot.png", dpi=140)
    plt.close(fig)


def export_visuals(output, results, animate=False):
    output = Path(output)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlepad": 12})
    events, batches, validations = [], [], []
    for name, episode in results.items():
        for p in POLICIES:
            event_audit(episode[p])
            _, gaps = clearance_trace(episode[p])
            validations.append({"scenario": name, "policy": p, "events_match_trace": True,
                                "sampled_gap_matches_metric": True, "min_sampled_gap_m": float(min(gaps))})
            events += [{"scenario": name, "policy": p, **e} for e in events_for(episode[p][0])]
            batches += [{"scenario": name, "policy": p, "release_s": t,
                         "vehicles": " ".join(map(str, ids)), "batch_size": len(ids)}
                        for t, ids in episode[p][3]]
    write_csv(output / "vehicle_events.csv", events)
    write_csv(output / "dispatch_batches.csv", batches)
    write_csv(output / "visual_validation.csv", validations)
    plot_timelines(output, results)
    plot_vehicle_waits(output, results)
    plot_safety(output, results)
    plot_tradeoffs(output, results)
    if animate:
        for name, filename in (("simultaneous", "simultaneous.gif"),
                               ("blocked_pair", "fifo_vs_fair.gif"),
                               ("slow_approach", "tied_result.gif")):
            print(f"Rendering {filename} from verified traces", flush=True)
            animate_episode(output, results[name], name, filename)
