"""Wait-aware opposing-pair scheduling with constrained longitudinal MPC.

Run `python3 fair_mpc.py` for reproducible FIFO comparisons and plots.
The existing all_way_stop.py remains the team's original demonstration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

DT = 0.2
HORIZON = 25
V_DES = 8.0
A_MIN, A_MAX = -4.0, 2.5
CAR_LENGTH, CAR_WIDTH = 4.5, 2.0
EXIT = 12.0
REAR_CLEAR = EXIT + CAR_LENGTH + 0.5
DELETE = 30.0
MAX_WAIT = 6.0
CLEARANCE_TIME = 0.4
OPPOSITE = {"W": "E", "E": "W", "N": "S", "S": "N"}


@dataclass
class Vehicle:
    id: int
    approach: str
    p: float
    v: float
    state: str = "APPROACH"
    a: float = 0.0
    stop_time: float | None = None
    release_time: float | None = None
    exit_time: float | None = None
    priority: int | None = None
    plan: np.ndarray = field(default_factory=lambda: np.zeros(HORIZON))


def compatible(cars):
    """Straight-through routes only; a group has at most two opposite cars."""
    return len(cars) <= 1 or (len(cars) == 2 and
                             OPPOSITE[cars[0].approach] == cars[1].approach)


class Scheduler:
    def __init__(self, policy):
        if policy not in ("fifo", "fair"):
            raise ValueError("policy must be fifo or fair")
        self.policy = policy
        self.active = []
        self.next_priority = 0
        self.available_at = 0.0
        self.batches = []

    def update(self, cars, now):
        by_id = {c.id: c for c in cars}
        was_active = bool(self.active)
        self.active = [i for i in self.active if by_id[i].p < REAR_CLEAR]
        if was_active and not self.active:
            self.available_at = now + CLEARANCE_TIME

        for car in cars:
            if car.state == "APPROACH" and -0.10 <= car.p <= 1e-6 and car.v <= 0.05:
                # No position/speed reset: no teleportation or hidden braking.
                car.state = "WAIT"
                car.stop_time = now
                car.priority = self.next_priority
                self.next_priority += 1

        waiting = sorted((c for c in cars if c.state == "WAIT"),
                         key=lambda c: c.priority)
        if self.active or now < self.available_at - 1e-9 or not waiting:
            return
        if self.policy == "fifo":
            chosen = waiting[:1]
            if len(waiting) > 1 and compatible(waiting[:2]):
                chosen = waiting[:2]
        else:
            # Enumerate the independent sets for this four-route conflict graph.
            groups = [(c,) for c in waiting]
            groups += [pair for pair in combinations(waiting, 2) if compatible(pair)]
            oldest = waiting[0]
            if now - oldest.stop_time >= MAX_WAIT - 1e-9:
                groups = [g for g in groups if any(c.id == oldest.id for c in g)]
            # Each additional served vehicle has value 1; age adds urgency.
            # Stable priority tie-breaks make runs deterministic.
            chosen = max(groups, key=lambda g: (
                sum(1.0 + now - c.stop_time for c in g),
                -min(c.priority for c in g),
                tuple(-c.priority for c in g)))

        assert compatible(chosen)
        for car in chosen:
            car.state = "GO"
            car.release_time = now
        self.active = [c.id for c in chosen]
        self.batches.append((now, tuple(self.active)))


class ConstrainedMPC:
    """Convex quadratic cost and linear bounds, solved with analytic gradients.

    Acceleration is constant within a sample. Nonnegative endpoint speeds imply
    monotone position within that sample, so p[k] <= 0 also protects the stop
    line between samples. Jerk is penalized, not constrained.
    """

    def __init__(self):
        lower = np.tril(np.ones((HORIZON, HORIZON)))
        self.V = DT * lower
        k, j = np.indices((HORIZON, HORIZON))
        self.P = DT**2 * np.maximum(k - j + 0.5, 0.0)
        self.t = DT * np.arange(1, HORIZON + 1)
        self.D = np.eye(HORIZON) - np.eye(HORIZON, k=-1)
        self.solve_times = []
        self.reused_plans = 0

    def problem(self, car):
        p0 = car.p + self.t * car.v
        v0 = np.full(HORIZON, car.v)
        previous = np.zeros(HORIZON)
        previous[0] = car.a
        # Acceleration and changes of acceleration affect both STOP and GO.
        Q = 0.3 * np.eye(HORIZON) + 3.0 * self.D.T @ self.D
        q = -3.0 * self.D.T @ previous
        if car.state in ("APPROACH", "WAIT"):
            Q += 100.0 * np.outer(self.P[-1], self.P[-1])
            Q += 150.0 * np.outer(self.V[-1], self.V[-1])
            q += 100.0 * (p0[-1] + 0.03) * self.P[-1]
            q += 150.0 * v0[-1] * self.V[-1]
            matrix = np.vstack((self.V, self.P))
            low = np.r_[-v0, np.full(HORIZON, -np.inf)]
            high = np.r_[V_DES - v0, -p0]
        else:
            Q += 15.0 * self.V.T @ self.V
            q += 15.0 * self.V.T @ (v0 - V_DES)
            matrix, low, high = self.V, -v0, V_DES - v0
        return Q, q, matrix, low, high

    @staticmethod
    def feasible(plan, matrix, low, high):
        return (np.all(np.isfinite(plan)) and
                np.all(plan >= A_MIN - 1e-6) and
                np.all(plan <= A_MAX + 1e-6) and
                np.all(matrix @ plan >= low - 1e-6) and
                np.all(matrix @ plan <= high + 1e-6))

    def solve(self, car):
        Q, q, matrix, low, high = self.problem(car)
        guess = np.r_[car.plan[1:], 0.0]
        # Positive rescaling leaves the minimizer unchanged and improves SLSQP
        # conditioning when terminal position weights dominate near the line.
        scale = max(1.0, float(np.max(np.diag(Q))))
        started = time.perf_counter()
        result = minimize(lambda u: (u @ Q @ u + 2.0 * q @ u) / scale, guess,
                          jac=lambda u: 2.0 * (Q @ u + q) / scale, method="SLSQP",
                          bounds=Bounds(A_MIN, A_MAX),
                          constraints=LinearConstraint(matrix, low, high),
                          options={"maxiter": 150, "ftol": 1e-10})
        self.solve_times.append(time.perf_counter() - started)
        if result.success and self.feasible(result.x, matrix, low, high):
            car.plan = result.x
        elif self.feasible(guess, matrix, low, high):
            # Reuse only after checking against the CURRENT state/constraints.
            self.reused_plans += 1
            car.plan = guess
        else:
            raise RuntimeError(f"No feasible MPC plan for car {car.id}: {result.message}")
        return float(car.plan[0])


def scenario_specs(seed_count=8):
    # Each episode has one straight-through vehicle per approach. Relative
    # stopping order W,N,E,S deliberately exposes FIFO's head-of-line blocking.
    yield "blocked_pair", [("W", -22., 7.), ("N", -26., 7.),
                            ("E", -27., 7.), ("S", -30., 7.)]
    yield "simultaneous", [(a, -28., 7.) for a in ("W", "N", "E", "S")]
    yield "staggered", [("W", -24., 7.), ("E", -30., 7.5),
                         ("S", -36., 8.), ("N", -42., 8.)]
    yield "slow_approach", [("N", -18., 3.), ("E", -32., 8.),
                             ("S", -25., 5.), ("W", -38., 7.)]
    for seed in range(seed_count):
        rng = np.random.default_rng(seed)
        yield f"seed_{seed:02d}", [(a, float(-rng.uniform(24., 44.)),
                                    float(rng.uniform(4., 8.)))
                                   for a in ("W", "N", "E", "S")]


def xy(car, p):
    # p denotes front bumper; return body centre in world coordinates.
    p = p - CAR_LENGTH / 2
    return {"W": (p, 3.), "E": (12. - p, 9.),
            "S": (9., p), "N": (3., 12. - p)}[car.approach]


def body_clearance(c1, p1, c2, p2):
    """Euclidean gap between axis-aligned rectangles; negative means overlap."""
    x1, y1 = xy(c1, p1)
    x2, y2 = xy(c2, p2)
    hx1, hy1 = (CAR_LENGTH / 2, CAR_WIDTH / 2) if c1.approach in "WE" else (CAR_WIDTH / 2, CAR_LENGTH / 2)
    hx2, hy2 = (CAR_LENGTH / 2, CAR_WIDTH / 2) if c2.approach in "WE" else (CAR_WIDTH / 2, CAR_LENGTH / 2)
    dx, dy = abs(x1 - x2) - hx1 - hx2, abs(y1 - y2) - hy1 - hy2
    return float(np.hypot(max(dx, 0), max(dy, 0))) if max(dx, dy) >= 0 else max(dx, dy)


def simulate(spec, policy, limit=60.0):
    if len({a for a, _, _ in spec}) != len(spec):
        raise ValueError("This experiment supports one vehicle per approach")
    cars = [Vehicle(i + 1, *entry) for i, entry in enumerate(spec)]
    if any(c.approach not in OPPOSITE or c.p >= 0 or not 0 <= c.v <= V_DES for c in cars):
        raise ValueError("Invalid initial approach, position or speed")
    scheduler, mpc, rows = Scheduler(policy), ConstrainedMPC(), []
    min_gap = np.inf
    for step in range(round(limit / DT)):
        now = step * DT
        scheduler.update(cars, now)
        controls = [mpc.solve(c) if c.state != "DONE" else 0.0 for c in cars]
        # Measure body clearance at 20 ms intervals, including within a step.
        for tau in np.linspace(0., DT, 11):
            positions = [c.p + c.v * tau + 0.5 * a * tau**2 for c, a in zip(cars, controls)]
            for i, j in combinations(range(len(cars)), 2):
                min_gap = min(min_gap, body_clearance(cars[i], positions[i], cars[j], positions[j]))
        for c, a in zip(cars, controls):
            rows.append({"time_s": now, "vehicle": c.id, "approach": c.approach,
                         "state": c.state, "position_m": c.p, "speed_mps": c.v,
                         "accel_mps2": a, "jerk_mps3": (a - c.a) / DT})
            new_p, new_v = c.p + DT * c.v + 0.5 * a * DT**2, c.v + DT * a
            if c.state in ("APPROACH", "WAIT") and new_p > 1e-5:
                raise AssertionError("Vehicle crossed stop line before release")
            if not (-1e-5 <= new_v <= V_DES + 1e-5 and A_MIN - 1e-5 <= a <= A_MAX + 1e-5):
                raise AssertionError("Actuator/speed constraint violated")
            c.p, c.v, c.a = new_p, new_v, a
            if c.exit_time is None and c.p >= REAR_CLEAR:
                c.exit_time = now + DT
            if c.p >= DELETE:
                c.state = "DONE"
        if all(c.state == "DONE" for c in cars):
            break
    if any(c.exit_time is None for c in cars):
        raise RuntimeError("Simulation timed out before every vehicle cleared")
    if min_gap < -1e-5:
        raise AssertionError(f"Sampled vehicle-body overlap: {min_gap}")
    waits = [c.release_time - c.stop_time for c in cars]
    metrics = {"policy": policy, "mean_wait_s": float(np.mean(waits)),
               "max_wait_s": max(waits), "last_rear_clear_s": max(c.exit_time for c in cars),
               "rms_jerk_mps3": float(np.sqrt(np.mean([r["jerk_mps3"]**2 for r in rows]))),
               "min_sampled_body_gap_m": min_gap, "mpc_reused_plans": mpc.reused_plans,
               "mean_solve_ms": 1000 * float(np.mean(mpc.solve_times))}
    return cars, rows, metrics, scheduler.batches


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(output, results, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(results)
    x = np.arange(len(names))
    for policy, offset, color in (("fifo", -0.2, "#64748b"), ("fair", 0.2, "#0d9488")):
        subset = [m for m in metrics if m["policy"] == policy]
        for ax, key, title in zip(axes, ("mean_wait_s", "last_rear_clear_s"),
                                  ("Mean stopped waiting time", "Last vehicle rear clears intersection")):
            ax.bar(x + offset, [m[key] for m in subset], width=0.4, color=color, label=policy.upper())
            ax.set(title=title, ylabel="Seconds", xticks=x, xticklabels=names)
            ax.tick_params(axis="x", rotation=55, labelsize=8)
            ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Wait-aware batch scheduling vs FIFO · identical constrained MPC", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey="row")
    for col, policy in enumerate(("fifo", "fair")):
        _, rows, _, _ = results["blocked_pair"][policy]
        for vid in range(1, 5):
            data = [r for r in rows if r["vehicle"] == vid and r["state"] != "DONE"]
            for ax, key, label in zip(axes[:, col], ("position_m", "speed_mps", "accel_mps2"),
                                      ("Front position [m]", "Speed [m/s]", "Acceleration [m/s²]")):
                ax.plot([r["time_s"] for r in data], [r[key] for r in data], label=f"V{vid} ({data[0]['approach']})")
                ax.set_ylabel(label)
                ax.grid(alpha=0.2)
        axes[0, col].set_title(policy.upper())
        axes[0, col].axhline(0, color="black", linestyle="--", linewidth=0.7)
        axes[0, col].axhline(REAR_CLEAR, color="black", linestyle=":", linewidth=0.7)
        axes[0, col].legend(fontsize=8)
        axes[2, col].set_xlabel("Time [s]")
    fig.suptitle("Head-of-line blocking scenario · position, speed and control")
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fair_mpc"))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--animate", action="store_true", help="also export three annotated comparison GIFs")
    args = parser.parse_args()
    if args.seeds < 0:
        parser.error("--seeds must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    results, metrics = {}, []
    for name, spec in scenario_specs(args.seeds):
        results[name] = {}
        for policy in ("fifo", "fair"):
            result = simulate(spec, policy)
            results[name][policy] = result
            metric = {"scenario": name, **result[2]}
            metrics.append(metric)
            write_csv(args.output / f"{name}_{policy}.csv", result[1])
            print(f"{name:16} {policy:4} wait={metric['mean_wait_s']:.2f}s "
                  f"clear={metric['last_rear_clear_s']:.2f}s gap={metric['min_sampled_body_gap_m']:.2f}m", flush=True)
    write_csv(args.output / "metrics.csv", metrics)
    plot_comparison(args.output, results, metrics)
    from fair_visuals import export_visuals
    export_visuals(args.output, results, animate=args.animate)
    print(f"Saved traces, metrics and plots to {args.output}")


if __name__ == "__main__":
    main()
