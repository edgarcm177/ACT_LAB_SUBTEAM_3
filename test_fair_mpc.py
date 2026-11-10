"""Controller and dispatch invariants; run python3 -m unittest -v."""

import unittest
import copy
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from fair_mpc import (A_MAX, A_MIN, CLEARANCE_TIME, ConstrainedMPC, DT,
                      HORIZON, MAX_WAIT, REAR_CLEAR, Scheduler, Vehicle,
                      body_clearance, compatible, simulate)


def waiting(vid, approach, stopped=0.):
    return Vehicle(vid, approach, -0.03, 0., state="WAIT",
                   stop_time=stopped, priority=vid - 1)


class SchedulingTests(unittest.TestCase):
    def test_looks_past_conflicting_queue_head(self):
        for policy, expected in (("fifo", [1]), ("fair", [1, 3])):
            cars = [waiting(1, "W"), waiting(2, "N"), waiting(3, "E")]
            scheduler = Scheduler(policy)
            scheduler.update(cars, 1.)
            self.assertEqual(scheduler.active, expected)

    def test_overdue_oldest_overrides_higher_scoring_pair(self):
        cars = [waiting(1, "N"), waiting(2, "W", 0.1), waiting(3, "E", 0.1)]
        scheduler = Scheduler("fair")
        scheduler.update(cars, MAX_WAIT)
        self.assertEqual(scheduler.active, [1])

    def test_same_approach_cannot_form_batch(self):
        self.assertFalse(compatible([waiting(1, "W"), waiting(2, "W")]))

    def test_opposing_routes_are_compatible(self):
        for a, b in (("W", "E"), ("N", "S")):
            self.assertTrue(compatible([waiting(1, a), waiting(2, b)]))

    def test_waits_for_rear_clearance_and_guard_time(self):
        first, second = waiting(1, "W"), waiting(2, "N")
        scheduler = Scheduler("fair")
        scheduler.update([first, second], 0.)
        first.p = REAR_CLEAR - .01
        scheduler.update([first, second], 1.)
        self.assertEqual(second.state, "WAIT")
        first.p = REAR_CLEAR
        scheduler.update([first, second], 2.)
        scheduler.update([first, second], 2. + CLEARANCE_TIME - .01)
        self.assertEqual(second.state, "WAIT")
        scheduler.update([first, second], 2. + CLEARANCE_TIME)
        self.assertEqual(second.state, "GO")

    def test_does_not_release_moving_approach_vehicle(self):
        car = Vehicle(1, "W", -.05, 2.)
        scheduler = Scheduler("fair")
        scheduler.update([car], 0.)
        self.assertEqual(scheduler.active, [])


class ControlTests(unittest.TestCase):
    def test_prediction_matches_stepwise_dynamics(self):
        mpc = ConstrainedMPC()
        u = np.random.default_rng(10).uniform(-1., 1., HORIZON)
        p, v, ps, vs = -30., 7., [], []
        for a in u:
            p, v = p + DT * v + .5 * DT**2 * a, v + DT * a
            ps.append(p)
            vs.append(v)
        np.testing.assert_allclose(ps, -30 + mpc.t * 7 + mpc.P @ u)
        np.testing.assert_allclose(vs, 7 + mpc.V @ u)

    def test_objective_gradient(self):
        mpc = ConstrainedMPC()
        for state in ("APPROACH", "WAIT", "GO"):
            car = Vehicle(1, "W", -25., 7., state=state, a=-.4)
            Q, q, *_ = mpc.problem(car)
            u = np.random.default_rng(4).normal(size=HORIZON)
            epsilon = 1e-5
            numeric = []
            for i in range(HORIZON):
                delta = np.eye(HORIZON)[i] * epsilon
                cost = lambda x: x @ Q @ x + 2 * q @ x
                numeric.append((cost(u + delta) - cost(u - delta)) / (2 * epsilon))
            np.testing.assert_allclose(numeric, 2 * (Q @ u + q), rtol=1e-6, atol=1e-4)
            self.assertGreater(np.linalg.eigvalsh(Q).min(), 0.)

    def test_stop_plan_obeys_bounds(self):
        mpc, car = ConstrainedMPC(), Vehicle(1, "W", -24., 8.)
        a = mpc.solve(car)
        _, _, matrix, low, high = mpc.problem(car)
        self.assertTrue(mpc.feasible(car.plan, matrix, low, high))
        self.assertTrue(A_MIN <= a <= A_MAX)

    def test_failed_solver_reuses_only_feasible_plan(self):
        mpc, car = ConstrainedMPC(), Vehicle(1, "W", -.03, 0.)
        with patch("fair_mpc.minimize", return_value=SimpleNamespace(success=False, message="injected failure")):
            self.assertEqual(mpc.solve(car), 0.)
        self.assertEqual(mpc.reused_plans, 1)

    def test_failed_solver_with_no_feasible_backup_raises(self):
        mpc, car = ConstrainedMPC(), Vehicle(1, "W", -.01, 8.)
        with patch("fair_mpc.minimize", return_value=SimpleNamespace(success=False, message="injected failure")):
            with self.assertRaisesRegex(RuntimeError, "No feasible MPC"):
                mpc.solve(car)

    def test_body_overlap_uses_rectangles(self):
        w, s = waiting(1, "W"), waiting(2, "S")
        self.assertLess(body_clearance(w, 11.25, s, 5.25), 0.)
        self.assertGreater(body_clearance(w, -1., s, -1.), 0.)

    def test_rejects_multiple_cars_per_approach(self):
        with self.assertRaisesRegex(ValueError, "one vehicle per approach"):
            simulate([("W", -24., 7.), ("W", -40., 7.)], "fair")


class VisualEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = simulate([("W", -28., 7.), ("N", -28., 7.),
                               ("E", -28., 7.), ("S", -28., 7.)], "fair")

    def test_zero_wait_and_waiting_events_match_trace(self):
        from fair_visuals import event_audit
        event_audit(self.result)
        waits = [c.release_time - c.stop_time for c in self.result[0]]
        self.assertIn(0., waits)
        self.assertGreater(max(waits), 0.)

    def test_incorrect_event_time_is_rejected(self):
        from fair_visuals import event_audit
        result = copy.deepcopy(self.result)
        result[0][0].release_time += DT
        with self.assertRaises(AssertionError):
            event_audit(result)

    def test_plotted_clearance_reconstructs_simulation_measurement(self):
        from fair_visuals import clearance_trace
        times, gaps = clearance_trace(self.result)
        self.assertAlmostEqual(min(gaps), self.result[2]["min_sampled_body_gap_m"])
        self.assertGreaterEqual(min(gaps), 0.)
        self.assertAlmostEqual(times[-1], max(r["time_s"] for r in self.result[1]) + DT)


if __name__ == "__main__":
    unittest.main()
