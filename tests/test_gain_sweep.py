"""Pure-function tests for analysis.gain_sweep -- no sim stepping (that
is analysis.live's job, already covered by tests/test_live.py). Covers
grid construction, disqualification, winner selection (incl. the tie
chain and the all-disqualified raise), headroom vs.
analysis.dashboard.headroom_frac, non-finite headroom sanitization, and
sweep_state.json round-trip / fingerprint refusal.
"""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analysis import dashboard, gain_sweep


def _row(config_id, stage, *, pos_rmse=0.01, rot_rmse=0.01, sat_pct=0.0,
         settle=1.0, settled=True, valid=True, warning_reasons=(),
         gains=None, headroom_mean=0.1, headroom_p95=0.2,
         disqualification=()):
    return {
        "stage": stage,
        "config_id": config_id,
        "gains": dict(gains) if gains else dict(gain_sweep.BASELINE_GAINS),
        "settled": settled,
        "settle_duration_s": settle,
        "valid": valid,
        "warning_reasons": list(warning_reasons),
        "arms": {
            "right": {
                "position_error_norm_mean_m": pos_rmse,
                "position_error_norm_rmse_m": pos_rmse,
                "position_error_norm_peak_m": pos_rmse,
                "rotation_error_norm_mean_rad": rot_rmse,
                "rotation_error_norm_rmse_rad": rot_rmse,
                "rotation_error_norm_peak_rad": rot_rmse,
                "velocity_saturation_overall_pct": sat_pct,
            },
        },
        "velocity_saturation_overall_pct": sat_pct,
        "headroom_mean_frac": headroom_mean,
        "headroom_p95_frac": headroom_p95,
        "worst_arm_pos_rmse_m": pos_rmse,
        "worst_arm_rot_rmse_rad": rot_rmse,
        "disqualification": list(disqualification),
    }


class GridTest(unittest.TestCase):
    def test_stage_sizes(self):
        self.assertEqual(len(gain_sweep.stage_grid(0, {})), 1)
        self.assertEqual(len(gain_sweep.stage_grid(1, {})), 56)

        winner1 = dict(gain_sweep.BASELINE_GAINS)
        self.assertEqual(len(gain_sweep.stage_grid(2, {1: winner1})), 56)

        winner3 = dict(gain_sweep.BASELINE_GAINS)
        self.assertEqual(len(gain_sweep.stage_grid(4, {3: winner3})), 42)

    def test_stage0_is_baseline(self):
        [(config_id, gains)] = gain_sweep.stage_grid(0, {})
        self.assertEqual(config_id, "s0_baseline")
        self.assertEqual(gains, gain_sweep.BASELINE_GAINS)

    def test_stage1_config_ids_are_deterministic_and_unique(self):
        grid = gain_sweep.stage_grid(1, {})
        ids = [config_id for config_id, _ in grid]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(cid.startswith("s1_kp") for cid in ids))
        # deterministic: rebuilding gives the exact same IDs, same order
        self.assertEqual(ids, [cid for cid, _ in gain_sweep.stage_grid(1, {})])

    def test_stage1_carries_baseline_for_untouched_gains(self):
        _, gains = gain_sweep.stage_grid(1, {})[0]
        self.assertEqual(gains["KP_ROT"], gain_sweep.BASELINE_GAINS["KP_ROT"])
        self.assertEqual(gains["DAMPING"], gain_sweep.BASELINE_GAINS["DAMPING"])

    def test_stage2_carries_stage1_winner_forward(self):
        winner1 = dict(gain_sweep.BASELINE_GAINS, KP_POS=7.0, KD_POS=0.6)
        _, gains = gain_sweep.stage_grid(2, {1: winner1})[0]
        self.assertEqual(gains["KP_POS"], 7.0)
        self.assertEqual(gains["KD_POS"], 0.6)

    def test_stage3_skips_scales_pushing_kd_over_ceiling(self):
        winner2 = dict(gain_sweep.BASELINE_GAINS, KD_POS=0.8, KD_ROT=0.8)
        grid = gain_sweep.stage_grid(3, {2: winner2})
        scales = [gain_sweep._parse_stage3_scale(cid) for cid, _ in grid]
        # 0.8 * 1.3 = 1.04 and 0.8 * 1.5 = 1.2 both clear KD_CEILING=0.95
        self.assertNotIn(1.3, scales)
        self.assertNotIn(1.5, scales)
        # 0.8 * 0.6 = 0.48 stays well under the ceiling
        self.assertIn(0.6, scales)
        self.assertLessEqual(len(grid), 7)

    def test_stage3_scales_all_four_task_gains_uniformly(self):
        winner2 = dict(gain_sweep.BASELINE_GAINS, KP_POS=2.0, KD_POS=0.2,
                        KP_ROT=4.0, KD_ROT=0.4)
        grid = gain_sweep.stage_grid(3, {2: winner2})
        _, gains = next((cid, g) for cid, g in grid
                        if gain_sweep._parse_stage3_scale(cid) == 1.15)
        self.assertAlmostEqual(gains["KP_POS"], 2.3, places=3)
        self.assertAlmostEqual(gains["KD_POS"], 0.23, places=3)
        self.assertAlmostEqual(gains["KP_ROT"], 4.6, places=3)
        self.assertAlmostEqual(gains["KD_ROT"], 0.46, places=3)

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            gain_sweep.stage_grid(99, {})


class DisqualifyReasonsTest(unittest.TestCase):
    def test_qualified_row_has_no_reasons(self):
        row = _row("c", 1, sat_pct=5.0)
        self.assertEqual(gain_sweep.disqualify_reasons(row, 1), [])

    def test_not_settled_flagged(self):
        row = _row("c", 1, settled=False)
        self.assertIn("not settled", gain_sweep.disqualify_reasons(row, 1))

    def test_invalid_log_flagged_with_reasons(self):
        row = _row("c", 1, valid=False, warning_reasons=("contact detected",))
        reasons = gain_sweep.disqualify_reasons(row, 1)
        self.assertTrue(any("invalid" in r and "contact detected" in r
                            for r in reasons))

    def test_saturation_over_threshold_flagged(self):
        row = _row("c", 1, sat_pct=50.0)
        reasons = gain_sweep.disqualify_reasons(row, 1, sat_threshold=20.0)
        self.assertTrue(any("saturation" in r for r in reasons))

    def test_saturation_under_threshold_not_flagged(self):
        row = _row("c", 1, sat_pct=5.0)
        reasons = gain_sweep.disqualify_reasons(row, 1, sat_threshold=20.0)
        self.assertEqual(reasons, [])

    def test_rotation_guard_applies_only_to_stages_3_and_4(self):
        row1 = _row("c", 1, rot_rmse=1.0)
        self.assertEqual(
            gain_sweep.disqualify_reasons(row1, 1, stage2_winner_rot_rmse=0.1),
            [])

        row3 = _row("c", 3, rot_rmse=1.0)
        reasons = gain_sweep.disqualify_reasons(
            row3, 3, stage2_winner_rot_rmse=0.1)
        self.assertTrue(any("rotation" in r for r in reasons))

    def test_rotation_guard_respects_factor(self):
        # guard = ROT_GUARD_FACTOR(2.0) * 0.1 = 0.2 rad
        row_ok = _row("c", 4, rot_rmse=0.19)
        row_bad = _row("d", 4, rot_rmse=0.21)
        self.assertEqual(
            gain_sweep.disqualify_reasons(row_ok, 4, stage2_winner_rot_rmse=0.1),
            [])
        self.assertTrue(gain_sweep.disqualify_reasons(
            row_bad, 4, stage2_winner_rot_rmse=0.1))


class SelectWinnerTest(unittest.TestCase):
    def test_lowest_metric_wins(self):
        rows = [_row("a", 1, pos_rmse=0.05), _row("b", 1, pos_rmse=0.01)]
        winner = gain_sweep.select_winner(rows, "worst_arm_pos_rmse_m")
        self.assertEqual(winner["config_id"], "b")

    def test_disqualified_rows_are_excluded(self):
        rows = [
            _row("a", 1, pos_rmse=0.01, disqualification=["not settled"]),
            _row("b", 1, pos_rmse=0.05),
        ]
        winner = gain_sweep.select_winner(rows, "worst_arm_pos_rmse_m")
        self.assertEqual(winner["config_id"], "b")

    def test_tie_chain_saturation_then_settle_then_grid_order(self):
        rows = [
            _row("a", 1, pos_rmse=0.01, sat_pct=10.0, settle=2.0),
            _row("b", 1, pos_rmse=0.01, sat_pct=5.0, settle=3.0),
            _row("c", 1, pos_rmse=0.01, sat_pct=5.0, settle=1.0),
            _row("d", 1, pos_rmse=0.01, sat_pct=5.0, settle=1.0),
        ]
        winner = gain_sweep.select_winner(rows, "worst_arm_pos_rmse_m")
        # c and d tie on metric+saturation+settle -> grid order (c first)
        self.assertEqual(winner["config_id"], "c")

    def test_all_disqualified_raises(self):
        rows = [
            _row("a", 1, disqualification=["not settled"]),
            _row("b", 1, disqualification=["not settled"]),
        ]
        with self.assertRaises(RuntimeError):
            gain_sweep.select_winner(rows, "worst_arm_pos_rmse_m")

    def test_non_strict_returns_none_instead_of_raising(self):
        rows = [_row("a", 1, disqualification=["not settled"])]
        self.assertIsNone(
            gain_sweep.select_winner(rows, "worst_arm_pos_rmse_m", strict=False))


class HeadroomTest(unittest.TestCase):
    def test_matches_dashboard_headroom_frac_per_step(self):
        from controller import servo

        qdot = np.zeros((3, 7))
        qdot[0, 2] = 0.4 * servo.QDOT_LIMIT[2]
        qdot[1, 5] = 1.2 * servo.QDOT_LIMIT[5]
        # row 2 stays all-zero

        series = gain_sweep._headroom_series(qdot)
        expected = [dashboard.headroom_frac(row) for row in qdot]
        np.testing.assert_allclose(series, expected)


class NonFiniteHeadroomTest(unittest.TestCase):
    """An unstable gain corner can produce non-finite qdot_raw; np.mean/
    np.percentile then yield NaN, which write_state_atomic's
    json.dumps(..., allow_nan=False) would crash on -- and since the
    episode is disqualified anyway (live.py already flags a non-finite
    log invalid), the crash would just repeat on every resume. Row
    construction must sanitize to None instead (metrics.py's
    _finite_float convention)."""

    def test_nan_qdot_raw_yields_none_headroom_and_json_safe_row(self):
        from analysis.live import ExperimentConfig, ExperimentLog

        config = ExperimentConfig(
            arms=("right",), linear_amplitude=np.zeros(3),
            linear_frequency=0.0, rotational_amplitude=np.zeros(3),
            rotational_frequency=0.0, evaluation_seconds=0.01)
        n = 2
        arm_data = {
            "right": {
                "e_pos": np.zeros((n, 3)),
                "e_rot": np.zeros((n, 3)),
                "speed_saturated": np.zeros((n, 7), dtype=bool),
                "joint_margin": np.full((n, 7), np.inf),
                # step 0: a diverging corner (all-NaN qdot_raw); step 1: normal
                "qdot_raw": np.vstack([np.full(7, np.nan), np.full(7, 0.1)]),
            },
        }
        log = ExperimentLog(
            arms=("right",),
            config=config,
            sim_time=np.arange(n, dtype=float),
            contact_time=np.arange(n, dtype=float),
            eval_time=np.arange(n, dtype=float),
            phase=np.asarray(["evaluation"] * n, dtype="U10"),
            gain_segment=np.zeros(n, dtype=int),
            base_displacement=np.zeros((n, 3)),
            base_linear_velocity=np.zeros((n, 3)),
            base_angular_velocity=np.zeros((n, 3)),
            arm_data=arm_data,
            contact_count=np.zeros(n, dtype=int),
            torso_contact={"right": np.zeros(n, dtype=bool)},
            contact_pairs=tuple(() for _ in range(n)),
            gain_snapshots=({"KP_POS": 2.0},),
            settled=True,
            settle_duration=0.0,
            valid=True,
            warning_reasons=(),
        )

        row = gain_sweep._episode_row(log, dict(gain_sweep.BASELINE_GAINS))

        self.assertIsNone(row["headroom_mean_frac"])
        self.assertIsNone(row["headroom_p95_frac"])
        json.dumps(row, allow_nan=False)  # must not raise


class StateRoundTripTest(unittest.TestCase):
    def setUp(self):
        self._orig_out = gain_sweep.OUT

    def tearDown(self):
        gain_sweep.OUT = self._orig_out

    def test_round_trip_and_fingerprint_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            gain_sweep.OUT = Path(tmp)
            args = argparse.Namespace(fresh=False, sat_threshold=20.0)

            state = gain_sweep.load_or_init_state(args)
            state["stages"]["1"] = {
                "rows": {"s1_kp1_kd0": _row("s1_kp1_kd0", 1)},
                "winner_config_id": "s1_kp1_kd0",
            }
            gain_sweep.write_state_atomic(state)

            reloaded = gain_sweep.load_or_init_state(args)
            self.assertEqual(reloaded["fingerprint"], state["fingerprint"])
            self.assertIn("1", reloaded["stages"])
            self.assertEqual(
                reloaded["stages"]["1"]["winner_config_id"], "s1_kp1_kd0")

            mismatched_threshold = argparse.Namespace(
                fresh=False, sat_threshold=5.0)
            with self.assertRaises(RuntimeError):
                gain_sweep.load_or_init_state(mismatched_threshold)

            fresh_args = argparse.Namespace(fresh=True, sat_threshold=5.0)
            fresh_state = gain_sweep.load_or_init_state(fresh_args)
            self.assertEqual(fresh_state["stages"], {})
            self.assertEqual(fresh_state["sat_threshold"], 5.0)


if __name__ == "__main__":
    unittest.main()
