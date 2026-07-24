"""Pure-function tests for analysis.gain_sweep -- no sim stepping (that
is analysis.live's job, already covered by tests/test_live.py). Covers
grid construction, disqualification, winner selection (incl. the tie
chain and the all-disqualified raise), headroom vs.
analysis.dashboard.headroom_frac, non-finite headroom sanitization,
sweep_state.json round-trip / resume-refusal / threshold-reselection,
and run_stage's plot-vs-winner-storage ordering (with run_episode and
live.run_experiment/save_run stubbed -- still no real sim stepping).
"""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analysis import dashboard, gain_sweep


def _row(config_id, stage, *, pos_rmse=0.01, rot_rmse=0.01, sat_pct=0.0,
         settle=1.0, settled=True, valid=True, warning_reasons=(),
         gains=None, headroom_mean=0.1, headroom_p95=0.2,
         joint_margin_min_rad=None, disqualification=()):
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
                "joint_margin_min_rad": joint_margin_min_rad,
            },
        },
        "velocity_saturation_overall_pct": sat_pct,
        "headroom_mean_frac": headroom_mean,
        "headroom_p95_frac": headroom_p95,
        "worst_arm_pos_rmse_m": pos_rmse,
        "worst_arm_rot_rmse_rad": rot_rmse,
        "worst_arm_joint_margin_min_rad": joint_margin_min_rad,
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

    def test_contact_detected_is_a_warning_not_disqualification(self):
        row = _row("c", 1, valid=False, warning_reasons=("contact detected",))
        self.assertEqual(gain_sweep.disqualify_reasons(row, 1), [])

    def test_torso_contact_is_a_warning_not_disqualification(self):
        row = _row("c", 1, valid=False,
                   warning_reasons=("torso contact detected",))
        self.assertEqual(gain_sweep.disqualify_reasons(row, 1), [])

    def test_non_finite_data_flagged(self):
        row = _row("c", 1, valid=False, warning_reasons=("non-finite data",))
        self.assertIn("non-finite data", gain_sweep.disqualify_reasons(row, 1))

    def test_short_evaluation_warning_ignored(self):
        row = _row("c", 1, valid=False, warning_reasons=(
            "evaluation shorter than one disturbance period",))
        self.assertEqual(gain_sweep.disqualify_reasons(row, 1), [])

    def test_settling_timeout_warning_not_double_counted(self):
        # "not settled" already covers this case; the warning_reasons
        # text itself is not one of the specifically-matched reasons.
        row = _row("c", 1, settled=False, valid=False,
                   warning_reasons=("settling timeout",))
        self.assertEqual(gain_sweep.disqualify_reasons(row, 1), ["not settled"])

    def test_baseline_soft_limit_penetration_not_disqualified(self):
        # Observed at baseline gains in the real 0.3 m @ 0.1 Hz scenario:
        # joint 6 rides its soft limit at ~-0.007 rad (-0.40 deg) on both
        # arms during evaluation -- MuJoCo soft-limit constraint
        # compliance, not instability. live.py's own "negative joint
        # margin" warning fires for this (any negative margin), but must
        # not gate the sweep.
        row = _row("c", 1, valid=False,
                   warning_reasons=("negative joint margin",),
                   joint_margin_min_rad=-0.007)
        self.assertEqual(gain_sweep.disqualify_reasons(row, 1), [])

    def test_deep_joint_limit_penetration_disqualified(self):
        row = _row("c", 1, joint_margin_min_rad=-0.03)
        reasons = gain_sweep.disqualify_reasons(row, 1)
        self.assertTrue(any("joint limit" in r for r in reasons))

    def test_missing_joint_margin_field_not_disqualifying(self):
        # Rows persisted before this field existed lack the key entirely.
        row = _row("c", 1)
        del row["worst_arm_joint_margin_min_rad"]
        self.assertEqual(gain_sweep.disqualify_reasons(row, 1), [])

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
        limits = np.asarray(servo.LIMITS.joint_velocity_rad_s)
        qdot[0, 2] = 0.4 * limits[2]
        qdot[1, 5] = 1.2 * limits[5]
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
            metrics_computable=False,
            accepted=False,
            contact_observed=False,
            joint_limit_within_tolerance=True,
            limit_penetration_rad=0.0,
            warning_reasons=(),
        )

        row = gain_sweep._episode_row(log, dict(gain_sweep.BASELINE_GAINS))

        self.assertIsNone(row["headroom_mean_frac"])
        self.assertIsNone(row["headroom_p95_frac"])
        json.dumps(row, allow_nan=False)  # must not raise


class RunStageProgressionOrderingTest(unittest.TestCase):
    """run_stage must regenerate sweep_progression.png AFTER this stage's
    winner_config_id is stored, not before. Previously it was drawn
    between _make_stage_plots and select_winner, so "after stage N"
    never appeared until stage N+1 ran and redrew it (a fresh full run's
    progression plot always ended one stage short; confirmed on both a
    smoke run and the committed real-sweep figure).

    run_episode and live.run_experiment/save_run are stubbed so this
    exercises real run_stage control flow (real stage_grid, real
    disqualify_reasons/select_winner, real state dict, real plot calls)
    without any actual sim stepping -- only _make_progression_plot is
    replaced, with a spy that records whether this stage's
    winner_config_id is already in `state` at the moment it's called."""

    def setUp(self):
        self._orig_out = gain_sweep.OUT

    def tearDown(self):
        gain_sweep.OUT = self._orig_out

    def test_progression_plot_sees_this_stages_winner(self):
        progression_calls = []

        def fake_progression_plot(state):
            stage_state = state["stages"].get("1", {})
            progression_calls.append(stage_state.get("winner_config_id"))

        def fake_run_episode(gains):
            # Deterministic and distinguishable per grid point (by
            # KP_POS) so select_winner's pick is unambiguous; unrelated
            # to the real sim.
            return _row("ignored", 1, pos_rmse=gains["KP_POS"])

        with tempfile.TemporaryDirectory() as tmp:
            gain_sweep.OUT = Path(tmp)
            args = argparse.Namespace(fresh=False, sat_threshold=200.0)
            state = gain_sweep.load_or_init_state(args)

            with mock.patch.object(gain_sweep, "run_episode",
                                    fake_run_episode), \
                 mock.patch.object(gain_sweep, "_make_progression_plot",
                                    fake_progression_plot), \
                 mock.patch.object(gain_sweep.live, "run_experiment",
                                    return_value=object()), \
                 mock.patch.object(gain_sweep.live, "save_run"):
                winner = gain_sweep.run_stage(1, state, args)

        self.assertEqual(len(progression_calls), 1)
        self.assertIsNotNone(progression_calls[0])
        self.assertEqual(progression_calls[0], winner["config_id"])


class StateRoundTripTest(unittest.TestCase):
    """sweep_state.json load/save, and the resume-refusal rules: a
    scenario/grid mismatch refuses (needs --fresh); a sat_threshold-only
    change is allowed and reselects winners from the stored rows without
    rerunning any episode -- except a stage whose inherited base gains
    moved, which is dropped so it reruns from scratch."""

    def setUp(self):
        self._orig_out = gain_sweep.OUT

    def tearDown(self):
        gain_sweep.OUT = self._orig_out

    def test_round_trip_and_fresh_discards_state(self):
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
            self.assertIn("effective_control_config", reloaded)
            self.assertIn("control_config_sha256", reloaded)
            self.assertIn("1", reloaded["stages"])
            self.assertEqual(
                reloaded["stages"]["1"]["winner_config_id"], "s1_kp1_kd0")

            fresh_args = argparse.Namespace(fresh=True, sat_threshold=5.0)
            fresh_state = gain_sweep.load_or_init_state(fresh_args)
            self.assertEqual(fresh_state["stages"], {})
            self.assertEqual(fresh_state["sat_threshold"], 5.0)

    def test_fingerprint_mismatch_refuses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            gain_sweep.OUT = Path(tmp)
            args = argparse.Namespace(fresh=False, sat_threshold=20.0)
            state = gain_sweep.load_or_init_state(args)
            state["fingerprint"] = "not-the-real-fingerprint"
            gain_sweep.write_state_atomic(state)

            with self.assertRaises(RuntimeError):
                gain_sweep.load_or_init_state(args)

    def test_threshold_change_reselects_from_stored_rows_without_rerunning(self):
        # Real stage-1 config_ids/gains -- reconciliation aligns stored
        # rows against a freshly regenerated stage_grid(), so fabricated
        # IDs that aren't part of the real grid would just be dropped.
        grid1 = gain_sweep.stage_grid(1, {})
        cid_a, gains_a = grid1[0]
        cid_b, gains_b = grid1[1]

        with tempfile.TemporaryDirectory() as tmp:
            gain_sweep.OUT = Path(tmp)
            args20 = argparse.Namespace(fresh=False, sat_threshold=20.0)
            state = gain_sweep.load_or_init_state(args20)

            # Both rows already run and stored; both disqualified for
            # 25% > 20% saturation (as select_winner(strict=False) would
            # have left them: no winner yet).
            reason_20 = ["worst-arm saturation 25.0% > 20.0% threshold"]
            row_a = _row(cid_a, 1, gains=gains_a, pos_rmse=0.05, sat_pct=25.0,
                         disqualification=reason_20)
            row_b = _row(cid_b, 1, gains=gains_b, pos_rmse=0.01, sat_pct=25.0,
                         disqualification=reason_20)
            state["stages"]["1"] = {
                "rows": {cid_a: row_a, cid_b: row_b},
                "winner_config_id": None,
            }
            gain_sweep.write_state_atomic(state)

            args30 = argparse.Namespace(fresh=False, sat_threshold=30.0)
            reloaded = gain_sweep.load_or_init_state(args30)

            self.assertEqual(reloaded["sat_threshold"], 30.0)
            # both episodes kept -- no rerun (this test never touches the
            # sim: run_episode/live.run_experiment are never called)
            self.assertEqual(set(reloaded["stages"]["1"]["rows"]),
                             {cid_a, cid_b})
            self.assertEqual(
                reloaded["stages"]["1"]["rows"][cid_a]["disqualification"], [])
            self.assertEqual(
                reloaded["stages"]["1"]["rows"][cid_b]["disqualification"], [])
            # b has the lower worst_arm_pos_rmse_m -> wins once qualified
            self.assertEqual(reloaded["stages"]["1"]["winner_config_id"], cid_b)

    def test_threshold_change_invalidates_dependent_stage_when_winner_moves(self):
        grid1 = gain_sweep.stage_grid(1, {})
        cid_a, gains_a = grid1[0]
        cid_b, gains_b = grid1[-1]  # far apart in the grid -> distinct gains

        with tempfile.TemporaryDirectory() as tmp:
            gain_sweep.OUT = Path(tmp)
            args20 = argparse.Namespace(fresh=False, sat_threshold=20.0)
            state = gain_sweep.load_or_init_state(args20)

            # Under threshold=20, only A qualifies (worse metric) -> A won.
            row_a = _row(cid_a, 1, gains=gains_a, pos_rmse=0.05, sat_pct=5.0)
            row_b = _row(cid_b, 1, gains=gains_b, pos_rmse=0.01, sat_pct=25.0,
                         disqualification=[
                             "worst-arm saturation 25.0% > 20.0% threshold"])
            state["stages"]["1"] = {
                "rows": {cid_a: row_a, cid_b: row_b},
                "winner_config_id": cid_a,
            }

            # Stage 2 was built on top of A's gains.
            grid2 = gain_sweep.stage_grid(2, {1: gains_a})
            cid2, gains2 = grid2[0]
            row2 = _row(cid2, 2, gains=gains2, rot_rmse=0.02, sat_pct=1.0)
            state["stages"]["2"] = {
                "rows": {cid2: row2},
                "winner_config_id": cid2,
                "grid_signature": gain_sweep._stage_grid_signature(grid2),
            }
            gain_sweep.write_state_atomic(state)

            # Under threshold=30, B also qualifies and has the better
            # metric -> B wins stage 1 instead -- stage 2's stored grid
            # (built on A) no longer matches and must be dropped.
            args30 = argparse.Namespace(fresh=False, sat_threshold=30.0)
            reloaded = gain_sweep.load_or_init_state(args30)

            self.assertEqual(reloaded["stages"]["1"]["winner_config_id"], cid_b)
            self.assertNotIn("2", reloaded["stages"])


if __name__ == "__main__":
    unittest.main()
