import csv
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _config(arms=("right",)):
    from analysis.live import ExperimentConfig

    return ExperimentConfig(
        arms=arms,
        linear_amplitude=np.zeros(3),
        linear_frequency=0.0,
        rotational_amplitude=np.zeros(3),
        rotational_frequency=0.0,
        evaluation_seconds=0.01,
    )


def _log(
    *,
    phase,
    base_displacement,
    e_pos,
    speed_saturated,
    joint_margin,
    gain_segment=None,
    contact_count=None,
    contact_pairs=None,
    torso_contact=None,
):
    from analysis.live import ExperimentLog

    phase = np.asarray(phase, dtype="U10")
    sample_count = len(phase)
    arms = tuple(e_pos)
    if gain_segment is None:
        gain_segment = np.zeros(sample_count, dtype=int)
    if contact_pairs is None:
        contact_pairs = tuple(() for _ in range(sample_count))
    else:
        contact_pairs = tuple(tuple(pairs) for pairs in contact_pairs)
    if contact_count is None:
        contact_count = np.asarray([len(pairs) for pairs in contact_pairs])
    if torso_contact is None:
        torso_contact = {
            side: np.zeros(sample_count, dtype=bool) for side in arms
        }
    arm_data = {
        side: {
            "e_pos": np.asarray(e_pos[side], dtype=float),
            "speed_saturated": np.asarray(speed_saturated[side], dtype=bool),
            "joint_margin": np.asarray(joint_margin[side], dtype=float),
        }
        for side in arms
    }
    config = _config(arms)
    return ExperimentLog(
        arms=arms,
        config=config,
        sim_time=np.arange(sample_count, dtype=float) * 0.002,
        contact_time=np.arange(sample_count, dtype=float) * 0.002,
        eval_time=np.where(phase == "evaluation", np.arange(sample_count), -1.0),
        phase=phase,
        gain_segment=np.asarray(gain_segment, dtype=int),
        base_displacement=np.asarray(base_displacement, dtype=float),
        base_linear_velocity=np.zeros((sample_count, 3)),
        base_angular_velocity=np.zeros((sample_count, 3)),
        arm_data=arm_data,
        contact_count=np.asarray(contact_count, dtype=int),
        torso_contact={
            side: np.asarray(torso_contact[side], dtype=bool) for side in arms
        },
        contact_pairs=contact_pairs,
        gain_snapshots=tuple({"KP_POS": float(i)} for i in range(
            int(np.max(gain_segment)) + 1 if sample_count else 1)),
        settled=True,
        settle_duration=0.0,
        valid=True,
        warning_reasons=(),
    )


class ExperimentMetricsTest(unittest.TestCase):
    def test_evaluation_only_multi_axis_metrics_are_exact(self):
        from analysis.metrics import experiment_metrics

        log = _log(
            phase=("settling", "evaluation", "evaluation", "evaluation"),
            base_displacement=((99.0, 0.0, 0.0), (3.0, 4.0, 0.0),
                               (0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
            e_pos={"right": ((99.0, 99.0, 99.0), (1.0, -2.0, 2.0),
                             (3.0, 0.0, -4.0), (-2.0, 2.0, 1.0))},
            speed_saturated={"right": (
                (True,) * 7,
                (True, False, False, False, False, False, False),
                (True, True, False, False, False, False, False),
                (False,) * 7,
            )},
            joint_margin={"right": (
                (-99.0, np.inf),
                (0.2, np.inf),
                (0.1, np.inf),
                (0.3, np.inf),
            )},
            gain_segment=(9, 0, 0, 1),
            contact_count=(5, 0, 2, 0),
            contact_pairs=(
                ("right_hand/a <> floor/f",) * 5,
                (),
                ("right_hand/a <> floor/f", "box/b <> floor/f"),
                (),
            ),
            torso_contact={"right": (True, False, True, True)},
        )

        result = experiment_metrics(log)

        self.assertEqual(result["evaluation_sample_count"], 3)
        self.assertEqual(result["peak_base_displacement_m"], 5.0)
        self.assertAlmostEqual(result["contact_occupancy_total_pct"], 100.0 / 3.0)
        arm = result["arms"]["right"]
        np.testing.assert_allclose(
            arm["position_error_axis_mean_m"], [2.0 / 3.0, 0.0, -1.0 / 3.0])
        self.assertAlmostEqual(arm["position_error_norm_mean_m"], 11.0 / 3.0)
        np.testing.assert_allclose(
            arm["position_error_axis_rmse_m"], np.sqrt([14.0 / 3.0, 8.0 / 3.0, 7.0]))
        self.assertAlmostEqual(
            arm["position_error_norm_rmse_m"], np.sqrt(43.0 / 3.0))
        self.assertEqual(arm["position_error_axis_abs_peak_m"], [3.0, 2.0, 4.0])
        self.assertEqual(arm["position_error_norm_peak_m"], 5.0)
        self.assertEqual(arm["rejection_pct"], 0.0)
        self.assertAlmostEqual(arm["velocity_saturation_overall_pct"], 100.0 / 7.0)
        self.assertEqual(arm["joint_margin_min_rad"], 0.1)
        self.assertAlmostEqual(arm["joint_margin_p01_rad"], 0.102)
        self.assertAlmostEqual(arm["contact_occupancy_total_pct"], 100.0 / 3.0)
        self.assertAlmostEqual(arm["torso_contact_occupancy_pct"], 200.0 / 3.0)

    def test_gain_segments_repeat_all_metrics_without_mixing_samples(self):
        from analysis.metrics import experiment_metrics

        log = _log(
            phase=("evaluation", "evaluation", "evaluation"),
            base_displacement=((2.0, 0.0, 0.0),) * 3,
            e_pos={"right": ((1.0, 0.0, 0.0),
                             (3.0, 0.0, 0.0),
                             (-2.0, 0.0, 0.0))},
            speed_saturated={"right": ((False,) * 7,) * 3},
            joint_margin={"right": ((0.5,), (0.4,), (0.3,))},
            gain_segment=(0, 0, 1),
            contact_count=(0, 1, 1),
            contact_pairs=(
                (),
                ("right_hand/a <> floor/f",),
                ("torso/t <> right_forearm/a",),
            ),
            torso_contact={"right": (False, False, True)},
        )

        result = experiment_metrics(log)

        self.assertEqual(set(result["gain_segments"]), {"0", "1"})
        first = result["gain_segments"]["0"]
        second = result["gain_segments"]["1"]
        self.assertEqual(first["evaluation_sample_count"], 2)
        self.assertEqual(first["arms"]["right"]["position_error_axis_mean_m"],
                         [2.0, 0.0, 0.0])
        self.assertEqual(first["contact_occupancy_total_pct"], 50.0)
        self.assertEqual(
            first["arms"]["right"]["contact_occupancy_total_pct"], 50.0)
        self.assertEqual(second["evaluation_sample_count"], 1)
        self.assertEqual(second["arms"]["right"]["position_error_axis_mean_m"],
                         [-2.0, 0.0, 0.0])
        self.assertEqual(second["arms"]["right"]["rejection_pct"], 0.0)
        self.assertEqual(
            second["arms"]["right"]["contact_occupancy_total_pct"], 100.0)
        self.assertEqual(
            second["arms"]["right"]["torso_contact_occupancy_pct"], 100.0)

    def test_per_arm_contact_occupancy_uses_only_that_sides_pairs(self):
        from analysis.metrics import experiment_metrics

        pairs = (
            ("left_hand/finger <> floor/floor",),
            ("right_hand/finger <> floor/floor",),
            ("torso/torso <> right_forearm_link/forearm",),
            ("box/box <> floor/floor",),
        )
        log = _log(
            phase=("evaluation",) * 4,
            base_displacement=((1.0, 0.0, 0.0),) * 4,
            e_pos={
                "right": ((0.0, 0.0, 0.0),) * 4,
                "left": ((0.0, 0.0, 0.0),) * 4,
            },
            speed_saturated={
                "right": ((False,) * 7,) * 4,
                "left": ((False,) * 7,) * 4,
            },
            joint_margin={
                "right": ((0.1,),) * 4,
                "left": ((0.1,),) * 4,
            },
            gain_segment=(0, 0, 1, 1),
            contact_pairs=pairs,
            torso_contact={
                "right": (False, False, True, False),
                "left": (False,) * 4,
            },
        )

        result = experiment_metrics(log)

        self.assertEqual(result["contact_occupancy_total_pct"], 100.0)
        self.assertEqual(
            result["arms"]["right"]["contact_occupancy_total_pct"], 50.0)
        self.assertEqual(
            result["arms"]["left"]["contact_occupancy_total_pct"], 25.0)
        self.assertEqual(
            result["gain_segments"]["0"]["arms"]["right"]
            ["contact_occupancy_total_pct"],
            50.0,
        )
        self.assertEqual(
            result["gain_segments"]["0"]["arms"]["left"]
            ["contact_occupancy_total_pct"],
            50.0,
        )
        self.assertEqual(
            result["gain_segments"]["1"]["arms"]["right"]
            ["contact_occupancy_total_pct"],
            50.0,
        )
        self.assertEqual(
            result["gain_segments"]["1"]["arms"]["left"]
            ["contact_occupancy_total_pct"],
            0.0,
        )

    def test_static_base_and_empty_evaluation_use_json_safe_none(self):
        from analysis.metrics import experiment_metrics

        static = _log(
            phase=("evaluation",),
            base_displacement=((0.0, 0.0, 0.0),),
            e_pos={"right": ((1.0, 0.0, 0.0),)},
            speed_saturated={"right": ((False,) * 7,)},
            joint_margin={"right": ((np.inf,),)},
        )
        empty = _log(
            phase=("settling",),
            base_displacement=((0.0, 0.0, 0.0),),
            e_pos={"right": ((1.0, 0.0, 0.0),)},
            speed_saturated={"right": ((False,) * 7,)},
            joint_margin={"right": ((np.inf,),)},
        )

        static_result = experiment_metrics(static)
        self.assertIsNone(static_result["arms"]["right"]["rejection_pct"])
        self.assertIsNone(static_result["arms"]["right"]["joint_margin_min_rad"])
        self.assertIsNone(static_result["arms"]["right"]["joint_margin_p01_rad"])
        json.dumps(static_result, allow_nan=False)

        empty_result = experiment_metrics(empty)
        self.assertEqual(empty_result["evaluation_sample_count"], 0)
        self.assertIsNone(empty_result["peak_base_displacement_m"])
        self.assertIsNone(empty_result["contact_occupancy_total_pct"])
        self.assertEqual(empty_result["gain_segments"], {})
        self.assertTrue(all(value is None
                            for value in empty_result["arms"]["right"].values()))
        json.dumps(empty_result, allow_nan=False)


class LegacyMetricsTest(unittest.TestCase):
    def test_legacy_stats_and_window_are_static_base_safe(self):
        from analysis.metrics import stats, windowed_stats

        legacy = stats({
            "base_disp": np.zeros((2, 3)),
            "right_e": np.array([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0]]),
            "left_e": np.zeros((2, 3)),
        })

        self.assertEqual(set(legacy), {"peak_base", "right", "left"})
        self.assertIsNone(legacy["right"]["rejection"])
        self.assertIsNone(legacy["left"]["rejection"])
        self.assertEqual(windowed_stats(
            np.array([1.0, 2.0]), np.zeros(2)),
            (np.sqrt(2.5), 2.0, None),
        )

    def test_print_stats_keeps_legacy_alignment_and_formats_none(self):
        from analysis.metrics import print_stats

        values = {
            "peak_base": 2.0,
            "right": {
                "rms": np.zeros(3), "peak": np.zeros(3),
                "peak_norm": 1.0, "rejection": 50.0,
            },
            "left": {
                "rms": np.zeros(3), "peak": np.zeros(3),
                "peak_norm": 0.0, "rejection": None,
            },
        }
        stream = io.StringIO()

        with redirect_stdout(stream):
            print_stats(values)

        lines = stream.getvalue().splitlines()
        self.assertTrue(lines[1].endswith(f"{50.0:11.1f}%  [mm]"))
        self.assertTrue(lines[2].endswith(f"{'n/a':>12s}  [mm]"))


class MetricsPersistenceTest(unittest.TestCase):
    def test_save_run_writes_json_and_flat_csv(self):
        from analysis.live import save_run
        from analysis.metrics import experiment_metrics

        log = _log(
            phase=("evaluation",),
            base_displacement=((2.0, 0.0, 0.0),),
            e_pos={"right": ((1.0, -2.0, 0.0),)},
            speed_saturated={"right": ((True, False, False, False,
                                         False, False, False),)},
            joint_margin={"right": ((0.25, np.inf),)},
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = save_run(log, log.config, Path(tmp))
            stored_json = json.loads((run_dir / "metrics.json").read_text())
            with (run_dir / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(stored_json, experiment_metrics(log))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evaluation_sample_count"], "1")
        self.assertEqual(rows[0]["arms.right.position_error_axis_mean_m.x"], "1.0")
        self.assertEqual(rows[0]["arms.right.position_error_axis_mean_m.y"], "-2.0")
        self.assertEqual(rows[0]["arms.right.rejection_pct"],
                         str((1.0 - np.sqrt(5.0) / 2.0) * 100.0))


if __name__ == "__main__":
    unittest.main()
