import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np


def _config(**changes):
    from analysis.live import ExperimentConfig

    values = dict(
        arms=("right", "left"),
        linear_amplitude=np.zeros(3),
        linear_frequency=0.0,
        rotational_amplitude=np.zeros(3),
        rotational_frequency=0.0,
        evaluation_seconds=0.006,
        settle_pos_tol=10.0,
        settle_rot_tol=10.0,
        settle_dwell=0.0,
        settle_timeout=0.01,
    )
    values.update(changes)
    return ExperimentConfig(**values)


class ExperimentConfigContractTest(unittest.TestCase):
    def test_exact_fields_and_defaults(self):
        from analysis.live import ExperimentConfig

        self.assertEqual(
            ExperimentConfig._fields,
            ("arms", "linear_amplitude", "linear_frequency",
             "rotational_amplitude", "rotational_frequency",
             "evaluation_seconds", "settle_pos_tol", "settle_rot_tol",
             "settle_dwell", "settle_timeout"),
        )
        config = ExperimentConfig(
            ("right",), np.zeros(3), 0.0, np.zeros(3), 0.0, 0.01)
        self.assertEqual(config.settle_pos_tol, 0.002)
        self.assertEqual(config.settle_rot_tol, np.deg2rad(0.5))
        self.assertEqual(config.settle_dwell, 0.5)
        self.assertEqual(config.settle_timeout, 15.0)

    def test_rejects_invalid_public_inputs(self):
        from analysis.live import run_experiment

        for config in (
            _config(arms=()),
            _config(linear_amplitude=np.zeros(2)),
            _config(linear_frequency=-1.0),
            _config(evaluation_seconds=0.0),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_experiment(config)


class ResetAndRunTest(unittest.TestCase):
    def tearDown(self):
        from analysis.live import _reset_simulation

        _reset_simulation()

    def test_complete_reset_restores_plant_state(self):
        from analysis.live import _reset_simulation
        from controller import frames, servo
        from runtime_config import CONFIG
        from sim import world

        world.data.qpos[:] = 1.0
        world.data.qvel[:] = 2.0
        world.data.ctrl[:] = 3.0
        world.data.mocap_pos[:] = 4.0
        world.data.mocap_quat[:] = np.array([0.0, 1.0, 0.0, 0.0])
        world.data.time = 9.0

        _reset_simulation()

        np.testing.assert_array_equal(world.data.qpos, world.model.qpos0)
        np.testing.assert_array_equal(world.data.qvel, np.zeros(world.model.nv))
        self.assertEqual(world.data.time, 0.0)
        self.assertIs(servo.CONTROL, CONFIG.reactive_pose)
        for side in world.SIDES:
            np.testing.assert_array_equal(
                world.data.ctrl[world.ctrl_adrs[side]],
                world.data.qpos[world.qpos_adrs[side]],
            )
        self.assertTrue(np.all(np.isfinite(world.data.mocap_pos)))
        self.assertTrue(np.all(np.isfinite(world.data.mocap_quat)))

    def test_threshold_dwell_then_evaluation_phase_zero(self):
        from analysis.live import run_experiment
        from sim import world

        log = run_experiment(_config(settle_dwell=0.004))
        mask = log.evaluation_mask
        self.assertTrue(log.settled)
        self.assertGreaterEqual(log.settle_duration, 0.004)
        self.assertTrue(np.all(log.phase[~mask] == "settling"))
        self.assertTrue(np.all(log.phase[mask] == "evaluation"))
        self.assertEqual(log.eval_time[mask][0], 0.0)
        self.assertEqual(log.eval_time[mask][1], world.model.opt.timestep)
        np.testing.assert_array_equal(log.base_displacement[mask][0], np.zeros(3))

    def test_timeout_and_short_period_warn_but_return_data(self):
        from analysis.live import run_experiment

        log = run_experiment(_config(
            linear_amplitude=np.array([0.01, 0.0, 0.0]),
            linear_frequency=1.0,
            settle_pos_tol=0.0,
            settle_rot_tol=0.0,
            settle_timeout=0.002,
        ))
        self.assertFalse(log.settled)
        self.assertFalse(log.valid)
        self.assertIn("settling timeout", log.warning_reasons)
        self.assertIn("evaluation shorter than one disturbance period",
                      log.warning_reasons)
        self.assertGreater(np.count_nonzero(log.evaluation_mask), 0)

    def test_same_process_repeatability(self):
        from analysis.live import run_experiment

        first = run_experiment(_config())
        second = run_experiment(_config())
        for field in ("sim_time", "contact_time", "eval_time", "base_displacement",
                      "base_linear_velocity", "base_angular_velocity",
                      "contact_count"):
            np.testing.assert_allclose(getattr(first, field), getattr(second, field),
                                       atol=1e-12, rtol=0.0)
        for side in first.arms:
            for name in first.arm_data[side]:
                np.testing.assert_allclose(
                    first.arm_data[side][name], second.arm_data[side][name],
                    atol=1e-12, rtol=0.0)


class ContactAndPersistenceTest(unittest.TestCase):
    def test_pair_specific_torso_classification(self):
        from analysis.live import _torso_contact_from_pairs

        pairs = (
            "torso/geom_1 <> right_forearm_link/geom_2",
            "left_hand/finger <> floor/floor",
        )
        self.assertEqual(
            _torso_contact_from_pairs(pairs, ("right", "left")),
            {"right": True, "left": False},
        )

    def test_npz_json_round_trip(self):
        from analysis.live import load_run, run_experiment, save_run
        from runtime_config import CONFIG

        config = _config(arms=("right",))
        log = run_experiment(config)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = save_run(log, config, Path(tmp))
            self.assertTrue((run_dir / "run.npz").is_file())
            self.assertTrue((run_dir / "metadata.json").is_file())
            self.assertTrue((run_dir / "manifest.json").is_file())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["classification"], "exploratory")
            self.assertIn("run.npz", manifest["artifacts"])
            self.assertIn("controller_configuration", manifest)
            self.assertIn("effective_control_config", manifest)
            self.assertIn("control_config_sha256", manifest)
            self.assertEqual(
                manifest["effective_control_config"]["controller"][
                    "reactive_pose"
                ]["kd_position"],
                CONFIG.reactive_pose.kd_position,
            )
            self.assertEqual(
                manifest["control_config_sha256"], CONFIG.source_sha256)
            loaded = load_run(run_dir)
        for field in dataclasses.fields(log):
            expected = getattr(log, field.name)
            actual = getattr(loaded, field.name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            elif field.name == "config":
                self.assertEqual(actual.arms, expected.arms)
                np.testing.assert_array_equal(
                    actual.linear_amplitude, expected.linear_amplitude)
                np.testing.assert_array_equal(
                    actual.rotational_amplitude, expected.rotational_amplitude)
                for name in (
                    "linear_frequency", "rotational_frequency",
                    "evaluation_seconds", "settle_pos_tol", "settle_rot_tol",
                    "settle_dwell", "settle_timeout",
                ):
                    self.assertEqual(getattr(actual, name),
                                     getattr(expected, name))
            elif field.name == "arm_data":
                for side in log.arms:
                    for name, value in expected[side].items():
                        np.testing.assert_array_equal(actual[side][name], value)
            elif field.name == "torso_contact":
                for side in log.arms:
                    np.testing.assert_array_equal(actual[side], expected[side])
            else:
                self.assertEqual(actual, expected, field.name)

    def test_save_rejects_mismatched_configuration(self):
        from analysis.live import run_experiment, save_run

        config = _config(arms=("right",))
        log = run_experiment(config)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            save_run(log, config._replace(evaluation_seconds=1.0), Path(tmp))

    def test_canonical_save_rejects_dirty_worktree(self):
        from analysis import provenance
        from analysis.live import run_experiment, save_run

        config = _config(arms=("right",))
        log = run_experiment(config)
        dirty = {
            "revision": "abc", "branch": "main", "worktree_clean": False,
            "worktree_status_sha256": "dirty",
        }
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(provenance, "git_provenance",
                                  return_value=dirty), \
                self.assertRaisesRegex(RuntimeError, "clean Git worktree"):
            save_run(log, config, Path(tmp), canonical=True)

    def test_update_callback_receives_exact_logged_sample_count(self):
        from analysis.live import ExperimentUpdate, run_experiment

        updates = []
        log = run_experiment(_config(arms=("left",)), updates.append)
        self.assertEqual(len(updates), len(log.sim_time))
        self.assertTrue(all(isinstance(update, ExperimentUpdate)
                            for update in updates))
        self.assertEqual(updates[-1].phase, "evaluation")

    def test_contact_samples_align_with_the_control_state_timestamp(self):
        from analysis.live import run_experiment

        log = run_experiment(_config(arms=("right",)))
        np.testing.assert_allclose(
            log.contact_time,
            log.sim_time,
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            log.contact_count,
            np.asarray([len(pairs) for pairs in log.contact_pairs]),
        )


class GainOverrideTest(unittest.TestCase):
    def tearDown(self):
        from analysis.live import _reset_simulation

        _reset_simulation()

    def test_overrides_land_in_first_gain_snapshot(self):
        from analysis.live import run_experiment

        log = run_experiment(_config(), gains={"KP_POS": 5.0, "K_NULL": 2.0})
        self.assertEqual(log.gain_snapshots[0]["KP_POS"], 5.0)
        self.assertEqual(log.gain_snapshots[0]["K_NULL"], 2.0)

    def test_override_is_run_local_and_does_not_mutate_startup_config(self):
        from analysis.live import run_experiment
        from controller import servo
        from runtime_config import CONFIG

        log = run_experiment(
            _config(arms=("right",)), gains={"K_NULL": 3.5})
        self.assertEqual(log.gain_snapshots[0]["K_NULL"], 3.5)
        self.assertIs(servo.CONTROL, CONFIG.reactive_pose)
        self.assertEqual(CONFIG.reactive_pose.null_gain_s_inv, 1.0)

    def test_next_run_reconstructs_from_startup_config(self):
        from analysis.live import run_experiment
        from controller import servo
        from runtime_config import CONFIG

        changed = run_experiment(_config(), gains={"KP_POS": 9.0})
        default = run_experiment(_config())
        self.assertEqual(changed.gain_snapshots[0]["KP_POS"], 9.0)
        self.assertEqual(
            default.gain_snapshots[0]["KP_POS"],
            CONFIG.reactive_pose.kp_position_s_inv,
        )
        self.assertIs(servo.CONTROL, CONFIG.reactive_pose)

    def test_bogus_gain_name_raises(self):
        from analysis.live import run_experiment

        with self.assertRaises(ValueError):
            run_experiment(_config(), gains={"NOT_A_GAIN": 1.0})

    def test_negative_gain_value_raises(self):
        from analysis.live import run_experiment

        with self.assertRaises(ValueError):
            run_experiment(_config(), gains={"KP_POS": -1.0})


if __name__ == "__main__":
    unittest.main()
