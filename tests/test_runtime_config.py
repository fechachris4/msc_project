import tempfile
from pathlib import Path
import unittest
from unittest import mock

from runtime_config import (
    control_with_legacy_overrides,
    effective_config_dict,
    load_config,
    print_effective_config,
)

FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "control.toml"
CONFIG = load_config(FIXTURE_CONFIG)



class RuntimeConfigTest(unittest.TestCase):
    def test_committed_config_contains_complete_baseline(self):
        self.assertEqual(CONFIG.run.nominal_dt_s, 0.002)
        self.assertEqual(CONFIG.run.arm, "both")
        self.assertEqual(CONFIG.reactive_pose.kp_position_s_inv, 2.0)
        self.assertEqual(CONFIG.reactive_pose.kp_rotation_s_inv, 2.0)
        self.assertEqual(CONFIG.reactive_pose.kd_position, 0.3)
        self.assertEqual(CONFIG.reactive_pose.kd_rotation, 0.3)
        self.assertEqual(CONFIG.reactive_pose.null_gain_s_inv, 1.0)
        self.assertEqual(CONFIG.reactive_pose.dls_damping, 0.05)
        self.assertEqual(len(CONFIG.limits.joint_velocity_rad_s), 7)
        self.assertTrue(CONFIG.human_safety.enabled)
        self.assertEqual(
            CONFIG.human_safety.center_xy_torso_m, (0.0, 0.0)
        )
        self.assertEqual(CONFIG.human_safety.radius_m, 0.25)
        self.assertEqual(CONFIG.human_safety.clearance_m, 0.02)
        self.assertEqual(CONFIG.human_safety.control_margin_m, 0.02)
        self.assertEqual(CONFIG.right_target.reference_frame, "world")
        self.assertEqual(CONFIG.left_target.reference_frame, "world")

    def test_config_is_immutable(self):
        with self.assertRaises(AttributeError):
            CONFIG.reactive_pose.kd_position = 0.9
        with self.assertRaises(TypeError):
            CONFIG.limits.joint_velocity_rad_s[0] = 0.1

    def test_override_reconstructs_instead_of_mutating(self):
        changed = control_with_legacy_overrides(
            CONFIG.reactive_pose, {"KD_POS": 0.8, "K_NULL": 2.0}
        )
        self.assertEqual(changed.kd_position, 0.8)
        self.assertEqual(changed.null_gain_s_inv, 2.0)
        self.assertEqual(CONFIG.reactive_pose.kd_position, 0.3)
        self.assertEqual(CONFIG.reactive_pose.null_gain_s_inv, 1.0)

    def test_unknown_or_missing_key_fails(self):
        source = Path(CONFIG.source_path).read_text()
        invalid = source.replace(
            "nominal_dt_s = 0.002",
            "nominal_dt_s = 0.002\nunexpected = 1",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(invalid)
            with self.assertRaisesRegex(ValueError, "run keys differ"):
                load_config(path)

    def test_run_arm_accepts_only_supported_choices(self):
        source = Path(CONFIG.source_path).read_text()
        configured_arm = CONFIG.run.arm
        configured_line = f'arm = "{configured_arm}"'
        with tempfile.TemporaryDirectory() as directory:
            for arm in ("right", "left", "both"):
                path = Path(directory) / f"{arm}.toml"
                path.write_text(source.replace(
                    configured_line,
                    f'arm = "{arm}"',
                    1,
                ))
                self.assertEqual(load_config(path).run.arm, arm)

            path = Path(directory) / "invalid.toml"
            path.write_text(source.replace(
                configured_line,
                'arm = "upper"',
                1,
            ))
            with self.assertRaisesRegex(ValueError, "run.arm"):
                load_config(path)

    def test_enabled_pose_channels_require_positive_kp(self):
        source = Path(CONFIG.source_path).read_text().replace(
            "kp_rotation_s_inv = 2.0",
            "kp_rotation_s_inv = 0.0",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(source)
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                load_config(path)

    def test_enabled_velocity_feedback_requires_kd_below_one(self):
        source = Path(CONFIG.source_path).read_text().replace(
            "kd_position = 0.3",
            "kd_position = 1.0",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(source)
            with self.assertRaisesRegex(ValueError, "less than one"):
                load_config(path)

    def test_step_three_accepts_only_implemented_target_frames(self):
        source = Path(CONFIG.source_path).read_text()
        with tempfile.TemporaryDirectory() as directory:
            for frame in ("world", "base", "torso"):
                path = Path(directory) / f"{frame}.toml"
                path.write_text(source.replace(
                    'reference_frame = "world"',
                    f'reference_frame = "{frame}"',
                    1,
                ))
                self.assertEqual(
                    load_config(path).right_target.reference_frame, frame)

            path = Path(directory) / "invalid.toml"
            path.write_text(source.replace(
                'reference_frame = "world"',
                'reference_frame = "end_effector"',
                1,
            ))
            with self.assertRaisesRegex(ValueError, "reference_frame"):
                load_config(path)

    def test_effective_dictionary_contains_no_loader_metadata(self):
        values = effective_config_dict(CONFIG)
        self.assertNotIn("source_path", values)
        self.assertNotIn("source_sha256", values)
        self.assertIn("reactive_pose", values["controller"])
        self.assertIn("targets", values)
        self.assertIn("limits", values)
        self.assertIn("human_safety", values)

    def test_startup_print_contains_effective_config_and_source_hash(self):
        with mock.patch("builtins.print") as output:
            print_effective_config(CONFIG)
        rendered = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in output.call_args_list
        )
        self.assertIn('"kd_position": 0.3', rendered)
        self.assertIn(f"config_sha256={CONFIG.source_sha256}", rendered)


if __name__ == "__main__":
    unittest.main()
