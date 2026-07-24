import unittest


class ProvenanceContractTest(unittest.TestCase):
    def test_current_environment_matches_the_pinned_primary(self):
        from analysis import provenance

        self.assertTrue(provenance.primary_environment_matches())

    def test_controller_identity_contains_all_runtime_levers(self):
        from analysis import provenance

        config = provenance.controller_configuration()
        self.assertEqual(
            set(config["gains"]),
            {"KP_POS", "KP_ROT", "KD_POS", "KD_ROT", "K_NULL", "DAMPING"},
        )
        self.assertEqual(
            set(config["components"]),
            {"position_enabled", "orientation_enabled", "velocity_enabled"},
        )
        self.assertEqual(len(config["joint_speed_limits_rad_s"]), 7)
        self.assertEqual(
            config["command_interface"],
            "integrated_clipped_joint_velocity_to_position",
        )

    def test_environment_mismatch_is_not_primary(self):
        from analysis import provenance

        snapshot = provenance.environment_snapshot()
        snapshot["dependencies"] = dict(snapshot["dependencies"])
        snapshot["dependencies"]["numpy"] = "different"
        self.assertFalse(provenance.primary_environment_matches(snapshot))

    def test_identity_stamps_effective_config_and_raw_toml_hash(self):
        from analysis import provenance
        from runtime_config import CONFIG

        identity = provenance.experiment_identity(
            {"scenario": "test"}, {"KD_POS": 0.8})
        self.assertEqual(
            identity["effective_control_config"]["controller"][
                "reactive_pose"
            ]["kd_position"],
            0.8,
        )
        self.assertEqual(
            identity["control_config_sha256"], CONFIG.source_sha256)


if __name__ == "__main__":
    unittest.main()
