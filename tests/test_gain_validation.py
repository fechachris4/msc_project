import unittest


class GainPromotionGateTest(unittest.TestCase):
    def test_policy_is_explicitly_pending_and_empty(self):
        from analysis import gain_validation

        policy = gain_validation.load_policy()
        self.assertEqual(policy["status"], "policy_pending")
        self.assertEqual(policy["scenario_matrix"], [])
        self.assertEqual(policy["acceptance_thresholds"], {})
        self.assertFalse(policy["promotion_enabled"])

    def test_candidates_can_be_recorded_as_exploratory(self):
        from analysis import gain_validation

        report = gain_validation.exploratory_report(
            "working_tree_32_2",
            evidence=({"kind": "position_gain_sweep", "manifest": "abc"},),
        )
        self.assertEqual(report["classification"], "exploratory")
        self.assertEqual(report["gains"]["KP_POS"], 32.0)
        self.assertFalse(report["promotion_requested"])

    def test_pending_policy_refuses_promotion(self):
        from analysis import gain_validation

        report = gain_validation.exploratory_report("committed_20_0.9")
        decision = gain_validation.promotion_decision(report)
        self.assertFalse(decision.eligible)
        self.assertIn("joint validation policy is pending", decision.reasons)
        with self.assertRaisesRegex(RuntimeError, "promotion refused"):
            gain_validation.require_promotion_eligible(report)

    def test_diagnostic_sweeps_alone_cannot_promote(self):
        from analysis import gain_validation

        policy = gain_validation.load_policy()
        policy.update({
            "status": "approved",
            "scenario_matrix": [{"id": "joint"}],
            "acceptance_thresholds": {"position_rmse_m": 0.01},
            "promotion_enabled": True,
        })
        report = {
            "classification": "joint_validation",
            "evidence": [
                {"kind": "position_gain_sweep"},
                {"kind": "orientation_gain_sweep"},
            ],
            "all_thresholds_passed": True,
            "canonical_manifest_sha256": "abc",
        }
        decision = gain_validation.promotion_decision(report, policy)
        self.assertFalse(decision.eligible)
        self.assertIn("diagnostic sweeps cannot promote gains", decision.reasons)


if __name__ == "__main__":
    unittest.main()
