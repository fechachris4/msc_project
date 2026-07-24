import unittest

import numpy as np


class RunAcceptancePolicyTest(unittest.TestCase):
    def _assess(self, *, margin=0.1, contact=False, finite=True):
        from analysis import policy
        from analysis.live import ExperimentConfig

        config = ExperimentConfig(
            arms=("right",),
            linear_amplitude=np.zeros(3),
            linear_frequency=0.0,
            rotational_amplitude=np.zeros(3),
            rotational_frequency=0.0,
            evaluation_seconds=0.002,
        )
        arrays = {
            "phase": np.asarray(["evaluation"]),
            "sim_time": np.asarray([0.0]),
            "contact_time": np.asarray([0.0]),
            "eval_time": np.asarray([0.0]),
            "base_displacement": np.zeros((1, 3)),
            "base_linear_velocity": np.zeros((1, 3)),
            "base_angular_velocity": np.zeros((1, 3)),
            "contact_count": np.asarray([1 if contact else 0]),
        }
        arm_data = {
            "right": {
                "e_pos": np.asarray([[0.0, 0.0,
                                      0.0 if finite else np.nan]]),
                "joint_margin": np.asarray([[margin, np.inf]]),
            }
        }
        return policy.assess_run(
            config, arrays, arm_data,
            {"right": np.asarray([contact])}, settled=True)

    def test_contact_warns_without_automatic_rejection(self):
        status = self._assess(contact=True)
        self.assertTrue(status.contact_observed)
        self.assertTrue(status.accepted)
        self.assertIn("contact detected", status.warning_reasons)

    def test_penetration_at_tolerance_is_accepted(self):
        from analysis import policy

        status = self._assess(
            margin=-policy.JOINT_LIMIT_MAX_PENETRATION_RAD)
        self.assertTrue(status.joint_limit_within_tolerance)
        self.assertTrue(status.accepted)

    def test_penetration_above_tolerance_is_rejected(self):
        from analysis import policy

        status = self._assess(
            margin=-(policy.JOINT_LIMIT_MAX_PENETRATION_RAD + 1e-6))
        self.assertFalse(status.joint_limit_within_tolerance)
        self.assertFalse(status.accepted)

    def test_noncomputable_metrics_cannot_be_accepted(self):
        status = self._assess(finite=False)
        self.assertFalse(status.metrics_computable)
        self.assertFalse(status.accepted)


if __name__ == "__main__":
    unittest.main()
