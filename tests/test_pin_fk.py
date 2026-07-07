"""Pinocchio FK vs the analytical fk() from controller.kinematics.

Both compute T_K_E (base_link -> pinch_site) for one Gen3 arm. The
analytical chain is extracted from the full scene (right arm); the
Pinocchio model is built from the arm-only MJCF — same kinematics.
"""

import unittest

import numpy as np


ARM_MJCF = "sim/assets/kinova_gen3/gen3.xml"
N_SAMPLES = 20
POS_TOL = 1e-9   # metres
ROT_TOL = 1e-9   # radians, geodesic


def geodesic_angle(R_a, R_b):
    # atan2(sin, cos) form: arccos((tr-1)/2) alone loses precision below
    # ~1e-7 rad near identity, which is coarser than the 1e-9 tolerance.
    R = R_a.T @ R_b
    sin_theta = 0.5 * np.linalg.norm(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return float(np.arctan2(sin_theta, cos_theta))


class PinFKTest(unittest.TestCase):
    def test_pin_fk_matches_analytical_fk(self):
        from controller import kinematics
        from controller.pin_fk import build_pin_model, pin_T_K_E
        from sim import world

        chain = kinematics.extract_chain(
            world.model, "right_base_link", "right_pinch_site"
        )
        pin_model, pin_data, frame_id = build_pin_model(
            ARM_MJCF, "base_link", "pinch_site"
        )
        self.assertEqual(pin_model.nq, len(chain.joint_ids))

        rng = np.random.default_rng(123)
        qpos = np.zeros(world.model.nq)
        failures = []
        for _ in range(N_SAMPLES):
            q = np.empty(7)
            for i, (jnt_id, adr) in enumerate(
                zip(chain.joint_ids, chain.qpos_adrs)
            ):
                if world.model.jnt_limited[jnt_id]:
                    low, high = world.model.jnt_range[jnt_id]
                else:
                    low, high = -np.pi, np.pi
                q[i] = rng.uniform(low, high)
                qpos[adr] = q[i]

            T_analytical = kinematics.fk(chain, qpos)
            T_pin = pin_T_K_E(pin_model, pin_data, frame_id, q)

            pos_err = np.linalg.norm(T_pin[:3, 3] - T_analytical[:3, 3])
            rot_err = geodesic_angle(T_pin[:3, :3], T_analytical[:3, :3])
            if pos_err > POS_TOL or rot_err > ROT_TOL:
                failures.append((q.copy(), T_analytical, T_pin,
                                 pos_err, rot_err))

        for q, T_a, T_p, pos_err, rot_err in failures:
            print(f"\nFAIL q={q}\n"
                  f"pos_err={pos_err:.3e} m, rot_err={rot_err:.3e} rad\n"
                  f"analytical fk():\n{T_a}\npin_T_K_E:\n{T_p}")
        self.assertEqual(
            len(failures), 0,
            f"{len(failures)}/{N_SAMPLES} configurations exceeded tolerance",
        )


if __name__ == "__main__":
    unittest.main()
