"""Pure control math for the reactive controller. NumPy + Pinocchio only —
no MuJoCo, no scene knowledge.

Error convention: e = reference - actual, expressed in the world frame.
e_pos in meters. e_rot is the axis-angle vector (radians) of
R_err = R_ref @ R_ee.T, the world-frame rotation taking the actual
orientation to the reference.

Proportional law (resolved-rate): v = Kp * e is a commanded world twist;
qdot comes from a damped-least-squares solve of v = J qdot.
"""

import numpy as np
import pinocchio as pin

KP_POS = 2.0    # 1/s task-space bandwidth
KP_ROT = 2.0    # 1/s
DAMPING = 0.05  # DLS lambda


def position_error(ref_pos, ee_pos):
    """World-frame position error, meters: reference - actual."""
    return np.asarray(ref_pos, dtype=float) - np.asarray(ee_pos, dtype=float)


def rotation_error(ref_rot, ee_rot):
    """World-frame axis-angle error, radians: log3(R_ref @ R_ee.T)."""
    return pin.log3(ref_rot @ ee_rot.T)


def qdot_from_error(J, e_pos, e_rot, damping=DAMPING):
    """Joint rates from the world-frame pose error: DLS solve of v = Kp*e."""
    v = np.concatenate([KP_POS * np.asarray(e_pos, dtype=float),
                        KP_ROT * np.asarray(e_rot, dtype=float)])
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), v)
