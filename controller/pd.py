"""Pure control math for the reactive controller. NumPy + Pinocchio only —
no MuJoCo, no scene knowledge.

Error convention: e = reference - actual, expressed in the world frame.
e_pos in meters. e_rot is the axis-angle vector (radians) of
R_err = R_ref @ R_ee.T, the world-frame rotation taking the actual
orientation to the reference.

Twist ordering is linear-first, matching the Jacobian rows:
v = [vx, vy, vz (m/s); wx, wy, wz (rad/s)], world frame.
"""

import numpy as np
import pinocchio as pin

KP_POS = 2.0    # 1/s task-space bandwidth
KP_ROT = 2.0    # 1/s
DAMPING = 0.05  # DLS lambda


def position_error(ref_pos, ee_pos):
    """World-frame position error, meters: reference - actual."""
    return ref_pos - ee_pos


def rotation_error(ref_rot, ee_rot):
    """World-frame axis-angle error, radians: log3(R_ref @ R_ee.T)."""
    return pin.log3(ref_rot @ ee_rot.T)


def qdot_from_error(J, e_pos, e_rot, damping=DAMPING):
    """World-frame pose error -> joint rates (rad/s), two equations:

    1. control law:   v = [KP_POS*e_pos; KP_ROT*e_rot]  (commanded twist)
    2. DLS inversion: qdot = J^T (J J^T + damping^2 I)^-1 v
       — bounded qdot through singularities at the cost of a small bias."""
    v = np.concatenate([KP_POS * e_pos, KP_ROT * e_rot])
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), v)
