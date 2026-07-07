"""Pure control math for the reactive controller. NumPy + Pinocchio only —
no MuJoCo, no scene knowledge.

Error convention: e = reference - actual, expressed in the world frame.
e_pos in meters. e_rot is the axis-angle vector (radians) of
R_err = R_ref @ R_ee.T, the world-frame rotation taking the actual
orientation to the reference.

Proportional law (resolved-rate), two separable equations:
  1. control law:        v = Kp * e         (pose error -> commanded twist)
  2. kinematic inversion: qdot = J^+_dls v  (twist -> joint rates)

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
    return np.asarray(ref_pos, dtype=float) - np.asarray(ee_pos, dtype=float)


def rotation_error(ref_rot, ee_rot):
    """World-frame axis-angle error, radians: log3(R_ref @ R_ee.T)."""
    return pin.log3(ref_rot @ ee_rot.T)


def twist_from_error(e_pos, e_rot):
    """Control law: commanded world twist v = [KP_POS*e_pos; KP_ROT*e_rot].

    e_pos in meters, e_rot in radians (axis-angle) -> v in [m/s; rad/s],
    ordered linear-first to match the Jacobian rows."""
    return np.concatenate([KP_POS * np.asarray(e_pos, dtype=float),
                           KP_ROT * np.asarray(e_rot, dtype=float)])


def qdot_from_twist(J, v, damping=DAMPING):
    """Kinematic inversion: joint rates (rad/s) realizing the twist v.

    Damped least squares: qdot = J^T (J J^T + damping^2 I)^-1 v, which
    solves min ||J qdot - v||^2 + damping^2 ||qdot||^2 — bounded qdot
    through singularities at the cost of a small tracking bias."""
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), v)


def qdot_from_error(J, e_pos, e_rot, damping=DAMPING):
    """Full step: world-frame pose error -> commanded twist -> joint rates."""
    return qdot_from_twist(J, twist_from_error(e_pos, e_rot), damping)
