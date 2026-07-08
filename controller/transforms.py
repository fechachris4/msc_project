"""Pure rigid-transform math. NumPy only — no MuJoCo, no scene knowledge."""

import numpy as np


def transform_from_pose(pos, rot):
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform


def pose_from_transform(transform):
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def rotation_from_quat(quat):
    """Rotation matrix from a MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotation_about_axis(axis, angle):
    """Rodrigues rotation about a unit axis."""
    kx, ky, kz = axis
    cross = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def rotation_from_rpy(rpy):
    """Rotation matrix from [roll, pitch, yaw], composed as Rz @ Ry @ Rx."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def angular_velocity_from_rpy_rates(rpy, rpy_dot):
    """World-frame angular velocity (rad/s) of a frame following
    rotation_from_rpy(rpy(t)) — the derivative counterpart of its
    Rz @ Ry @ Rx composition.

    Each rate spins about its own current axis, expressed in the world:
    w = yaw_dot*ez + pitch_dot*(Rz ey) + roll_dot*(Rz Ry ex). The columns
    of E are those three axes (roll does not appear: it rotates about the
    already-transformed x axis)."""
    _, pitch, yaw = rpy
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    E = np.array([
        [cy * cp, -sy, 0.0],
        [sy * cp,  cy, 0.0],
        [-sp,     0.0, 1.0],
    ])
    return E @ np.asarray(rpy_dot, dtype=float)
