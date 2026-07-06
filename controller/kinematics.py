import numpy as np

from sim import world

def transform_from_pose(pos, rot):
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform

def pose_from_transform(transform):
    return transform[:3, 3].copy(), transform[:3, :3].copy()

def inverse_transform(transform):
    inverse = np.eye(4)
    rot = transform[:3, :3]
    pos = transform[:3, 3]
    inverse[:3, :3] = rot.T
    inverse[:3, 3] = -rot.T @ pos
    return inverse

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

def direct_left_ee_pose():
    pos = world.data.site_xpos[world.left_ee_id].copy()
    rot = world.data.site_xmat[world.left_ee_id].reshape(3, 3).copy()
    return pos, rot

def direct_right_ee_pose():
    pos = world.data.site_xpos[world.right_ee_id].copy()
    rot = world.data.site_xmat[world.right_ee_id].reshape(3, 3).copy()
    return pos, rot

def torso_mocap_position():
    pos = world.data.xpos[world.torso_mocap_id].copy()
    rot = world.data.xmat[world.torso_mocap_id].reshape(3, 3).copy()
    return pos, rot

def kinova_right_base_link_position():
    pos = world.data.xpos[world.kinova_right_base_id].copy()
    rot = world.data.xmat[world.kinova_right_base_id].reshape(3, 3).copy()
    return pos, rot

def kinova_left_base_link_position():
    pos = world.data.xpos[world.kinova_left_base_id].copy()
    rot = world.data.xmat[world.kinova_left_base_id].reshape(3, 3).copy()
    return pos, rot

def right_ee_positions():
    """
    W = world
    T = torso mocap body
    K = Kinova base_link
    E = end-effector site, right_pinch_site
    R = right
    L = left

    T_W_E(q, t) = T_W_T(t) · T_T_K(t) · T_K_E(q)
    """
    T_W_T = transform_from_pose(*torso_mocap_position())
    T_W_KR = transform_from_pose(*kinova_right_base_link_position())
    T_W_E_direct = transform_from_pose(*direct_right_ee_pose())

    T_T_KR = inverse_transform(T_W_T) @ T_W_KR
    T_KR_E = inverse_transform(T_W_KR) @ T_W_E_direct
    T_W_E = T_W_T @ T_T_KR @ T_KR_E

    return pose_from_transform(T_W_E)
