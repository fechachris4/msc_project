import mujoco
import numpy as np
import pinocchio as pin

from controller.pin_fk import build_pin_model, pin_T_K_E
from controller.transforms import (
    pose_from_transform,
    rotation_from_quat,
    transform_from_pose,
)
from sim import world


def mount_transform(model, base_body_id):
    """T_T_K: base_link pose in the torso frame, from model constants
    (base_link's parent in sim/scene.xml is the torso mocap body)."""
    return transform_from_pose(
        model.body_pos[base_body_id].copy(),
        rotation_from_quat(model.body_quat[base_body_id]),
    )

def arm_qpos_adrs(model, prefix):
    """Global qpos addresses of one arm's 7 joints, base to tip."""
    adrs = []
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint_{i}"
        )
        assert jnt_id >= 0, f"{prefix}joint_{i} not in model"
        adrs.append(int(model.jnt_qposadr[jnt_id]))
    return adrs


# One arm model serves both arms: left/right are the same MJCF, and the
# mount difference lives in T_T_K, not in T_K_E.
pin_model, pin_data, ee_frame_id = build_pin_model(
    "sim/assets/kinova_gen3/gen3.xml", "base_link", "pinch_site"
)
right_qpos_adrs = arm_qpos_adrs(world.model, "right_")
left_qpos_adrs = arm_qpos_adrs(world.model, "left_")

T_T_KR = mount_transform(world.model, world.kinova_right_base_id)
T_T_KL = mount_transform(world.model, world.kinova_left_base_id)


def torso_pose():
    """T_W_T: torso pose in the world frame, as MuJoCo measures it"""
    pos = world.data.xpos[world.torso_mocap_id].copy()
    rot = world.data.xmat[world.torso_mocap_id].reshape(3, 3).copy()
    return pos, rot


def _ee_pose_world(T_T_K, qpos_adrs):
    """T_W_E(q, t) = T_W_T(t) · T_T_K · T_K_E(q).

    T_W_T is the torso pose in the world frame, as MuJoCo measures it; T_T_K is
    a fixed mount from model constants; T_K_E(q) is Pinocchio FK.
    No EE pose is read from MuJoCo.
    """
    T_W_T = transform_from_pose(*torso_pose())
    T_K_E = pin_T_K_E(pin_model, pin_data, ee_frame_id,
                      world.data.qpos[qpos_adrs])
    return pose_from_transform(T_W_T @ T_T_K @ T_K_E)


def right_ee_pose():
    return _ee_pose_world(T_T_KR, right_qpos_adrs)


def left_ee_pose():
    return _ee_pose_world(T_T_KL, left_qpos_adrs)


def _jacobian_world(T_T_K, qpos_adrs):
    """World-aligned EE frame Jacobian, 6x7: linear rows then angular.

    Pinocchio computes it aligned with the arm base K; both blocks are
    rotated by R_W_K = R_W_T @ R_T_K. The base is kinematic (mocap), so
    the joint columns are the complete Jacobian."""
    q = np.asarray(world.data.qpos[qpos_adrs], dtype=float)
    J_K = pin.computeFrameJacobian(pin_model, pin_data, q, ee_frame_id,
                                   pin.LOCAL_WORLD_ALIGNED)
    _, torso_rot = torso_pose()
    R_W_K = torso_rot @ T_T_K[:3, :3]
    J = np.empty_like(J_K)
    J[:3] = R_W_K @ J_K[:3]
    J[3:] = R_W_K @ J_K[3:]
    return J


def right_jacobian_world():
    return _jacobian_world(T_T_KR, right_qpos_adrs)


def left_jacobian_world():
    return _jacobian_world(T_T_KL, left_qpos_adrs)


def measured_right_ee_pose():
    """Right EE pose as MuJoCo measures it (the comparison side, not FK)."""
    pos = world.data.site_xpos[world.right_ee_id].copy()
    rot = world.data.site_xmat[world.right_ee_id].reshape(3, 3).copy()
    return pos, rot


def measured_left_ee_pose():
    pos = world.data.site_xpos[world.left_ee_id].copy()
    rot = world.data.site_xmat[world.left_ee_id].reshape(3, 3).copy()
    return pos, rot
