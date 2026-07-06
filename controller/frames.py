"""Frame compositions for the torso + dual-Gen3 scene. Scene-specific glue.

This is the disposable layer: it knows about sim.world, the "right_"/"left_"
naming convention, and the torso mount. The reusable layers live in
controller/transforms.py (pure math) and controller/kinematics.py (chain FK).

Frames: W = world, T = torso mocap body, K = arm base_link, E = EE site.
"""

from controller.kinematics import extract_chain, fk
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


right_chain = extract_chain(world.model, "right_base_link", "right_pinch_site")
left_chain = extract_chain(world.model, "left_base_link", "left_pinch_site")

T_T_KR = mount_transform(world.model, world.kinova_right_base_id)
T_T_KL = mount_transform(world.model, world.kinova_left_base_id)


def torso_pose():
    """Commanded torso mocap pose in the world frame."""
    pos = world.data.xpos[world.torso_mocap_id].copy()
    rot = world.data.xmat[world.torso_mocap_id].reshape(3, 3).copy()
    return pos, rot


def _ee_pose_world(T_T_K, chain):
    """T_W_E(q, t) = T_W_T(t) · T_T_K · T_K_E(q).

    T_W_T is the commanded torso pose (an input, not an FK output); T_T_K is
    a fixed mount from model constants; T_K_E(q) is the analytical chain FK.
    No EE pose is read from MuJoCo.
    """
    T_W_T = transform_from_pose(*torso_pose())
    return pose_from_transform(T_W_T @ T_T_K @ fk(chain, world.data.qpos))


def right_ee_pose():
    return _ee_pose_world(T_T_KR, right_chain)


def left_ee_pose():
    return _ee_pose_world(T_T_KL, left_chain)


def measured_right_ee_pose():
    """Right EE pose as MuJoCo measures it (the comparison side, not FK)."""
    pos = world.data.site_xpos[world.right_ee_id].copy()
    rot = world.data.site_xmat[world.right_ee_id].reshape(3, 3).copy()
    return pos, rot


def measured_left_ee_pose():
    pos = world.data.site_xpos[world.left_ee_id].copy()
    rot = world.data.site_xmat[world.left_ee_id].reshape(3, 3).copy()
    return pos, rot
