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


def arm_dof_adrs(model, prefix):
    """Global qvel (dof) addresses of one arm's 7 joints, base to tip."""
    adrs = []
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint_{i}"
        )
        assert jnt_id >= 0, f"{prefix}joint_{i} not in model"
        adrs.append(int(model.jnt_dofadr[jnt_id]))
    return adrs


# One arm model serves both arms: left/right are the same MJCF, and the
# mount difference lives in T_T_K, not in T_K_E.
pin_model, pin_data, ee_frame_id = build_pin_model(
    "sim/assets/kinova_gen3/gen3.xml", "base_link", "pinch_site"
)
qpos_adrs = {s: arm_qpos_adrs(world.model, f"{s}_") for s in world.SIDES}
dof_adrs = {s: arm_dof_adrs(world.model, f"{s}_") for s in world.SIDES}
_T_T_K = {s: mount_transform(world.model, world.arm_base_id[s])
          for s in world.SIDES}


def torso_pose():
    """T_W_T: torso pose in the world frame, as MuJoCo measures it.

    Returns (pos (3,) meters, R 3x3) — the (pos, rot) pair convention
    used throughout."""
    pos = world.data.xpos[world.torso_body_id].copy()
    rot = world.data.xmat[world.torso_body_id].reshape(3, 3).copy()
    return pos, rot


def ee_pose(side):
    """T_W_E(q, t) = T_W_T(t) · T_T_K · T_K_E(q).

    T_W_T is the torso pose in the world frame, as MuJoCo measures it; T_T_K is
    a fixed mount from model constants; T_K_E(q) is Pinocchio FK.
    No EE pose is read from MuJoCo.
    """
    T_W_T = transform_from_pose(*torso_pose())
    T_K_E = pin_T_K_E(pin_model, pin_data, ee_frame_id,
                      world.data.qpos[qpos_adrs[side]])
    return pose_from_transform(T_W_T @ _T_T_K[side] @ T_K_E)


def jacobian_world(side):
    """World-aligned EE frame Jacobian, 6x7: maps joint rates qdot (rad/s, 7)
    to the EE world twist, rows [vx vy vz (m/s); wx wy wz (rad/s)].

    Pinocchio computes it aligned with the arm base K; both blocks are
    rotated by R_W_K = R_W_T @ R_T_K. The base is kinematic (mocap), so
    the joint columns are the complete Jacobian."""
    q = np.asarray(world.data.qpos[qpos_adrs[side]], dtype=float)
    J_K = pin.computeFrameJacobian(pin_model, pin_data, q, ee_frame_id,
                                   pin.LOCAL_WORLD_ALIGNED)
    _, torso_rot = torso_pose()
    R_W_K = torso_rot @ _T_T_K[side][:3, :3]
    J = np.empty_like(J_K)
    J[:3] = R_W_K @ J_K[:3]
    J[3:] = R_W_K @ J_K[3:]
    return J


def ee_velocity(side, base_twist, J=None):
    """EE world twist: measured joint rates plus the given base twist.

        v_E = v_T + w_T x (p_E - p_T) + (J_world(q) qdot)_lin
        w_E = w_T + (J_world(q) qdot)_ang

    — the time derivative of ee_pose's T_W_E = T_W_T · T_T_K · T_K_E(q),
    with T_T_K constant. base_twist = (v_T (3,) m/s, w_T (3,) rad/s) is
    the torso world twist, which MuJoCo cannot provide (the mocap torso
    teleports, it never has velocity): scripted-motion derivative in sim
    (motion.torso_twist_at), Vicon estimate on hardware, zeros for a
    stationary base. Required on purpose — no default, so a forgotten
    base twist fails loudly instead of returning a silently wrong zero.
    Returns (v_E (3,) m/s, w_E (3,) rad/s), of the EE relative to the
    world, expressed in the world frame. J, if given, must be
    jacobian_world(side) at the current state — lets the control loop
    reuse the one it already computed."""
    v_T, w_T = (np.asarray(x, dtype=float) for x in base_twist)
    qdot = np.asarray(world.data.qvel[dof_adrs[side]], dtype=float)
    if J is None:
        J = jacobian_world(side)
    arm = J @ qdot
    p_E, _ = ee_pose(side)
    p_T, _ = torso_pose()
    return v_T + np.cross(w_T, p_E - p_T) + arm[:3], w_T + arm[3:]


def measured_ee_pose(side):
    """EE pose as MuJoCo measures it (the comparison side, not FK)."""
    pos = world.data.site_xpos[world.ee_site_id[side]].copy()
    rot = world.data.site_xmat[world.ee_site_id[side]].reshape(3, 3).copy()
    return pos, rot
