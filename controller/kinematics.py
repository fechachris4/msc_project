import mujoco
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

class KinematicChain:
    """Analytical FK for one arm: {prefix}base_link -> {prefix}pinch_site.

    Built once from MjModel constants (body offsets, joint axes/anchors,
    qpos addresses). fk() reads nothing from MjData — only the passed qpos.
    """

    def __init__(self, model, prefix):
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, prefix + "pinch_site"
        )
        base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link"
        )
        if site_id < 0 or base_id < 0:
            raise ValueError(f"missing site or base_link for prefix {prefix!r}")

        # Walk parents from the site's body up to base_link (exclusive),
        # then reverse to get base-to-tip order.
        body_ids = []
        body_id = model.site_bodyid[site_id]
        while body_id != base_id:
            if body_id == 0:
                raise ValueError(
                    f"{prefix}pinch_site does not descend from {prefix}base_link"
                )
            body_ids.append(body_id)
            body_id = model.body_parentid[body_id]
        body_ids.reverse()

        # Per body: fixed parent offset, plus (axis, anchor, qpos adr) if jointed.
        self.joint_ids = []
        self.qpos_adrs = []
        self._steps = []
        for body_id in body_ids:
            T_fixed = transform_from_pose(
                model.body_pos[body_id].copy(),
                rotation_from_quat(model.body_quat[body_id]),
            )
            if model.body_jntnum[body_id] == 0:
                self._steps.append((T_fixed, None, None, None))
                continue
            if model.body_jntnum[body_id] != 1:
                raise ValueError(f"body {body_id} has multiple joints")
            jnt_id = model.body_jntadr[body_id]
            if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"joint {jnt_id} is not a hinge")
            self.joint_ids.append(jnt_id)
            self.qpos_adrs.append(int(model.jnt_qposadr[jnt_id]))
            self._steps.append((
                T_fixed,
                model.jnt_axis[jnt_id].copy(),
                model.jnt_pos[jnt_id].copy(),
                int(model.jnt_qposadr[jnt_id]),
            ))

        self._T_site = transform_from_pose(
            model.site_pos[site_id].copy(),
            rotation_from_quat(model.site_quat[site_id]),
        )

    def fk(self, qpos):
        """T_K_E(q): EE site pose in the base_link frame, as a 4x4 transform."""
        T = np.eye(4)
        for T_fixed, axis, anchor, adr in self._steps:
            T = T @ T_fixed
            if adr is None:
                continue
            rot = rotation_about_axis(axis, qpos[adr])
            T_joint = np.eye(4)
            T_joint[:3, :3] = rot
            T_joint[:3, 3] = anchor - rot @ anchor
            T = T @ T_joint
        return T @ self._T_site


right_chain = KinematicChain(world.model, "right_")
left_chain = KinematicChain(world.model, "left_")


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
