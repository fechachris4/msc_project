"""Analytical forward kinematics for a hinge-joint chain in a MuJoCo model.

Generic over any model: the chain is named by an explicit base body and
end-effector site. Scene-specific wiring (which arms exist, how they are
mounted) lives in controller/frames.py.
"""

import mujoco
import numpy as np

from controller.transforms import (
    rotation_about_axis,
    rotation_from_quat,
    transform_from_pose,
)


class KinematicChain:
    """Analytical FK for one chain: base_body -> ee_site.

    Built once from MjModel constants (body offsets, joint axes/anchors,
    qpos addresses). fk() reads nothing from MjData — only the passed qpos.
    """

    def __init__(self, model, base_body, ee_site):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if site_id < 0:
            raise ValueError(f"no site named {ee_site!r}")
        if base_id < 0:
            raise ValueError(f"no body named {base_body!r}")

        # Walk parents from the site's body up to base_body (exclusive),
        # then reverse to get base-to-tip order.
        body_ids = []
        body_id = model.site_bodyid[site_id]
        while body_id != base_id:
            if body_id == 0:
                raise ValueError(f"{ee_site!r} does not descend from {base_body!r}")
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
        """EE site pose in the base_body frame, as a 4x4 transform."""
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
