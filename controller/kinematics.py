"""Analytical forward kinematics for a hinge-joint chain in a MuJoCo model.

Generic over any model: the chain is named by an explicit base body and
end-effector site. Scene-specific wiring (which arms exist, how they are
mounted) lives in controller/frames.py.

Usage:
    chain = extract_chain(model, "right_base_link", "right_pinch_site")
    T_K_E = fk(chain, data.qpos)
"""

from typing import NamedTuple

import mujoco
import numpy as np

from controller.transforms import (
    rotation_about_axis,
    rotation_from_quat,
    transform_from_pose,
)


class Chain(NamedTuple):
    """Constants of one kinematic chain, read once from MjModel."""

    steps: list        # per body: (T_fixed 4x4, axis 3, anchor 3, qpos_adr) — axis/anchor/adr are None for jointless bodies
    T_site: np.ndarray  # EE site offset on the last body, 4x4
    joint_ids: list    # MuJoCo joint ids, base to tip
    qpos_adrs: list    # global qpos address per chain joint


def extract_chain(model, base_body, ee_site):
    """Read the base_body -> ee_site chain constants from the model.

    Uses only MjModel constants (body offsets, joint axes/anchors, qpos
    addresses); fk() then needs nothing but the returned Chain and qpos.
    """
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
        if body_id == 0:  # world's parent is itself — without this the walk never ends
            raise ValueError(f"{ee_site!r} does not descend from {base_body!r}")
        body_ids.append(body_id)
        body_id = model.body_parentid[body_id]
    body_ids.reverse()

    # Per body: fixed parent offset, plus (axis, anchor, qpos adr) if jointed.
    steps = []
    joint_ids = []
    qpos_adrs = []
    for body_id in body_ids:
        T_fixed = transform_from_pose(
            model.body_pos[body_id].copy(),
            rotation_from_quat(model.body_quat[body_id]),
        )
        if model.body_jntnum[body_id] == 0:
            steps.append((T_fixed, None, None, None))
            continue
        jnt_id = model.body_jntadr[body_id]
        if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"joint {jnt_id} is not a hinge")
        joint_ids.append(jnt_id)
        qpos_adrs.append(int(model.jnt_qposadr[jnt_id]))
        steps.append((
            T_fixed,
            model.jnt_axis[jnt_id].copy(),
            model.jnt_pos[jnt_id].copy(),
            int(model.jnt_qposadr[jnt_id]),
        ))

    T_site = transform_from_pose(
        model.site_pos[site_id].copy(),
        rotation_from_quat(model.site_quat[site_id]),
    )
    return Chain(steps, T_site, joint_ids, qpos_adrs)


def fk(chain, qpos):
    """EE site pose in the base body frame, as a 4x4 transform."""
    T = np.eye(4)
    for T_fixed, axis, anchor, adr in chain.steps:
        T = T @ T_fixed
        if adr is None:
            continue
        rot = rotation_about_axis(axis, qpos[adr])
        T_joint = np.eye(4)
        T_joint[:3, :3] = rot
        T_joint[:3, 3] = anchor - rot @ anchor
        T = T @ T_joint
    return T @ chain.T_site
