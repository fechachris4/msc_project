"""Pinocchio-backed forward kinematics for one Gen3 arm.

Loads the arm-only MJCF (sim/assets/kinova_gen3/gen3.xml), where base_link
is fixed at the origin. The torso mocap body and world-frame composition
are MuJoCo-side concerns (controller/frames.py); Pinocchio never sees them.

Usage:
    arm = build_pin_model("sim/assets/kinova_gen3/gen3.xml",
                          "base_link", "pinch_site")
    T_K_E = pin_T_K_E(arm.model, arm.data, arm.frame_id, q)
"""

from typing import NamedTuple

import numpy as np
import pinocchio as pin


class PinArm(NamedTuple):
    model: "pin.Model"
    data: "pin.Data"
    frame_id: int


def build_pin_model(mjcf_path, base_body_name, ee_site_name):
    """Load an arm MJCF into Pinocchio and locate the EE site frame.

    Fails loudly if the base body or EE site does not map to a Pinocchio
    frame; never substitutes a nearby body frame for the site.
    """
    model = pin.buildModelFromMJCF(str(mjcf_path))

    if not model.existFrame(base_body_name):
        raise ValueError(
            f"Pinocchio model from {mjcf_path!r} has no frame named "
            f"{base_body_name!r}; frames: {[f.name for f in model.frames]}"
        )
    base_frame = model.frames[model.getFrameId(base_body_name)]
    if not (base_frame.placement == pin.SE3.Identity()
            and model.frames[base_frame.parentFrame].name == "universe"):
        raise ValueError(
            f"{base_body_name!r} is not fixed at the origin of the Pinocchio "
            "model; pin_T_K_E assumes oMf is expressed in the base frame."
        )

    if not model.existFrame(ee_site_name):
        raise ValueError(
            f"MJCF site {ee_site_name!r} did not map to a Pinocchio frame "
            f"in {mjcf_path!r}; frames: {[f.name for f in model.frames]}"
        )
    frame_id = model.getFrameId(ee_site_name)
    if model.frames[frame_id].type != pin.FrameType.OP_FRAME:
        raise ValueError(
            f"frame {ee_site_name!r} exists but is type "
            f"{model.frames[frame_id].type}, not OP_FRAME — refusing to "
            "silently use a body frame in place of the MJCF site."
        )

    return PinArm(model, pin.Data(model), frame_id)


def pin_T_K_E(model, data, frame_id, q):
    """EE site pose in the arm base_link frame, as a 4x4 transform.

    q: 7-vector of joint angles, base to tip. Pinocchio's MJCF parser
    keeps MuJoCo's joint order, axes, and angle signs, so q is used as-is.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (model.nq,):
        raise ValueError(f"expected q of shape ({model.nq},), got {q.shape}")
    pin.framesForwardKinematics(model, data, q)
    return data.oMf[frame_id].homogeneous.copy()
