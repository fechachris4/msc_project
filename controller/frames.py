"""Pure world-frame kinematics and target-frame conversion.

No function in this module reads MuJoCo state. Dynamic torso state and fixed
mount calibration are explicit inputs; arm FK/Jacobians come from Pinocchio.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin

from controller.pin_fk import build_pin_model, pin_T_K_E
from controller.state import (
    ArmControllerState,
    DualArmControllerStates,
    DualArmFramedTargets,
    DualArmWorldTargets,
    FramedTarget,
    MountCalibration,
    PlantState,
    Pose,
    TargetFrame,
    Twist,
    WorldTarget,
)
from controller.transforms import pose_from_transform, transform_from_pose


_GEN3_PATH = (
    Path(__file__).resolve().parents[1]
    / "sim"
    / "assets"
    / "kinova_gen3"
    / "gen3.xml"
)
pin_model, pin_data, ee_frame_id = build_pin_model(
    _GEN3_PATH, "base_link", "pinch_site"
)


def compose_pose(parent_to_child, child_to_object):
    transform = (
        transform_from_pose(
            parent_to_child.position_m, parent_to_child.rotation
        )
        @ transform_from_pose(
            child_to_object.position_m, child_to_object.rotation
        )
    )
    return Pose(*pose_from_transform(transform))


def arm_controller_state(plant, side, calibration):
    """Derive one arm's world pose, twist, and Jacobian from explicit state."""
    if not isinstance(plant, PlantState):
        raise TypeError("plant must be a PlantState")
    if not isinstance(calibration, MountCalibration):
        raise TypeError("calibration must be a MountCalibration")

    joints = plant.arm(side)
    mount = calibration.for_arm(side)
    T_W_T = transform_from_pose(
        plant.torso_pose_world.position_m,
        plant.torso_pose_world.rotation,
    )
    T_T_B = transform_from_pose(mount.position_m, mount.rotation)
    T_B_E = pin_T_K_E(
        pin_model, pin_data, ee_frame_id, joints.position_rad
    )
    ee_pose_world = Pose(*pose_from_transform(T_W_T @ T_T_B @ T_B_E))

    J_B = pin.computeFrameJacobian(
        pin_model,
        pin_data,
        joints.position_rad,
        ee_frame_id,
        pin.LOCAL_WORLD_ALIGNED,
    )
    R_W_B = plant.torso_pose_world.rotation @ mount.rotation
    jacobian_world = np.empty_like(J_B)
    jacobian_world[:3] = R_W_B @ J_B[:3]
    jacobian_world[3:] = R_W_B @ J_B[3:]

    arm_twist = jacobian_world @ joints.velocity_rad_s
    torso_twist = plant.torso_twist_world
    offset_world = (
        ee_pose_world.position_m - plant.torso_pose_world.position_m
    )
    ee_twist_world = Twist(
        torso_twist.linear_m_s
        + np.cross(torso_twist.angular_rad_s, offset_world)
        + arm_twist[:3],
        torso_twist.angular_rad_s + arm_twist[3:],
    )
    return ArmControllerState(
        joints=joints,
        ee_pose_world=ee_pose_world,
        ee_twist_world=ee_twist_world,
        jacobian_world=jacobian_world,
    )


def controller_states(plant, calibration):
    return DualArmControllerStates(
        right=arm_controller_state(plant, "right", calibration),
        left=arm_controller_state(plant, "left", calibration),
    )


def _world_reference_frame(plant, side, calibration, frame):
    """Return world pose/twist of the selected reference frame."""
    if frame == TargetFrame.WORLD:
        return Pose(np.zeros(3), np.eye(3)), Twist.zero()
    if frame == TargetFrame.TORSO:
        return plant.torso_pose_world, plant.torso_twist_world
    if frame == TargetFrame.BASE:
        mount = calibration.for_arm(side)
        pose_world = compose_pose(plant.torso_pose_world, mount)
        offset_world = (
            pose_world.position_m - plant.torso_pose_world.position_m
        )
        torso_twist = plant.torso_twist_world
        twist_world = Twist(
            torso_twist.linear_m_s
            + np.cross(torso_twist.angular_rad_s, offset_world),
            torso_twist.angular_rad_s,
        )
        return pose_world, twist_world
    raise ValueError(f"unsupported target frame: {frame!r}")


def resolve_target_world(plant, side, calibration, target):
    """Convert a world/base/torso target once at the frame boundary.

    Controllers receive only the returned ``WorldTarget`` and never see the
    frame selector.
    """
    if not isinstance(plant, PlantState):
        raise TypeError("plant must be a PlantState")
    if not isinstance(calibration, MountCalibration):
        raise TypeError("calibration must be a MountCalibration")
    if not isinstance(target, FramedTarget):
        raise TypeError("target must be a FramedTarget")

    frame_pose, frame_twist = _world_reference_frame(
        plant, side, calibration, target.reference_frame
    )
    pose_world = compose_pose(frame_pose, target.pose)
    offset_world = frame_pose.rotation @ target.pose.position_m
    twist_world = Twist(
        frame_twist.linear_m_s
        + np.cross(frame_twist.angular_rad_s, offset_world)
        + frame_pose.rotation @ target.twist.linear_m_s,
        frame_twist.angular_rad_s
        + frame_pose.rotation @ target.twist.angular_rad_s,
    )
    return WorldTarget(pose_world, twist_world)


def resolve_targets_world(plant, calibration, targets):
    if not isinstance(targets, DualArmFramedTargets):
        raise TypeError("targets must be DualArmFramedTargets")
    return DualArmWorldTargets(
        right=resolve_target_world(
            plant, "right", calibration, targets.right),
        left=resolve_target_world(
            plant, "left", calibration, targets.left),
    )
