#include "kinematics/Frames.h"

#include "math/Transforms.h"

namespace srl::kinematics {
namespace {

// World pose/twist of the selected reference frame (frames._world_reference_frame).
std::pair<Pose, Twist> WorldReferenceFrame(const PlantState& plant, Side side,
                                           const MountCalibration& calibration,
                                           TargetFrame frame) {
  switch (frame) {
    case TargetFrame::World:
      return {Pose{}, Twist::Zero()};
    case TargetFrame::Torso:
      return {plant.torso_pose_world, plant.torso_twist_world};
    case TargetFrame::Base: {
      const Pose& mount = calibration.for_arm(side);
      const Pose pose_world = ComposePose(plant.torso_pose_world, mount);
      const Eigen::Vector3d offset_world =
          pose_world.position_m - plant.torso_pose_world.position_m;
      const Twist& torso = plant.torso_twist_world;
      Twist twist_world;
      twist_world.linear_m_s =
          torso.linear_m_s + torso.angular_rad_s.cross(offset_world);
      twist_world.angular_rad_s = torso.angular_rad_s;
      return {pose_world, twist_world};
    }
  }
  throw std::invalid_argument("unsupported target frame");
}

}  // namespace

Pose ComposePose(const Pose& parent_to_child, const Pose& child_to_object) {
  const Eigen::Matrix4d transform =
      transforms::TransformFromPose(parent_to_child.position_m,
                                    parent_to_child.rotation) *
      transforms::TransformFromPose(child_to_object.position_m,
                                    child_to_object.rotation);
  Pose pose;
  transforms::PoseFromTransform(transform, pose.position_m, pose.rotation);
  return pose;
}

ArmControllerState ArmControllerStateOf(PinModel& pin, const PlantState& plant,
                                        Side side,
                                        const MountCalibration& calibration) {
  const ArmJointState& joints = plant.arm(side);
  const Pose& mount = calibration.for_arm(side);

  const Eigen::Matrix4d transform_world_torso = transforms::TransformFromPose(
      plant.torso_pose_world.position_m, plant.torso_pose_world.rotation);
  const Eigen::Matrix4d transform_torso_base =
      transforms::TransformFromPose(mount.position_m, mount.rotation);
  const Eigen::Matrix4d transform_world_base =
      transform_world_torso * transform_torso_base;

  pin.Update(joints.position_rad);
  const Eigen::Matrix4d transform_base_ee = pin.EeTransform();

  ArmControllerState state;
  state.joints = joints;
  transforms::PoseFromTransform(transform_world_base * transform_base_ee,
                                state.ee_pose_world.position_m,
                                state.ee_pose_world.rotation);

  const Matrix6x7 jacobian_base = pin.EeJacobian();
  // Computed as R_W_T * R_T_B rather than reused from the 4x4 product, exactly
  // as the Python does -- the two agree mathematically but not bit-for-bit.
  const Eigen::Matrix3d rotation_world_base =
      plant.torso_pose_world.rotation * mount.rotation;
  state.jacobian_world.topRows<3>() =
      rotation_world_base * jacobian_base.topRows<3>();
  state.jacobian_world.bottomRows<3>() =
      rotation_world_base * jacobian_base.bottomRows<3>();

  const Vector6 arm_twist = state.jacobian_world * joints.velocity_rad_s;
  const Twist& torso_twist = plant.torso_twist_world;
  const Eigen::Vector3d offset_world =
      state.ee_pose_world.position_m - plant.torso_pose_world.position_m;
  state.ee_twist_world.linear_m_s = torso_twist.linear_m_s +
                                    torso_twist.angular_rad_s.cross(offset_world) +
                                    arm_twist.head<3>();
  state.ee_twist_world.angular_rad_s =
      torso_twist.angular_rad_s + arm_twist.tail<3>();

  pin.ResolveLinkSafetyPoints(transform_world_base.topRightCorner<3, 1>(),
                              rotation_world_base, state.link_safety_points);
  return state;
}

DualArmControllerStates ControllerStates(PinModel& pin, const PlantState& plant,
                                         const MountCalibration& calibration) {
  DualArmControllerStates states;
  states.right = ArmControllerStateOf(pin, plant, Side::Right, calibration);
  states.left = ArmControllerStateOf(pin, plant, Side::Left, calibration);
  return states;
}

Pose ExpressPoseInTargetFrame(const PlantState& plant, Side side,
                              const MountCalibration& calibration,
                              TargetFrame frame, const Pose& pose_world) {
  const auto [frame_pose_world, unused_twist] =
      WorldReferenceFrame(plant, side, calibration, frame);
  (void)unused_twist;
  const Eigen::Matrix3d rotation_frame_world = frame_pose_world.rotation.transpose();
  return Pose(
      rotation_frame_world *
          (pose_world.position_m - frame_pose_world.position_m),
      rotation_frame_world * pose_world.rotation);
}

WorldTarget ResolveTargetWorld(const PlantState& plant, Side side,
                               const MountCalibration& calibration,
                               const FramedTarget& target) {
  const auto [frame_pose, frame_twist] =
      WorldReferenceFrame(plant, side, calibration, target.reference_frame);

  WorldTarget resolved;
  resolved.pose_world = ComposePose(frame_pose, target.pose);
  const Eigen::Vector3d offset_world =
      frame_pose.rotation * target.pose.position_m;
  resolved.twist_world.linear_m_s =
      frame_twist.linear_m_s + frame_twist.angular_rad_s.cross(offset_world) +
      frame_pose.rotation * target.twist.linear_m_s;
  resolved.twist_world.angular_rad_s =
      frame_twist.angular_rad_s + frame_pose.rotation * target.twist.angular_rad_s;
  return resolved;
}

DualArmWorldTargets ResolveTargetsWorld(const PlantState& plant,
                                        const MountCalibration& calibration,
                                        const DualArmFramedTargets& targets) {
  DualArmWorldTargets resolved;
  resolved.right =
      ResolveTargetWorld(plant, Side::Right, calibration, targets.right);
  resolved.left =
      ResolveTargetWorld(plant, Side::Left, calibration, targets.left);
  return resolved;
}

}  // namespace srl::kinematics
