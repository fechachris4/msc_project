#include "sim/Targets.h"

#include <array>

#include <mujoco/mujoco.h>

#include "math/Transforms.h"

namespace srl::sim {

Eigen::Vector4d QuatFromRotation(const Eigen::Matrix3d& rotation) {
  std::array<mjtNum, 9> matrix{};
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(matrix.data()) =
      rotation;
  std::array<mjtNum, 4> quaternion{};
  mju_mat2Quat(quaternion.data(), matrix.data());
  return Eigen::Vector4d(quaternion[0], quaternion[1], quaternion[2],
                         quaternion[3]);
}

void SetTarget(MujocoBackend& backend, Side side,
               const Eigen::Vector3d& position_m) {
  backend.SetTargetPose(side, position_m, backend.TargetQuaternion(side));
}

void SetTargetQuat(MujocoBackend& backend, Side side,
                   const Eigen::Vector4d& quaternion_wxyz) {
  backend.SetTargetPose(side, backend.TargetPosition(side), quaternion_wxyz);
}

Eigen::Vector3d TargetPosition(const MujocoBackend& backend, Side side) {
  return backend.TargetPosition(side);
}

Eigen::Vector4d TargetQuat(const MujocoBackend& backend, Side side) {
  return backend.TargetQuaternion(side);
}

WorldTarget WorldTargetOf(const MujocoBackend& backend, Side side) {
  WorldTarget target;
  target.pose_world.position_m = TargetPosition(backend, side);
  target.pose_world.rotation =
      transforms::RotationFromQuat(TargetQuat(backend, side));
  target.twist_world = Twist::Zero();
  return target;
}

DualArmFramedTargets FramedWorldTargets(const MujocoBackend& backend) {
  DualArmFramedTargets targets;
  for (Side side : kSides) {
    FramedTarget& target = targets.for_arm(side);
    target.reference_frame = TargetFrame::World;
    target.pose = WorldTargetOf(backend, side).pose_world;
    target.twist = Twist::Zero();
  }
  return targets;
}

}  // namespace srl::sim
