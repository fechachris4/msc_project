#include "sim/TargetMotion.h"

#include <stdexcept>

#include "math/Transforms.h"
#include "sim/Targets.h"

namespace srl::sim {

void TargetMotion::CaptureHome(const MujocoBackend& backend) {
  for (Side side : kSides) {
    home_position_.for_arm(side) = TargetPosition(backend, side);
    home_rotation_.for_arm(side) =
        transforms::RotationFromQuat(TargetQuat(backend, side));
  }
  captured_ = true;
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> TargetMotion::PoseAt(
    double t, Side side, const Eigen::Vector3d& linear_amplitude,
    double linear_frequency, const Eigen::Vector3d& rotational_amplitude,
    double rotational_frequency) const {
  if (!captured_) {
    throw std::runtime_error("TargetMotion::CaptureHome was never called");
  }
  const Eigen::Vector3d position =
      home_position_.for_arm(side) +
      transforms::SineOffset(t, linear_amplitude, linear_frequency);
  const Eigen::Matrix3d rotation =
      home_rotation_.for_arm(side) *
      transforms::RotationFromRpy(transforms::SineOffset(
          t, rotational_amplitude, rotational_frequency));
  return {position, rotation};
}

void TargetMotion::SetTargetPose(MujocoBackend& backend, double t, Side side,
                                 const Eigen::Vector3d& linear_amplitude,
                                 double linear_frequency,
                                 const Eigen::Vector3d& rotational_amplitude,
                                 double rotational_frequency) const {
  if (linear_amplitude.isZero(0.0) && rotational_amplitude.isZero(0.0)) return;
  const auto [position, rotation] =
      PoseAt(t, side, linear_amplitude, linear_frequency, rotational_amplitude,
             rotational_frequency);
  backend.SetTargetPose(side, position, QuatFromRotation(rotation));
}

}  // namespace srl::sim
