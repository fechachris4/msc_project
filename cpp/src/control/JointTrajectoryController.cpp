#include "control/JointTrajectoryController.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace srl::control {

void JointTrackingConfig::Validate() const {
  if (!kp_s_inv.array().isFinite().all() || (kp_s_inv.array() < 0.0).any()) {
    throw std::invalid_argument(
        "joint tracking gains must be finite and non-negative");
  }
  if (!std::isfinite(start_tolerance_rad) || start_tolerance_rad < 0.0) {
    throw std::invalid_argument(
        "start_tolerance_rad must be finite and non-negative");
  }
  if (!std::isfinite(max_tracking_error_rad) ||
      max_tracking_error_rad <= 0.0) {
    throw std::invalid_argument(
        "max_tracking_error_rad must be finite and positive");
  }
  if (start_tolerance_rad > max_tracking_error_rad) {
    throw std::invalid_argument(
        "start_tolerance_rad must not exceed max_tracking_error_rad");
  }
}

JointTrajectoryController::JointTrajectoryController(
    JointTrackingConfig config)
    : config_(std::move(config)) {
  config_.Validate();
}

JointTrackingOutput JointTrajectoryController::Compute(
    const planning::JointTrajectorySample& reference,
    const ArmJointState& measured) const {
  if (!measured.position_rad.array().isFinite().all() ||
      !measured.velocity_rad_s.array().isFinite().all()) {
    throw std::invalid_argument("measured joint state must be finite");
  }

  JointTrackingOutput output;
  output.reference = reference;
  output.position_error_rad =
      reference.position_rad - measured.position_rad;
  if (output.position_error_rad.cwiseAbs().maxCoeff() >
      config_.max_tracking_error_rad) {
    output.stopped = true;
    output.reason = "tracking_error_limit";
    return output;
  }

  output.feedback_velocity_rad_s =
      (config_.kp_s_inv.array() * output.position_error_rad.array()).matrix();
  output.requested_velocity_rad_s =
      reference.velocity_rad_s + output.feedback_velocity_rad_s;
  return output;
}

}  // namespace srl::control
