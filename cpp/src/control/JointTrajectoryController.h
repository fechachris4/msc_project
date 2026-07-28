// Joint-space tracking law used to execute an optimized trajectory.
//
// qdot_request = qdot_reference + Kp * (q_reference - q_measured)
//
// This remains separate from planning, safety projection and command
// integration so each boundary can be tested independently.

#pragma once

#include <string>

#include "core/Types.h"
#include "planning/JointTrajectory.h"

namespace srl::control {

struct JointTrackingConfig {
  Vector7 kp_s_inv{Vector7::Constant(4.0)};
  double start_tolerance_rad{0.05};
  double max_tracking_error_rad{0.35};

  void Validate() const;
};

struct JointTrackingOutput {
  planning::JointTrajectorySample reference;
  Vector7 position_error_rad{Vector7::Zero()};
  Vector7 feedback_velocity_rad_s{Vector7::Zero()};
  Vector7 requested_velocity_rad_s{Vector7::Zero()};
  bool stopped{false};
  std::string reason{"tracking"};
};

class JointTrajectoryController {
 public:
  explicit JointTrajectoryController(JointTrackingConfig config);

  const JointTrackingConfig& config() const { return config_; }

  JointTrackingOutput Compute(
      const planning::JointTrajectorySample& reference,
      const ArmJointState& measured) const;

 private:
  JointTrackingConfig config_;
};

}  // namespace srl::control
