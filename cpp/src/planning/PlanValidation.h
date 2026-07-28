// Independent post-validation for any generated joint trajectory.
//
// GPMP2 factor costs are soft penalties. This validator turns the execution
// contract into hard checks using the simulator's actual joint limits and
// Pinocchio-derived 18-sphere arm geometry.

#pragma once

#include <cstddef>
#include <string>

#include "config/RuntimeConfig.h"
#include "control/PositionActuation.h"
#include "core/Types.h"
#include "kinematics/PinModel.h"
#include "planning/JointTrajectory.h"

namespace srl::planning {

struct PlanValidationReport {
  bool valid{false};
  std::size_t checked_samples{0};
  double minimum_joint_margin_rad{0.0};
  double maximum_velocity_ratio{0.0};
  double maximum_acceleration_rad_s2{0.0};
  double minimum_human_clearance_m{0.0};
  std::string reason;
};

PlanValidationReport ValidateForExecution(
    const JointTrajectory& trajectory, Side side,
    const PlantState& planning_state,
    const MountCalibration& calibration, kinematics::PinModel& pin,
    const control::PositionActuationLimits& limits,
    const config::HumanSafetyConfig& human_safety,
    double sample_period_s);

}  // namespace srl::planning
