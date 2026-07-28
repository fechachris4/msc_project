#include "planning/PlanValidation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "control/HumanSafety.h"
#include "kinematics/Frames.h"

namespace srl::planning {

PlanValidationReport ValidateForExecution(
    const JointTrajectory& trajectory, Side side,
    const PlantState& planning_state,
    const MountCalibration& calibration, kinematics::PinModel& pin,
    const control::PositionActuationLimits& limits,
    const config::HumanSafetyConfig& human_safety,
    double sample_period_s) {
  if (!std::isfinite(sample_period_s) || sample_period_s <= 0.0) {
    throw std::invalid_argument(
        "validation sample period must be finite and positive");
  }
  limits.Validate();

  PlanValidationReport report;
  report.minimum_joint_margin_rad =
      std::numeric_limits<double>::infinity();
  report.minimum_human_clearance_m =
      std::numeric_limits<double>::infinity();

  const std::size_t intervals = static_cast<std::size_t>(
      std::ceil(trajectory.duration_s() / sample_period_s));
  PlantState state = planning_state;
  for (std::size_t index = 0; index <= intervals; ++index) {
    const double fraction =
        intervals == 0
            ? 0.0
            : static_cast<double>(index) / static_cast<double>(intervals);
    const double time_s =
        trajectory.start_time_s() + fraction * trajectory.duration_s();
    const JointTrajectorySample sample = trajectory.Sample(time_s);
    state.arms.for_arm(side).position_rad = sample.position_rad;
    state.arms.for_arm(side).velocity_rad_s = sample.velocity_rad_s;

    const double lower_margin =
        (sample.position_rad - limits.lower_position_rad).minCoeff();
    const double upper_margin =
        (limits.upper_position_rad - sample.position_rad).minCoeff();
    report.minimum_joint_margin_rad =
        std::min(report.minimum_joint_margin_rad,
                 std::min(lower_margin, upper_margin));
    report.maximum_velocity_ratio =
        std::max(report.maximum_velocity_ratio,
                 (sample.velocity_rad_s.array().abs() /
                  limits.velocity_rad_s.array())
                     .maxCoeff());
    report.maximum_acceleration_rad_s2 =
        std::max(report.maximum_acceleration_rad_s2,
                 sample.acceleration_rad_s2.cwiseAbs().maxCoeff());

    const ArmControllerState arm_state =
        kinematics::ArmControllerStateOf(pin, state, side, calibration);
    const ArmHumanSafetyState safety =
        control::EvaluateArm(state.torso_pose_world, arm_state, human_safety);
    report.minimum_human_clearance_m =
        std::min(report.minimum_human_clearance_m,
                 safety.minimum_clearance_m());
    ++report.checked_samples;
  }

  constexpr double kTolerance = 1e-9;
  if (report.minimum_joint_margin_rad < -kTolerance) {
    report.reason = "joint_position_limit";
  } else if (report.maximum_velocity_ratio > 1.0 + kTolerance) {
    report.reason = "joint_velocity_limit";
  } else if (report.minimum_human_clearance_m < -kTolerance) {
    report.reason = "human_clearance";
  } else {
    report.valid = true;
    report.reason = "valid";
  }
  return report;
}

}  // namespace srl::planning
