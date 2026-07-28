// Simulation initialisation for configured target trajectories.
//
// Runs before the control loop: resets the model, writes the configured
// initial joint posture, resolves the trajectory start pose in its declared
// frame, and compiles the segment list into a program.
// One-to-one with sim/target_trajectory.py. Contract clause K4.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "config/RuntimeConfig.h"
#include "control/TargetSource.h"
#include "core/Types.h"
#include "kinematics/PinModel.h"
#include "sim/MujocoBackend.h"
#include "trajectory/TrajectoryConfig.h"

namespace srl::sim {

struct TargetTrajectorySetup {
  std::shared_ptr<control::DualArmTargetSource> source;
  std::shared_ptr<trajectory::TimedTargetSource> selected_source;
  Pose start_pose_reference;
  Pose end_pose_reference;
  Pose start_pose_world;
  Pose end_pose_world;
  double duration_s{0.0};
  std::vector<double> boundary_times_s;
  trajectory::TrajectoryRateBounds rate_bounds;
  Vector6 initial_singular_values{Vector6::Zero()};
  double initial_joint_margin_rad{0.0};
};

TargetTrajectorySetup PrepareTargetTrajectory(
    MujocoBackend& backend, kinematics::PinModel& pin,
    const MountCalibration& calibration, Side side,
    const config::TargetTrajectoryConfig& trajectory_config,
    const std::optional<Vector7>& initial_joint_position_rad,
    const DualArmFramedTargets& static_targets);

void PrintTargetTrajectorySetup(
    Side side, const config::TargetTrajectoryConfig& trajectory_config,
    const TargetTrajectorySetup& setup);

}  // namespace srl::sim
