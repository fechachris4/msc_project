// Compile validated structured trajectory intent into pure target sources.
//
// The request-to-control seam. Structured data only: no natural-language
// interpretation, MuJoCo access, or marker state belongs here.
// One-to-one with controller/trajectory_config.py.

#pragma once

#include <memory>
#include <vector>

#include "config/RuntimeConfig.h"
#include "core/Types.h"
#include "trajectory/Trajectory.h"

namespace srl::trajectory {

// A complete one-arm trajectory ready for Runner composition.
struct MaterializedTrajectory {
  std::shared_ptr<TimedTargetSource> source;
  std::shared_ptr<TargetProgram> program;
  double duration_s{0.0};
  std::vector<double> boundary_times_s;
  TrajectoryRateBounds rate_bounds;
};

// Build and validate one structured trajectory from its start pose.
MaterializedTrajectory MaterializeTrajectory(
    const config::TargetTrajectoryConfig& config, const Pose& start_pose);

}  // namespace srl::trajectory
