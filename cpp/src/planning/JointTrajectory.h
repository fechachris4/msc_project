// Time-parameterized seven-joint trajectory shared by planners and executors.
//
// Positions are radians, velocities are radians/second and time is seconds.
// Between planner samples, cubic Hermite interpolation preserves both the
// position and velocity at each endpoint.

#pragma once

#include <cstddef>
#include <vector>

#include "core/Types.h"

namespace srl::planning {

struct JointTrajectoryPoint {
  double time_s{0.0};
  Vector7 position_rad{Vector7::Zero()};
  Vector7 velocity_rad_s{Vector7::Zero()};
};

struct JointTrajectorySample {
  double requested_time_s{0.0};
  double trajectory_time_s{0.0};
  Vector7 position_rad{Vector7::Zero()};
  Vector7 velocity_rad_s{Vector7::Zero()};
  Vector7 acceleration_rad_s2{Vector7::Zero()};
  std::size_t interval_index{0};
  bool before_start{false};
  bool finished{false};
};

class JointTrajectory {
 public:
  explicit JointTrajectory(std::vector<JointTrajectoryPoint> points);

  const std::vector<JointTrajectoryPoint>& points() const { return points_; }
  double start_time_s() const { return points_.front().time_s; }
  double end_time_s() const { return points_.back().time_s; }
  double duration_s() const { return end_time_s() - start_time_s(); }

  JointTrajectorySample Sample(double time_s) const;

 private:
  JointTrajectorySample SampleInterval(std::size_t index, double time_s) const;

  std::vector<JointTrajectoryPoint> points_;
};

}  // namespace srl::planning
