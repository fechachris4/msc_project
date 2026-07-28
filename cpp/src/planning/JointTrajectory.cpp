#include "planning/JointTrajectory.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace srl::planning {
namespace {

bool Finite(const Vector7& value) { return value.array().isFinite().all(); }

}  // namespace

JointTrajectory::JointTrajectory(std::vector<JointTrajectoryPoint> points)
    : points_(std::move(points)) {
  if (points_.size() < 2) {
    throw std::invalid_argument("joint trajectory requires at least two points");
  }
  for (std::size_t index = 0; index < points_.size(); ++index) {
    const JointTrajectoryPoint& point = points_[index];
    if (!std::isfinite(point.time_s) || !Finite(point.position_rad) ||
        !Finite(point.velocity_rad_s)) {
      throw std::invalid_argument("joint trajectory values must be finite");
    }
    if (index > 0 && point.time_s <= points_[index - 1].time_s) {
      throw std::invalid_argument(
          "joint trajectory times must be strictly increasing");
    }
  }
}

JointTrajectorySample JointTrajectory::SampleInterval(std::size_t index,
                                                       double time_s) const {
  const JointTrajectoryPoint& start = points_[index];
  const JointTrajectoryPoint& end = points_[index + 1];
  const double duration = end.time_s - start.time_s;
  const double u = (time_s - start.time_s) / duration;
  const double u2 = u * u;
  const double u3 = u2 * u;

  const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
  const double h10 = u3 - 2.0 * u2 + u;
  const double h01 = -2.0 * u3 + 3.0 * u2;
  const double h11 = u3 - u2;

  JointTrajectorySample sample;
  sample.requested_time_s = time_s;
  sample.trajectory_time_s = time_s;
  sample.interval_index = index;
  sample.position_rad =
      h00 * start.position_rad + h10 * duration * start.velocity_rad_s +
      h01 * end.position_rad + h11 * duration * end.velocity_rad_s;

  sample.velocity_rad_s =
      ((6.0 * u2 - 6.0 * u) / duration) * start.position_rad +
      (3.0 * u2 - 4.0 * u + 1.0) * start.velocity_rad_s +
      ((-6.0 * u2 + 6.0 * u) / duration) * end.position_rad +
      (3.0 * u2 - 2.0 * u) * end.velocity_rad_s;

  sample.acceleration_rad_s2 =
      ((12.0 * u - 6.0) / (duration * duration)) * start.position_rad +
      ((6.0 * u - 4.0) / duration) * start.velocity_rad_s +
      ((-12.0 * u + 6.0) / (duration * duration)) * end.position_rad +
      ((6.0 * u - 2.0) / duration) * end.velocity_rad_s;
  return sample;
}

JointTrajectorySample JointTrajectory::Sample(double time_s) const {
  if (!std::isfinite(time_s)) {
    throw std::invalid_argument("joint trajectory sample time must be finite");
  }
  if (time_s <= start_time_s()) {
    JointTrajectorySample sample = SampleInterval(0, start_time_s());
    sample.requested_time_s = time_s;
    sample.before_start = time_s < start_time_s();
    return sample;
  }
  if (time_s >= end_time_s()) {
    JointTrajectorySample sample =
        SampleInterval(points_.size() - 2, end_time_s());
    sample.requested_time_s = time_s;
    sample.finished = time_s >= end_time_s();
    return sample;
  }

  const auto upper = std::upper_bound(
      points_.begin(), points_.end(), time_s,
      [](double requested, const JointTrajectoryPoint& point) {
        return requested < point.time_s;
      });
  const std::size_t index =
      static_cast<std::size_t>(std::distance(points_.begin(), upper) - 1);
  return SampleInterval(index, time_s);
}

}  // namespace srl::planning
