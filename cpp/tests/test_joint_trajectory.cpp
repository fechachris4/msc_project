#include "TestSupport.h"

#include <limits>
#include <vector>

#include "planning/JointTrajectory.h"

namespace {

using srl::Vector7;
using srl::planning::JointTrajectory;
using srl::planning::JointTrajectoryPoint;

Vector7 Constant(double value) { return Vector7::Constant(value); }

}  // namespace

int main() {
  {
    const JointTrajectory trajectory({
        JointTrajectoryPoint{0.0, Constant(0.0), Constant(0.0)},
        JointTrajectoryPoint{2.0, Constant(1.0), Constant(0.0)},
    });

    const auto start = trajectory.Sample(0.0);
    CHECK_MATRIX(start.position_rad, Constant(0.0), 0.0,
                 "start position is exact");
    CHECK_MATRIX(start.velocity_rad_s, Constant(0.0), 0.0,
                 "start velocity is exact");

    const auto middle = trajectory.Sample(1.0);
    CHECK_MATRIX(middle.position_rad, Constant(0.5), 1e-15,
                 "Hermite midpoint position");
    CHECK_MATRIX(middle.velocity_rad_s, Constant(0.75), 1e-15,
                 "Hermite midpoint velocity");
    CHECK_MATRIX(middle.acceleration_rad_s2, Constant(0.0), 1e-15,
                 "Hermite midpoint acceleration");

    const auto end = trajectory.Sample(3.0);
    CHECK_MATRIX(end.position_rad, Constant(1.0), 0.0,
                 "sampling after end clamps position");
    CHECK_MATRIX(end.velocity_rad_s, Constant(0.0), 0.0,
                 "sampling after end clamps velocity");
    CHECK_TRUE(end.finished, "sampling at or after end is finished");
    CHECK_CLOSE(end.trajectory_time_s, 2.0, 0.0,
                "clamped trajectory time is the final timestamp");
  }

  {
    const JointTrajectory trajectory({
        JointTrajectoryPoint{1.0, Constant(-0.2), Constant(0.1)},
        JointTrajectoryPoint{2.0, Constant(0.3), Constant(-0.1)},
        JointTrajectoryPoint{4.0, Constant(0.7), Constant(0.0)},
    });
    const auto knot = trajectory.Sample(2.0);
    CHECK_MATRIX(knot.position_rad, Constant(0.3), 1e-15,
                 "interior knot position is continuous");
    CHECK_MATRIX(knot.velocity_rad_s, Constant(-0.1), 1e-15,
                 "interior knot velocity is continuous");
    CHECK_TRUE(knot.interval_index == 1,
               "an interior knot starts the following interval");

    const auto before = trajectory.Sample(-1.0);
    CHECK_TRUE(before.before_start, "sampling before start is marked");
    CHECK_MATRIX(before.position_rad, Constant(-0.2), 0.0,
                 "sampling before start clamps position");
  }

  CHECK_THROWS(
      JointTrajectory(std::vector<JointTrajectoryPoint>{
          JointTrajectoryPoint{0.0, Constant(0.0), Constant(0.0)}}),
      "a trajectory needs two points");
  CHECK_THROWS(
      JointTrajectory({
          JointTrajectoryPoint{0.0, Constant(0.0), Constant(0.0)},
          JointTrajectoryPoint{0.0, Constant(1.0), Constant(0.0)},
      }),
      "timestamps must increase");
  Vector7 bad = Constant(0.0);
  bad(3) = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(
      JointTrajectory({
          JointTrajectoryPoint{0.0, Constant(0.0), Constant(0.0)},
          JointTrajectoryPoint{1.0, bad, Constant(0.0)},
      }),
      "non-finite joint data is rejected");

  return srl::test::Finish("joint trajectory");
}
