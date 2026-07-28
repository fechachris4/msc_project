// Samples a configured trajectory on a fixed grid and prints pose, twist and
// acceleration at full precision, so cpp/tools/compare_trajectory.py can diff
// it against the Python implementation. Exercises contract clauses H1-H10.

#include <cstdio>
#include <exception>
#include <string>

#include "config/RuntimeConfig.h"
#include "trajectory/TrajectoryConfig.h"

namespace {

// Deterministic start pose, so the comparison does not depend on the sim.
srl::Pose StartPose() {
  srl::Pose pose;
  pose.position_m = Eigen::Vector3d(0.45, 0.30, 1.20);
  pose.rotation = Eigen::Matrix3d::Identity();
  return pose;
}

void PrintVector(const char* name, double t, const Eigen::Vector3d& value) {
  std::printf("%s,%.17g,%.17g,%.17g,%.17g\n", name, t, value(0), value(1),
              value(2));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: srl_trajectory_dump CONFIG.toml [SAMPLES]\n");
    return 2;
  }
  const int samples = argc > 2 ? std::stoi(argv[2]) : 400;

  try {
    const srl::config::ProjectConfig config = srl::config::LoadConfig(argv[1]);
    if (!config.left_target.trajectory) {
      std::fprintf(stderr, "config has no left trajectory\n");
      return 2;
    }
    const srl::trajectory::MaterializedTrajectory materialized =
        srl::trajectory::MaterializeTrajectory(*config.left_target.trajectory,
                                               StartPose());

    std::printf("duration,%.17g\n", materialized.duration_s);
    for (double boundary : materialized.boundary_times_s) {
      std::printf("boundary,%.17g\n", boundary);
    }
    std::printf("bounds,%.17g,%.17g,%.17g,%.17g\n",
                materialized.rate_bounds.max_linear_speed_m_s,
                materialized.rate_bounds.max_linear_acceleration_m_s2,
                materialized.rate_bounds.max_angular_speed_rad_s,
                materialized.rate_bounds.max_angular_acceleration_rad_s2);

    // Sample slightly past the end so the terminal clamp is covered too.
    const double horizon = materialized.duration_s * 1.05;
    for (int index = 0; index <= samples; ++index) {
      const double t = horizon * index / samples;
      const auto sample = materialized.source->SampleKinematics(t);
      PrintVector("pos", t, sample.target.pose.position_m);
      for (int row = 0; row < 3; ++row) {
        std::printf("rot,%.17g,%.17g,%.17g,%.17g\n", t,
                    sample.target.pose.rotation(row, 0),
                    sample.target.pose.rotation(row, 1),
                    sample.target.pose.rotation(row, 2));
      }
      PrintVector("vel", t, sample.target.twist.linear_m_s);
      PrintVector("omg", t, sample.target.twist.angular_rad_s);
      PrintVector("acc", t, sample.linear_acceleration_m_s2);
      PrintVector("aac", t, sample.angular_acceleration_rad_s2);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "trajectory dump failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
