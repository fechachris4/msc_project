#include "planning/JointTrajectoryCsv.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace srl::planning {

void WriteJointTrajectoryCsv(const JointTrajectory& trajectory,
                             const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open trajectory output: " + path);
  }
  output << "time_s";
  for (int joint = 1; joint <= kJoints; ++joint) {
    output << ",q" << joint << "_rad";
  }
  for (int joint = 1; joint <= kJoints; ++joint) {
    output << ",qdot" << joint << "_rad_s";
  }
  output << '\n' << std::setprecision(17);
  for (const JointTrajectoryPoint& point : trajectory.points()) {
    output << point.time_s;
    for (int joint = 0; joint < kJoints; ++joint) {
      output << ',' << point.position_rad(joint);
    }
    for (int joint = 0; joint < kJoints; ++joint) {
      output << ',' << point.velocity_rad_s(joint);
    }
    output << '\n';
  }
  if (!output) {
    throw std::runtime_error("failed while writing trajectory output: " +
                             path);
  }
}

}  // namespace srl::planning
