#pragma once

#include <string>

#include "planning/JointTrajectory.h"

namespace srl::planning {

// Writes planner knots in SI units with explicit column names.
void WriteJointTrajectoryCsv(const JointTrajectory& trajectory,
                             const std::string& path);

}  // namespace srl::planning
