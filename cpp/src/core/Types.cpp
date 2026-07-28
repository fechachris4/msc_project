#include "core/Types.h"

#include <cmath>
#include <limits>

namespace srl {

void Pose::Validate(const char* what) const {
  const Eigen::Matrix3d gram = rotation.transpose() * rotation;
  const bool orthonormal =
      ((gram - Eigen::Matrix3d::Identity()).array().abs() <= 1e-9).all();
  const bool right_handed = std::abs(rotation.determinant() - 1.0) <= 1e-9;
  if (!orthonormal || !right_handed) {
    throw std::invalid_argument(
        std::string(what) + " must be a proper orthonormal matrix");
  }
  if (!position_m.allFinite() || !rotation.allFinite()) {
    throw std::invalid_argument(std::string(what) + " must be finite");
  }
}

TargetFrame TargetFrameFromName(std::string_view name) {
  if (name == "world") return TargetFrame::World;
  if (name == "base") return TargetFrame::Base;
  if (name == "torso") return TargetFrame::Torso;
  throw std::invalid_argument("unsupported target frame: " +
                              std::string(name));
}

std::string_view TargetFrameName(TargetFrame frame) {
  switch (frame) {
    case TargetFrame::World:
      return "world";
    case TargetFrame::Base:
      return "base";
    case TargetFrame::Torso:
      return "torso";
  }
  return "world";
}

double ArmHumanSafetyState::minimum_clearance_m() const {
  if (!evaluated) return std::numeric_limits<double>::infinity();
  double smallest = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < constraints.size(); ++index) {
    if (constraints.mount_exempt[index]) continue;
    smallest = std::min(smallest, constraints.signed_clearance_m[index]);
  }
  return smallest;
}

}  // namespace srl
