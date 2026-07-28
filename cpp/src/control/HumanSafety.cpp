#include "control/HumanSafety.h"

#include <cmath>
#include <stdexcept>

namespace srl::control {
namespace {

constexpr double kAxisEpsilon = 1e-12;

}  // namespace

void FiniteCylinderDistanceGradient(const Eigen::Vector3d& point_torso_m,
                                    const config::HumanSafetyConfig& config,
                                    double& distance,
                                    Eigen::Vector3d& gradient) {
  const Eigen::Vector2d radial_vector(
      point_torso_m(0) - config.center_xy_torso_m[0],
      point_torso_m(1) - config.center_xy_torso_m[1]);
  const double radial_distance = radial_vector.norm();

  Eigen::Vector3d radial_gradient(1.0, 0.0, 0.0);
  if (radial_distance > kAxisEpsilon) {
    radial_gradient(0) = radial_vector(0) / radial_distance;
    radial_gradient(1) = radial_vector(1) / radial_distance;
    radial_gradient(2) = 0.0;
  }

  const double radial_signed = radial_distance - config.radius_m;
  const double below = config.z_min_torso_m - point_torso_m(2);
  const double above = point_torso_m(2) - config.z_max_torso_m;
  const double vertical_signed = std::max(below, above);
  const Eigen::Vector3d vertical_gradient(0.0, 0.0, below >= above ? -1.0 : 1.0);

  // Exterior corner: both excesses positive -> hypotenuse and blended normal.
  if (radial_signed > 0.0 && vertical_signed > 0.0) {
    distance = std::hypot(radial_signed, vertical_signed);
    gradient = radial_gradient * (radial_signed / distance) +
               vertical_gradient * (vertical_signed / distance);
    return;
  }
  if (radial_signed >= vertical_signed) {
    distance = radial_signed;
    gradient = radial_gradient;
    return;
  }
  distance = vertical_signed;
  gradient = vertical_gradient;
}

ArmHumanSafetyState EvaluateArm(const Pose& torso_pose_world,
                                const ArmControllerState& arm_state,
                                const config::HumanSafetyConfig& config) {
  const auto& points = arm_state.link_safety_points;
  if (points.size() == 0) {
    throw std::invalid_argument("arm_state has no link safety geometry");
  }
  const Eigen::Matrix3d rotation_torso_world = torso_pose_world.rotation.transpose();

  ArmHumanSafetyState state;
  state.evaluated = true;
  HumanDistanceConstraints& constraints = state.constraints;
  const std::size_t count = points.size();
  constraints.signed_clearance_m.resize(count);
  constraints.distance_jacobian_m_rad.resize(count);
  constraints.active.resize(count);
  constraints.mount_exempt.resize(count);

  for (std::size_t index = 0; index < count; ++index) {
    const Eigen::Vector3d point_torso =
        rotation_torso_world *
        (points.position_world_m[index] - torso_pose_world.position_m);
    const Matrix3x7 jacobian_torso =
        rotation_torso_world * points.jacobian_world_m_rad[index];

    double centre_distance = 0.0;
    Eigen::Vector3d gradient_torso;
    FiniteCylinderDistanceGradient(point_torso, config, centre_distance,
                                   gradient_torso);

    const auto& sphere = kinematics::kLinkSpheres[index];
    const double signed_clearance =
        centre_distance - sphere.radius_m - config.clearance_m;
    constraints.signed_clearance_m[index] = signed_clearance;
    constraints.distance_jacobian_m_rad[index] =
        (gradient_torso.transpose() * jacobian_torso).transpose();
    constraints.mount_exempt[index] = sphere.mount_exempt ? 1 : 0;
    constraints.active[index] =
        (config.enabled && !sphere.mount_exempt &&
         signed_clearance <= config.activation_distance_m)
            ? 1
            : 0;
  }
  return state;
}

DualArmHumanSafetyStates EvaluateDualArm(
    const PlantState& plant, const DualArmControllerStates& states,
    const config::HumanSafetyConfig& config) {
  DualArmHumanSafetyStates result;
  result.right = EvaluateArm(plant.torso_pose_world, states.right, config);
  result.left = EvaluateArm(plant.torso_pose_world, states.left, config);
  return result;
}

}  // namespace srl::control
