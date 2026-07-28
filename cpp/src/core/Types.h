// Fixed-shape control data contracts.
//
// One-to-one with controller/state.py. The Python records validate on every
// construction; here validation lives at the boundaries that can actually
// produce bad data (config load, backend read, target sampling) so the
// control path stays allocation- and branch-free. See docs/03-cpp-design.md §4.

#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace srl {

inline constexpr int kJoints = 7;

using Vector7 = Eigen::Matrix<double, kJoints, 1>;
using Matrix6x7 = Eigen::Matrix<double, 6, kJoints>;
using Matrix3x7 = Eigen::Matrix<double, 3, kJoints>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

// ------------------------------------------------------------------- arms

enum class Side { Right, Left };

inline constexpr std::array<Side, 2> kSides{Side::Right, Side::Left};

inline constexpr std::string_view SideName(Side side) {
  return side == Side::Right ? "right" : "left";
}

inline Side SideFromName(std::string_view name) {
  if (name == "right") return Side::Right;
  if (name == "left") return Side::Left;
  throw std::invalid_argument("unknown arm: " + std::string(name));
}

// Fixed two-arm container; deliberately not a map. The Python records are
// equally fixed ("the fixed two-arm records are deliberate scope").
template <typename T>
struct DualArm {
  T right{};
  T left{};

  T& for_arm(Side side) { return side == Side::Right ? right : left; }
  const T& for_arm(Side side) const {
    return side == Side::Right ? right : left;
  }
};

// ------------------------------------------------------------------ poses

struct Pose {
  Eigen::Vector3d position_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};

  Pose() = default;
  Pose(Eigen::Vector3d position, Eigen::Matrix3d rotation_matrix)
      : position_m(std::move(position)), rotation(std::move(rotation_matrix)) {}

  // Mirrors Pose.__post_init__: proper orthonormal, atol 1e-9.
  void Validate(const char* what = "rotation") const;
};

struct Twist {
  Eigen::Vector3d linear_m_s{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_rad_s{Eigen::Vector3d::Zero()};

  static Twist Zero() { return Twist{}; }
};

struct ArmJointState {
  Vector7 position_rad{Vector7::Zero()};
  Vector7 velocity_rad_s{Vector7::Zero()};
};

struct PlantState {
  double sample_time_s{0.0};
  double nominal_dt_s{0.0};
  Pose torso_pose_world{};
  Twist torso_twist_world{};
  DualArm<ArmJointState> arms{};

  const ArmJointState& arm(Side side) const { return arms.for_arm(side); }
};

// ---------------------------------------------------------------- targets

enum class TargetFrame { World, Base, Torso };

TargetFrame TargetFrameFromName(std::string_view name);
std::string_view TargetFrameName(TargetFrame frame);

// Target pose/twist relative to and expressed in `reference_frame`.
struct FramedTarget {
  TargetFrame reference_frame{TargetFrame::World};
  Pose pose{};
  Twist twist{};
};

// Controller-facing Cartesian target; pose and twist are world-aligned.
struct WorldTarget {
  Pose pose_world{};
  Twist twist_world{};
};

using DualArmFramedTargets = DualArm<FramedTarget>;
using DualArmWorldTargets = DualArm<WorldTarget>;

// Fixed poses T_T_B for the right and left arm base frames.
using MountCalibration = DualArm<Pose>;

// ------------------------------------------------------------ arm safety

// Fixed arrays for all conservative arm spheres in one plant sample.
// Sized at runtime from the generated LINK_SPHERES table.
struct LinkSafetyPoints {
  std::vector<Eigen::Vector3d> position_world_m;
  std::vector<Matrix3x7> jacobian_world_m_rad;

  std::size_t size() const { return position_world_m.size(); }
};

// Vectorised distance and joint-rate sensitivity for all arm spheres.
// `mount_exempt` is carried here rather than looked up in the static sphere
// table so this record stays self-describing, exactly as the Python one is.
struct HumanDistanceConstraints {
  std::vector<double> signed_clearance_m;
  std::vector<Vector7> distance_jacobian_m_rad;
  std::vector<char> active;        // char, not bool: addressable storage
  std::vector<char> mount_exempt;

  std::size_t size() const { return signed_clearance_m.size(); }
};

struct ArmHumanSafetyState {
  bool evaluated{false};
  HumanDistanceConstraints constraints{};

  // min over non-mount-exempt spheres; +inf when unevaluated or empty.
  double minimum_clearance_m() const;
};

using DualArmHumanSafetyStates = DualArm<ArmHumanSafetyState>;

// ------------------------------------------------------- controller state

// World-aligned controller input derived from one explicit plant sample.
struct ArmControllerState {
  ArmJointState joints{};
  Pose ee_pose_world{};
  Twist ee_twist_world{};
  Matrix6x7 jacobian_world{Matrix6x7::Zero()};
  LinkSafetyPoints link_safety_points{};
};

using DualArmControllerStates = DualArm<ArmControllerState>;

// Backend-facing position command in radians for both seven-DOF arms.
struct JointPositionCommand {
  DualArm<Vector7> position_rad{Vector7::Zero(), Vector7::Zero()};

  const Vector7& for_arm(Side side) const {
    return position_rad.for_arm(side);
  }
};

}  // namespace srl
