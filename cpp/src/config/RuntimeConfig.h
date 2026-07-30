// Strict loader for the Python/C++ shared control TOML.
//
// One-to-one with runtime_config.py, including its validation rules and error
// messages: exact key sets at every level, unknown keys rejected, cross-field
// constraints checked even for disabled features. Contract clauses I1-I5.

#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "core/Types.h"

namespace srl::config {

struct RunConfig {
  double nominal_dt_s{};
  std::string arm;
};

struct ReactivePoseConfig {
  double kp_position_s_inv{};
  double kp_rotation_s_inv{};
  double kd_position{};
  double kd_rotation{};
  double null_gain_s_inv{};
  double dls_damping{};
  bool position_enabled{};
  bool orientation_enabled{};
  bool velocity_enabled{};
};

struct LimitConfig {
  Vector7 joint_velocity_rad_s{Vector7::Zero()};
  double position_lead_rad{};
};

// Field names stay comparable with basic_control/src/app/Options.h, but this
// simulation interprets centre and height in WORLD coordinates.
struct CylinderKeepoutConfig {
  bool cylinder_keepout_enabled{};
  double cylinder_keepout_center_x_m{};
  double cylinder_keepout_center_y_m{};
  double cylinder_keepout_radius_m{};
  double cylinder_keepout_z_min_m{};
  double cylinder_keepout_z_max_m{};
  double cylinder_keepout_clearance_m{};
  double cylinder_waypoint_tolerance_m{};
};

struct HumanSafetyConfig {
  bool enabled{};
  std::array<double, 2> center_xy_torso_m{};
  double radius_m{};
  double z_min_torso_m{};
  double z_max_torso_m{};
  double clearance_m{};
  double control_margin_m{};
  double activation_distance_m{};
  double recovery_gain_s_inv{};
  double approach_velocity_damping{};
  int projection_iterations{};
  double constraint_tolerance_m_s{};
};

// Shared configuration for the native Cartesian planning stage. srl_sim
// validates every field, plans the selected arm(s) at startup, and gives each
// resulting WORLD-frame source to the unchanged reactive controller.
struct PlanningConfig {
  bool enabled{};
  std::string arm;
  int waypoint_count{};
  int dense_samples{};
  double clearance_margin_m{};
  double smoothness_weight{};
  double obstacle_weight{};
  int max_iterations{};
  double tool_radius_m{};
  double deviation_weight{};
  double reach_allowance_m{};
  bool lead_compensation_enabled{};
  double replan_clearance_trigger_m{};
  bool include_floor{};
  double floor_height_world_m{};
  bool include_torso_box{};
  Eigen::Vector3d torso_box_half_extent_m{Eigen::Vector3d::Zero()};
  double max_linear_speed_m_s{};
  double max_linear_acceleration_m_s2{};
  double max_angular_speed_rad_s{};
  double max_angular_acceleration_rad_s2{};
};

struct TrajectoryConstraintsConfig {
  std::optional<double> max_linear_speed_m_s;
  std::optional<double> max_linear_acceleration_m_s2;
  std::optional<double> max_angular_speed_rad_s;
  std::optional<double> max_angular_acceleration_rad_s2;
};

enum class SegmentType { Hold, Line, Waypoints, Circle };

struct TrajectorySegmentConfig {
  SegmentType type{SegmentType::Hold};
  std::optional<double> duration_s;
  std::optional<Eigen::Vector3d> displacement_m;
  std::optional<Eigen::Vector3d> end_position_m;
  std::vector<Eigen::Vector3d> offsets_m;
  std::vector<Eigen::Vector3d> positions_m;
  std::vector<double> durations_s;
  std::vector<Eigen::Vector3d> rpy_rad;
  std::optional<Eigen::Vector3d> end_rpy_rad;
  std::optional<double> radius_m;
  std::optional<Eigen::Vector3d> normal;
  std::optional<Eigen::Vector3d> start_direction;
  std::optional<int> revolutions;
  std::optional<bool> clockwise;

  bool has_offsets{false};
  bool has_positions{false};
  bool has_durations{false};
  bool has_rpy{false};
};

struct TargetTrajectoryConfig {
  std::string reference_frame;
  std::string start;
  bool loop{};
  bool open_live_path_plot{};
  TrajectoryConstraintsConfig constraints;
  std::vector<TrajectorySegmentConfig> segments;
};

struct TargetConfig {
  std::string reference_frame;
  Eigen::Vector3d position_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rpy_rad{Eigen::Vector3d::Zero()};
  std::optional<TargetTrajectoryConfig> trajectory;
};

struct LookAtObjectMotionConfig {
  std::string body_name;
  Eigen::Vector3d home_position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d linear_amplitude_m{Eigen::Vector3d::Zero()};
  double linear_frequency_hz{};
};

struct SimulationConfig {
  std::optional<Vector7> right_initial_joint_position_rad;
  std::optional<Vector7> left_initial_joint_position_rad;
  std::optional<LookAtObjectMotionConfig> look_at_object_motion;

  const std::optional<Vector7>& initial_joint_position(Side side) const {
    return side == Side::Right ? right_initial_joint_position_rad
                               : left_initial_joint_position_rad;
  }
};

struct ProjectConfig {
  RunConfig run;
  ReactivePoseConfig reactive_pose;
  LimitConfig limits;
  CylinderKeepoutConfig cylinder_keepout;
  HumanSafetyConfig human_safety;
  PlanningConfig planning;
  TargetConfig right_target;
  TargetConfig left_target;
  SimulationConfig simulation;
  std::string source_path;
  std::string source_sha256;

  const TargetConfig& target(Side side) const {
    return side == Side::Right ? right_target : left_target;
  }
};

// Load and validate. Throws std::invalid_argument with the same message text
// the Python raises, so a malformed config fails identically in both.
ProjectConfig LoadConfig(const std::string& path);

// Default location: <python root>/config/control.toml.
std::string DefaultConfigPath();

// Sorted-key, 2-space-indented JSON matching json.dumps(..., indent=2,
// sort_keys=True) on effective_config_dict(), including CPython float repr.
std::string EffectiveConfigJson(const ProjectConfig& config);

void PrintEffectiveConfig(const ProjectConfig& config);

// CPython's repr() for a double, used by the JSON writer above.
std::string PythonFloatRepr(double value);

}  // namespace srl::config
