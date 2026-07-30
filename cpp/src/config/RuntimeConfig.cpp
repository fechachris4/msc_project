#include "config/RuntimeConfig.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>

#include <toml++/toml.hpp>

namespace srl::config {
namespace {

// ------------------------------------------------------------------ sha256
// Small self-contained SHA-256 so provenance stamping needs no crypto
// dependency. Hashes the raw file bytes, exactly like hashlib.sha256(raw).
class Sha256 {
 public:
  void Update(const unsigned char* data, std::size_t length) {
    for (std::size_t index = 0; index < length; ++index) {
      buffer_[buffer_length_++] = data[index];
      if (buffer_length_ == 64) {
        Transform(buffer_.data());
        bit_length_ += 512;
        buffer_length_ = 0;
      }
    }
  }

  std::string HexDigest() {
    std::uint64_t total_bits = bit_length_ + buffer_length_ * 8ULL;
    std::size_t index = buffer_length_;
    buffer_[index++] = 0x80;
    if (index > 56) {
      while (index < 64) buffer_[index++] = 0x00;
      Transform(buffer_.data());
      index = 0;
    }
    while (index < 56) buffer_[index++] = 0x00;
    for (int shift = 7; shift >= 0; --shift) {
      buffer_[index++] = static_cast<unsigned char>(total_bits >> (shift * 8));
    }
    Transform(buffer_.data());

    std::string hex;
    hex.reserve(64);
    char octet[3];
    for (std::uint32_t word : state_) {
      for (int shift = 3; shift >= 0; --shift) {
        std::snprintf(octet, sizeof(octet), "%02x",
                      static_cast<unsigned>((word >> (shift * 8)) & 0xFF));
        hex.append(octet);
      }
    }
    return hex;
  }

 private:
  static std::uint32_t RotateRight(std::uint32_t value, std::uint32_t count) {
    return (value >> count) | (value << (32 - count));
  }

  void Transform(const unsigned char* chunk) {
    static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    std::array<std::uint32_t, 64> schedule{};
    for (int index = 0; index < 16; ++index) {
      schedule[index] = (static_cast<std::uint32_t>(chunk[index * 4]) << 24) |
                        (static_cast<std::uint32_t>(chunk[index * 4 + 1]) << 16) |
                        (static_cast<std::uint32_t>(chunk[index * 4 + 2]) << 8) |
                        static_cast<std::uint32_t>(chunk[index * 4 + 3]);
    }
    for (int index = 16; index < 64; ++index) {
      const std::uint32_t s0 = RotateRight(schedule[index - 15], 7) ^
                               RotateRight(schedule[index - 15], 18) ^
                               (schedule[index - 15] >> 3);
      const std::uint32_t s1 = RotateRight(schedule[index - 2], 17) ^
                               RotateRight(schedule[index - 2], 19) ^
                               (schedule[index - 2] >> 10);
      schedule[index] =
          schedule[index - 16] + s0 + schedule[index - 7] + s1;
    }

    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int index = 0; index < 64; ++index) {
      const std::uint32_t s1 =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temp1 =
          h + s1 + choice + kRoundConstants[index] + schedule[index];
      const std::uint32_t s0 =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + majority;
      h = g; g = f; f = e; e = d + temp1;
      d = c; c = b; b = a; a = temp1 + temp2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                      0xa54ff53a, 0x510e527f, 0x9b05688c,
                                      0x1f83d9ab, 0x5be0cd19};
  std::array<unsigned char, 64> buffer_{};
  std::size_t buffer_length_{0};
  std::uint64_t bit_length_{0};
};

// ------------------------------------------------------------- validation

[[noreturn]] void Fail(const std::string& message) {
  throw std::invalid_argument(message);
}

std::string FormatList(const std::vector<std::string>& values) {
  std::string text = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) text += ", ";
    text += "'" + values[index] + "'";
  }
  return text + "]";
}

const toml::table& RequireTable(const toml::node& node,
                                const std::string& location) {
  if (!node.is_table()) Fail(location + " must be a TOML table");
  return *node.as_table();
}

std::set<std::string> KeysOf(const toml::table& table) {
  std::set<std::string> keys;
  for (const auto& [key, _] : table) keys.insert(std::string(key.str()));
  return keys;
}

void RequireExactKeys(const toml::table& table,
                      const std::set<std::string>& expected,
                      const std::string& location) {
  const std::set<std::string> actual = KeysOf(table);
  if (actual == expected) return;
  std::vector<std::string> missing, extra;
  std::set_difference(expected.begin(), expected.end(), actual.begin(),
                      actual.end(), std::back_inserter(missing));
  std::set_difference(actual.begin(), actual.end(), expected.begin(),
                      expected.end(), std::back_inserter(extra));
  Fail(location + " keys differ; missing=" + FormatList(missing) +
       ", extra=" + FormatList(extra));
}

struct NumberOptions {
  bool positive{false};
  bool nonnegative{false};
};

double FiniteNumber(const toml::node& node, const std::string& location,
                    NumberOptions options = {}) {
  // Python rejects bool explicitly before the numeric check.
  if (node.is_boolean() || !(node.is_floating_point() || node.is_integer())) {
    Fail(location + " must be a number");
  }
  const double value = node.is_integer()
                           ? static_cast<double>(node.as_integer()->get())
                           : node.as_floating_point()->get();
  if (!std::isfinite(value)) Fail(location + " must be finite");
  if (options.positive && value <= 0.0) {
    Fail(location + " must be greater than zero");
  }
  if (options.nonnegative && value < 0.0) Fail(location + " must be non-negative");
  return value;
}

bool Boolean(const toml::node& node, const std::string& location) {
  if (!node.is_boolean()) Fail(location + " must be true or false");
  return node.as_boolean()->get();
}

std::string Choice(const toml::node& node,
                   const std::vector<std::string>& choices,
                   const std::string& location) {
  if (!node.is_string()) Fail(location + " must be one of " + FormatList(choices));
  const std::string value = node.as_string()->get();
  if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
    Fail(location + " must be one of " + FormatList(choices));
  }
  return value;
}

std::string NonEmptyString(const toml::node& node,
                           const std::string& location) {
  if (!node.is_string()) Fail(location + " must be a non-empty string");
  const std::string value = node.as_string()->get();
  if (value.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
    Fail(location + " must be a non-empty string");
  }
  return value;
}

std::vector<double> Vector(const toml::node& node, std::size_t size,
                           const std::string& location, bool positive = false) {
  if (!node.is_array() || node.as_array()->size() != size) {
    Fail(location + " must contain exactly " + std::to_string(size) +
         " numbers");
  }
  std::vector<double> values;
  values.reserve(size);
  std::size_t index = 0;
  for (const auto& item : *node.as_array()) {
    values.push_back(FiniteNumber(item, location + "[" +
                                            std::to_string(index) + "]",
                                  {positive, false}));
    ++index;
  }
  return values;
}

Eigen::Vector3d Vector3(const toml::node& node, const std::string& location) {
  const std::vector<double> values = Vector(node, 3, location);
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

std::vector<Eigen::Vector3d> Vectors3(const toml::node& node,
                                      const std::string& location,
                                      std::size_t minimum_length) {
  if (!node.is_array() || node.as_array()->size() < minimum_length) {
    Fail(location + " must contain at least " +
         std::to_string(minimum_length) + " vectors");
  }
  std::vector<Eigen::Vector3d> values;
  std::size_t index = 0;
  for (const auto& item : *node.as_array()) {
    values.push_back(Vector3(item, location + "[" + std::to_string(index) + "]"));
    ++index;
  }
  return values;
}

int PositiveInteger(const toml::node& node, const std::string& location) {
  if (node.is_boolean() || !node.is_integer() ||
      node.as_integer()->get() <= 0) {
    Fail(location + " must be a positive integer");
  }
  return static_cast<int>(node.as_integer()->get());
}

std::optional<double> OptionalPositive(const toml::table& table,
                                       const std::string& key,
                                       const std::string& location) {
  const auto* node = table.get(key);
  if (node == nullptr) return std::nullopt;
  return FiniteNumber(*node, location + "." + key, {true, false});
}

const toml::node& Require(const toml::table& table, const std::string& key,
                          const std::string& location) {
  const auto* node = table.get(key);
  if (node == nullptr) Fail(location + "." + key + " is required");
  return *node;
}

// --------------------------------------------------------------- key sets

const std::set<std::string> kRootKeys = {
    "run", "controller", "limits", "cylinder_keepout",
    "human_safety", "planning", "targets", "simulation"};
const std::set<std::string> kRunKeys = {"nominal_dt_s", "arm"};
const std::set<std::string> kControllerKeys = {"reactive_pose"};
const std::set<std::string> kReactiveKeys = {
    "kp_position_s_inv", "kp_rotation_s_inv", "kd_position", "kd_rotation",
    "null_gain_s_inv", "dls_damping", "position_enabled",
    "orientation_enabled", "velocity_enabled"};
const std::set<std::string> kLimitKeys = {"joint_velocity_rad_s",
                                          "position_lead_rad"};
const std::set<std::string> kCylinderKeys = {
    "cylinder_keepout_enabled", "cylinder_keepout_center_x_m",
    "cylinder_keepout_center_y_m", "cylinder_keepout_radius_m",
    "cylinder_keepout_z_min_m", "cylinder_keepout_z_max_m",
    "cylinder_keepout_clearance_m", "cylinder_waypoint_tolerance_m"};
const std::set<std::string> kHumanSafetyKeys = {
    "enabled", "center_xy_torso_m", "radius_m", "z_min_torso_m",
    "z_max_torso_m", "clearance_m", "control_margin_m",
    "activation_distance_m", "recovery_gain_s_inv",
    "approach_velocity_damping", "projection_iterations",
    "constraint_tolerance_m_s"};
const std::set<std::string> kPlanningKeys = {
    "enabled",
    "arm",
    "waypoint_count",
    "dense_samples",
    "clearance_margin_m",
    "smoothness_weight",
    "obstacle_weight",
    "max_iterations",
    "tool_radius_m",
    "deviation_weight",
    "reach_allowance_m",
    "lead_compensation_enabled",
    "replan_clearance_trigger_m",
    "include_floor",
    "floor_height_world_m",
    "include_torso_box",
    "torso_box_half_extent_m",
    "max_linear_speed_m_s",
    "max_linear_acceleration_m_s2",
    "max_angular_speed_rad_s",
    "max_angular_acceleration_rad_s2"};
const std::set<std::string> kTargetKeys = {"reference_frame", "position_m",
                                           "rpy_rad"};
const std::set<std::string> kTrajectoryRequiredKeys = {
    "reference_frame", "start", "loop", "open_live_path_plot", "segments"};
const std::set<std::string> kTrajectoryConstraintKeys = {
    "max_linear_speed_m_s", "max_linear_acceleration_m_s2",
    "max_angular_speed_rad_s", "max_angular_acceleration_rad_s2"};
const std::set<std::string> kSimulationRequiredKeys = {
    "initial_joint_position_rad"};
const std::set<std::string> kSimulationAllowedKeys = {
    "initial_joint_position_rad", "look_at_object_motion"};
const std::set<std::string> kLookAtObjectMotionKeys = {
    "body_name", "home_position_world_m", "linear_amplitude_m",
    "linear_frequency_hz"};

const std::vector<std::string> kTargetFrames = {"world", "base", "torso"};
const std::vector<std::string> kTrajectoryStarts = {"measured",
                                                    "configured_target"};

const std::map<std::string, std::pair<Eigen::Vector3d, Eigen::Vector3d>>&
CirclePlaneDirections() {
  static const std::map<std::string, std::pair<Eigen::Vector3d, Eigen::Vector3d>>
      table = {
          {"xy", {{0, 0, 1}, {1, 0, 0}}},
          {"horizontal", {{0, 0, 1}, {1, 0, 0}}},
          {"xz", {{0, -1, 0}, {1, 0, 0}}},
          {"vertical_xz", {{0, -1, 0}, {1, 0, 0}}},
          {"yz", {{1, 0, 0}, {0, 1, 0}}},
          {"vertical_yz", {{1, 0, 0}, {0, 1, 0}}},
      };
  return table;
}

void CheckMissingExtra(const std::set<std::string>& actual,
                       const std::set<std::string>& required,
                       const std::set<std::string>& allowed,
                       const std::string& location) {
  std::vector<std::string> missing, extra;
  std::set_difference(required.begin(), required.end(), actual.begin(),
                      actual.end(), std::back_inserter(missing));
  std::set_difference(actual.begin(), actual.end(), allowed.begin(),
                      allowed.end(), std::back_inserter(extra));
  if (!missing.empty() || !extra.empty()) {
    Fail(location + " keys differ; missing=" + FormatList(missing) +
         ", extra=" + FormatList(extra));
  }
}

// ------------------------------------------------------------- parse steps

ReactivePoseConfig ParseReactivePose(const toml::node& node) {
  const std::string location = "controller.reactive_pose";
  const toml::table& table = RequireTable(node, location);
  RequireExactKeys(table, kReactiveKeys, location);
  ReactivePoseConfig config;
  const NumberOptions nonneg{false, true};
  config.kp_position_s_inv = FiniteNumber(
      *table.get("kp_position_s_inv"), location + ".kp_position_s_inv", nonneg);
  config.kp_rotation_s_inv = FiniteNumber(
      *table.get("kp_rotation_s_inv"), location + ".kp_rotation_s_inv", nonneg);
  config.kd_position = FiniteNumber(*table.get("kd_position"),
                                    location + ".kd_position", nonneg);
  config.kd_rotation = FiniteNumber(*table.get("kd_rotation"),
                                    location + ".kd_rotation", nonneg);
  config.null_gain_s_inv = FiniteNumber(
      *table.get("null_gain_s_inv"), location + ".null_gain_s_inv", nonneg);
  config.dls_damping = FiniteNumber(*table.get("dls_damping"),
                                    location + ".dls_damping", {true, false});
  config.position_enabled =
      Boolean(*table.get("position_enabled"), location + ".position_enabled");
  config.orientation_enabled = Boolean(*table.get("orientation_enabled"),
                                       location + ".orientation_enabled");
  config.velocity_enabled =
      Boolean(*table.get("velocity_enabled"), location + ".velocity_enabled");

  if (config.position_enabled && config.kp_position_s_inv <= 0.0) {
    Fail("controller.reactive_pose.kp_position_s_inv must be greater than "
         "zero when position control is enabled");
  }
  if (config.orientation_enabled && config.kp_rotation_s_inv <= 0.0) {
    Fail("controller.reactive_pose.kp_rotation_s_inv must be greater than "
         "zero when orientation control is enabled");
  }
  if (config.velocity_enabled &&
      (config.kd_position >= 1.0 || config.kd_rotation >= 1.0)) {
    Fail("controller.reactive_pose Kd gains must be less than one when "
         "velocity feedback is enabled");
  }
  return config;
}

CylinderKeepoutConfig ParseCylinderKeepout(const toml::node& node) {
  // Bounds are checked whether or not the keep-out is enabled, exactly as the
  // hardware controller does, so a disabled-but-wrong config still fails loud.
  const std::string location = "cylinder_keepout";
  const toml::table& table = RequireTable(node, location);
  RequireExactKeys(table, kCylinderKeys, location);
  CylinderKeepoutConfig config;
  config.cylinder_keepout_enabled =
      Boolean(*table.get("cylinder_keepout_enabled"),
              location + ".cylinder_keepout_enabled");
  config.cylinder_keepout_center_x_m =
      FiniteNumber(*table.get("cylinder_keepout_center_x_m"),
                   location + ".cylinder_keepout_center_x_m");
  config.cylinder_keepout_center_y_m =
      FiniteNumber(*table.get("cylinder_keepout_center_y_m"),
                   location + ".cylinder_keepout_center_y_m");
  config.cylinder_keepout_radius_m =
      FiniteNumber(*table.get("cylinder_keepout_radius_m"),
                   location + ".cylinder_keepout_radius_m", {true, false});
  config.cylinder_keepout_z_min_m =
      FiniteNumber(*table.get("cylinder_keepout_z_min_m"),
                   location + ".cylinder_keepout_z_min_m");
  config.cylinder_keepout_z_max_m =
      FiniteNumber(*table.get("cylinder_keepout_z_max_m"),
                   location + ".cylinder_keepout_z_max_m");
  config.cylinder_keepout_clearance_m =
      FiniteNumber(*table.get("cylinder_keepout_clearance_m"),
                   location + ".cylinder_keepout_clearance_m", {false, true});
  config.cylinder_waypoint_tolerance_m =
      FiniteNumber(*table.get("cylinder_waypoint_tolerance_m"),
                   location + ".cylinder_waypoint_tolerance_m", {true, false});
  if (config.cylinder_keepout_z_max_m <= config.cylinder_keepout_z_min_m) {
    Fail(location +
         ".cylinder_keepout_z_max_m must be greater than "
         "cylinder_keepout_z_min_m");
  }
  return config;
}

HumanSafetyConfig ParseHumanSafety(const toml::node& node) {
  const std::string location = "human_safety";
  const toml::table& table = RequireTable(node, location);
  RequireExactKeys(table, kHumanSafetyKeys, location);
  HumanSafetyConfig config;
  config.enabled = Boolean(*table.get("enabled"), location + ".enabled");
  const std::vector<double> center =
      Vector(*table.get("center_xy_torso_m"), 2, location + ".center_xy_torso_m");
  config.center_xy_torso_m = {center[0], center[1]};
  config.radius_m =
      FiniteNumber(*table.get("radius_m"), location + ".radius_m", {true, false});
  config.z_min_torso_m =
      FiniteNumber(*table.get("z_min_torso_m"), location + ".z_min_torso_m");
  config.z_max_torso_m =
      FiniteNumber(*table.get("z_max_torso_m"), location + ".z_max_torso_m");
  config.clearance_m = FiniteNumber(*table.get("clearance_m"),
                                    location + ".clearance_m", {false, true});
  config.control_margin_m = FiniteNumber(
      *table.get("control_margin_m"), location + ".control_margin_m", {false, true});
  config.activation_distance_m =
      FiniteNumber(*table.get("activation_distance_m"),
                   location + ".activation_distance_m", {true, false});
  config.recovery_gain_s_inv =
      FiniteNumber(*table.get("recovery_gain_s_inv"),
                   location + ".recovery_gain_s_inv", {true, false});
  config.approach_velocity_damping =
      FiniteNumber(*table.get("approach_velocity_damping"),
                   location + ".approach_velocity_damping", {false, true});
  config.projection_iterations = PositiveInteger(
      *table.get("projection_iterations"), location + ".projection_iterations");
  config.constraint_tolerance_m_s =
      FiniteNumber(*table.get("constraint_tolerance_m_s"),
                   location + ".constraint_tolerance_m_s", {true, false});
  if (config.z_max_torso_m <= config.z_min_torso_m) {
    Fail(location + ".z_max_torso_m must be greater than z_min_torso_m");
  }
  return config;
}

PlanningConfig ParsePlanning(const toml::node& node) {
  const std::string location = "planning";
  const toml::table& table = RequireTable(node, location);
  RequireExactKeys(table, kPlanningKeys, location);

  PlanningConfig config;
  config.enabled = Boolean(*table.get("enabled"), location + ".enabled");
  config.arm =
      Choice(*table.get("arm"), {"right", "left", "both"}, location + ".arm");
  config.waypoint_count =
      PositiveInteger(*table.get("waypoint_count"),
                      location + ".waypoint_count");
  config.dense_samples =
      PositiveInteger(*table.get("dense_samples"),
                      location + ".dense_samples");
  config.clearance_margin_m =
      FiniteNumber(*table.get("clearance_margin_m"),
                   location + ".clearance_margin_m", {false, true});
  config.smoothness_weight =
      FiniteNumber(*table.get("smoothness_weight"),
                   location + ".smoothness_weight", {true, false});
  config.obstacle_weight =
      FiniteNumber(*table.get("obstacle_weight"),
                   location + ".obstacle_weight", {true, false});
  config.max_iterations =
      PositiveInteger(*table.get("max_iterations"),
                      location + ".max_iterations");
  config.tool_radius_m =
      FiniteNumber(*table.get("tool_radius_m"),
                   location + ".tool_radius_m", {false, true});
  config.deviation_weight =
      FiniteNumber(*table.get("deviation_weight"),
                   location + ".deviation_weight", {true, false});
  config.reach_allowance_m =
      FiniteNumber(*table.get("reach_allowance_m"),
                   location + ".reach_allowance_m", {true, false});
  config.lead_compensation_enabled =
      Boolean(*table.get("lead_compensation_enabled"),
              location + ".lead_compensation_enabled");
  config.replan_clearance_trigger_m =
      FiniteNumber(*table.get("replan_clearance_trigger_m"),
                   location + ".replan_clearance_trigger_m", {false, true});
  config.include_floor =
      Boolean(*table.get("include_floor"), location + ".include_floor");
  config.floor_height_world_m =
      FiniteNumber(*table.get("floor_height_world_m"),
                   location + ".floor_height_world_m");
  config.include_torso_box =
      Boolean(*table.get("include_torso_box"),
              location + ".include_torso_box");
  config.torso_box_half_extent_m =
      Vector3(*table.get("torso_box_half_extent_m"),
              location + ".torso_box_half_extent_m");
  for (int index = 0; index < 3; ++index) {
    if (config.torso_box_half_extent_m(index) <= 0.0) {
      Fail(location + ".torso_box_half_extent_m[" +
           std::to_string(index) + "] must be greater than zero");
    }
  }
  config.max_linear_speed_m_s =
      FiniteNumber(*table.get("max_linear_speed_m_s"),
                   location + ".max_linear_speed_m_s", {true, false});
  config.max_linear_acceleration_m_s2 =
      FiniteNumber(*table.get("max_linear_acceleration_m_s2"),
                   location + ".max_linear_acceleration_m_s2", {true, false});
  config.max_angular_speed_rad_s =
      FiniteNumber(*table.get("max_angular_speed_rad_s"),
                   location + ".max_angular_speed_rad_s", {true, false});
  config.max_angular_acceleration_rad_s2 =
      FiniteNumber(*table.get("max_angular_acceleration_rad_s2"),
                   location + ".max_angular_acceleration_rad_s2",
                   {true, false});
  return config;
}

TrajectorySegmentConfig ParseTrajectorySegment(const toml::node& node,
                                               const std::string& trajectory,
                                               std::size_t index) {
  const std::string location =
      trajectory + ".segments[" + std::to_string(index) + "]";
  const toml::table& table = RequireTable(node, location);
  if (table.get("type") == nullptr) Fail(location + ".type is required");
  const std::string kind =
      Choice(*table.get("type"),
             {"hold", "line", "waypoints", "circle"}, location + ".type");
  const std::set<std::string> actual = KeysOf(table);
  TrajectorySegmentConfig segment;

  if (kind == "hold") {
    RequireExactKeys(table, {"type", "duration_s"}, location);
    segment.type = SegmentType::Hold;
    segment.duration_s = FiniteNumber(*table.get("duration_s"),
                                      location + ".duration_s", {true, false});
    return segment;
  }

  if (kind == "line") {
    const std::set<std::string> allowed = {
        "type", "duration_s", "displacement_m", "end_position_m", "end_rpy_rad"};
    std::vector<std::string> extra;
    std::set_difference(actual.begin(), actual.end(), allowed.begin(),
                        allowed.end(), std::back_inserter(extra));
    const int position_keys = (table.get("displacement_m") != nullptr) +
                              (table.get("end_position_m") != nullptr);
    if (!extra.empty() || position_keys != 1) {
      Fail(location +
           " line requires exactly one of displacement_m/end_position_m and "
           "no unknown keys; extra=" + FormatList(extra));
    }
    segment.type = SegmentType::Line;
    if (table.get("displacement_m") != nullptr) {
      segment.displacement_m =
          Vector3(*table.get("displacement_m"), location + ".displacement_m");
      if (segment.displacement_m->isZero(0.0) &&
          table.get("end_rpy_rad") == nullptr) {
        Fail(location + " zero displacement is a hold, not a line");
      }
    }
    segment.duration_s = OptionalPositive(table, "duration_s", location);
    if (table.get("end_position_m") != nullptr) {
      segment.end_position_m =
          Vector3(*table.get("end_position_m"), location + ".end_position_m");
    }
    if (table.get("end_rpy_rad") != nullptr) {
      segment.end_rpy_rad =
          Vector3(*table.get("end_rpy_rad"), location + ".end_rpy_rad");
    }
    return segment;
  }

  if (kind == "waypoints") {
    const std::set<std::string> allowed = {"type", "durations_s", "offsets_m",
                                           "positions_m", "rpy_rad"};
    std::vector<std::string> extra;
    std::set_difference(actual.begin(), actual.end(), allowed.begin(),
                        allowed.end(), std::back_inserter(extra));
    const int position_keys = (table.get("offsets_m") != nullptr) +
                              (table.get("positions_m") != nullptr);
    if (!extra.empty() || position_keys != 1) {
      Fail(location +
           " waypoints requires exactly one of offsets_m/positions_m and no "
           "unknown keys; extra=" + FormatList(extra));
    }
    segment.type = SegmentType::Waypoints;
    const bool use_offsets = table.get("offsets_m") != nullptr;
    const std::string points_key = use_offsets ? "offsets_m" : "positions_m";
    std::vector<Eigen::Vector3d> points =
        Vectors3(*table.get(points_key), location + "." + points_key, 2);
    segment.has_offsets = use_offsets;
    segment.has_positions = !use_offsets;
    (use_offsets ? segment.offsets_m : segment.positions_m) = points;

    if (table.get("durations_s") != nullptr) {
      const auto* array = table.get("durations_s")->as_array();
      if (array == nullptr) Fail(location + ".durations_s must be an array");
      std::size_t duration_index = 0;
      for (const auto& item : *array) {
        segment.durations_s.push_back(FiniteNumber(
            item,
            location + ".durations_s[" + std::to_string(duration_index) + "]",
            {true, false}));
        ++duration_index;
      }
      segment.has_durations = true;
      if (segment.durations_s.size() != points.size() - 1) {
        Fail(location + ".durations_s must have one entry per leg");
      }
    }
    if (table.get("rpy_rad") != nullptr) {
      segment.rpy_rad = Vectors3(*table.get("rpy_rad"), location + ".rpy_rad", 2);
      segment.has_rpy = true;
      if (segment.rpy_rad.size() != points.size()) {
        Fail(location + ".rpy_rad must match the waypoint count");
      }
    }
    return segment;
  }

  // circle
  const std::set<std::string> allowed = {
      "type", "duration_s", "radius_m", "plane", "normal",
      "start_direction", "revolutions", "clockwise", "end_rpy_rad"};
  const std::set<std::string> required = {"type", "radius_m", "revolutions",
                                          "clockwise"};
  CheckMissingExtra(actual, required, allowed, location + " circle");

  const bool has_plane = table.get("plane") != nullptr;
  const bool has_normal = table.get("normal") != nullptr;
  const bool has_start_direction = table.get("start_direction") != nullptr;
  if (has_plane && (has_normal || has_start_direction)) {
    Fail(location +
         " circle requires either plane or normal/start_direction, not both");
  }
  if (!has_plane && !(has_normal && has_start_direction)) {
    Fail(location + " circle requires plane or both normal and start_direction");
  }
  segment.type = SegmentType::Circle;
  if (has_plane) {
    std::vector<std::string> plane_names;
    for (const auto& [name, _] : CirclePlaneDirections()) {
      plane_names.push_back(name);
    }
    const std::string plane =
        Choice(*table.get("plane"), plane_names, location + ".plane");
    segment.normal = CirclePlaneDirections().at(plane).first;
    segment.start_direction = CirclePlaneDirections().at(plane).second;
  } else {
    segment.normal = Vector3(*table.get("normal"), location + ".normal");
    segment.start_direction =
        Vector3(*table.get("start_direction"), location + ".start_direction");
  }
  segment.duration_s = OptionalPositive(table, "duration_s", location);
  segment.radius_m =
      FiniteNumber(*table.get("radius_m"), location + ".radius_m", {true, false});
  segment.revolutions =
      PositiveInteger(*table.get("revolutions"), location + ".revolutions");
  segment.clockwise = Boolean(*table.get("clockwise"), location + ".clockwise");
  if (table.get("end_rpy_rad") != nullptr) {
    segment.end_rpy_rad =
        Vector3(*table.get("end_rpy_rad"), location + ".end_rpy_rad");
  }
  return segment;
}

TargetTrajectoryConfig ParseTargetTrajectory(const toml::node& node,
                                             const std::string& side) {
  const std::string location = "targets." + side + ".trajectory";
  const toml::table& table = RequireTable(node, location);
  if (table.get("shape") != nullptr) {
    Fail(location +
         " legacy 'shape' trajectories are not supported by the C++ port; "
         "use the segment form");
  }
  std::set<std::string> allowed = kTrajectoryRequiredKeys;
  allowed.insert("constraints");
  CheckMissingExtra(KeysOf(table), kTrajectoryRequiredKeys, allowed, location);

  TargetTrajectoryConfig config;
  TrajectoryConstraintsConfig constraints;
  if (table.get("constraints") != nullptr) {
    const toml::table& constraint_table =
        RequireTable(*table.get("constraints"), location + ".constraints");
    std::vector<std::string> extra;
    const std::set<std::string> actual = KeysOf(constraint_table);
    std::set_difference(actual.begin(), actual.end(),
                        kTrajectoryConstraintKeys.begin(),
                        kTrajectoryConstraintKeys.end(),
                        std::back_inserter(extra));
    if (!extra.empty()) {
      Fail(location + ".constraints has unknown keys: " + FormatList(extra));
    }
    const std::string where = location + ".constraints";
    constraints.max_linear_speed_m_s =
        OptionalPositive(constraint_table, "max_linear_speed_m_s", where);
    constraints.max_linear_acceleration_m_s2 = OptionalPositive(
        constraint_table, "max_linear_acceleration_m_s2", where);
    constraints.max_angular_speed_rad_s =
        OptionalPositive(constraint_table, "max_angular_speed_rad_s", where);
    constraints.max_angular_acceleration_rad_s2 = OptionalPositive(
        constraint_table, "max_angular_acceleration_rad_s2", where);
  }
  config.constraints = constraints;

  const auto* segments = table.get("segments");
  if (segments == nullptr || !segments->is_array() ||
      segments->as_array()->empty()) {
    Fail(location + ".segments must be a non-empty array");
  }
  std::size_t index = 0;
  for (const auto& item : *segments->as_array()) {
    config.segments.push_back(ParseTrajectorySegment(item, location, index));
    ++index;
  }
  config.reference_frame = Choice(*table.get("reference_frame"), kTargetFrames,
                                  location + ".reference_frame");
  config.start =
      Choice(*table.get("start"), kTrajectoryStarts, location + ".start");
  config.loop = Boolean(*table.get("loop"), location + ".loop");
  config.open_live_path_plot = Boolean(*table.get("open_live_path_plot"),
                                       location + ".open_live_path_plot");
  return config;
}

TargetConfig ParseTarget(const toml::node& node, const std::string& side) {
  const std::string location = "targets." + side;
  const toml::table& table = RequireTable(node, location);
  std::set<std::string> allowed = kTargetKeys;
  allowed.insert("trajectory");
  CheckMissingExtra(KeysOf(table), kTargetKeys, allowed, location);

  TargetConfig config;
  const auto* frame = table.get("reference_frame");
  if (frame == nullptr || !frame->is_string() ||
      std::find(kTargetFrames.begin(), kTargetFrames.end(),
                frame->as_string()->get()) == kTargetFrames.end()) {
    Fail(location + ".reference_frame must be one of " +
         FormatList(kTargetFrames));
  }
  config.reference_frame = frame->as_string()->get();
  config.position_m = Vector3(*table.get("position_m"), location + ".position_m");
  config.rpy_rad = Vector3(*table.get("rpy_rad"), location + ".rpy_rad");
  if (table.get("trajectory") != nullptr) {
    config.trajectory = ParseTargetTrajectory(*table.get("trajectory"), side);
  }
  return config;
}

SimulationConfig ParseSimulation(const toml::node& node) {
  const toml::table& table = RequireTable(node, "simulation");
  CheckMissingExtra(KeysOf(table), kSimulationRequiredKeys,
                    kSimulationAllowedKeys, "simulation");
  const toml::table& positions =
      RequireTable(Require(table, "initial_joint_position_rad", "simulation"),
                   "simulation.initial_joint_position_rad");
  std::vector<std::string> extra;
  const std::set<std::string> arms = {"right", "left"};
  const std::set<std::string> actual = KeysOf(positions);
  std::set_difference(actual.begin(), actual.end(), arms.begin(), arms.end(),
                      std::back_inserter(extra));
  if (!extra.empty()) {
    Fail("simulation.initial_joint_position_rad has unknown arms: " +
         FormatList(extra));
  }
  SimulationConfig config;
  for (Side side : kSides) {
    const std::string name(SideName(side));
    const auto* node_for_side = positions.get(name);
    if (node_for_side == nullptr) continue;
    const std::vector<double> values = Vector(
        *node_for_side, 7, "simulation.initial_joint_position_rad." + name);
    Vector7 joints;
    for (int index = 0; index < kJoints; ++index) joints(index) = values[index];
    (side == Side::Right ? config.right_initial_joint_position_rad
                         : config.left_initial_joint_position_rad) = joints;
  }
  if (table.get("look_at_object_motion") != nullptr) {
    const std::string location = "simulation.look_at_object_motion";
    const toml::table& motion =
        RequireTable(*table.get("look_at_object_motion"), location);
    RequireExactKeys(motion, kLookAtObjectMotionKeys, location);
    LookAtObjectMotionConfig parsed;
    parsed.body_name =
        NonEmptyString(*motion.get("body_name"), location + ".body_name");
    parsed.home_position_world_m =
        Vector3(*motion.get("home_position_world_m"),
                location + ".home_position_world_m");
    parsed.linear_amplitude_m =
        Vector3(*motion.get("linear_amplitude_m"),
                location + ".linear_amplitude_m");
    if (parsed.linear_amplitude_m.norm() < 1e-12) {
      Fail(location + ".linear_amplitude_m must be non-zero");
    }
    parsed.linear_frequency_hz =
        FiniteNumber(*motion.get("linear_frequency_hz"),
                     location + ".linear_frequency_hz", {true, false});
    config.look_at_object_motion = std::move(parsed);
  }
  return config;
}

}  // namespace

// ---------------------------------------------------------- python floats

std::string PythonFloatRepr(double value) {
  if (std::isnan(value)) return "NaN";
  if (std::isinf(value)) return value > 0 ? "Infinity" : "-Infinity";

  // Shortest round-trip digits, then CPython's repr formatting rules:
  // scientific iff decimal exponent <= -4 or > 16, and ".0" appended when the
  // result would otherwise look like an integer.
  std::array<char, 64> buffer{};
  const auto result =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                    std::chars_format::scientific);
  std::string scientific(buffer.data(), result.ptr);

  const std::size_t exponent_position = scientific.find('e');
  std::string mantissa = scientific.substr(0, exponent_position);
  const int exponent = std::stoi(scientific.substr(exponent_position + 1));

  bool negative = false;
  if (!mantissa.empty() && mantissa.front() == '-') {
    negative = true;
    mantissa.erase(mantissa.begin());
  }
  std::string digits;
  for (char character : mantissa) {
    if (character != '.') digits.push_back(character);
  }
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();

  const int decimal_point = exponent + 1;  // digits before the decimal point
  std::string text;
  if (decimal_point <= -4 || decimal_point > 17) {
    text = digits.substr(0, 1);
    if (digits.size() > 1) text += "." + digits.substr(1);
    const int printed_exponent = decimal_point - 1;
    char exponent_text[16];
    std::snprintf(exponent_text, sizeof(exponent_text), "e%c%02d",
                  printed_exponent < 0 ? '-' : '+', std::abs(printed_exponent));
    text += exponent_text;
  } else if (decimal_point <= 0) {
    text = "0." + std::string(static_cast<std::size_t>(-decimal_point), '0') +
           digits;
  } else if (static_cast<std::size_t>(decimal_point) >= digits.size()) {
    text = digits +
           std::string(static_cast<std::size_t>(decimal_point) - digits.size(),
                       '0') +
           ".0";
  } else {
    text = digits.substr(0, static_cast<std::size_t>(decimal_point)) + "." +
           digits.substr(static_cast<std::size_t>(decimal_point));
  }
  return negative ? "-" + text : text;
}

// ---------------------------------------------------------------- loading

std::string DefaultConfigPath() {
  return std::string(SRL_PYTHON_ROOT) + "/config/control.toml";
}

ProjectConfig LoadConfig(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) Fail("cannot read config file: " + path);
  const std::string raw((std::istreambuf_iterator<char>(stream)),
                        std::istreambuf_iterator<char>());

  toml::table parsed;
  try {
    parsed = toml::parse(raw);
  } catch (const toml::parse_error& error) {
    Fail(std::string("invalid TOML: ") + error.description().data());
  }
  RequireExactKeys(parsed, kRootKeys, "root");

  const toml::table& run = RequireTable(*parsed.get("run"), "run");
  RequireExactKeys(run, kRunKeys, "run");
  const toml::table& controller =
      RequireTable(*parsed.get("controller"), "controller");
  RequireExactKeys(controller, kControllerKeys, "controller");
  const toml::table& limits = RequireTable(*parsed.get("limits"), "limits");
  RequireExactKeys(limits, kLimitKeys, "limits");
  const toml::table& targets = RequireTable(*parsed.get("targets"), "targets");
  RequireExactKeys(targets, {"right", "left"}, "targets");

  ProjectConfig config;
  config.run.nominal_dt_s = FiniteNumber(*run.get("nominal_dt_s"),
                                         "run.nominal_dt_s", {true, false});
  config.run.arm = Choice(*run.get("arm"), {"right", "left", "both"}, "run.arm");
  config.reactive_pose = ParseReactivePose(*controller.get("reactive_pose"));

  const std::vector<double> velocity = Vector(
      *limits.get("joint_velocity_rad_s"), 7, "limits.joint_velocity_rad_s",
      /*positive=*/true);
  for (int index = 0; index < kJoints; ++index) {
    config.limits.joint_velocity_rad_s(index) = velocity[index];
  }
  config.limits.position_lead_rad = FiniteNumber(
      *limits.get("position_lead_rad"), "limits.position_lead_rad", {true, false});

  config.cylinder_keepout = ParseCylinderKeepout(*parsed.get("cylinder_keepout"));
  config.human_safety = ParseHumanSafety(*parsed.get("human_safety"));
  config.planning = ParsePlanning(*parsed.get("planning"));
  config.right_target = ParseTarget(*targets.get("right"), "right");
  config.left_target = ParseTarget(*targets.get("left"), "left");
  config.simulation = ParseSimulation(*parsed.get("simulation"));

  config.source_path = path;
  Sha256 hash;
  hash.Update(reinterpret_cast<const unsigned char*>(raw.data()), raw.size());
  config.source_sha256 = hash.HexDigest();
  return config;
}

// ------------------------------------------------------------------- json

namespace {

std::string Indent(int level) { return std::string(level * 2, ' '); }

std::string JsonBool(bool value) { return value ? "true" : "false"; }

std::string JsonVector(const std::vector<double>& values, int level) {
  if (values.empty()) return "[]";
  std::string text = "[\n";
  for (std::size_t index = 0; index < values.size(); ++index) {
    text += Indent(level + 1) + PythonFloatRepr(values[index]);
    if (index + 1 < values.size()) text += ",";
    text += "\n";
  }
  return text + Indent(level) + "]";
}

std::vector<double> ToVector(const Eigen::Vector3d& value) {
  return {value(0), value(1), value(2)};
}

std::string JsonSegment(const TrajectorySegmentConfig& segment, int level);

std::string JsonTrajectory(const TargetTrajectoryConfig& trajectory, int level) {
  // Keys emitted in sorted order to match json.dumps(sort_keys=True).
  std::string text = "{\n";
  const auto field = [&](const std::string& key, const std::string& value,
                         bool last) {
    text += Indent(level + 1) + "\"" + key + "\": " + value +
            (last ? "\n" : ",\n");
  };
  std::string constraints = "{\n";
  const auto optional_field = [&](const std::string& key,
                                  const std::optional<double>& value,
                                  bool last) {
    constraints += Indent(level + 2) + "\"" + key + "\": " +
                   (value ? PythonFloatRepr(*value) : "null") +
                   (last ? "\n" : ",\n");
  };
  optional_field("max_angular_acceleration_rad_s2",
                 trajectory.constraints.max_angular_acceleration_rad_s2, false);
  optional_field("max_angular_speed_rad_s",
                 trajectory.constraints.max_angular_speed_rad_s, false);
  optional_field("max_linear_acceleration_m_s2",
                 trajectory.constraints.max_linear_acceleration_m_s2, false);
  optional_field("max_linear_speed_m_s",
                 trajectory.constraints.max_linear_speed_m_s, true);
  constraints += Indent(level + 1) + "}";

  field("constraints", constraints, false);
  field("loop", JsonBool(trajectory.loop), false);
  field("open_live_path_plot", JsonBool(trajectory.open_live_path_plot), false);
  field("reference_frame", "\"" + trajectory.reference_frame + "\"", false);

  std::string segments = "[\n";
  for (std::size_t index = 0; index < trajectory.segments.size(); ++index) {
    segments += Indent(level + 2) + JsonSegment(trajectory.segments[index],
                                                level + 2);
    if (index + 1 < trajectory.segments.size()) segments += ",";
    segments += "\n";
  }
  segments += Indent(level + 1) + "]";
  field("segments", segments, false);
  field("start", "\"" + trajectory.start + "\"", true);
  return text + Indent(level) + "}";
}

std::string SegmentTypeName(SegmentType type) {
  switch (type) {
    case SegmentType::Hold: return "hold";
    case SegmentType::Line: return "line";
    case SegmentType::Waypoints: return "waypoints";
    case SegmentType::Circle: return "circle";
  }
  return "hold";
}

std::string JsonSegment(const TrajectorySegmentConfig& segment, int level) {
  // Mirrors asdict(TrajectorySegmentConfig): every field present, unset ones
  // serialised as null, keys sorted.
  std::vector<std::pair<std::string, std::string>> fields;
  const auto optional_number = [](const std::optional<double>& value) {
    return value ? PythonFloatRepr(*value) : std::string("null");
  };
  const auto optional_vector = [&](const std::optional<Eigen::Vector3d>& value) {
    return value ? JsonVector(ToVector(*value), level + 1) : std::string("null");
  };
  const auto vector_list = [&](const std::vector<Eigen::Vector3d>& values,
                               bool present) {
    if (!present) return std::string("null");
    std::string text = "[\n";
    for (std::size_t index = 0; index < values.size(); ++index) {
      text += Indent(level + 2) + JsonVector(ToVector(values[index]), level + 2);
      if (index + 1 < values.size()) text += ",";
      text += "\n";
    }
    return text + Indent(level + 1) + "]";
  };

  fields.emplace_back("clockwise", segment.clockwise
                                       ? JsonBool(*segment.clockwise)
                                       : std::string("null"));
  fields.emplace_back("displacement_m", optional_vector(segment.displacement_m));
  fields.emplace_back("duration_s", optional_number(segment.duration_s));
  fields.emplace_back("durations_s",
                      segment.has_durations
                          ? JsonVector(segment.durations_s, level + 1)
                          : std::string("null"));
  fields.emplace_back("end_position_m", optional_vector(segment.end_position_m));
  fields.emplace_back("end_rpy_rad", optional_vector(segment.end_rpy_rad));
  fields.emplace_back("normal", optional_vector(segment.normal));
  fields.emplace_back("offsets_m",
                      vector_list(segment.offsets_m, segment.has_offsets));
  fields.emplace_back("positions_m",
                      vector_list(segment.positions_m, segment.has_positions));
  fields.emplace_back("radius_m", optional_number(segment.radius_m));
  fields.emplace_back("revolutions",
                      segment.revolutions ? std::to_string(*segment.revolutions)
                                          : std::string("null"));
  fields.emplace_back("rpy_rad", vector_list(segment.rpy_rad, segment.has_rpy));
  fields.emplace_back("start_direction", optional_vector(segment.start_direction));
  fields.emplace_back("type", "\"" + SegmentTypeName(segment.type) + "\"");

  std::string text = "{\n";
  for (std::size_t index = 0; index < fields.size(); ++index) {
    text += Indent(level + 1) + "\"" + fields[index].first + "\": " +
            fields[index].second;
    if (index + 1 < fields.size()) text += ",";
    text += "\n";
  }
  return text + Indent(level) + "}";
}

std::string JsonTarget(const TargetConfig& target, int level) {
  std::string text = "{\n";
  text += Indent(level + 1) + "\"position_m\": " +
          JsonVector(ToVector(target.position_m), level + 1) + ",\n";
  text += Indent(level + 1) + "\"reference_frame\": \"" +
          target.reference_frame + "\",\n";
  text += Indent(level + 1) + "\"rpy_rad\": " +
          JsonVector(ToVector(target.rpy_rad), level + 1) + ",\n";
  text += Indent(level + 1) + "\"trajectory\": " +
          (target.trajectory ? JsonTrajectory(*target.trajectory, level + 1)
                             : std::string("null")) +
          "\n";
  return text + Indent(level) + "}";
}

std::string JsonJoints(const std::optional<Vector7>& joints, int level) {
  if (!joints) return "null";
  std::vector<double> values(joints->data(), joints->data() + kJoints);
  return JsonVector(values, level);
}

std::string JsonLookAtObjectMotion(
    const std::optional<LookAtObjectMotionConfig>& motion, int level) {
  if (!motion) return "null";
  std::string text = "{\n";
  text += Indent(level + 1) + "\"body_name\": \"" + motion->body_name + "\",\n";
  text += Indent(level + 1) + "\"home_position_world_m\": " +
          JsonVector(ToVector(motion->home_position_world_m), level + 1) + ",\n";
  text += Indent(level + 1) + "\"linear_amplitude_m\": " +
          JsonVector(ToVector(motion->linear_amplitude_m), level + 1) + ",\n";
  text += Indent(level + 1) + "\"linear_frequency_hz\": " +
          PythonFloatRepr(motion->linear_frequency_hz) + "\n";
  return text + Indent(level) + "}";
}

}  // namespace

std::string EffectiveConfigJson(const ProjectConfig& config) {
  std::ostringstream out;
  const auto& reactive = config.reactive_pose;
  const auto& keepout = config.cylinder_keepout;
  const auto& safety = config.human_safety;

  out << "{\n";
  out << "  \"controller\": {\n    \"reactive_pose\": {\n";
  out << "      \"dls_damping\": " << PythonFloatRepr(reactive.dls_damping) << ",\n";
  out << "      \"kd_position\": " << PythonFloatRepr(reactive.kd_position) << ",\n";
  out << "      \"kd_rotation\": " << PythonFloatRepr(reactive.kd_rotation) << ",\n";
  out << "      \"kp_position_s_inv\": " << PythonFloatRepr(reactive.kp_position_s_inv) << ",\n";
  out << "      \"kp_rotation_s_inv\": " << PythonFloatRepr(reactive.kp_rotation_s_inv) << ",\n";
  out << "      \"null_gain_s_inv\": " << PythonFloatRepr(reactive.null_gain_s_inv) << ",\n";
  out << "      \"orientation_enabled\": " << JsonBool(reactive.orientation_enabled) << ",\n";
  out << "      \"position_enabled\": " << JsonBool(reactive.position_enabled) << ",\n";
  out << "      \"velocity_enabled\": " << JsonBool(reactive.velocity_enabled) << "\n";
  out << "    }\n  },\n";

  out << "  \"cylinder_keepout\": {\n";
  out << "    \"cylinder_keepout_center_x_m\": " << PythonFloatRepr(keepout.cylinder_keepout_center_x_m) << ",\n";
  out << "    \"cylinder_keepout_center_y_m\": " << PythonFloatRepr(keepout.cylinder_keepout_center_y_m) << ",\n";
  out << "    \"cylinder_keepout_clearance_m\": " << PythonFloatRepr(keepout.cylinder_keepout_clearance_m) << ",\n";
  out << "    \"cylinder_keepout_enabled\": " << JsonBool(keepout.cylinder_keepout_enabled) << ",\n";
  out << "    \"cylinder_keepout_radius_m\": " << PythonFloatRepr(keepout.cylinder_keepout_radius_m) << ",\n";
  out << "    \"cylinder_keepout_z_max_m\": " << PythonFloatRepr(keepout.cylinder_keepout_z_max_m) << ",\n";
  out << "    \"cylinder_keepout_z_min_m\": " << PythonFloatRepr(keepout.cylinder_keepout_z_min_m) << ",\n";
  out << "    \"cylinder_waypoint_tolerance_m\": " << PythonFloatRepr(keepout.cylinder_waypoint_tolerance_m) << "\n";
  out << "  },\n";

  out << "  \"human_safety\": {\n";
  out << "    \"activation_distance_m\": " << PythonFloatRepr(safety.activation_distance_m) << ",\n";
  out << "    \"approach_velocity_damping\": " << PythonFloatRepr(safety.approach_velocity_damping) << ",\n";
  out << "    \"center_xy_torso_m\": "
      << JsonVector({safety.center_xy_torso_m[0], safety.center_xy_torso_m[1]}, 2) << ",\n";
  out << "    \"clearance_m\": " << PythonFloatRepr(safety.clearance_m) << ",\n";
  out << "    \"constraint_tolerance_m_s\": " << PythonFloatRepr(safety.constraint_tolerance_m_s) << ",\n";
  out << "    \"control_margin_m\": " << PythonFloatRepr(safety.control_margin_m) << ",\n";
  out << "    \"enabled\": " << JsonBool(safety.enabled) << ",\n";
  out << "    \"projection_iterations\": " << safety.projection_iterations << ",\n";
  out << "    \"radius_m\": " << PythonFloatRepr(safety.radius_m) << ",\n";
  out << "    \"recovery_gain_s_inv\": " << PythonFloatRepr(safety.recovery_gain_s_inv) << ",\n";
  out << "    \"z_max_torso_m\": " << PythonFloatRepr(safety.z_max_torso_m) << ",\n";
  out << "    \"z_min_torso_m\": " << PythonFloatRepr(safety.z_min_torso_m) << "\n";
  out << "  },\n";

  std::vector<double> velocity(config.limits.joint_velocity_rad_s.data(),
                               config.limits.joint_velocity_rad_s.data() + kJoints);
  out << "  \"limits\": {\n";
  out << "    \"joint_velocity_rad_s\": " << JsonVector(velocity, 2) << ",\n";
  out << "    \"position_lead_rad\": " << PythonFloatRepr(config.limits.position_lead_rad) << "\n";
  out << "  },\n";

  const auto& planning = config.planning;
  out << "  \"planning\": {\n";
  out << "    \"arm\": \"" << planning.arm << "\",\n";
  out << "    \"clearance_margin_m\": " << PythonFloatRepr(planning.clearance_margin_m) << ",\n";
  out << "    \"dense_samples\": " << planning.dense_samples << ",\n";
  out << "    \"deviation_weight\": " << PythonFloatRepr(planning.deviation_weight) << ",\n";
  out << "    \"enabled\": " << JsonBool(planning.enabled) << ",\n";
  out << "    \"floor_height_world_m\": " << PythonFloatRepr(planning.floor_height_world_m) << ",\n";
  out << "    \"include_floor\": " << JsonBool(planning.include_floor) << ",\n";
  out << "    \"include_torso_box\": " << JsonBool(planning.include_torso_box) << ",\n";
  out << "    \"lead_compensation_enabled\": " << JsonBool(planning.lead_compensation_enabled) << ",\n";
  out << "    \"max_angular_acceleration_rad_s2\": " << PythonFloatRepr(planning.max_angular_acceleration_rad_s2) << ",\n";
  out << "    \"max_angular_speed_rad_s\": " << PythonFloatRepr(planning.max_angular_speed_rad_s) << ",\n";
  out << "    \"max_iterations\": " << planning.max_iterations << ",\n";
  out << "    \"max_linear_acceleration_m_s2\": " << PythonFloatRepr(planning.max_linear_acceleration_m_s2) << ",\n";
  out << "    \"max_linear_speed_m_s\": " << PythonFloatRepr(planning.max_linear_speed_m_s) << ",\n";
  out << "    \"obstacle_weight\": " << PythonFloatRepr(planning.obstacle_weight) << ",\n";
  out << "    \"reach_allowance_m\": " << PythonFloatRepr(planning.reach_allowance_m) << ",\n";
  out << "    \"replan_clearance_trigger_m\": " << PythonFloatRepr(planning.replan_clearance_trigger_m) << ",\n";
  out << "    \"smoothness_weight\": " << PythonFloatRepr(planning.smoothness_weight) << ",\n";
  out << "    \"tool_radius_m\": " << PythonFloatRepr(planning.tool_radius_m) << ",\n";
  out << "    \"torso_box_half_extent_m\": "
      << JsonVector(ToVector(planning.torso_box_half_extent_m), 2) << ",\n";
  out << "    \"waypoint_count\": " << planning.waypoint_count << "\n";
  out << "  },\n";

  out << "  \"run\": {\n";
  out << "    \"arm\": \"" << config.run.arm << "\",\n";
  out << "    \"nominal_dt_s\": " << PythonFloatRepr(config.run.nominal_dt_s) << "\n";
  out << "  },\n";

  out << "  \"simulation\": {\n";
  out << "    \"left_initial_joint_position_rad\": "
      << JsonJoints(config.simulation.left_initial_joint_position_rad, 2) << ",\n";
  out << "    \"look_at_object_motion\": "
      << JsonLookAtObjectMotion(config.simulation.look_at_object_motion, 2)
      << ",\n";
  out << "    \"right_initial_joint_position_rad\": "
      << JsonJoints(config.simulation.right_initial_joint_position_rad, 2) << "\n";
  out << "  },\n";

  out << "  \"targets\": {\n";
  out << "    \"left\": " << JsonTarget(config.left_target, 2) << ",\n";
  out << "    \"right\": " << JsonTarget(config.right_target, 2) << "\n";
  out << "  }\n";
  out << "}";
  return out.str();
}

void PrintEffectiveConfig(const ProjectConfig& config) {
  std::cout << "Effective control configuration:\n"
            << EffectiveConfigJson(config) << "\n"
            << "config_sha256=" << config.source_sha256 << "\n";
}

}  // namespace srl::config
