// Strict TOML loading and provenance. Contract clauses I1-I4.

#include "TestSupport.h"

#include <cstdio>
#include <fstream>
#include <string>

#include "config/RuntimeConfig.h"

using namespace srl;
using namespace srl::config;

namespace {

std::string ReadAll(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(stream)),
                     std::istreambuf_iterator<char>());
}

std::string WriteTemp(const std::string& text) {
  static int counter = 0;
  const std::string path =
      "/tmp/srl_config_test_" + std::to_string(counter++) + ".toml";
  std::ofstream stream(path, std::ios::binary);
  stream << text;
  return path;
}

// Load the committed config with one textual substitution applied.
std::string MutatedConfig(const std::string& from, const std::string& to) {
  std::string text = ReadAll(DefaultConfigPath());
  const std::size_t position = text.find(from);
  if (position == std::string::npos) {
    std::fprintf(stderr, "test setup: '%s' not found in control.toml\n",
                 from.c_str());
    std::abort();
  }
  text.replace(position, from.size(), to);
  return WriteTemp(text);
}

void RewriteAssignment(std::string& text, const std::string& section,
                       const std::string& key,
                       const std::string& replacement) {
  const std::string header = "\n[" + section + "]\n";
  const std::size_t header_start = text.find(header);
  if (header_start == std::string::npos) {
    std::fprintf(stderr, "test setup: section [%s] not found\n",
                 section.c_str());
    std::abort();
  }
  const std::size_t section_start = header_start + 1;
  const std::size_t section_end =
      text.find("\n[", section_start + header.size() - 2);
  const std::string assignment_prefix = "\n" + key + " =";
  const std::size_t assignment =
      text.find(assignment_prefix, section_start + header.size() - 2);
  if (assignment == std::string::npos ||
      (section_end != std::string::npos && assignment >= section_end)) {
    std::fprintf(stderr, "test setup: '%s' not found in [%s]\n", key.c_str(),
                 section.c_str());
    std::abort();
  }
  const std::size_t line_start = assignment + 1;
  const std::size_t line_end = text.find('\n', line_start);
  text.replace(line_start, line_end - line_start, replacement);
}

std::string MutatedAssignment(const std::string& section,
                              const std::string& key,
                              const std::string& literal) {
  std::string text = ReadAll(DefaultConfigPath());
  RewriteAssignment(text, section, key, key + " = " + literal);
  return WriteTemp(text);
}

std::string RemovedAssignment(const std::string& section,
                              const std::string& key) {
  std::string text = ReadAll(DefaultConfigPath());
  RewriteAssignment(text, section, key, "");
  return WriteTemp(text);
}

void LoadsTheCurrentConfigIncludingPlanning() {
  const ProjectConfig config = LoadConfig(DefaultConfigPath());
  CHECK_CLOSE(config.run.nominal_dt_s, 0.002, 0.0, "nominal timestep");
  CHECK_TRUE(config.run.arm == "right" || config.run.arm == "left" ||
                 config.run.arm == "both",
             "configured arm");
  CHECK_CLOSE(config.reactive_pose.dls_damping, 0.05, 0.0, "DLS damping");
  CHECK_TRUE(config.planning.arm == "right" ||
                 config.planning.arm == "left" ||
                 config.planning.arm == "both",
             "planning arm is parsed and stored");
  CHECK_TRUE(config.planning.waypoint_count > 0,
             "planning waypoint count is validated");
  CHECK_TRUE(config.planning.torso_box_half_extent_m.minCoeff() > 0.0,
             "planning torso dimensions are validated");
  CHECK_TRUE(EffectiveConfigJson(config).find("\"planning\":") !=
                 std::string::npos,
             "effective config records the planning section");
  CHECK_TRUE(config.simulation.look_at_object_motion.has_value(),
             "the current simulation object-motion section is parsed");
  CHECK_TRUE(config.source_sha256.size() == 64, "sha256 is 64 hex characters");
}

void RejectsUnknownAndMissingKeys() {
  // Clause I1: exact key sets, both directions.
  CHECK_THROWS(LoadConfig(MutatedConfig("[run]", "[run]\nunexpected = 1")),
               "an unknown key in [run] is rejected");
  CHECK_THROWS(LoadConfig(RemovedAssignment("run", "nominal_dt_s")),
               "a missing required key is rejected");
  CHECK_THROWS(
      LoadConfig(MutatedConfig("[limits]", "[limits]\nextra_limit = 2.0")),
      "an unknown key in [limits] is rejected");
  CHECK_THROWS(
      LoadConfig(MutatedConfig("\n[planning]\n",
                               "\n[planning]\nunexpected_planner_key = 1\n")),
      "an unknown key in [planning] is rejected");
  CHECK_THROWS(LoadConfig(RemovedAssignment("planning", "waypoint_count")),
               "a missing required planning key is rejected");
  CHECK_THROWS(
      LoadConfig(MutatedConfig(
          "\n[simulation.look_at_object_motion]\n",
          "\n[simulation.look_at_object_motion]\nunexpected_motion_key = 1\n")),
      "an unknown look-at object-motion key is rejected");
}

void RejectsOutOfRangeValues() {
  CHECK_THROWS(
      LoadConfig(MutatedAssignment("run", "nominal_dt_s", "0.0")),
      "a non-positive timestep is rejected");
  CHECK_THROWS(LoadConfig(MutatedAssignment(
                   "controller.reactive_pose", "dls_damping", "-0.05")),
               "a negative DLS damping is rejected");
  CHECK_THROWS(LoadConfig(MutatedAssignment("run", "arm", "\"middle\"")),
               "an unknown arm selection is rejected");
  CHECK_THROWS(
      LoadConfig(MutatedAssignment("planning", "dense_samples", "0")),
      "a non-positive planning sample count is rejected");
  CHECK_THROWS(
      LoadConfig(MutatedAssignment("planning", "arm", "\"middle\"")),
      "an unknown planning arm selection is rejected");
  CHECK_THROWS(LoadConfig(MutatedAssignment(
                   "simulation.look_at_object_motion",
                   "linear_amplitude_m", "[0.0, 0.0, 0.0]")),
               "a zero object-motion amplitude is rejected");
}

void EnforcesCrossFieldRules() {
  // Clause I2.
  CHECK_THROWS(LoadConfig(MutatedAssignment(
                   "controller.reactive_pose", "kp_position_s_inv", "0.0")),
               "zero position gain with position control enabled is rejected");
  CHECK_THROWS(LoadConfig(MutatedAssignment(
                   "controller.reactive_pose", "kd_position", "1.5")),
               "a Kd >= 1 with velocity feedback enabled is rejected");
  CHECK_THROWS(LoadConfig(MutatedAssignment(
                   "human_safety", "z_max_torso_m", "-2.0")),
               "an inverted human-safety height band is rejected");
  CHECK_THROWS(LoadConfig(MutatedAssignment(
                   "cylinder_keepout", "cylinder_keepout_z_max_m", "-1.0")),
               "an inverted keep-out height band is rejected");
}

void ValidatesDisabledFeaturesToo() {
  // Clause I2: the cylinder bounds are checked even when it is switched off,
  // so a disabled-but-wrong config fails loudly rather than lying in wait.
  std::string text = ReadAll(DefaultConfigPath());
  RewriteAssignment(text, "cylinder_keepout", "cylinder_keepout_enabled",
                    "cylinder_keepout_enabled = false");
  RewriteAssignment(text, "cylinder_keepout", "cylinder_keepout_z_max_m",
                    "cylinder_keepout_z_max_m = -5.0");
  CHECK_THROWS(LoadConfig(WriteTemp(text)),
               "a disabled keep-out with bad bounds is still rejected");

  text = ReadAll(DefaultConfigPath());
  RewriteAssignment(text, "planning", "enabled", "enabled = false");
  RewriteAssignment(text, "planning", "smoothness_weight",
                    "smoothness_weight = 0.0");
  CHECK_THROWS(LoadConfig(WriteTemp(text)),
               "disabled planning settings remain strictly validated");
}

void PythonFloatReprMatchesCPython() {
  // Clause I4: the effective-config JSON has to reproduce CPython's repr.
  CHECK_TRUE(PythonFloatRepr(2.0) == "2.0", "integral floats keep a .0");
  CHECK_TRUE(PythonFloatRepr(0.002) == "0.002", "short decimals stay decimal");
  CHECK_TRUE(PythonFloatRepr(1.1) == "1.1", "shortest round-trip is used");
  CHECK_TRUE(PythonFloatRepr(1e-06) == "1e-06", "small values use exponent form");
  CHECK_TRUE(PythonFloatRepr(0.0) == "0.0", "zero renders as 0.0");
  CHECK_TRUE(PythonFloatRepr(-0.25) == "-0.25", "negatives keep their sign");
  CHECK_TRUE(PythonFloatRepr(1.3892820845874863) == "1.3892820845874863",
             "17-digit values round-trip exactly");
}

void ProvenanceIsStable() {
  const ProjectConfig first = LoadConfig(DefaultConfigPath());
  const ProjectConfig second = LoadConfig(DefaultConfigPath());
  CHECK_TRUE(first.source_sha256 == second.source_sha256,
             "the config hash is deterministic");
  CHECK_TRUE(EffectiveConfigJson(first) == EffectiveConfigJson(second),
             "the effective-config JSON is deterministic");

  // A one-character change must move the hash.
  const ProjectConfig mutated = LoadConfig(
      MutatedAssignment("controller.reactive_pose", "kd_rotation", "0.4"));
  CHECK_TRUE(mutated.source_sha256 != first.source_sha256,
             "editing the file changes the recorded hash");
}

}  // namespace

int main() {
  LoadsTheCurrentConfigIncludingPlanning();
  RejectsUnknownAndMissingKeys();
  RejectsOutOfRangeValues();
  EnforcesCrossFieldRules();
  ValidatesDisabledFeaturesToo();
  PythonFloatReprMatchesCPython();
  ProvenanceIsStable();
  return srl::test::Finish("test_config");
}
