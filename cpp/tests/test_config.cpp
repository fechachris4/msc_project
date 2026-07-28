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

void LoadsTheCommittedConfig() {
  const ProjectConfig config = LoadConfig(DefaultConfigPath());
  CHECK_CLOSE(config.run.nominal_dt_s, 0.002, 0.0, "nominal timestep");
  CHECK_TRUE(config.run.arm == "both", "configured arm");
  CHECK_CLOSE(config.reactive_pose.dls_damping, 0.05, 0.0, "DLS damping");
  CHECK_TRUE(config.human_safety.enabled, "human safety enabled by default");
  CHECK_TRUE(config.cylinder_keepout.cylinder_keepout_enabled,
             "cylinder keep-out enabled by default");
  CHECK_TRUE(config.source_sha256.size() == 64, "sha256 is 64 hex characters");
  CHECK_TRUE(config.left_target.trajectory.has_value(),
             "the left arm has a configured trajectory");
}

void RejectsUnknownAndMissingKeys() {
  // Clause I1: exact key sets, both directions.
  CHECK_THROWS(LoadConfig(MutatedConfig("[run]", "[run]\nunexpected = 1")),
               "an unknown key in [run] is rejected");
  CHECK_THROWS(LoadConfig(MutatedConfig("nominal_dt_s = 0.002\n", "")),
               "a missing required key is rejected");
  CHECK_THROWS(
      LoadConfig(MutatedConfig("[limits]", "[limits]\nextra_limit = 2.0")),
      "an unknown key in [limits] is rejected");
}

void RejectsOutOfRangeValues() {
  CHECK_THROWS(
      LoadConfig(MutatedConfig("nominal_dt_s = 0.002", "nominal_dt_s = 0.0")),
      "a non-positive timestep is rejected");
  CHECK_THROWS(LoadConfig(MutatedConfig("dls_damping = 0.05",
                                        "dls_damping = -0.05")),
               "a negative DLS damping is rejected");
  CHECK_THROWS(LoadConfig(MutatedConfig("arm = \"both\"", "arm = \"middle\"")),
               "an unknown arm selection is rejected");
}

void EnforcesCrossFieldRules() {
  // Clause I2.
  CHECK_THROWS(LoadConfig(MutatedConfig("kp_position_s_inv = 2.0",
                                        "kp_position_s_inv = 0.0")),
               "zero position gain with position control enabled is rejected");
  CHECK_THROWS(LoadConfig(MutatedConfig("kd_position = 0.3",
                                        "kd_position = 1.5")),
               "a Kd >= 1 with velocity feedback enabled is rejected");
  CHECK_THROWS(LoadConfig(MutatedConfig("z_max_torso_m = 0.70",
                                        "z_max_torso_m = -2.0")),
               "an inverted human-safety height band is rejected");
  CHECK_THROWS(LoadConfig(MutatedConfig("cylinder_keepout_z_max_m = 1.8",
                                        "cylinder_keepout_z_max_m = -1.0")),
               "an inverted keep-out height band is rejected");
}

void ValidatesDisabledFeaturesToo() {
  // Clause I2: the cylinder bounds are checked even when it is switched off,
  // so a disabled-but-wrong config fails loudly rather than lying in wait.
  std::string text = ReadAll(DefaultConfigPath());
  const std::size_t enabled = text.find("cylinder_keepout_enabled = true");
  text.replace(enabled, std::string("cylinder_keepout_enabled = true").size(),
               "cylinder_keepout_enabled = false");
  const std::size_t bound = text.find("cylinder_keepout_z_max_m = 1.8");
  text.replace(bound, std::string("cylinder_keepout_z_max_m = 1.8").size(),
               "cylinder_keepout_z_max_m = -5.0");
  CHECK_THROWS(LoadConfig(WriteTemp(text)),
               "a disabled keep-out with bad bounds is still rejected");
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
  const ProjectConfig mutated =
      LoadConfig(MutatedConfig("kd_rotation = 0.3", "kd_rotation = 0.4"));
  CHECK_TRUE(mutated.source_sha256 != first.source_sha256,
             "editing the file changes the recorded hash");
}

}  // namespace

int main() {
  LoadsTheCommittedConfig();
  RejectsUnknownAndMissingKeys();
  RejectsOutOfRangeValues();
  EnforcesCrossFieldRules();
  ValidatesDisabledFeaturesToo();
  PythonFloatReprMatchesCPython();
  ProvenanceIsStable();
  return srl::test::Finish("test_config");
}
