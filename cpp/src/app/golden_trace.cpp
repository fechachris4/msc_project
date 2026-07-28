// C++ counterpart of tests/golden_trace.py.
//
// Runs the identical 250-cycle scenario through the ported pipeline and writes
// the identical 210-column CSV, so the two implementations can be compared
// column-by-column by tools/compare_trace.py. This is the acceptance gate for
// the port (docs/03-cpp-design.md stage 8).
//
// Scenario, matching the Python harness exactly:
//   * human safety DISABLED (so qdot_safety_filtered == qdot_raw),
//   * cylinder keep-out ENABLED (from config/control.toml),
//   * torso driven by the golden sinusoid amplitudes,
//   * both EE targets driven by their own sinusoids,
//   * targets fed back in as explicitly world-framed marker poses.

#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "config/RuntimeConfig.h"
#include "control/Runner.h"
#include "control/TargetSource.h"
#include "kinematics/Frames.h"
#include "kinematics/PinModel.h"
#include "sim/DesiredPos.h"
#include "sim/Motion.h"
#include "sim/MujocoBackend.h"
#include "sim/TargetMotion.h"
#include "sim/Targets.h"
#include "telemetry/TraceWriter.h"

namespace {

using srl::Side;
using srl::kSides;

constexpr int kSteps = 250;

const Eigen::Vector3d kBaseLinearAmplitudeM{0.18, 0.04, 0.05};
const Eigen::Vector3d kBaseRotationalAmplitudeRad{0.0, 0.0, -0.2};
constexpr double kBaseLinearFrequencyHz = 0.5;
constexpr double kBaseRotationalFrequencyHz = 0.5;

constexpr double kTargetLinearFrequencyHz = 0.7;
constexpr double kTargetRotationalFrequencyHz = 0.4;

Eigen::Vector3d TargetLinearAmplitude(Side side) {
  return side == Side::Right ? Eigen::Vector3d{0.012, -0.008, 0.005}
                             : Eigen::Vector3d{-0.009, 0.011, -0.004};
}

Eigen::Vector3d TargetRotationalAmplitude(Side side) {
  return side == Side::Right ? Eigen::Vector3d{0.015, -0.010, 0.008}
                             : Eigen::Vector3d{-0.012, 0.007, -0.009};
}

std::string PythonRoot() { return std::string(SRL_PYTHON_ROOT); }

}  // namespace

int main(int argc, char** argv) {
  std::string output_path = PythonRoot() + "/cpp/build/cpp_trace.csv";
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--out" && index + 1 < argc) {
      output_path = argv[++index];
    } else {
      std::fprintf(stderr, "usage: srl_golden_trace [--out PATH]\n");
      return 2;
    }
  }

  try {
    const srl::config::ProjectConfig config =
        srl::config::LoadConfig(srl::config::DefaultConfigPath());

    srl::sim::MujocoBackend backend(PythonRoot() + "/sim/scene.xml", config);
    srl::kinematics::PinModel pin(
        PythonRoot() + "/sim/assets/kinova_gen3/gen3.xml", "base_link",
        "pinch_site");

    // _reset_current_code(): reset, apply configured targets, capture target
    // homes, then install the golden torso driver.
    backend.Reset();
    srl::sim::ApplyDesiredPos(backend, pin, config);
    srl::sim::TargetMotion target_motion;
    target_motion.CaptureHome(backend);

    srl::sim::TorsoMotion torso = srl::sim::CaptureTorsoHome(backend);
    torso.linear_amplitude = kBaseLinearAmplitudeM;
    torso.rotational_amplitude = kBaseRotationalAmplitudeRad;
    torso.linear_frequency = kBaseLinearFrequencyHz;
    torso.rotational_frequency = kBaseRotationalFrequencyHz;
    srl::sim::InstallTorsoDriver(backend, torso);

    // The golden scenario disables the human-safety filter.
    srl::config::HumanSafetyConfig human_safety = config.human_safety;
    human_safety.enabled = false;

    auto initial_source = std::make_shared<srl::control::StaticDualArmTargetSource>(
        srl::sim::FramedWorldTargets(backend));
    srl::control::ReactivePositionRunner runner(
        backend, pin, backend.mount_calibration(), backend.pipeline_setup(),
        initial_source, std::vector<Side>{Side::Right, Side::Left},
        config.reactive_pose,
        srl::control::KeepoutFromConfig(config.cylinder_keepout), human_safety);
    runner.Start();

    const double dt = backend.model()->opt.timestep;
    srl::telemetry::TraceWriter writer(output_path, config.reactive_pose,
                                       backend.mount_calibration(),
                                       backend.pipeline_setup());

    for (int cycle = 0; cycle < kSteps; ++cycle) {
      const double t = runner.current_state().sample_time_s;
      for (Side side : kSides) {
        target_motion.SetTargetPose(
            backend, t, side, TargetLinearAmplitude(side),
            kTargetLinearFrequencyHz, TargetRotationalAmplitude(side),
            kTargetRotationalFrequencyHz);
      }
      srl::control::StaticDualArmTargetSource source(
          srl::sim::FramedWorldTargets(backend));
      const srl::control::RunnerCycle result = runner.Cycle(&source);

      for (Side side : kSides) {
        writer.WriteCycle(cycle, side, dt, result, backend, pin);
      }
    }
    runner.Close();
    writer.Close();

    std::printf("wrote %d rows to %s\n", kSteps * 2, output_path.c_str());
    std::printf("qp_fallback_entries=%ld\n",
                srl::control::QpFallbackEntryCount());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "golden trace failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
