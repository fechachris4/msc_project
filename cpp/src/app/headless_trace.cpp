// Headless parity harness for the DEFAULT configuration.
//
// The golden trace deliberately disables human safety, so the safety-geometry
// evaluation, the projection ladder and the cylinder router's replanning are
// untested by it. This harness runs exactly what `srl_sim` runs -- configured
// targets, the configured left-arm trajectory, cylinder routing and the
// whole-arm safety filter all enabled -- and dumps the quantities that path
// produces, so cpp/tools/dump_headless.py can be diffed against it.

#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "config/RuntimeConfig.h"
#include "control/Runner.h"
#include "control/TargetSource.h"
#include "kinematics/PinModel.h"
#include "sim/DesiredPos.h"
#include "sim/Motion.h"
#include "sim/MujocoBackend.h"
#include "sim/TargetTrajectory.h"

namespace {

using srl::Side;

std::string PythonRoot() { return std::string(SRL_PYTHON_ROOT); }

void Number(std::ofstream& out, double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  out << "," << buffer;
}

template <typename Derived>
void Vector(std::ofstream& out, const Eigen::MatrixBase<Derived>& value) {
  for (int index = 0; index < value.size(); ++index) Number(out, value(index));
}

void Names(std::vector<std::string>& names, const std::string& prefix,
           int count) {
  for (int index = 0; index < count; ++index) {
    names.push_back(prefix + "_" + std::to_string(index));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string output_path = "/tmp/cpp_headless.csv";
  int steps = 2000;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--out" && index + 1 < argc) {
      output_path = argv[++index];
    } else if (argument == "--steps" && index + 1 < argc) {
      steps = std::stoi(argv[++index]);
    } else {
      std::fprintf(stderr,
                   "usage: srl_headless_trace [--out PATH] [--steps N]\n");
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

    const srl::DualArmFramedTargets static_targets =
        srl::sim::ConfiguredTargets(config);
    srl::sim::TorsoMotion torso = srl::sim::CaptureTorsoHome(backend);

    const std::vector<Side> arms{Side::Right, Side::Left};
    std::vector<Side> trajectory_arms;
    for (Side side : arms) {
      if (config.target(side).trajectory) trajectory_arms.push_back(side);
    }

    std::shared_ptr<srl::control::DualArmTargetSource> target_source;
    if (!trajectory_arms.empty()) {
      const Side side = trajectory_arms.front();
      const srl::sim::TargetTrajectorySetup setup =
          srl::sim::PrepareTargetTrajectory(
              backend, pin, backend.mount_calibration(), side,
              *config.target(side).trajectory,
              config.simulation.initial_joint_position(side), static_targets);
      target_source = setup.source;
    } else {
      target_source =
          std::make_shared<srl::control::StaticDualArmTargetSource>(
              static_targets);
    }
    srl::sim::InstallTorsoDriver(backend, torso);

    srl::control::ReactivePositionRunner runner(
        backend, pin, backend.mount_calibration(), backend.pipeline_setup(),
        target_source, arms, config.reactive_pose,
        srl::control::KeepoutFromConfig(config.cylinder_keepout),
        config.human_safety);
    runner.Start();

    std::ofstream out(output_path, std::ios::binary);
    std::vector<std::string> names = {"cycle", "arm", "sample_time_s"};
    Names(names, "q", 7);
    Names(names, "qdot_measured", 7);
    Names(names, "qdot_raw", 7);
    Names(names, "qdot_safety_filtered", 7);
    Names(names, "ctrl_after", 7);
    Names(names, "e_pos", 3);
    Names(names, "e_rot", 3);
    Names(names, "target_world_m", 3);
    Names(names, "routed_world_m", 3);
    names.push_back("min_clearance_m");
    names.push_back("active_count");
    names.push_back("human_adjusted");
    names.push_back("limit_adjusted");
    names.push_back("stopped");
    names.push_back("reason");
    names.push_back("route_kind");
    names.push_back("waypoint_count");
    names.push_back("target_adjusted");
    for (std::size_t index = 0; index < names.size(); ++index) {
      out << (index > 0 ? "," : "") << names[index];
    }
    out << "\n";

    for (int cycle = 0; cycle < steps; ++cycle) {
      const srl::control::RunnerCycle result = runner.Cycle();
      for (Side side : arms) {
        const auto& trace = *result.traces.for_arm(side);
        const auto& safety = *result.human_safety_statuses.for_arm(side);
        const auto& route = result.cylinder_routes.for_arm(side);

        out << cycle << "," << SideName(side);
        Number(out, result.input_state.sample_time_s);
        Vector(out, trace.q);
        Vector(out, trace.qdot_measured);
        Vector(out, trace.qdot_raw);
        Vector(out, trace.qdot_safety_filtered);
        Vector(out, trace.ctrl_after);
        Vector(out, trace.e_pos);
        Vector(out, trace.e_rot);
        Vector(out, result.resolved_targets.for_arm(side).pose_world.position_m);
        Vector(out, result.routed_targets.for_arm(side).pose_world.position_m);
        Number(out, safety.minimum_clearance_m);
        out << "," << safety.active_constraint_count;
        out << "," << (safety.human_adjusted ? "True" : "False");
        out << "," << (safety.limit_adjusted ? "True" : "False");
        out << "," << (safety.stopped ? "True" : "False");
        out << "," << safety.reason;
        out << "," << (route ? route->kind : std::string("none"));
        out << "," << (route ? route->waypoint_count : 0);
        out << "," << (route && route->target_adjusted ? "True" : "False");
        out << "\n";
      }
    }
    runner.Close();
    out.close();

    std::printf("wrote %d rows to %s\n", steps * 2, output_path.c_str());
    std::printf("qp_fallback_entries=%ld\n",
                srl::control::QpFallbackEntryCount());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "headless trace failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
