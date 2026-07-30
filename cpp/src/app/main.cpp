// Viewer entry point: world-frame EE pose hold, closed loop.
//
// Per-step data flow (all SI: metres, radians; mm only in the printout):
//   sampled TOML target source + backend PlantState  [trajectory, sim]
//   -> FK EE pose  T_W_E = T_W_T . T_T_K . T_K_E(q)  [kinematics/Frames]
//   -> pose + twist errors; PD + DLS qdot            [control/ReactiveController]
//   -> whole-arm human-distance safety projection    [control/SafetyFilter]
//   -> integrate joint-position command (rad)        [control/PositionActuation]
//   -> backend.Exchange: apply command, mj_step, return next state
//
// usage: srl_sim [right|left|both]
//
// With no positional argument, the arm is read from config/control.toml.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "config/RuntimeConfig.h"
#include "control/Runner.h"
#include "control/TargetSource.h"
#include "kinematics/PinModel.h"
#include "math/LinAlg.h"
#include "planning/CartesianPlanner.h"
#include "render/Overlays.h"
#include "render/Viewer.h"
#include "sim/DesiredPos.h"
#include "sim/Motion.h"
#include "sim/MujocoBackend.h"
#include "sim/TargetTrajectory.h"

namespace {

using srl::Side;

// Steps between error printouts (0.5 s at the 2 ms timestep).
constexpr int kPrintEvery = 250;

struct AppOptions {
  std::vector<Side> arms;
  bool planning_smoke{false};
};

AppOptions ParseOptions(int argc, char** argv, const std::string& fallback) {
  std::string choice = fallback;
  bool planning_smoke = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--planning-smoke") {
      planning_smoke = true;
      continue;
    }
    if (argument == "--trajectory-plot") {
      // Accepted for command-line compatibility with main.py. The live
      // matplotlib path plot is not part of the C++ port; use
      // srl_golden_trace / the trace CSV for offline plotting instead.
      std::printf(
          "note: --trajectory-plot is a Python-only live plot; the C++ port "
          "emits trace CSV instead\n");
      continue;
    }
    if (argument == "right" || argument == "left" || argument == "both") {
      choice = argument;
      continue;
    }
    throw std::invalid_argument(
        "usage: srl_sim [right|left|both] [--planning-smoke]");
  }
  if (choice == "right") return {{Side::Right}, planning_smoke};
  if (choice == "left") return {{Side::Left}, planning_smoke};
  return {{Side::Right, Side::Left}, planning_smoke};
}

std::string PythonRoot() { return std::string(SRL_PYTHON_ROOT); }

}  // namespace

int main(int argc, char** argv) {
  try {
    const srl::config::ProjectConfig config =
        srl::config::LoadConfig(srl::config::DefaultConfigPath());
    const AppOptions options = ParseOptions(argc, argv, config.run.arm);
    const std::vector<Side>& arms = options.arms;
    srl::config::PrintEffectiveConfig(config);

    srl::sim::MujocoBackend backend(PythonRoot() + "/sim/scene.xml", config);
    srl::kinematics::PinModel pin(
        PythonRoot() + "/sim/assets/kinova_gen3/gen3.xml", "base_link",
        "pinch_site");

    const srl::DualArmFramedTargets static_targets =
        srl::sim::ConfiguredTargets(config);

    // Module-default amplitudes are all zero: a static torso unless edited.
    srl::sim::TorsoMotion torso = srl::sim::CaptureTorsoHome(backend);

    srl::DualArm<std::shared_ptr<srl::control::TargetSource>> arm_sources{
        std::make_shared<srl::control::StaticTargetSource>(
            static_targets.right),
        std::make_shared<srl::control::StaticTargetSource>(
            static_targets.left)};

    // At most one configured trajectory arm is supported by the current C++
    // trajectory initialisation path.
    std::vector<Side> trajectory_arms;
    for (Side side : srl::kSides) {
      if (config.target(side).trajectory) trajectory_arms.push_back(side);
    }
    if (trajectory_arms.size() > 1) {
      throw std::invalid_argument(
          "simulation currently supports one configured trajectory arm per run");
    }

    std::vector<Side> cylinder_routing_bypass;
    if (!trajectory_arms.empty()) {
      const Side side = trajectory_arms.front();
      const srl::sim::TargetTrajectorySetup setup =
          srl::sim::PrepareTargetTrajectory(
              backend, pin, backend.mount_calibration(), side,
              *config.target(side).trajectory,
              config.simulation.initial_joint_position(side), static_targets);
      srl::sim::PrintTargetTrajectorySetup(side, *config.target(side).trajectory,
                                           setup);
      arm_sources.for_arm(side) = setup.selected_source;
      cylinder_routing_bypass.push_back(side);
    }

    const srl::PlantState planning_state =
        backend.ReadState(srl::Twist::Zero());
    std::vector<srl::planning::CartesianArmPlan> plans;
    if (config.planning.enabled) {
      for (Side side : srl::kSides) {
        if (!srl::planning::PlanningIncludesSide(config.planning, side)) {
          continue;
        }
        if (config.target(side).trajectory) {
          throw std::invalid_argument(
              "[planning] and [targets." + std::string(SideName(side)) +
              ".trajectory] both drive the " +
              std::string(SideName(side)) + " arm; disable one");
        }
        plans.push_back(srl::planning::PlanCartesianArm(
            pin, planning_state, backend.mount_calibration(), side, config));
        arm_sources.for_arm(side) = plans.back().source;
        cylinder_routing_bypass.push_back(side);
      }
      srl::planning::PrintCartesianPlans(config.planning, plans);
    }
    if (options.planning_smoke) {
      if (!config.planning.enabled || plans.empty()) {
        throw std::invalid_argument(
            "--planning-smoke requires an enabled [planning] arm");
      }
    }

    std::shared_ptr<srl::control::DualArmTargetSource> target_source =
        std::make_shared<srl::control::IndependentArmTargetSource>(
            arm_sources.right, arm_sources.left);

    // Installed after any trajectory preparation, which resets the backend.
    srl::sim::InstallTorsoDriver(backend, torso);

    srl::control::ReactivePositionRunner runner(
        backend, pin, backend.mount_calibration(), backend.pipeline_setup(),
        target_source, arms, config.reactive_pose,
        srl::control::KeepoutFromConfig(config.cylinder_keepout),
        config.human_safety, cylinder_routing_bypass);
    runner.Start();

    std::printf("%s\n", srl::render::DescribeKeepout(runner.cylinder_keepout(),
                                                     arms)
                            .c_str());
    std::printf("%s\n",
                srl::render::DescribeHumanSafety(config.human_safety).c_str());

    if (options.planning_smoke) {
      constexpr int kPlanningSmokeCycles = 50;
      srl::control::RunnerCycle cycle;
      for (int index = 0; index < kPlanningSmokeCycles; ++index) {
        cycle = runner.Cycle();
      }
      runner.Close();
      backend.ConfigureTorsoDriver(nullptr, nullptr);
      std::printf(
          "PLANNING SMOKE OK: native Cartesian plan executed for %d "
          "closed-loop srl_sim cycles; final simulation time=%.3f s\n",
          kPlanningSmokeCycles, cycle.next_state.sample_time_s);
      return 0;
    }

    srl::render::Viewer viewer(backend.model(), backend.data(),
                               "SRL dual Gen3 - reactive pose hold");

    int step = 0;
    const double timestep = backend.model()->opt.timestep;
    while (viewer.IsRunning()) {
      const auto step_start = std::chrono::steady_clock::now();
      const srl::control::RunnerCycle cycle = runner.Cycle();
      srl::sim::ShowTargets(backend, cycle.resolved_targets);

      viewer.UpdateScene();
      // Visualisation only: overlay geometry never contacts the arms.
      srl::render::DrawKeepout(viewer.scene(), runner.cylinder_keepout(),
                               cycle.cylinder_routes, arms);
      srl::render::DrawCartesianPlans(viewer.scene(), plans);
      srl::render::DrawHumanSafety(viewer.scene(), cycle.input_state,
                                   cycle.controller_states,
                                   cycle.human_safety_states,
                                   config.human_safety, arms);

      if (step % kPrintEvery == 0) {
        for (Side side : arms) {
          const auto& trace = *cycle.traces.for_arm(side);
          const Eigen::Vector3d error_mm = trace.e_pos * 1000.0;
          const srl::Vector6 sigma = srl::linalg::SingularValues(trace.J);
          std::printf(
              "t=%6.2fs  %-5s |e|=%.1f mm  e_pos=[%7.1f %7.1f %7.1f]  sigma=[",
              cycle.input_state.sample_time_s, std::string(SideName(side)).c_str(),
              error_mm.norm(), error_mm(0), error_mm(1), error_mm(2));
          for (int index = 0; index < 6; ++index) {
            std::printf("%s%.3f", index > 0 ? " " : "", sigma(index));
          }
          std::printf("]\n");
        }
        for (Side side : arms) {
          const auto& route = cycle.cylinder_routes.for_arm(side);
          if (!route) continue;
          std::printf(
              "        %-5s route=%-17s waypoint %zu/%zu  final=%s  "
              "target_adjusted=%s\n",
              std::string(SideName(side)).c_str(), route->kind.c_str(),
              route->waypoint_index + 1, route->waypoint_count,
              route->at_final_waypoint ? "True" : "False",
              route->target_adjusted ? "True" : "False");
        }
        for (Side side : arms) {
          const auto& status = cycle.human_safety_statuses.for_arm(side);
          if (!status) continue;
          std::printf(
              "        %-5s %s: clearance=%.1f mm  active=%d  adjusted=%s  "
              "reason=%s\n",
              std::string(SideName(side)).c_str(),
              status->stopped ? "SAFETY STOP" : "human safety",
              status->minimum_clearance_m * 1000.0,
              status->active_constraint_count,
              status->human_adjusted ? "True" : "False",
              status->reason.c_str());
        }
      }
      ++step;

      viewer.Render();

      // Real-time pacing only; presentation, never physics (clause J5).
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - step_start;
      const double remaining = timestep - elapsed.count();
      if (remaining > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(remaining));
      }
    }
    runner.Close();
    backend.ConfigureTorsoDriver(nullptr, nullptr);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "simulation failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
