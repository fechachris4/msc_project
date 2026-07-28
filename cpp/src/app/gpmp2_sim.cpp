// Visible in-simulation GPMP2 planning and joint-space execution.
//
// MuJoCo starts first and holds the measured joint positions. GPMP2 and exact
// post-validation run on a background thread with their own Pinocchio model.
// Only a validated trajectory is handed to PlannedJointRunner; MuJoCo and its
// viewer remain owned by this main thread.

#include <chrono>
#include <cstdio>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "config/RuntimeConfig.h"
#include "control/HumanSafety.h"
#include "control/PlannedJointRunner.h"
#include "kinematics/Frames.h"
#include "kinematics/PinModel.h"
#include "planning/Gpmp2Planner.h"
#include "planning/PlanValidation.h"
#include "render/Overlays.h"
#include "render/Viewer.h"
#include "sim/MujocoBackend.h"

namespace {

using srl::Side;

constexpr int kPrintEveryCycles = 250;
constexpr double kRenderPeriodS = 1.0 / 60.0;
constexpr double kGoalToleranceRad = 0.01;

std::string PythonRoot() { return std::string(SRL_PYTHON_ROOT); }

struct Arguments {
  Side side{Side::Left};
  srl::Vector7 goal{srl::Vector7::Zero()};
};

Arguments ParseArguments(int argc, char** argv) {
  if (argc != 9) {
    throw std::invalid_argument(
        "usage: srl_gpmp2_sim <left|right> "
        "<goal_q1_rad> ... <goal_q7_rad>");
  }
  Arguments arguments;
  arguments.side = srl::SideFromName(argv[1]);
  for (int joint = 0; joint < srl::kJoints; ++joint) {
    arguments.goal(joint) = std::stod(argv[2 + joint]);
  }
  return arguments;
}

srl::JointPositionCommand HoldMeasured(const srl::PlantState& state) {
  srl::JointPositionCommand command;
  command.position_rad.right = state.arms.right.position_rad;
  command.position_rad.left = state.arms.left.position_rad;
  return command;
}

struct PlanningOutcome {
  srl::planning::Gpmp2Result result;
  srl::planning::PlanValidationReport validation;
  std::string error;

  bool valid() const {
    return error.empty() && result.trajectory != nullptr && validation.valid;
  }
};

PlanningOutcome PlanAndValidate(
    srl::planning::Gpmp2Request request,
    srl::PlantState planning_state,
    std::string arm_model_path) noexcept {
  PlanningOutcome outcome;
  try {
    outcome.result = srl::planning::PlanWithGpmp2(request);
    // The background thread owns this model. It never shares mutable
    // Pinocchio state with the main simulation/control thread.
    srl::kinematics::PinModel validation_pin(
        arm_model_path, "base_link", "pinch_site");
    outcome.validation = srl::planning::ValidateForExecution(
        *outcome.result.trajectory, request.side, planning_state,
        request.mount_calibration, validation_pin, request.limits,
        request.human_safety, planning_state.nominal_dt_s);
    if (!outcome.validation.valid) {
      outcome.error = "exact post-validation failed: " +
                      outcome.validation.reason;
    }
  } catch (const std::exception& error) {
    outcome.error = error.what();
  }
  return outcome;
}

enum class SimulationStage {
  Planning,
  Executing,
  Complete,
  PlanningFailed,
  SafetyStopped,
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    const srl::config::ProjectConfig config =
        srl::config::LoadConfig(srl::config::DefaultConfigPath());
    const std::string arm_model_path =
        PythonRoot() + "/sim/assets/kinova_gen3/gen3.xml";

    srl::sim::MujocoBackend backend(PythonRoot() + "/sim/scene.xml", config);
    // Main-thread model used only by the executor and viewer overlays.
    srl::kinematics::PinModel control_pin(
        arm_model_path, "base_link", "pinch_site");

    srl::PlantState state = backend.Takeover();
    bool hold_backend_active = true;
    srl::JointPositionCommand hold_command = HoldMeasured(state);

    srl::planning::Gpmp2Request request;
    request.side = arguments.side;
    request.start_position_rad = state.arm(arguments.side).position_rad;
    request.start_velocity_rad_s = state.arm(arguments.side).velocity_rad_s;
    request.goal_position_rad = arguments.goal;
    request.mount_calibration = backend.mount_calibration();
    request.limits =
        backend.pipeline_setup().for_arm(arguments.side).actuation_limits;
    request.human_safety = config.human_safety;

    srl::render::Viewer viewer(
        backend.model(), backend.data(),
        "SRL GPMP2 - planning in running simulation", false);
    const std::vector<Side> displayed_arms{arguments.side};
    std::printf(
        "MuJoCo running: holding %s arm while GPMP2 plans in background\n",
        std::string(srl::SideName(arguments.side)).c_str());
    std::printf("%s\n",
                srl::render::DescribeHumanSafety(config.human_safety).c_str());

    std::future<PlanningOutcome> planning_future = std::async(
        std::launch::async, PlanAndValidate, request, state, arm_model_path);
    SimulationStage stage = SimulationStage::Planning;
    std::unique_ptr<srl::control::PlannedJointRunner> runner;

    srl::PlantState display_state = state;
    srl::DualArmControllerStates display_controller_states;
    srl::DualArmHumanSafetyStates display_safety_states;
    bool display_geometry_ready = false;
    bool completion_reported = false;
    std::size_t control_cycle = 0;
    auto next_render = std::chrono::steady_clock::now();

    while (viewer.IsRunning()) {
      const auto cycle_start = std::chrono::steady_clock::now();

      if (stage == SimulationStage::Planning &&
          planning_future.wait_for(std::chrono::seconds(0)) ==
              std::future_status::ready) {
        const PlanningOutcome outcome = planning_future.get();
        if (!outcome.valid()) {
          stage = SimulationStage::PlanningFailed;
          viewer.SetTitle("SRL GPMP2 - planning failed, holding position");
          std::fprintf(stderr, "GPMP2 plan rejected: %s\n",
                       outcome.error.c_str());
        } else {
          std::printf(
              "GPMP2 accepted: graph error %.9g -> %.9g, "
              "exact clearance %.3f mm, %zu validation samples\n",
              outcome.result.initial_graph_error,
              outcome.result.final_graph_error,
              outcome.validation.minimum_human_clearance_m * 1000.0,
              outcome.validation.checked_samples);

          // Transfer backend ownership at a cycle boundary. The simulated
          // arm has been held at the trajectory's measured start meanwhile.
          backend.Release();
          hold_backend_active = false;
          runner = std::make_unique<srl::control::PlannedJointRunner>(
              backend, control_pin, backend.mount_calibration(),
              backend.pipeline_setup(), outcome.result.trajectory,
              arguments.side, srl::control::JointTrackingConfig{},
              config.human_safety);
          try {
            state = runner->Start();
            stage = SimulationStage::Executing;
            viewer.SetTitle("SRL GPMP2 - executing validated trajectory");
            std::printf("trajectory handed to 2 ms joint-controller loop\n");
          } catch (const std::exception& error) {
            runner.reset();
            state = backend.Takeover();
            hold_backend_active = true;
            hold_command = HoldMeasured(state);
            stage = SimulationStage::PlanningFailed;
            viewer.SetTitle(
                "SRL GPMP2 - start changed, execution refused");
            std::fprintf(stderr, "trajectory handoff refused: %s\n",
                         error.what());
          }
        }
      }

      if (runner != nullptr) {
        const srl::control::PlannedRunnerCycle cycle = runner->Cycle();
        display_state = cycle.input_state;
        display_controller_states.for_arm(arguments.side) =
            cycle.controller_state;
        display_safety_states.for_arm(arguments.side) =
            cycle.human_safety_state;
        display_geometry_ready = true;
        state = cycle.next_state;

        if (cycle.execution_stopped) {
          std::fprintf(stderr, "execution safety stop: %s\n",
                       cycle.stop_reason.c_str());
          runner->Close();
          runner.reset();
          state = backend.Takeover();
          hold_backend_active = true;
          hold_command = HoldMeasured(state);
          stage = SimulationStage::SafetyStopped;
          viewer.SetTitle("SRL GPMP2 - safety stop, holding position");
        } else {
          const double error_rad =
              (arguments.goal - state.arm(arguments.side).position_rad)
                  .cwiseAbs()
                  .maxCoeff();
          if (cycle.tracking.reference.finished &&
              error_rad <= kGoalToleranceRad) {
            stage = SimulationStage::Complete;
            if (!completion_reported) {
              completion_reported = true;
              viewer.SetTitle(
                  "SRL GPMP2 - trajectory complete, holding goal");
              std::printf(
                  "trajectory complete: max joint error %.6f rad\n",
                  error_rad);
            }
          }
          if (control_cycle % kPrintEveryCycles == 0) {
            std::printf(
                "t=%6.2fs  %s  max_joint_error=%.5f rad  "
                "clearance=%.1f mm  safety=%s\n",
                cycle.input_state.sample_time_s,
                stage == SimulationStage::Complete ? "complete " : "executing",
                error_rad,
                cycle.human_safety_status.minimum_clearance_m * 1000.0,
                cycle.human_safety_status.reason.c_str());
          }
        }
      } else {
        // Planning, rejection and safety-stop modes all fail safe to a
        // measured-position hold while MuJoCo continues stepping.
        display_state = state;
        display_controller_states.for_arm(arguments.side) =
            srl::kinematics::ArmControllerStateOf(
                control_pin, state, arguments.side,
                backend.mount_calibration());
        display_safety_states.for_arm(arguments.side) =
            srl::control::EvaluateArm(
                state.torso_pose_world,
                display_controller_states.for_arm(arguments.side),
                config.human_safety);
        display_geometry_ready = true;
        state = backend.Exchange(hold_command);

        if (control_cycle % kPrintEveryCycles == 0) {
          const char* label =
              stage == SimulationStage::Planning
                  ? "planning"
                  : (stage == SimulationStage::SafetyStopped ? "safety stop"
                                                              : "plan rejected");
          std::printf("t=%6.2fs  %-13s holding measured joints\n",
                      display_state.sample_time_s, label);
        }
      }
      ++control_cycle;

      const auto now = std::chrono::steady_clock::now();
      if (now >= next_render) {
        viewer.UpdateScene();
        if (display_geometry_ready) {
          srl::render::DrawHumanSafety(
              viewer.scene(), display_state, display_controller_states,
              display_safety_states, config.human_safety, displayed_arms);
        }
        viewer.Render();
        next_render = now + std::chrono::duration_cast<
                                std::chrono::steady_clock::duration>(
                                std::chrono::duration<double>(kRenderPeriodS));
      }

      // The controller owns one 2 ms simulation step per wall-clock cycle.
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - cycle_start;
      const double remaining = state.nominal_dt_s - elapsed.count();
      if (remaining > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(remaining));
      }
    }

    if (runner != nullptr) {
      runner->Close();
    } else if (hold_backend_active) {
      backend.Release();
    }
    if (planning_future.valid()) {
      std::printf("viewer closed while planning; waiting for planner shutdown\n");
      planning_future.wait();
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "GPMP2 simulation failed: %s\n", error.what());
    return 2;
  }
}
