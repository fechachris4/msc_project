#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "config/RuntimeConfig.h"
#include "control/PlannedJointRunner.h"
#include "kinematics/PinModel.h"
#include "planning/Gpmp2Planner.h"
#include "planning/PlanValidation.h"
#include "sim/MujocoBackend.h"

namespace {

std::string PythonRoot() { return std::string(SRL_PYTHON_ROOT); }

struct Arguments {
  srl::Side side{srl::Side::Left};
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    const srl::config::ProjectConfig config =
        srl::config::LoadConfig(srl::config::DefaultConfigPath());
    srl::sim::MujocoBackend backend(PythonRoot() + "/sim/scene.xml", config);
    srl::kinematics::PinModel pin(
        PythonRoot() + "/sim/assets/kinova_gen3/gen3.xml", "base_link",
        "pinch_site");

    const srl::PlantState planning_state = backend.Takeover();
    backend.Release();
    srl::planning::Gpmp2Request request;
    request.side = arguments.side;
    request.start_position_rad =
        planning_state.arm(arguments.side).position_rad;
    request.start_velocity_rad_s =
        planning_state.arm(arguments.side).velocity_rad_s;
    request.goal_position_rad = arguments.goal;
    request.mount_calibration = backend.mount_calibration();
    request.limits =
        backend.pipeline_setup().for_arm(arguments.side).actuation_limits;
    request.human_safety = config.human_safety;

    const srl::planning::Gpmp2Result planned =
        srl::planning::PlanWithGpmp2(request);
    const srl::planning::PlanValidationReport validation =
        srl::planning::ValidateForExecution(
            *planned.trajectory, arguments.side, planning_state,
            backend.mount_calibration(), pin, request.limits,
            config.human_safety, planning_state.nominal_dt_s);
    if (!validation.valid) {
      throw std::runtime_error(
          "refusing execution after exact post-validation: " +
          validation.reason);
    }

    srl::control::JointTrackingConfig tracking;
    srl::control::PlannedJointRunner runner(
        backend, pin, backend.mount_calibration(), backend.pipeline_setup(),
        planned.trajectory, arguments.side, tracking, config.human_safety);
    const srl::PlantState started = runner.Start();
    const std::size_t maximum_cycles = static_cast<std::size_t>(
        std::ceil((planned.trajectory->duration_s() + 2.0) /
                  started.nominal_dt_s));
    double final_error_rad = std::numeric_limits<double>::infinity();
    std::size_t cycles = 0;
    for (; cycles < maximum_cycles; ++cycles) {
      const srl::control::PlannedRunnerCycle cycle = runner.Cycle();
      if (cycle.execution_stopped) {
        runner.Close();
        throw std::runtime_error("execution stopped: " + cycle.stop_reason);
      }
      final_error_rad =
          (arguments.goal -
           cycle.next_state.arm(arguments.side).position_rad)
              .cwiseAbs()
              .maxCoeff();
      if (cycle.tracking.reference.finished && final_error_rad <= 0.01) {
        ++cycles;
        break;
      }
    }
    runner.Close();
    if (final_error_rad > 0.01) {
      throw std::runtime_error(
          "trajectory execution did not settle within 0.01 rad");
    }

    std::printf("planned_and_executed=true arm=%s cycles=%zu\n",
                std::string(srl::SideName(arguments.side)).c_str(), cycles);
    std::printf("final_max_joint_error_rad=%.9g\n", final_error_rad);
    std::printf("exact_minimum_clearance_m=%.9g\n",
                validation.minimum_human_clearance_m);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "GPMP2 simulation failed: %s\n", error.what());
    return 2;
  }
}
