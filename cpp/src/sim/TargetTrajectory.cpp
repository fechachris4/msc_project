#include "sim/TargetTrajectory.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "kinematics/Frames.h"
#include "math/LinAlg.h"

namespace srl::sim {
namespace {

double LimitedJointMargin(const MujocoBackend& backend, Side side,
                          const Vector7& joint_position_rad) {
  const JointRange range = backend.JointRangeOf(side);
  double smallest = std::numeric_limits<double>::infinity();
  bool any = false;
  for (int index = 0; index < kJoints; ++index) {
    if (!range.limited[index]) continue;
    any = true;
    smallest = std::min(smallest,
                        std::min(joint_position_rad(index) - range.lower(index),
                                 range.upper(index) - joint_position_rad(index)));
  }
  return any ? smallest : std::numeric_limits<double>::infinity();
}

void SetInitialJointPosture(
    MujocoBackend& backend, Side side,
    const std::optional<Vector7>& initial_joint_position_rad) {
  backend.Release();
  backend.Reset();
  if (!initial_joint_position_rad) {
    backend.Forward();
    return;
  }
  backend.SetJointPosition(side, *initial_joint_position_rad);
  backend.ZeroVelocities();
  backend.Forward();
}

Pose ConfiguredStartPose(const PlantState& plant, Side side,
                         const MountCalibration& calibration,
                         TargetFrame reference_frame,
                         const config::TargetTrajectoryConfig& trajectory_config,
                         const DualArmFramedTargets& static_targets,
                         const Pose& measured_world) {
  Pose world_pose;
  if (trajectory_config.start == "measured") {
    world_pose = measured_world;
  } else if (trajectory_config.start == "configured_target") {
    world_pose = kinematics::ResolveTargetWorld(plant, side, calibration,
                                                static_targets.for_arm(side))
                     .pose_world;
  } else {
    throw std::invalid_argument("unsupported trajectory start: " +
                                trajectory_config.start);
  }
  return kinematics::ExpressPoseInTargetFrame(plant, side, calibration,
                                              reference_frame, world_pose);
}

}  // namespace

TargetTrajectorySetup PrepareTargetTrajectory(
    MujocoBackend& backend, kinematics::PinModel& pin,
    const MountCalibration& calibration, Side side,
    const config::TargetTrajectoryConfig& trajectory_config,
    const std::optional<Vector7>& initial_joint_position_rad,
    const DualArmFramedTargets& static_targets) {
  SetInitialJointPosture(backend, side, initial_joint_position_rad);
  const PlantState plant = backend.ReadState(Twist::Zero());

  const DualArmControllerStates states =
      kinematics::ControllerStates(pin, plant, calibration);
  const ArmControllerState& selected = states.for_arm(side);
  const TargetFrame reference_frame =
      TargetFrameFromName(trajectory_config.reference_frame);

  const Pose start_reference =
      ConfiguredStartPose(plant, side, calibration, reference_frame,
                          trajectory_config, static_targets,
                          selected.ee_pose_world);
  const trajectory::MaterializedTrajectory materialized =
      trajectory::MaterializeTrajectory(trajectory_config, start_reference);

  // The other arm holds its configured static target.
  const Side other = side == Side::Right ? Side::Left : Side::Right;
  auto other_hold = std::make_shared<control::StaticTargetSource>(
      static_targets.for_arm(other));

  TargetTrajectorySetup setup;
  setup.selected_source = materialized.source;
  setup.source =
      side == Side::Left
          ? std::make_shared<control::IndependentArmTargetSource>(
                other_hold, materialized.source)
          : std::make_shared<control::IndependentArmTargetSource>(
                materialized.source, other_hold);

  setup.start_pose_reference = start_reference;
  setup.end_pose_reference =
      materialized.program->Sample(materialized.program->duration_s()).pose;
  setup.start_pose_world = selected.ee_pose_world;

  FramedTarget end_target;
  end_target.reference_frame = reference_frame;
  end_target.pose = setup.end_pose_reference;
  end_target.twist = Twist::Zero();
  setup.end_pose_world =
      kinematics::ResolveTargetWorld(plant, side, calibration, end_target)
          .pose_world;

  setup.duration_s = materialized.duration_s;
  setup.boundary_times_s = materialized.boundary_times_s;
  setup.rate_bounds = materialized.rate_bounds;
  setup.initial_singular_values = linalg::SingularValues(selected.jacobian_world);
  setup.initial_joint_margin_rad =
      LimitedJointMargin(backend, side, selected.joints.position_rad);
  return setup;
}

void PrintTargetTrajectorySetup(
    Side side, const config::TargetTrajectoryConfig& trajectory_config,
    const TargetTrajectorySetup& setup) {
  std::printf("Configured %s target trajectory:\n",
              std::string(SideName(side)).c_str());

  std::string segment_types = "[";
  for (std::size_t index = 0; index < trajectory_config.segments.size();
       ++index) {
    if (index > 0) segment_types += ", ";
    switch (trajectory_config.segments[index].type) {
      case config::SegmentType::Hold: segment_types += "'hold'"; break;
      case config::SegmentType::Line: segment_types += "'line'"; break;
      case config::SegmentType::Waypoints: segment_types += "'waypoints'"; break;
      case config::SegmentType::Circle: segment_types += "'circle'"; break;
    }
  }
  segment_types += "]";

  std::printf("reference_frame=%s start=%s loop=%s segments=%s duration_s=%.6f\n",
              trajectory_config.reference_frame.c_str(),
              trajectory_config.start.c_str(),
              trajectory_config.loop ? "True" : "False", segment_types.c_str(),
              setup.duration_s);
  std::printf("resolved_world_start_m=[%.9f %.9f %.9f]\n",
              setup.start_pose_world.position_m(0),
              setup.start_pose_world.position_m(1),
              setup.start_pose_world.position_m(2));
  std::printf("resolved_world_end_m=[%.9f %.9f %.9f]\n",
              setup.end_pose_world.position_m(0),
              setup.end_pose_world.position_m(1),
              setup.end_pose_world.position_m(2));
  std::printf("initial_jacobian_singular_values=[");
  for (int index = 0; index < 6; ++index) {
    std::printf("%s%.6f", index > 0 ? " " : "",
                setup.initial_singular_values(index));
  }
  std::printf("]\n");
  std::printf(
      "trajectory will execute in the real-time simulation loop; no preflight "
      "replay was run\n");
}

}  // namespace srl::sim
