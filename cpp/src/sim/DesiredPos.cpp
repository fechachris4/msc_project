#include "sim/DesiredPos.h"

#include "math/Transforms.h"
#include "sim/Targets.h"

namespace srl::sim {

DualArmFramedTargets ConfiguredTargets(const config::ProjectConfig& config) {
  DualArmFramedTargets targets;
  for (Side side : kSides) {
    const config::TargetConfig& target_config = config.target(side);
    FramedTarget& target = targets.for_arm(side);
    target.reference_frame = TargetFrameFromName(target_config.reference_frame);
    target.pose = Pose(target_config.position_m,
                       transforms::RotationFromRpy(target_config.rpy_rad));
    target.twist = Twist::Zero();
  }
  return targets;
}

void ShowTargets(MujocoBackend& backend, const DualArmWorldTargets& targets) {
  for (Side side : kSides) {
    const WorldTarget& resolved = targets.for_arm(side);
    backend.SetTargetPose(side, resolved.pose_world.position_m,
                          QuatFromRotation(resolved.pose_world.rotation));
  }
}

DualArmFramedTargets ApplyDesiredPos(MujocoBackend& backend,
                                     kinematics::PinModel& pin,
                                     const config::ProjectConfig& config) {
  const DualArmFramedTargets source_targets = ConfiguredTargets(config);
  const PlantState plant = backend.ReadState(Twist::Zero());
  ShowTargets(backend,
              kinematics::ResolveTargetsWorld(plant, backend.mount_calibration(),
                                              source_targets));
  (void)pin;
  return source_targets;
}

}  // namespace srl::sim
