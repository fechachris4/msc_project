// Pure world-frame kinematics and target-frame conversion.
//
// Nothing here reads MuJoCo state. Dynamic torso state and fixed mount
// calibration are explicit inputs; arm FK/Jacobians come from Pinocchio.
// One-to-one with controller/frames.py. Contract clauses B2-B4, C1-C4.

#pragma once

#include "core/Types.h"
#include "kinematics/PinModel.h"

namespace srl::kinematics {

Pose ComposePose(const Pose& parent_to_child, const Pose& child_to_object);

// Derive one arm's world pose, twist, Jacobian and link geometry from an
// explicit plant sample.
ArmControllerState ArmControllerStateOf(PinModel& pin, const PlantState& plant,
                                        Side side,
                                        const MountCalibration& calibration);

DualArmControllerStates ControllerStates(PinModel& pin, const PlantState& plant,
                                         const MountCalibration& calibration);

// Express a world pose in one declared target reference frame.
Pose ExpressPoseInTargetFrame(const PlantState& plant, Side side,
                              const MountCalibration& calibration,
                              TargetFrame frame, const Pose& pose_world);

// Convert a world/base/torso target once at the frame boundary. Controllers
// receive only the returned WorldTarget and never see the frame selector.
WorldTarget ResolveTargetWorld(const PlantState& plant, Side side,
                               const MountCalibration& calibration,
                               const FramedTarget& target);

DualArmWorldTargets ResolveTargetsWorld(const PlantState& plant,
                                        const MountCalibration& calibration,
                                        const DualArmFramedTargets& targets);

}  // namespace srl::kinematics
