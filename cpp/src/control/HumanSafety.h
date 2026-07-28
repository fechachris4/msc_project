// Torso-attached human envelope and whole-arm distance constraints.
//
// Geometry only -- controller policy stays in SafetyFilter. The finite
// cylinder is expressed in the torso frame, so it and the arm mounts share
// torso translation and rotation; only joint-induced relative motion appears
// in the distance Jacobian. One-to-one with controller/human_safety.py.
// Contract clauses E1-E6.

#pragma once

#include "config/RuntimeConfig.h"
#include "core/Types.h"
#include "kinematics/LinkSpheres.h"

namespace srl::control {

// Signed distance and outward gradient for a capped torso-z cylinder.
void FiniteCylinderDistanceGradient(const Eigen::Vector3d& point_torso_m,
                                    const config::HumanSafetyConfig& config,
                                    double& distance,
                                    Eigen::Vector3d& gradient);

// Build distance-rate constraints from one explicit plant sample.
ArmHumanSafetyState EvaluateArm(const Pose& torso_pose_world,
                                const ArmControllerState& arm_state,
                                const config::HumanSafetyConfig& config);

DualArmHumanSafetyStates EvaluateDualArm(
    const PlantState& plant, const DualArmControllerStates& states,
    const config::HumanSafetyConfig& config);

}  // namespace srl::control
