// Read and write the end-effector target mocap markers.
//
// The targets are mocap bodies in scene.xml (right_target: green sphere,
// left_target: red sphere), driven in world frame. They are DISPLAY markers:
// the Runner resolves retained framed targets and never reads marker state
// back as control input (clause L4) -- except in the golden-trace scenario,
// which deliberately routes marker poses back in as declared world targets.
// One-to-one with sim/targets.py.

#pragma once

#include "core/Types.h"
#include "sim/MujocoBackend.h"

namespace srl::sim {

void SetTarget(MujocoBackend& backend, Side side,
               const Eigen::Vector3d& position_m);
void SetTargetQuat(MujocoBackend& backend, Side side,
                   const Eigen::Vector4d& quaternion_wxyz);

Eigen::Vector3d TargetPosition(const MujocoBackend& backend, Side side);
Eigen::Vector4d TargetQuat(const MujocoBackend& backend, Side side);

// Controller-facing world target read from the simulation marker. The target
// twist is pinned to zero on purpose: the reactive baseline gets no reference
// velocity feedforward, so the D-term damps against the EE's own tracking
// velocity.
WorldTarget WorldTargetOf(const MujocoBackend& backend, Side side);

// Read markers as boundary inputs explicitly declared in the world frame.
DualArmFramedTargets FramedWorldTargets(const MujocoBackend& backend);

// Rotation matrix -> MuJoCo quaternion [w, x, y, z], via mju_mat2Quat.
Eigen::Vector4d QuatFromRotation(const Eigen::Matrix3d& rotation);

}  // namespace srl::sim
