// Desired end-effector poses -- the reference the controller tracks.
//
// Target placements come from config/control.toml: position in metres, rpy in
// radians with R = Rz(yaw) Ry(pitch) Rx(roll). Apply() returns the retained
// framed targets and initialises their world-frame mocap markers; the Runner
// resolves the retained records from the latest PlantState every cycle, so
// controller math receives only world-frame quantities.
// One-to-one with controller/desired_pos.py.

#pragma once

#include "config/RuntimeConfig.h"
#include "core/Types.h"
#include "kinematics/Frames.h"
#include "sim/MujocoBackend.h"

namespace srl::sim {

DualArmFramedTargets ConfiguredTargets(const config::ProjectConfig& config);

void ShowTargets(MujocoBackend& backend, const DualArmWorldTargets& targets);

// Initialise target markers and return the retained framed targets.
DualArmFramedTargets ApplyDesiredPos(MujocoBackend& backend,
                                     kinematics::PinModel& pin,
                                     const config::ProjectConfig& config);

}  // namespace srl::sim
