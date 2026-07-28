// Viewer-only geometry for the end-effector keep-out cylinder and the
// torso-attached whole-arm safety model.
//
// VISUALIZATION ONLY. Nothing here is added to the MJCF, so these shapes have
// no contype/conaffinity, generate no contacts, and cannot push, stop, or
// destabilise the arms. One-to-one with sim/cylinder_view.py and
// sim/human_safety_view.py.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "config/RuntimeConfig.h"
#include "control/CylinderRouter.h"
#include "control/Runner.h"
#include "core/Types.h"

namespace srl::render {

// One obvious startup banner line per state.
std::string DescribeKeepout(const control::CylinderKeepout& keepout,
                            const std::vector<Side>& sides);
std::string DescribeHumanSafety(const config::HumanSafetyConfig& config);

// Draw one shared cylinder plus each arm's active route. Returns the number of
// geoms added, so a caller can tell "disabled" from "drawn".
int DrawKeepout(
    mjvScene* scene, const control::CylinderKeepout& keepout,
    const DualArm<std::optional<control::CylinderRouteStatus>>& routes,
    const std::vector<Side>& sides);

// Draw the exact moving envelope and every conservative arm sphere.
int DrawHumanSafety(mjvScene* scene, const PlantState& plant,
                    const DualArmControllerStates& controller_states,
                    const DualArmHumanSafetyStates& safety_states,
                    const config::HumanSafetyConfig& config,
                    const std::vector<Side>& sides);

}  // namespace srl::render
