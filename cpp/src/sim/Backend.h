// Minimal plant boundary shared by MuJoCo and any future hardware backend.
//
// One command/state exchange per control cycle. `Takeover` returns the first
// feedback sample; `Exchange` applies one complete command and returns the next
// feedback sample; the backend owns whether that means advancing simulation
// time or waiting for a cyclic reply. One-to-one with controller/backend.py.

#pragma once

#include "core/Types.h"

namespace srl::sim {

class PlantBackend {
 public:
  virtual ~PlantBackend() = default;

  virtual PlantState Takeover() = 0;
  virtual PlantState Exchange(const JointPositionCommand& command) = 0;
  virtual void Release() = 0;
};

}  // namespace srl::sim
