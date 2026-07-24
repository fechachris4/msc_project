# Plant backend contract

The shared controller boundary has exactly three operations:

```text
takeover() -> PlantState
exchange(JointPositionCommand) -> PlantState
release()
```

`takeover` acquires backend-specific control ownership and returns the first
feedback sample. `exchange` atomically applies one complete command and returns
the next feedback sample. `release` relinquishes ownership. Backend-specific
inspection and scenario setup may exist outside this contract, but the Runner
and controller cannot depend on them.

## State and command

`PlantState` is a fixed record containing:

- `sample_time_s`: backend sample timestamp, seconds;
- `nominal_dt_s`: intended cycle period, seconds;
- `torso_pose_world`: position in metres and rotation matrix in world;
- `torso_twist_world`: linear m/s and angular rad/s, world-aligned;
- right and left joint position in radians and velocity in rad/s.

`JointPositionCommand` contains exactly seven commanded joint positions in
radians for each arm. Cartesian controller inputs are derived from
`PlantState`, calibrated `T_T_B`, and FK before entering controller math.

## The genuine backend difference

For MuJoCo, `exchange` writes `data.ctrl`, calls `mj_step`, refreshes scripted
torso kinematics, and reads the new `qpos/qvel`. MuJoCo time advances because
the backend explicitly steps it.

For BaseCyclic, `exchange` will send the command through `Refresh` and decode
the feedback reply. Robot time advances externally; the call waits for the
next cyclic feedback instead of advancing a model.

The Runner treats both as one cycle: current feedback → resolve frames →
controller → position integration → one `exchange` → next feedback. It uses
the current state's `nominal_dt_s` for controller integration and never calls
a simulator step, sleeps for a hardware cycle, or inspects a backend type.

MuJoCo model/data ownership, target-marker display, scripted torso setup, and
ground-truth reads are simulation mechanisms. BaseCyclic sessions, servoing
modes, transport errors, and `Refresh` are hardware mechanisms. None belongs
in controller math.
