# Whole-arm human safety filter

## What it does

The reactive controller still produces its normal requested seven-joint
velocity. Before that velocity is integrated into a position command, a
per-arm safety filter finds the nearest velocity that satisfies:

- every active link-point distance constraint;
- every joint-velocity limit;
- every one-step joint-position and position-lead limit.

If those constraints cannot all be satisfied, that arm is held and its status
is reported as a genuine safety stop. The other arm remains independent.

The existing `[cylinder_keepout]` router is unchanged. It may shape the
end-effector path at a high level, but it is neither required by nor trusted
as the whole-arm safety layer.

## Geometry and frames

`controller/link_spheres.py` contains 18 conservative spheres per arm,
covering all eight current Kinova collision-mesh bodies. The constants are
generated from `sim/assets/kinova_gen3/gen3.xml` by:

```bash
.venv/bin/python tools/derive_link_spheres.py
```

`test_sphere_chains_cover_every_collision_mesh_vertex` verifies that every
source collision-mesh vertex lies inside at least one sphere. Five spheres at
the fixed base/shoulder/innermost-upper-arm attachment are represented and
drawn but exempt from avoidance; they are the intentional robot/wearer mount
interface. The other 13 spheres are protected.

The human envelope is a finite capped cylinder in the torso frame:

```text
radius = 0.25 m
torso z range = [-1.10, 0.70] m
required clearance = 0.02 m
control margin = 0.02 m
```

Because the cylinder and the arm mounts share the torso frame, translating or
rotating the torso moves both together. Only joint-induced relative arm
motion enters the safety constraint.

## Constraint

For protected sphere `i`, let:

```text
h_i(q) = d_cylinder(p_i(q)) - sphere_radius_i - required_clearance
A_i(q) = outward_distance_gradient_i @ point_jacobian_i
```

The filter enforces:

```text
A_i(q) qdot >= -recovery_gain * (h_i(q) - control_margin)
```

alongside:

```text
qdot_lower <= qdot <= qdot_upper
```

The bounds are the intersection of configured joint-speed limits and the
one-cycle velocities that keep the persistent position command inside its
lead and joint-position limits.

Positive `h_i` means the sphere is outside the required clearance. Negative
`h_i` means that requirement has already been violated and the inequality
requests outward recovery. `control_margin` is a braking buffer for discrete
position-servo lag; it is not added to the reported required clearance.

The filter first checks or repairs the requested velocity directly. If needed,
a warm-started fixed-structure OSQP workspace computes the minimum-change
projection. Every candidate is independently checked against the original
unscaled constraints before use. A missing, non-finite, infeasible, or
insufficiently converged result is never accepted.

## Cycle order

One `ReactivePositionRunner` cycle is:

```text
PlantState
  -> world-frame EE and all link-sphere positions/Jacobians
  -> torso-frame human distances and active constraints
  -> requested reactive-controller joint velocity
  -> whole-arm safety and joint-limit projection
  -> persistent joint-position integration
  -> backend exchange
```

`ControlTrace` keeps requested, speed-clipped, safety-filtered, and actually
applied joint rates as separate quantities. The viewer draws the physical
human envelope, its clearance/control boundary, and every link sphere.

## Configuration

All safety settings live under `[human_safety]` in
`config/control.toml`. `enabled = false` bypasses the new filter and preserves
the recorded pre-filter controller output exactly. No separate launcher or
hidden startup replay is used.

## Current evidence

The automated evidence covers:

- mesh-to-sphere coverage;
- Pinocchio link-point positions and Jacobians against MuJoCo ground truth
  over random joint and torso poses;
- cylinder signed-distance gradients against finite differences;
- torso rigid-motion invariance;
- simultaneous human-distance and joint-limit constraints;
- infeasible-state hold and truthful stop reporting;
- a headless target-inside-human run with the waypoint router disabled;
- exact disabled-mode replay of the existing 500-row controller trace.

The final 2,200-cycle adversarial simulation had no torso/arm contacts and no
safety stops. Minimum reported clearance was 0.01996 m beyond the configured
required 0.02 m. The safety-projection phase measured 0.35 ms at the 99th
percentile on the development Mac after warm-up.

The complete Python/MuJoCo Runner cycle measured 3.13 ms at the 99th
percentile in that run, which is above the configured 2 ms nominal cycle.
The safety mathematics is inside its sub-millisecond budget, but end-to-end
500 Hz timing is not yet proven.

The complete Python suite passes 281 tests; three optional C++ router
cross-checks are skipped when the separate reference build is unavailable.

## Safety boundary

This is simulation evidence, not hardware certification and not proof that a
real robot cannot contact a person. The current human model is a torso-frame
cylinder, not live whole-body tracking; collision meshes, mount calibration,
timing, sensing latency, actuator tracking, and stop behavior must be
validated on the real system. Hardware commissioning should begin at reduced
speed with an independent emergency stop and conservative monitored tests.
The eventual C++/hardware loop also needs measured worst-case timing before
this filter can be treated as a real-time protection layer.
