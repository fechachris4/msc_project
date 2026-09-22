# Cartesian target trajectories

## How a trajectory gets in

A trajectory is a structured, SI-valued record under an arm's
`[targets.<arm>]` table in `config/control.toml`. There is no
natural-language parser in the controller process; runtime only accepts
strict, finite records.

The programmatic seam is:

```python
from controller.trajectory_config import materialize_trajectory

materialized = materialize_trajectory(validated_config, start_pose)
```

`validated_config` is a `runtime_config.TargetTrajectoryConfig`. The function
is deterministic and MuJoCo-independent. `sim.target_trajectory` is responsible
only for establishing the simulation start pose and preparing ONE arm's
source; per-arm composition into the dual-arm source the Runner samples
happens in `arm_flow.py`.

## Control-loop boundary

`controller/trajectory.py` contains the pure trajectory engine. A target source
is sampled with trajectory-local elapsed time and returns a `FramedTarget`.
The Runner samples it exactly once per cycle, then
`controller.frames.resolve_target_world` resolves the declared
world/base/torso frame. MuJoCo target bodies receive the resolved pose for
display only and are never controller truth.

```text
elapsed_time_s = current PlantState.sample_time_s
               - takeover PlantState.sample_time_s
```

No trajectory data means the existing `[targets.right]` or `[targets.left]`
pose is adapted to a static source. There is no separate mode switch.

## Mathematical model

Every dynamic source can provide pose, analytic twist, and analytic linear and
angular acceleration. Programs validate all of these at segment boundaries.

### Waypoint spline

For positions `p_i` at strictly increasing times `t_i`, each leg is a quintic
polynomial. Endpoint velocity and acceleration are fixed to zero. Interior
velocities and accelerations are solved globally to minimize

```text
sum over legs integral || d^3 p / dt^3 ||^2 dt
```

subject to exact waypoint interpolation and shared velocity/acceleration at
each interior knot. Translation is therefore C2 through waypoints and normally
flows through them instead of stopping at every point.

Orientation uses the principal SO(3) logarithm on each orientation leg:

```text
phi      = Log(R0^T R1)
h(u)     = 10u^3 - 15u^4 + 6u^5
R(t)     = R0 Exp(h phi)
omega(t) = dh/dt R0 phi
alpha(t) = d2h/dt2 R0 phi
```

Orientation knots are exact rest knots: angular velocity and acceleration are
zero on both sides. Omitting orientation values holds the segment's initial
orientation. This is explicit and portable; no Euler-angle interpolation is
used in the control path.

### Primitives

- `hold`: finite constant pose.
- `line`: two-pose waypoint convenience.
- `waypoints`: the globally solved C2 translation spline.
- `circle`: exact circle geometry with analytic tangent velocity and radial
  plus tangential acceleration. Phase uses the same quintic rest-to-rest law,
  so a repeated revolution has a C2 seam.

An ordered `TargetProgram` composes primitives. One frame is declared for the
complete program; segments cannot silently change it. Adjacent pose, twist,
and acceleration must match. `PeriodicTargetSource` additionally requires the
final boundary to match the start in all of those quantities.

Ordinary user paths can be represented as smooth waypoints without source
changes. A genuinely new mathematical family (such as a clothoid with a
specific curvature law) requires adding and testing a new primitive in code.

## Timing and limits

The structured specification supports:

- maximum linear speed, m/s;
- maximum linear acceleration, m/s²;
- maximum angular speed, rad/s;
- maximum angular acceleration, rad/s².

If duration is explicit, the engine computes polynomial rate bounds and rejects
timing that violates a limit. If duration is omitted, it derives conservative
timing from those limits and uniformly scales a waypoint spline when its
globally blended interior motion requires more time. Circle acceleration uses
a conservative analytic upper bound containing both tangential and centripetal
terms.

Endpoint and inter-primitive boundaries are rest boundaries unless a future
primitive explicitly exposes matching nonzero boundary derivatives. This makes
hold/line/circle/waypoint sequencing C2 by construction and prevents hidden
velocity jumps.

## Structured representation

The record lives under the existing per-arm target so runs are reproducible.
A side-facing vertical circle in the world `yz` plane can look like:

```toml
[targets.left.trajectory]
reference_frame = "world"
start = "measured"
loop = true
open_live_path_plot = true

[targets.left.trajectory.constraints]
max_linear_speed_m_s = 0.10
max_linear_acceleration_m_s2 = 0.20
max_angular_speed_rad_s = 0.50
max_angular_acceleration_rad_s2 = 1.00

[[targets.left.trajectory.segments]]
type = "circle"
plane = "vertical_yz"
radius_m = 0.05
revolutions = 1
clockwise = false
```

Named planes are `xy` (alias `horizontal`), `xz` (alias `vertical_xz`), and
`yz` (alias `vertical_yz`). The resolved measured or configured trajectory
start remains the first point on the circle. The centre is derived one radius
behind that point along the plane's first named axis, so selecting a plane
does not introduce a startup jump. With `clockwise = false`, the initial
geometric direction is from the first named axis toward the second: `+x` to
`+y` for `xy`, `+x` to `+z` for `xz`, and `+y` to `+z` for `yz`.

The advanced form remains available when a non-canonical plane or a particular
centre direction is required. It uses the same circle machinery, but
`normal` and `start_direction` must be supplied together and cannot be mixed
with `plane`:

```toml
[[targets.left.trajectory.segments]]
type = "circle"
radius_m = 0.05
normal = [1.0, 0.0, 0.0]
start_direction = [0.0, 0.0, 1.0]
revolutions = 1
clockwise = false
```

There is deliberately no `center_m` shorthand: a centre that does not already
place the resolved start on the circle would create an implicit position jump.

A smooth ordinary path can be materialized as:

```toml
[[targets.left.trajectory.segments]]
type = "waypoints"
offsets_m = [
  [0.00, 0.00, 0.00],
  [0.04, 0.03, 0.02],
  [0.08, 0.01, 0.05],
  [0.12, 0.00, 0.02],
]
```

When `durations_s` is omitted, the trajectory-level constraints derive timing.
`positions_m` may be used instead of `offsets_m`; its first position must equal
the previous segment endpoint. Optional `rpy_rad` orientation arrays must have
the same count and begin at the previous orientation.

The original `measured_start_displacement` table is still accepted and
translated into line segments internally so older configs still
load. It is compatibility input, not a second trajectory engine.
Simulation-only joint initialization remains separate under
`[simulation.initial_joint_position_rad]`.

## Real-time simulation and evidence

The entry point initializes the configured posture once and launches the
viewer. The trajectory advances only through the visible real-time control
loop; no hidden program replay runs before the viewer opens.

Aligned logs store the resolved desired world pose/twist and measured world
pose/twist from the same control input sample, plus the original source frame.
The live plot uses a bounded nonblocking snapshot queue; full-rate raw evidence
logging remains separate.

## C++ replay fixture

`tests/golden/cartesian_waypoint_replay.json` schema 2 records row-major
rotations plus position, linear/angular velocity, and linear/angular
acceleration. Its convention is ascending-power quintic coefficients, global
minimum-integrated-squared-jerk translation, zero endpoint derivatives, and
principal-log SO(3) orientation legs. A C++ port compares with relative
tolerance `1e-10` and absolute tolerance `1e-11`.
