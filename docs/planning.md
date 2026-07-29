# Collision-aware Cartesian path planning

## What it does

The reactive controller still receives targets through exactly the seam it
always used: a `DualArmTargetSource` sampled once per cycle. The planning
layer replaces what is on the other side of that seam.

At startup `planning.planner.plan_from_state` reads the measured
end-effector pose of each planned arm, resolves the existing
`[targets.<arm>]` goal into world coordinates, captures the torso pose,
and places the knots of the project's existing minimum-jerk C2 spline so
that the path between start and goal keeps clearance from the wearer. The
optimised spline is wrapped in a `PlannedDualArmSource` and handed to
`ReactivePositionRunner` as its `source_targets` argument. No file under
`controller/` is modified, imported into the control path in a new way, or
subclassed.

The planner carries no goal of its own. The goal stays in
`[targets.<arm>]`, in one place, so the planned and unplanned runs are
answering the same task.

Per arm, one plan produces:

- the world-frame trajectory the Runner will sample;
- the knots in both world and torso coordinates;
- the minimum clearance of the straight-line path it started from, and of
  the trajectory it finished with;
- the solver's own termination message and success flag.

An arm that is not planned holds its measured pose through a zero-twist
source, so `arm = "left"` never leaves the right arm undefined.

## What it deliberately does not do

- It does not control. It produces a reference; every joint-level
  decision, including the whole-arm safety projection, still belongs to
  the controller.
- It does not model the arm. The optimised quantity is the path of a
  single Cartesian point (inflated by `tool_radius_m`), not the swept
  volume of seven links.
- It does not solve smoothness, timing, or Cartesian rate limits. Those
  are already properties of `controller/trajectory.py`, and the optimiser
  only chooses where that spline's knots sit.
- It does not replan on its own. `plan_from_state` is a function a caller
  invokes; nothing in the shipped code calls it on a schedule.
- It does not refuse a goal. A plan that cannot reach the requested
  margin is returned with its measured clearance and a `success` flag,
  and the decision about what to do with it stays with the caller.

## Geometry and frames

The wearer envelope is `controller.human_safety`'s finite capped cylinder,
reused verbatim through `planning.obstacles.HumanCylinder`, including the
same `clearance_m` offset. A planned clearance of zero is therefore
exactly the real-time filter's activation boundary, and the two layers
cannot disagree about where the person is.

Every obstacle answers one question for a batch of points:

```text
distance_and_gradient(points_torso_m) -> (signed distance to surface, unit outward gradient)
```

`ObstacleSet` returns the elementwise minimum, which is the clearance the
optimiser must keep positive. Alongside the human cylinder the set can
carry a torso box, the floor as a half-space, and coarse spheres.

### Why collision is optimised in the torso frame

The human envelope is defined in the torso frame and is static there by
construction: when the wearer translates or rotates, the envelope and both
arm mounts move together. Optimising in that frame means the obstacle
field does not move during the solve, its distances and gradients are
exact analytic functions of the sample point, and the clearance number the
planner reports is measured against the same geometry the safety filter
uses.

The floor is the counter-example that proves the rule. It is world-static,
so its torso-frame description changes as the torso moves;
`world_plane_in_torso` converts it once, using the torso pose captured at
plan time, and it is only valid while the torso stays near that pose.

### Why the reference is delivered in the world frame

The task is a world-frame pose hold. `[targets.<arm>]` is declared in the
world frame and the scripted base motion is defined as a disturbance to
that hold, so a torso-frame reference would change the experiment: the
tool would follow the wearer instead of holding its place while the wearer
moves.

So the layer converts twice and only twice. Start, goal, and obstacles go
into the torso frame; the optimiser works there; the optimised knots come
back out through the same plan-time torso pose and the delivered
trajectory is world-frame from then on.

The consequence is explicit and is the reason `remaining_clearance` exists:
because the plan is world-frame and the wearer is not, the plan's
torso-frame clearance is exact only at the torso pose it was made at, and
decays as the wearer moves away from it.

## Adapted GPMP2 ideas

Three ideas are taken from the GPMP2 joint-space demonstration in
`examples/gpmp2_joint_space` and from HumanSL's planner:

- **A smoothness prior over the states.** Here it is a second-difference
  penalty on the knot positions, `k[i] - 2*k[i+1] + k[i+2]`, weighted by
  `smoothness_weight`. It keeps the free knots from bunching or zig-zagging
  when the obstacle cost is inactive.
- **A hinged obstacle cost.** The residual is
  `max(0, clearance_margin_m - clearance)`, weighted by `obstacle_weight`.
  It is exactly zero beyond the margin, so a path that is already clear
  contributes nothing and the solver is free to keep it straight.
- **Dense collision checking between the knots.** The cost is evaluated at
  `dense_samples` points along the spline, not at the knots, because a
  sparse plan is executed densely and the interesting collisions happen
  between support states. This mirrors the interpolated collision factors
  in the reference implementation.

### GTSAM and GPMP2 are not dependencies

The ideas are adapted; the libraries are not linked. `planning/` imports
numpy and `scipy.optimize.least_squares` and nothing else external. The
reasons are specific, not stylistic:

1. **The smoothness the prior would provide already exists.** A GP prior
   over position/velocity states exists to make a discretised trajectory
   smooth and dynamically consistent. This project already owns a globally
   solved minimum-jerk C2 spline that enforces continuity, endpoint rest,
   and Cartesian speed and acceleration limits. A GP prior would duplicate
   that and then fight it.
2. **The problem is tiny.** With the shipped `waypoint_count = 3` there are
   three free knots and nine decision variables. `least_squares` converges
   on the shipped scene in a few evaluations. A factor-graph library buys
   nothing at that size.
3. **The solver family is the same.** GTSAM solves these graphs with
   Gauss-Newton / Levenberg-Marquardt; `least_squares(method="trf")` is
   the same family with a trust region. Nothing about the mathematics
   requires the library.
4. **The domain is different.** GPMP2 plans in configuration space with
   kinematics factors and an SDF; this layer plans one Cartesian path and
   delegates whole-arm feasibility to the reactive filter. Most of the
   library would be unused.
5. **Reproducibility.** GTSAM and GPMP2 are native C++ builds with
   wrappers. The simulation is expected to run from one virtual
   environment on the project machines; the C++ demonstration stays where
   it belongs, in `examples/`, as the reference the adaptation is measured
   against.

## Why the Jacobian is exact

`CartesianWaypointTrajectory` solves for its minimum-jerk quintic
coefficients globally, from a linear system whose right-hand side is linear
in the knot positions. The sampled position is therefore **exactly linear
in the knots**:

```text
dense_positions = B(knot times) @ knot_positions
```

`B` depends only on the knot TIMES, never on the knot positions.
`_spline_basis` does not re-derive it: it builds `B` column by column by
constructing the real trajectory class with unit knot positions and
sampling it, so the basis is by construction the same spline the controller
will later be given.

Splitting the knots into the two fixed endpoints and the free interior
knots gives an affine map from decision variables to collision samples,
and the residual becomes

```text
r_smooth   = smoothness_weight * (D_free @ k_free + D_fixed)
r_obstacle = obstacle_weight * max(0, margin - (d(p(k_free)) - tool_radius))
```

The smoothness block is constant. The only nonlinearity in the whole
problem is the hinge, and its derivative is available in closed form
because each obstacle already returns the distance gradient:

```text
d(r_obstacle_j) / d(k_i) = -obstacle_weight * grad_j * B[j, i]   while active
                         =  0                                    otherwise
```

That is what `path_optimizer.jacobian` assembles. No finite differences are
taken anywhere in the solve, so the cost of an iteration does not scale
with the number of knots the way a numerical Jacobian would, and the
gradient is not an approximation of the model — it is the model.

One subtlety follows from the same property. `B` depends on the knot times,
and a minimum-jerk spline's SHAPE depends on its segment durations. The
delivered trajectory is timed afterwards from the Cartesian limits, so if
that re-timing costs the requested margin, `optimize_path` re-solves once
on the times the delivered trajectory actually uses. The reported
clearance is always measured on the final trajectory object
(`trajectory_min_clearance`), never on the optimiser's internal prediction.

## Delivery: phase ownership and lag compensation

`planning/plan_source.py` is the only file that touches the controller
boundary, and it does so without modifying it.

**Phase ownership.** The Runner's `elapsed_time_s` origin is frozen in
`start()` and can never be reset. A source that owned no phase would, after
a replan, be sampled at the OLD elapsed time and clamp straight to its
final waypoint. `PlannedArmPath` therefore treats the Runner's elapsed time
as a monotonic tick and maps it onto its own plan clock, so adopting a new
plan is continuous by construction.

**Lag compensation.** The reactive control law is

```text
task_twist = Kp * pose_error + Kd * (twist_ref - twist_measured)
```

The derivative term multiplies a velocity ERROR, which vanishes in steady
state, so a moving reference is tracked with a systematic lag. Writing
`e = r - p` for the reference-minus-measured error and taking the
velocity-resolved loop `pdot = task_twist`:

```text
(1 + Kd) * pdot = Kp * e + Kd * rdot
```

and substituting `pdot = rdot - edot`:

```text
rdot = Kp * e + (1 + Kd) * edot
```

which is a first-order lag,

```text
tau * edot + e = rdot / Kp,        tau = (1 + Kd) / Kp
```

Two things follow, and the second is the point:

- The time constant is `tau = (1 + Kd) / Kp`, so raising Kd makes the loop
  slower, not stiffer.
- In steady state `edot -> 0`, so the lag is `e = rdot / Kp`, **independent
  of Kd**. No amount of derivative gain removes it.

With the shipped gains (`Kp = 2.0 s^-1`) a reference moving at the
configured `max_linear_speed_m_s = 0.2` is followed 0.1 m behind. A planned
path is only worth planning if the arm is actually on it, so the layer
pre-inverts that lag at the reference instead:

```text
r = p_plan + pdot_plan / Kp + (Kd / Kp^2) * pddot_plan
```

The first correction cancels the steady-state lag; the second uses the
reference ACCELERATION, which the target seam would otherwise discard, to
correct while the velocity is still changing. The twist delivered alongside
is deliberately the plan's TRUE twist, not the derivative of the
prefiltered pose, because the controller's Kd term wants the real reference
velocity.

This is a prefilter on the reference, not a change to the controller, and
it is switchable (`lead_compensation_enabled`) so the uncompensated
reactive baseline stays measurable. When it is on, the commanded target is
NOT the planned path — it leads it — and both `sim/planning_view.describe`
and this document say so out loud, because a plot of the commanded target
will not lie on the planned line.

## Mutual exclusion with the cylinder keep-out router

`plan_from_state` raises `ValueError` if it is handed an enabled
`[cylinder_keepout]`.

The router rewrites the reference POSITION to its current waypoint while
passing the incoming TWIST through unchanged (`controller/runner.py`, the
`WorldTarget(Pose(waypoint_world, ...), target.twist_world)` construction).
That is coherent for a static goal, where the twist is zero. It is not
coherent for a planned path: the P term would act on the router's waypoint
while the D term acted on the planner's velocity, so the two halves of the
control law would reference different points and the executed path would be
the router's, not the plan's. The reported planned clearance would then
describe a path the robot never took, which is worse than having no plan.

The layers also overlap in intent. Both are end-effector-level obstacle
avoidance around the same central volume; the router does it reactively
with waypoints in the world frame, the planner does it once, in the torso
frame, with a smooth trajectory. Running both would make it impossible to
attribute a result to either. So the configuration is exclusive by
construction rather than by convention.

## Configuration

All keys live under `[planning]` in `config/control.toml`. They are read
into `runtime_config.PlanningConfig`; there is no separate launcher.

| Key | Ships as | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Whether a caller should build the planned source. `false` leaves the existing reactive/static path untouched. |
| `arm` | `"left"` | `"right"`, `"left"`, or `"both"`. Any arm not named holds its measured pose. |
| `waypoint_count` | `3` | Number of FREE interior knots. Total knots are `waypoint_count + 2`, including the two fixed endpoints. |
| `dense_samples` | `60` | Collision samples along the spline per residual evaluation. |
| `clearance_margin_m` | `0.05` | Hinge activation distance. The obstacle cost is exactly zero beyond it. |
| `smoothness_weight` | `1.0` | Weight on the second-difference knot penalty. |
| `obstacle_weight` | `40.0` | Weight on the hinge residual. Only the ratio to `smoothness_weight` matters. |
| `max_iterations` | `200` | Passed to `least_squares` as `max_nfev`. |
| `tool_radius_m` | `0.0` | Subtracted from every clearance. At `0.0` the end-effector is treated as a point. |
| `lead_compensation_enabled` | `true` | Enables the reference prefilter above. Turn it off to measure the uncompensated baseline. |
| `replan_clearance_trigger_m` | `0.01` | Clearance below which a caller SHOULD replan. Nothing in the shipped code polls it. |
| `include_floor` | `true` | Add the floor half-space to the obstacle set. |
| `floor_height_world_m` | `0.0` | World height of that floor, converted into the plan-time torso frame. |
| `include_torso_box` | `true` | Add an axis-aligned box at the torso origin. |
| `torso_box_half_extent_m` | `[0.12, 0.18, 0.28]` | Half extents of that box, in torso coordinates. |
| `max_linear_speed_m_s` | `0.2` | Cartesian limits used to time the optimised spline. |
| `max_linear_acceleration_m_s2` | `0.5` | " |
| `max_angular_speed_rad_s` | `0.5` | " |
| `max_angular_acceleration_rad_s2` | `1.0` | " |

The human envelope itself is NOT configured here. It is read from
`[human_safety]`, so the planner and the real-time filter cannot be
configured apart.

## Viewer and reporting

`sim/planning_view.py` is visualization only, on the same terms as
`sim/cylinder_view.py`: nothing is added to the MJCF, so the drawn geometry
has no contacts and cannot influence the arms.

`describe(planning_config, outcome)` prints the startup banner: which arms
were planned, the knot count, the clearance before and after optimisation
in millimetres, the plan duration, whether lead compensation is on, and, if
it is, an explicit warning that the delivered reference is prefiltered.
The reported `iterations` is the solver's function-evaluation count.

`draw(user_scn, outcome, torso_pose_world)` appends a polyline of the
planned path, a sphere at every knot with the two fixed endpoints coloured
distinctly, and — only when the current torso pose differs from the pose
the plan was made at — a faint second polyline through the same knots
re-expressed under the current torso pose. That ghost is where the path
would be if it had followed the wearer, so the gap between the two lines is
the drift that `remaining_clearance` measures numerically. It respects the
same `ngeom < maxgeom` bound as the other views and returns the number of
geoms it added.

It is deliberately not wired into `main.py`, which is fixed. It is
importable and testable on its own.

## Known limitations

Stated plainly, because each of these is a real gap rather than a tuning
detail.

1. **End-effector only.** The planner reasons about the path of one
   Cartesian point. A collision-free end-effector path does not imply a
   collision-free elbow or forearm. Whole-arm protection remains entirely
   the job of the reactive safety filter documented in
   `docs/human-safety.md`, and nothing here reduces that filter's
   responsibility.
2. **Orientation is not optimised.** Only knot POSITIONS are decision
   variables. Orientation follows a fixed geodesic interpolation from the
   start rotation to the goal rotation, so a tool whose clearance depends
   on its orientation is not modelled at all. With `tool_radius_m = 0.0`
   the tool is a point.
3. **No arm-arm collision model.** Each arm is planned independently and
   neither sees the other. `obstacles.Sphere` and the `extra_spheres`
   argument exist as a coarse stand-in, but `plan_from_state` passes none.
   Planning `arm = "both"` produces two paths that have not been checked
   against each other.
4. **The plan is world-frame, so its clearance decays.** The obstacle set
   is built from the torso pose at plan time. As the wearer moves, the
   world-frame path drifts relative to the torso-frame envelope and the
   guarantee weakens continuously. `remaining_clearance(outcome, plant,
   planning_config)` is the signal for that, and
   `replan_clearance_trigger_m` is the configured threshold — but no
   shipped code polls it, so the trigger is a documented intention, not a
   behaviour.
5. **Replanning latency is unmeasured.** `PlannedArmPath` makes a replan
   phase-continuous by construction, but the cost of `plan_from_state`
   (rebuilding the spline basis, then a nonlinear least-squares solve, per
   arm) has never been timed against the control cycle. It must not be
   called inline in the loop until it has been; the sibling document
   already records that the complete Python cycle exceeds its 2 ms nominal
   budget without any planner in it.
6. **The base-motion amplitudes ship at zero.** `sim/motion.py` has
   `LINEAR_AMPLITUDE` and `ROTATIONAL_AMPLITUDE` set to zero vectors, so
   every result obtained so far is for a STATIC base. The drift this layer
   is designed around — the reason `remaining_clearance` exists at all —
   is therefore currently untested. Raising those amplitudes is the first
   experiment this layer needs, not an optional extra.
7. **The acceleration term of the prefilter is an approximation.** The
   `1/Kp` term is the exact inverse of the steady-state lag. The
   `Kd/Kp^2` term is a second-order correction taken while the reference
   velocity is changing, and because the delivered twist is deliberately
   the plan's true twist rather than the derivative of the prefiltered
   pose, its exact form depends on which of those the controller's D term
   is fed. It has not been validated against a measured closed-loop run;
   with the shipped gains it is worth at most `0.3/2.0^2 * 0.5 = 0.0375 m`
   at full commanded acceleration, and it should be measured before it is
   trusted.
8. **`success` is weaker than the margin.** `PlanResult.success` requires
   the solver to converge and the final clearance to be non-negative, not
   to reach `clearance_margin_m`. A plan can succeed with less margin than
   was asked for; the reported clearance is the number to read, not the
   flag.

## Safety boundary

This is a planning layer in simulation. It shapes a reference; it is not a
protection mechanism and must never be treated as one. The wearer model is
a torso-frame cylinder, not live whole-body tracking. A plan is computed
once against the geometry visible at that instant, and the world is free to
change afterwards. Every claim above is about a MuJoCo simulation with a
static base and has no hardware evidence behind it.

On the real system the ordering is unchanged and non-negotiable: the
reactive whole-arm safety filter, the joint limits, and an independent
emergency stop are what keep a person safe. The planner's only safety
contribution is that it hands the controller a reference that does not
drive it into the person in the first place, which makes the filter's job
rarer, not unnecessary.
