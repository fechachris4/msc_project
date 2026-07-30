"""MuJoCo viewer geometry and startup reporting for the planning layer.

VISUALIZATION ONLY. Nothing here is added to the MJCF, so the drawn knots
and path have no contype/conaffinity, generate no contacts, and cannot push,
stop, or destabilize the arms. This module never touches the plan it renders:
it reads finished per-arm ``ArmPlan`` records and writes geoms into
``viewer.user_scn``.

What is drawn is the PLAN, sampled from the optimiser's own trajectory, which
is the path before lead compensation. The reference actually commanded is the
prefiltered one, so ``describe`` says so out loud rather than letting a viewer
picture imply the arm is tracking the drawn line exactly.

The plan is world-frame and the wearer envelope is torso-frame, so a moving
torso pulls the two apart. ``draw`` takes the CURRENT torso pose for that
reason: when it differs from the pose the plan was made at, the plan's
torso-frame knots are drawn a second time under the current torso pose. That
faint ghost is where the path would sit if it had followed the wearer, so the
gap between the two polylines is the drift that
``planning.planner.remaining_clearance`` measures numerically.

Lengths are metres; only printed output uses millimetres.
"""

import mujoco
import numpy as np

from controller.state import Pose
from planning.planner import ArmPlan
from runtime_config import PlanningConfig


# Planned path: bright and solid. Knots: the variables the optimiser moved,
# with the two fixed endpoints called out. Wearer-frame ghost: cool and
# near-transparent, so it can never be mistaken for the commanded path.
PATH_RGBA = (0.20, 0.85, 1.00, 0.90)
KNOT_RGBA = (1.00, 0.85, 0.10, 0.95)
START_KNOT_RGBA = (0.10, 1.00, 0.35, 0.95)
GOAL_KNOT_RGBA = (1.00, 0.30, 0.75, 0.95)
DRIFT_RGBA = (0.55, 0.55, 0.95, 0.22)

KNOT_RADIUS_M = 0.018
ENDPOINT_RADIUS_M = 0.028
DRIFT_KNOT_RADIUS_M = 0.014
PATH_LINE_WIDTH_M = 0.006
DRIFT_LINE_WIDTH_M = 0.004

# Polyline resolution. The plan is a C2 spline, so it is sampled rather than
# drawn knot-to-knot; between-knot curvature is exactly what the dense
# collision cost was evaluated on.
PATH_SAMPLES = 24

# Below these the current torso pose counts as the plan-time pose and the
# ghost is suppressed, because two coincident polylines only z-fight.
DRIFT_POSITION_TOLERANCE_M = 1e-3
DRIFT_ROTATION_TOLERANCE = 1e-3


def describe(planning_config, plans):
    """Startup banner: what was planned, how close it came, what is delivered.

    Clearances are the optimiser's own numbers: the straight-line clearance
    it started from and the clearance of the trajectory it finished with,
    both measured in the torso frame against the same wearer envelope the
    real-time safety filter uses.
    """
    if not isinstance(planning_config, PlanningConfig):
        raise TypeError("planning_config must be a PlanningConfig")
    plans = _checked_plans(plans)

    planned = tuple(plan.side for plan in plans)
    state = "ENABLED" if planning_config.enabled else "DISABLED in config"
    if not planning_config.enabled and planned:
        state = f"{state} (these plans were built explicitly)"
    unplanned = tuple(
        side for side in ("right", "left") if side not in planned
    )
    knot_count = planning_config.waypoint_count + 2

    lines = [
        f"planning: {state}  "
        f"arms={', '.join(planned) if planned else 'none'}  "
        f"waypoints={planning_config.waypoint_count} "
        f"({knot_count} knots incl. both endpoints)  "
        f"margin={planning_config.clearance_margin_m * 1000.0:.0f} mm",
        "  collision optimised in the TORSO frame; reference delivered in "
        "the WORLD frame",
    ]
    if unplanned:
        lines.append(
            f"  not planned: {', '.join(unplanned)} (that arm keeps its "
            "own configured motion)"
        )
    for plan in plans:
        result = plan.result
        status = "OK" if result.success else f"FAILED: {result.message}"
        lines.append(
            f"  {plan.side + ':':6s} clearance "
            f"{result.initial_min_clearance_m * 1000.0:.1f} -> "
            f"{result.final_min_clearance_m * 1000.0:.1f} mm  "
            f"duration={result.duration_s:.2f} s  "
            f"iterations={result.iterations}  {status}"
        )
    if not plans:
        lines.append("  no plan was built")

    if any(plan.lead.enabled for plan in plans):
        lines.append("  lead compensation: ENABLED")
        lines.append(
            "  WARNING: the delivered reference is PREFILTERED, not the "
            "plan itself:"
        )
        lines.append(
            "    r = p + pdot/Kp, so the commanded target LEADS the "
            "drawn path"
        )
        lines.append(
            "    by the controller's known steady-state lag; the arm, not "
            "the command, follows the plan"
        )
    else:
        lines.append(
            "  lead compensation: DISABLED (the plan is commanded "
            "unfiltered and will be tracked with lag)"
        )
    lines.append(
        "  END-EFFECTOR path only - NOT whole-arm avoidance; that stays "
        "with the [human_safety] filter"
    )
    return "\n".join(lines)


def _checked_plans(plans):
    plans = tuple(plans)
    if any(not isinstance(plan, ArmPlan) for plan in plans):
        raise TypeError("plans must contain ArmPlan values")
    return plans


def _add_geom(scene):
    """Claim the next free user_scn slot, or None when the scene is full."""
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _add_sphere(scene, position, radius, rgba):
    geom = _add_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0.0, 0.0]),
        np.asarray(position, dtype=float),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )


def _add_line(scene, start, end, width_m, rgba):
    geom = _add_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        width_m,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )


def _add_polyline(scene, points, width_m, rgba):
    for previous, point in zip(points, points[1:]):
        _add_line(scene, previous, point, width_m, rgba)


def _sampled_path_world_m(trajectory, samples=PATH_SAMPLES):
    """The plan itself, sampled densely, before any lead compensation."""
    times = np.linspace(0.0, trajectory.duration_s, int(samples))
    return [
        np.asarray(
            trajectory.sample(float(time_s)).pose.position_m, dtype=float
        )
        for time_s in times
    ]


def _torso_moved(current, planned):
    position_shift = float(
        np.linalg.norm(current.position_m - planned.position_m)
    )
    rotation_shift = float(
        np.max(np.abs(current.rotation - planned.rotation))
    )
    return (
        position_shift > DRIFT_POSITION_TOLERANCE_M
        or rotation_shift > DRIFT_ROTATION_TOLERANCE
    )


def draw(user_scn, plans, torso_pose_world):
    """Draw each planned path, its knots, and the wearer-frame drift ghost.

    ``user_scn`` is ``viewer.user_scn``; ``torso_pose_world`` is the CURRENT
    torso pose, not the pose a plan was made at. Returns the number of
    geoms added, so a caller can tell "nothing planned" from "drawn" without
    inspecting the scene.
    """
    plans = _checked_plans(plans)
    if not isinstance(torso_pose_world, Pose):
        raise TypeError("torso_pose_world must be a Pose")

    before = user_scn.ngeom
    for plan in plans:
        result = plan.result
        show_drift = _torso_moved(
            torso_pose_world, plan.torso_pose_world
        )
        _add_polyline(
            user_scn,
            _sampled_path_world_m(result.trajectory),
            PATH_LINE_WIDTH_M,
            PATH_RGBA,
        )
        knots = np.asarray(result.knots_world_m, dtype=float)
        last = len(knots) - 1
        for index, knot in enumerate(knots):
            if index == 0:
                _add_sphere(
                    user_scn, knot, ENDPOINT_RADIUS_M, START_KNOT_RGBA
                )
            elif index == last:
                _add_sphere(
                    user_scn, knot, ENDPOINT_RADIUS_M, GOAL_KNOT_RGBA
                )
            else:
                _add_sphere(user_scn, knot, KNOT_RADIUS_M, KNOT_RGBA)
        if not show_drift:
            continue
        # Same torso-frame knots, re-expressed under the torso pose NOW:
        # where the plan would be had it followed the wearer.
        followed = (
            np.asarray(result.knots_torso_m, dtype=float)
            @ torso_pose_world.rotation.T
            + torso_pose_world.position_m
        )
        _add_polyline(user_scn, followed, DRIFT_LINE_WIDTH_M, DRIFT_RGBA)
        for knot in followed:
            _add_sphere(user_scn, knot, DRIFT_KNOT_RADIUS_M, DRIFT_RGBA)
    return user_scn.ngeom - before
