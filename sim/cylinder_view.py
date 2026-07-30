"""MuJoCo viewer geometry for the end-effector keep-out cylinder.

VISUALIZATION ONLY. Nothing here is added to the MJCF, so the cylinder has no
contype/conaffinity, generates no contacts, and cannot push, stop, or
destabilize the arms. The simulation has no whole-arm collision-checking
architecture, so introducing a physical obstacle body would change closed-loop
behaviour rather than merely display it.

Geometry is written into ``viewer.user_scn`` each frame. The cylinder is drawn
once from the SAME ``CylinderKeepout`` the router consumes, directly in WORLD
coordinates. It is shared by both arms and never inherits either tilted arm
base's rotation.

Lengths are metres; the cylinder axis is world +z, perpendicular to the floor.
"""

import mujoco
import numpy as np


# Physical cylinder: warm and fairly solid. Inflated clearance boundary: cool
# and near-transparent, so the two are unmistakable at a glance.
PHYSICAL_RGBA = (0.90, 0.35, 0.20, 0.40)
CLEARANCE_RGBA = (0.25, 0.60, 0.95, 0.13)
ROUTE_RGBA = (1.00, 0.85, 0.10, 0.90)
ACTIVE_WAYPOINT_RGBA = (0.10, 1.00, 0.35, 0.95)
ROUTE_WAYPOINT_RADIUS_M = 0.018
ACTIVE_WAYPOINT_RADIUS_M = 0.030
ROUTE_LINE_WIDTH_M = 0.006


def describe(keepout, sides):
    """One obvious startup banner line per state."""
    if not keepout.enabled:
        return (
            "cylinder keep-out: DISABLED "
            "(direct targets; no routing, nothing drawn)"
        )
    center = keepout.center
    return (
        "cylinder keep-out: ENABLED  "
        f"centre=({center[0]:.3f}, {center[1]:.3f}) m  "
        f"radius={keepout.radius_m:.3f} m  "
        f"z=[{keepout.z_min_m:.3f}, {keepout.z_max_m:.3f}] m  "
        f"clearance={keepout.clearance_m:.3f} m  "
        f"tolerance={keepout.waypoint_tolerance_m:.3f} m\n"
        f"  frame: WORLD, axis=+z; one central cylinder shared by "
        f"{', '.join(sides)}\n"
        "  END-EFFECTOR routing only - NOT whole-arm/link collision avoidance"
    )


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


def _add_line(scene, start, end, rgba):
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
        ROUTE_LINE_WIDTH_M,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )


def draw(scene, keepout, cylinder_routes=None):
    """Draw one shared cylinder plus each arm's active route when routing.

    ``scene`` is ``viewer.user_scn``. Returns the number of geoms added, so a
    caller can tell "disabled" from "drawn" without inspecting the scene.
    """
    if not keepout.enabled:
        return 0

    before = scene.ngeom
    _add_cylinder_sized(
        scene,
        keepout.center,
        keepout.z_min_m,
        keepout.z_max_m,
        keepout.radius_m,
        PHYSICAL_RGBA,
    )
    _add_cylinder_sized(
        scene,
        keepout.center,
        keepout.obstacle_z_min_m,
        keepout.obstacle_z_max_m,
        keepout.obstacle_radius_m,
        CLEARANCE_RGBA,
    )

    for status in (() if cylinder_routes is None
                   else cylinder_routes.values()):
        points = [np.asarray(p, dtype=float) for p in status.waypoints_world_m]
        for previous, point in zip(points, points[1:]):
            _add_line(scene, previous, point, ROUTE_RGBA)
        for point in points:
            _add_sphere(scene, point, ROUTE_WAYPOINT_RADIUS_M, ROUTE_RGBA)
        # Composed routes are timed paths with no live cursor; only routes
        # that carry an active waypoint (the legacy follower demo) get the
        # highlight sphere.
        active = getattr(status, "active_waypoint_world_m", None)
        if active is not None:
            _add_sphere(
                scene,
                active,
                ACTIVE_WAYPOINT_RADIUS_M,
                ACTIVE_WAYPOINT_RGBA,
            )
    return scene.ngeom - before


def _add_cylinder_sized(
    scene, center_xy, z_low, z_high, radius, rgba
):
    geom = _add_geom(scene)
    if geom is None:
        return
    half_height = 0.5 * (z_high - z_low)
    center_world = np.array(
        [center_xy[0], center_xy[1], 0.5 * (z_low + z_high)], dtype=float
    )
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        np.array([radius, half_height, 0.0], dtype=float),
        center_world,
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )


def link_intersections(model, data, keepout):
    """Diagnostic only: arm link origins inside the inflated cylinder.

    The cylinder is not a physical body, so MuJoCo reports no contacts for it.
    This is the geometric stand-in: it reports which links currently sit inside
    the inflated volume. Reporting only — it never alters the route, and the
    router still never refuses a target.

    Returns a list of ``(side, body_name, depth_m)`` sorted deepest first,
    where depth is how far inside the inflated radius the link origin sits.
    """
    if not keepout.enabled:
        return []

    findings = []
    inflated_radius = keepout.obstacle_radius_m
    for side in ("right", "left"):
        prefix = f"{side}_"
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(
                model, int(mujoco.mjtObj.mjOBJ_BODY), body_id)
            if (
                not name
                or not name.startswith(prefix)
                or name == f"{side}_target"
            ):
                continue
            point_world = np.asarray(data.xpos[body_id], dtype=float)
            if not (
                keepout.obstacle_z_min_m
                <= point_world[2]
                <= keepout.obstacle_z_max_m
            ):
                continue
            radial = float(
                np.linalg.norm(point_world[:2] - keepout.center)
            )
            if radial < inflated_radius:
                findings.append((side, name, inflated_radius - radial))
    findings.sort(key=lambda item: item[2], reverse=True)
    return findings


def format_link_intersections(findings):
    if not findings:
        return None
    parts = ", ".join(
        f"{name} {depth * 1000.0:.0f} mm" for _, name, depth in findings[:4]
    )
    more = "" if len(findings) <= 4 else f" (+{len(findings) - 4} more)"
    return (
        f"cylinder diagnostic: {len(findings)} link origin(s) inside the "
        f"inflated volume: {parts}{more} "
        "[EE routing only; not whole-arm avoidance]"
    )
