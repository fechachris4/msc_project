"""Fixed-size Cartesian waypoint routing around one vertical keep-out cylinder.

Direct port of the hardware controller's ``src/control/CylinderRouter.{h,cpp}``
(HumanSL_MAIN/Christian_control/basic_control). Pure geometry: no robot, no
MuJoCo, no I/O. Every constant and branch mirrors the C++ so the two stay
comparable — ``tests/test_cylinder_router.py`` cross-checks this module against
the compiled C++ implementation.

Frames and units: all inputs and outputs are metres in ONE cylinder frame, the
frame the caller chose to express the keep-out in. The cylinder axis is that
frame's +z. Angles are radians. Nothing here converts frames; the caller
(``controller/runner.py``) does that once at its boundary.

SCOPE: this routes the END EFFECTOR point only. It is not whole-arm or
per-link collision avoidance, and it never refuses a target — a request inside
the cylinder is moved radially to the route boundary instead.
"""

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np


# --- C++ anonymous-namespace constants (CylinderRouter.cpp lines 11-13) ---
GEOMETRY_EPSILON = 1e-9
ARC_STEP_RAD = 15.0 * math.pi / 180.0  # arc waypoint spacing
ROUTE_PADDING_M = 0.01  # extra radial margin on top of chord compensation

# CylinderRoute::kMaxWaypoints — fixed route capacity, C++ line 32.
MAX_WAYPOINTS = 32


class CylinderRouteKind(Enum):
    """Mirrors the C++ ``enum class CylinderRouteKind``."""

    DIRECT = "direct"
    COUNTER_CLOCKWISE = "counter-clockwise"
    CLOCKWISE = "clockwise"
    OVER = "over"


def route_kind_name(kind):
    """Mirrors ``CylinderRouteKindName`` — the string used in logs and tests."""
    return CylinderRouteKind(kind).value


@dataclass(frozen=True, slots=True)
class CylinderKeepout:
    """One finite vertical cylinder; field-for-field the C++ ``CylinderKeepout``.

    center_xy_m      cylinder axis position in the cylinder frame, m
    radius_m         physical radius, m (inflated by clearance when routing)
    z_min_m/z_max_m  finite extent along the cylinder frame's z, m
    clearance_m      inflation applied to BOTH radius and height, m
    waypoint_tolerance_m  advance distance for the follower, m
    """

    enabled: bool = False
    center_xy_m: tuple = (0.0, 0.0)
    radius_m: float = 0.25
    z_min_m: float = 0.0
    z_max_m: float = 1.8
    clearance_m: float = 0.10
    waypoint_tolerance_m: float = 0.01

    def __post_init__(self):
        if len(self.center_xy_m) != 2:
            raise ValueError("center_xy_m must hold exactly two numbers")

    @property
    def center(self):
        """Cylinder axis as a 2-vector, m."""
        return np.asarray(self.center_xy_m, dtype=float)

    @property
    def obstacle_radius_m(self):
        """Inflated radius the router must keep segments outside of, m."""
        return self.radius_m + self.clearance_m

    @property
    def obstacle_z_min_m(self):
        """Inflated lower extent, m."""
        return self.z_min_m - self.clearance_m

    @property
    def obstacle_z_max_m(self):
        """Inflated upper extent, m."""
        return self.z_max_m + self.clearance_m

    @property
    def route_radius_m(self):
        """Radius the arc waypoints sit on, m.

        The extra radial amount keeps the straight chords between 15-degree arc
        waypoints outside the inflated obstacle rather than cutting through it
        (C++ lines 192-195).
        """
        return (
            self.obstacle_radius_m / math.cos(ARC_STEP_RAD / 2.0)
            + ROUTE_PADDING_M
        )


@dataclass(slots=True)
class CylinderRoute:
    """One planned route; mirrors the C++ ``CylinderRoute`` aggregate."""

    waypoints: list = field(default_factory=list)
    kind: CylinderRouteKind = CylinderRouteKind.DIRECT
    target_adjusted: bool = False
    requested_target: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    effective_target: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    length_m: float = 0.0

    @property
    def size(self):
        return len(self.waypoints)


def _positive_angle(angle):
    """Wrap to [0, 2*pi) — C++ ``PositiveAngle``."""
    two_pi = 2.0 * math.pi
    angle = math.fmod(angle, two_pi)
    return angle + two_pi if angle < 0.0 else angle


def _append(route, point):
    """C++ ``Append``: drop duplicates, respect the fixed capacity."""
    point = np.asarray(point, dtype=float)
    if route.waypoints and (
        np.linalg.norm(route.waypoints[-1] - point) < GEOMETRY_EPSILON
    ):
        return
    if len(route.waypoints) < MAX_WAYPOINTS:
        route.waypoints.append(point)


def _route_length(start, route):
    """C++ ``RouteLength``: start-to-first plus waypoint-to-waypoint, m."""
    length = 0.0
    previous = np.asarray(start, dtype=float)
    for point in route.waypoints:
        length += float(np.linalg.norm(point - previous))
        previous = point
    return length


def _radial_direction(point, center, fallback):
    """C++ ``RadialDirection``: unit radial, with two degenerate fallbacks."""
    radial = np.asarray(point, dtype=float)[:2] - center
    if np.linalg.norm(radial) > GEOMETRY_EPSILON:
        return radial / np.linalg.norm(radial)
    if np.linalg.norm(fallback) > GEOMETRY_EPSILON:
        return np.asarray(fallback, dtype=float) / np.linalg.norm(fallback)
    return np.array([1.0, 0.0])


def _arc_candidate(keepout, start, target, delta_angle, kind, route_radius):
    """C++ ``ArcCandidate``: 15-degree arc around the cylinder at route_radius."""
    route = CylinderRoute()
    route.kind = kind
    route.effective_target = np.asarray(target, dtype=float).copy()

    center = keepout.center
    target_fallback = np.asarray(target, dtype=float)[:2] - center
    start_direction = _radial_direction(start, center, target_fallback)
    target_direction = _radial_direction(target, center, start_direction)
    start_angle = math.atan2(start_direction[1], start_direction[0])
    target_angle = math.atan2(target_direction[1], target_direction[0])

    # The caller supplies direction but not magnitude. Recompute the exact
    # sweep so rounding around +/-pi cannot flip the route (C++ lines 74-78).
    ccw = _positive_angle(target_angle - start_angle)
    if delta_angle >= 0.0:
        sweep = ccw
    else:
        sweep = 0.0 if ccw == 0.0 else ccw - 2.0 * math.pi
    segments = max(1, int(math.ceil(abs(sweep) / ARC_STEP_RAD)))

    _append(route, np.array([
        center[0] + route_radius * math.cos(start_angle),
        center[1] + route_radius * math.sin(start_angle),
        float(start[2]),
    ]))

    for index in range(1, segments + 1):
        fraction = index / segments
        angle = start_angle + fraction * sweep
        _append(route, np.array([
            center[0] + route_radius * math.cos(angle),
            center[1] + route_radius * math.sin(angle),
            float(start[2]) + fraction * (float(target[2]) - float(start[2])),
        ]))
    _append(route, np.asarray(target, dtype=float))
    route.length_m = _route_length(start, route)
    return route


def _over_candidate(keepout, start, target):
    """C++ ``OverCandidate``: rise, translate, descend. Same top-height rule."""
    route = CylinderRoute()
    route.kind = CylinderRouteKind.OVER
    route.effective_target = np.asarray(target, dtype=float).copy()

    top_z = max(
        keepout.z_max_m + keepout.clearance_m + 0.05,
        float(start[2]),
        float(target[2]),
    )
    _append(route, np.array([float(start[0]), float(start[1]), top_z]))
    _append(route, np.array([float(target[0]), float(target[1]), top_z]))
    _append(route, np.asarray(target, dtype=float))
    route.length_m = _route_length(start, route)
    return route


class CylinderRouter:
    """Stateless planner; mirrors the C++ ``CylinderRouter`` class."""

    def __init__(self, keepout=None):
        self._keepout = CylinderKeepout() if keepout is None else keepout
        if not isinstance(self._keepout, CylinderKeepout):
            raise TypeError("keepout must be a CylinderKeepout")

    @property
    def keepout(self):
        return self._keepout

    def segment_intersects(self, start, end):
        """True when any part of the segment lies inside the inflated cylinder.

        C++ ``CylinderRouter::SegmentIntersects``: clip the segment to the
        inflated height band, then take the closest approach of the clipped
        span to the axis in the xy plane.
        """
        keepout = self._keepout
        if not keepout.enabled:
            return False

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        t_min = 0.0
        t_max = 1.0
        dz = end[2] - start[2]
        obstacle_z_min = keepout.obstacle_z_min_m
        obstacle_z_max = keepout.obstacle_z_max_m
        if abs(dz) < GEOMETRY_EPSILON:
            if start[2] < obstacle_z_min or start[2] > obstacle_z_max:
                return False
        else:
            t_a = (obstacle_z_min - start[2]) / dz
            t_b = (obstacle_z_max - start[2]) / dz
            t_min = max(0.0, min(t_a, t_b))
            t_max = min(1.0, max(t_a, t_b))
            if t_min > t_max:
                return False

        p0 = start[:2] - keepout.center
        direction = end[:2] - start[:2]
        nearest_t = t_min
        squared = float(direction @ direction)
        if squared > GEOMETRY_EPSILON:
            nearest_t = min(max(-float(p0 @ direction) / squared, t_min), t_max)
        nearest = p0 + nearest_t * direction
        obstacle_radius = keepout.obstacle_radius_m
        return float(nearest @ nearest) <= obstacle_radius * obstacle_radius

    def plan(self, start, requested_target):
        """Direct when clear, otherwise the shortest of ccw / cw / over.

        C++ ``CylinderRouter::Plan``. A target inside the keep-out is moved
        radially to the route boundary rather than rejected, so every finite
        request produces a route.
        """
        keepout = self._keepout
        start = np.asarray(start, dtype=float)
        requested_target = np.asarray(requested_target, dtype=float)

        direct = CylinderRoute()
        direct.requested_target = requested_target.copy()
        direct.effective_target = requested_target.copy()

        if not keepout.enabled:
            _append(direct, requested_target)
            direct.length_m = float(np.linalg.norm(requested_target - start))
            return direct

        obstacle_radius = keepout.obstacle_radius_m
        route_radius = keepout.route_radius_m
        center = keepout.center

        effective_target = requested_target.copy()
        target_in_height = (
            requested_target[2] >= keepout.obstacle_z_min_m
            and requested_target[2] <= keepout.obstacle_z_max_m
        )
        target_radial = requested_target[:2] - center
        if target_in_height and (
            np.linalg.norm(target_radial) <= obstacle_radius + GEOMETRY_EPSILON
        ):
            start_fallback = start[:2] - center
            direction = _radial_direction(
                requested_target, center, start_fallback)
            effective_target[0] = center[0] + route_radius * direction[0]
            effective_target[1] = center[1] + route_radius * direction[1]
            direct.target_adjusted = True
        direct.effective_target = effective_target

        if not self.segment_intersects(start, effective_target):
            _append(direct, effective_target)
            direct.length_m = float(np.linalg.norm(effective_target - start))
            return direct

        counter_clockwise = _arc_candidate(
            keepout, start, effective_target, 1.0,
            CylinderRouteKind.COUNTER_CLOCKWISE, route_radius)
        clockwise = _arc_candidate(
            keepout, start, effective_target, -1.0,
            CylinderRouteKind.CLOCKWISE, route_radius)
        over = _over_candidate(keepout, start, effective_target)

        # Strict '<' in the same order as the C++ so ties resolve identically.
        best = counter_clockwise
        if clockwise.length_m < best.length_m:
            best = clockwise
        if over.length_m < best.length_m:
            best = over
        best.requested_target = requested_target.copy()
        best.effective_target = effective_target
        best.target_adjusted = direct.target_adjusted
        return best


class CylinderRouteFollower:
    """Stateful cursor over one route; mirrors the C++ follower.

    Route construction happens only on a new target; ``update`` advances
    through the waypoints as the measured end effector reaches each one.
    """

    def __init__(self, keepout=None):
        self._router = CylinderRouter(keepout)
        self._route = CylinderRoute()
        self._index = 0

    @property
    def enabled(self):
        return self._router.keepout.enabled

    @property
    def route(self):
        return self._route

    @property
    def index(self):
        return self._index

    def reset(self, current):
        self._route = self._router.plan(current, current)
        self._index = 0

    def set_target(self, current, requested_target):
        self._route = self._router.plan(current, requested_target)
        self._index = 0

    def update(self, current):
        """Advance past every waypoint already reached; return the active one."""
        current = np.asarray(current, dtype=float)
        tolerance = self._router.keepout.waypoint_tolerance_m
        while (
            self._index + 1 < self._route.size
            and np.linalg.norm(current - self._route.waypoints[self._index])
            <= tolerance
        ):
            self._index += 1
        if self._route.size == 0:
            return current
        return self._route.waypoints[self._index]

    def at_final_waypoint(self):
        return self._route.size > 0 and self._index + 1 == self._route.size
