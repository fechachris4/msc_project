"""Strict loader for the Python/C++ shared control TOML.

The TOML file contains every controller-critical startup value. Loaded values
are frozen tuples/scalars so runtime code cannot retune a run by mutation.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "control.toml"
ARMS = ("right", "left")
TARGET_FRAMES = ("world", "base", "torso")
TRAJECTORY_STARTS = ("measured", "configured_target")
TRAJECTORY_SEGMENT_TYPES = ("hold", "line", "waypoints", "circle")
TRAJECTORY_ORIENTATION_POLICIES = (
    "look_at_fixed_world_point",
    "look_at_sim_object",
)
_CIRCLE_PLANE_DIRECTIONS = {
    "xy": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "horizontal": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "xz": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "vertical_xz": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "yz": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "vertical_yz": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}


@dataclass(frozen=True)
class RunConfig:
    nominal_dt_s: float
    arm: str


@dataclass(frozen=True)
class ReactivePoseConfig:
    kp_position_s_inv: float
    kp_rotation_s_inv: float
    kd_position: float
    kd_rotation: float
    null_gain_s_inv: float
    dls_damping: float
    position_enabled: bool
    orientation_enabled: bool
    velocity_enabled: bool
    velocity_feedforward_enabled: bool


@dataclass(frozen=True)
class LimitConfig:
    joint_velocity_rad_s: tuple[float, ...]
    position_lead_rad: float


@dataclass(frozen=True)
class CylinderKeepoutConfig:
    """One central world-frame end-effector keep-out cylinder.

    Field names remain comparable with ``basic_control/src/app/Options.h``,
    but this simulation interprets centre and height in WORLD coordinates.
    The axis is always world +z, perpendicular to the ground.
    """

    cylinder_keepout_enabled: bool
    cylinder_keepout_center_x_m: float
    cylinder_keepout_center_y_m: float
    cylinder_keepout_radius_m: float
    cylinder_keepout_z_min_m: float
    cylinder_keepout_z_max_m: float
    cylinder_keepout_clearance_m: float
    cylinder_waypoint_tolerance_m: float


@dataclass(frozen=True)
class HumanSafetyConfig:
    """Torso-attached finite cylinder used by the whole-arm safety filter."""

    enabled: bool
    center_xy_torso_m: tuple[float, float]
    radius_m: float
    z_min_torso_m: float
    z_max_torso_m: float
    clearance_m: float
    control_margin_m: float
    activation_distance_m: float
    recovery_gain_s_inv: float
    approach_velocity_damping: float
    projection_iterations: int
    constraint_tolerance_m_s: float


@dataclass(frozen=True)
class PlanningConfig:
    """Collision-aware Cartesian path planning above the controller.

    The planner carries no goal of its own: it plans from the measured
    end-effector pose to the existing ``[targets.<arm>]`` pose, so the
    goal representation stays in exactly one place.
    """

    enabled: bool
    arm: str
    waypoint_count: int
    dense_samples: int
    clearance_margin_m: float
    smoothness_weight: float
    obstacle_weight: float
    max_iterations: int
    tool_radius_m: float
    deviation_weight: float
    reach_allowance_m: float
    lead_compensation_enabled: bool
    replan_clearance_trigger_m: float
    include_floor: bool
    floor_height_world_m: float
    include_torso_box: bool
    torso_box_half_extent_m: tuple[float, float, float]
    max_linear_speed_m_s: float
    max_linear_acceleration_m_s2: float
    max_angular_speed_rad_s: float
    max_angular_acceleration_rad_s2: float


@dataclass(frozen=True)
class TrajectoryConstraintsConfig:
    max_linear_speed_m_s: float | None
    max_linear_acceleration_m_s2: float | None
    max_angular_speed_rad_s: float | None
    max_angular_acceleration_rad_s2: float | None


@dataclass(frozen=True)
class TrajectorySegmentConfig:
    type: str
    duration_s: float | None = None
    displacement_m: tuple[float, float, float] | None = None
    end_position_m: tuple[float, float, float] | None = None
    offsets_m: tuple[tuple[float, float, float], ...] | None = None
    positions_m: tuple[tuple[float, float, float], ...] | None = None
    durations_s: tuple[float, ...] | None = None
    rpy_rad: tuple[tuple[float, float, float], ...] | None = None
    end_rpy_rad: tuple[float, float, float] | None = None
    radius_m: float | None = None
    normal: tuple[float, float, float] | None = None
    start_direction: tuple[float, float, float] | None = None
    revolutions: int | None = None
    clockwise: bool | None = None


@dataclass(frozen=True)
class TrajectoryOrientationConfig:
    policy: str
    object_position_world_m: tuple[float, float, float] | None
    tool_forward_axis: tuple[float, float, float]
    tool_up_axis: tuple[float, float, float]
    world_up_direction: tuple[float, float, float]
    object_body: str | None = None


@dataclass(frozen=True)
class TargetTrajectoryConfig:
    reference_frame: str
    start: str
    loop: bool
    open_live_path_plot: bool
    constraints: TrajectoryConstraintsConfig
    segments: tuple[TrajectorySegmentConfig, ...]
    orientation: TrajectoryOrientationConfig | None = None


@dataclass(frozen=True)
class TargetConfig:
    reference_frame: str
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]
    trajectory: TargetTrajectoryConfig | None


@dataclass(frozen=True)
class LookAtObjectMotionConfig:
    body_name: str
    home_position_world_m: tuple[float, float, float]
    linear_amplitude_m: tuple[float, float, float]
    linear_frequency_hz: float


@dataclass(frozen=True)
class SimulationConfig:
    right_initial_joint_position_rad: tuple[float, ...] | None
    left_initial_joint_position_rad: tuple[float, ...] | None
    look_at_object_motion: LookAtObjectMotionConfig | None

    def initial_joint_position(self, side):
        if side == "right":
            return self.right_initial_joint_position_rad
        if side == "left":
            return self.left_initial_joint_position_rad
        raise ValueError(f"unknown arm: {side!r}")


@dataclass(frozen=True)
class ProjectConfig:
    run: RunConfig
    reactive_pose: ReactivePoseConfig
    limits: LimitConfig
    cylinder_keepout: CylinderKeepoutConfig
    human_safety: HumanSafetyConfig
    planning: PlanningConfig
    right_target: TargetConfig
    left_target: TargetConfig
    simulation: SimulationConfig
    source_path: str
    source_sha256: str

    def target(self, side):
        if side == "right":
            return self.right_target
        if side == "left":
            return self.left_target
        raise ValueError(f"unknown arm: {side!r}")


_ROOT_KEYS = {
    "run",
    "controller",
    "limits",
    "cylinder_keepout",
    "human_safety",
    "planning",
    "targets",
    "simulation",
}
_PLANNING_KEYS = {
    "enabled",
    "arm",
    "waypoint_count",
    "dense_samples",
    "clearance_margin_m",
    "smoothness_weight",
    "obstacle_weight",
    "max_iterations",
    "tool_radius_m",
    "deviation_weight",
    "reach_allowance_m",
    "lead_compensation_enabled",
    "replan_clearance_trigger_m",
    "include_floor",
    "floor_height_world_m",
    "include_torso_box",
    "torso_box_half_extent_m",
    "max_linear_speed_m_s",
    "max_linear_acceleration_m_s2",
    "max_angular_speed_rad_s",
    "max_angular_acceleration_rad_s2",
}
_RUN_KEYS = {"nominal_dt_s", "arm"}
_CONTROLLER_KEYS = {"reactive_pose"}
_REACTIVE_KEYS = {
    "kp_position_s_inv",
    "kp_rotation_s_inv",
    "kd_position",
    "kd_rotation",
    "null_gain_s_inv",
    "dls_damping",
    "position_enabled",
    "orientation_enabled",
    "velocity_enabled",
    "velocity_feedforward_enabled",
}
_LIMIT_KEYS = {"joint_velocity_rad_s", "position_lead_rad"}
_CYLINDER_KEEPOUT_KEYS = {
    "cylinder_keepout_enabled",
    "cylinder_keepout_center_x_m",
    "cylinder_keepout_center_y_m",
    "cylinder_keepout_radius_m",
    "cylinder_keepout_z_min_m",
    "cylinder_keepout_z_max_m",
    "cylinder_keepout_clearance_m",
    "cylinder_waypoint_tolerance_m",
}
_HUMAN_SAFETY_KEYS = {
    "enabled",
    "center_xy_torso_m",
    "radius_m",
    "z_min_torso_m",
    "z_max_torso_m",
    "clearance_m",
    "control_margin_m",
    "activation_distance_m",
    "recovery_gain_s_inv",
    "approach_velocity_damping",
    "projection_iterations",
    "constraint_tolerance_m_s",
}
_TARGET_KEYS = {"reference_frame", "position_m", "rpy_rad"}
_TRAJECTORY_REQUIRED_KEYS = {
    "reference_frame",
    "start",
    "loop",
    "open_live_path_plot",
    "segments",
}
_TRAJECTORY_ALLOWED_KEYS = _TRAJECTORY_REQUIRED_KEYS | {
    "constraints",
    "orientation",
}
_TRAJECTORY_CONSTRAINT_KEYS = {
    "max_linear_speed_m_s",
    "max_linear_acceleration_m_s2",
    "max_angular_speed_rad_s",
    "max_angular_acceleration_rad_s2",
}
_LOOK_AT_COMMON_KEYS = {
    "policy",
    "tool_forward_axis",
    "tool_up_axis",
    "world_up_direction",
}
_FIXED_LOOK_AT_KEYS = _LOOK_AT_COMMON_KEYS | {
    "object_position_world_m",
}
_SIM_OBJECT_LOOK_AT_KEYS = _LOOK_AT_COMMON_KEYS | {"object_body"}
_LEGACY_TRAJECTORY_KEYS = {
    "shape",
    "reference_frame",
    "displacement_m",
    "leg_duration_s",
    "repeat_out_and_back",
    "orientation_policy",
    "open_live_path_plot",
}
_SIMULATION_REQUIRED_KEYS = {"initial_joint_position_rad"}
_SIMULATION_ALLOWED_KEYS = _SIMULATION_REQUIRED_KEYS | {
    "look_at_object_motion",
}
_LOOK_AT_OBJECT_MOTION_KEYS = {
    "body_name",
    "home_position_world_m",
    "linear_amplitude_m",
    "linear_frequency_hz",
}


def _require_table(value, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a TOML table")
    return value


def _require_exact_keys(table, expected, location):
    actual = set(table)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{location} keys differ; missing={missing}, extra={extra}"
        )


def _finite_number(value, location, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{location} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{location} must be greater than zero")
    if nonnegative and value < 0.0:
        raise ValueError(f"{location} must be non-negative")
    return value


def _boolean(value, location):
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be true or false")
    return value


def _non_empty_string(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _choice(value, choices, location):
    if value not in choices:
        raise ValueError(f"{location} must be one of {choices}")
    return value


def _vector(value, size, location, *, positive=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{location} must contain exactly {size} numbers")
    return tuple(
        _finite_number(item, f"{location}[{index}]", positive=positive)
        for index, item in enumerate(value)
    )


def _vectors(value, size, location, *, minimum_length=1):
    if not isinstance(value, list) or len(value) < minimum_length:
        raise ValueError(
            f"{location} must contain at least {minimum_length} vectors"
        )
    return tuple(
        _vector(item, size, f"{location}[{index}]")
        for index, item in enumerate(value)
    )


def _positive_integer(value, location):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _optional_positive(table, key, location):
    return (
        None
        if key not in table
        else _finite_number(
            table[key], f"{location}.{key}", positive=True
        )
    )


def _parse_reactive_pose(table):
    table = _require_table(table, "controller.reactive_pose")
    _require_exact_keys(table, _REACTIVE_KEYS, "controller.reactive_pose")
    config = ReactivePoseConfig(
        kp_position_s_inv=_finite_number(
            table["kp_position_s_inv"],
            "controller.reactive_pose.kp_position_s_inv",
            nonnegative=True,
        ),
        kp_rotation_s_inv=_finite_number(
            table["kp_rotation_s_inv"],
            "controller.reactive_pose.kp_rotation_s_inv",
            nonnegative=True,
        ),
        kd_position=_finite_number(
            table["kd_position"],
            "controller.reactive_pose.kd_position",
            nonnegative=True,
        ),
        kd_rotation=_finite_number(
            table["kd_rotation"],
            "controller.reactive_pose.kd_rotation",
            nonnegative=True,
        ),
        null_gain_s_inv=_finite_number(
            table["null_gain_s_inv"],
            "controller.reactive_pose.null_gain_s_inv",
            nonnegative=True,
        ),
        dls_damping=_finite_number(
            table["dls_damping"],
            "controller.reactive_pose.dls_damping",
            positive=True,
        ),
        position_enabled=_boolean(
            table["position_enabled"],
            "controller.reactive_pose.position_enabled",
        ),
        orientation_enabled=_boolean(
            table["orientation_enabled"],
            "controller.reactive_pose.orientation_enabled",
        ),
        velocity_enabled=_boolean(
            table["velocity_enabled"],
            "controller.reactive_pose.velocity_enabled",
        ),
        velocity_feedforward_enabled=_boolean(
            table["velocity_feedforward_enabled"],
            "controller.reactive_pose.velocity_feedforward_enabled",
        ),
    )
    if config.position_enabled and config.kp_position_s_inv <= 0.0:
        raise ValueError(
            "controller.reactive_pose.kp_position_s_inv must be greater "
            "than zero when position control is enabled"
        )
    if config.orientation_enabled and config.kp_rotation_s_inv <= 0.0:
        raise ValueError(
            "controller.reactive_pose.kp_rotation_s_inv must be greater "
            "than zero when orientation control is enabled"
        )
    if config.velocity_enabled and (
        config.kd_position >= 1.0 or config.kd_rotation >= 1.0
    ):
        raise ValueError(
            "controller.reactive_pose Kd gains must be less than one "
            "when velocity feedback is enabled"
        )
    return config


def _parse_cylinder_keepout(table):
    """Mirror the C++ Options.cpp cylinder validation (lines 268-284).

    The bounds are checked whether or not the keep-out is enabled, exactly as
    the hardware controller does, so a disabled-but-wrong config still fails
    loudly instead of waiting until someone flips ``enabled``.
    """
    location = "cylinder_keepout"
    table = _require_table(table, location)
    _require_exact_keys(table, _CYLINDER_KEEPOUT_KEYS, location)
    config = CylinderKeepoutConfig(
        cylinder_keepout_enabled=_boolean(
            table["cylinder_keepout_enabled"],
            f"{location}.cylinder_keepout_enabled",
        ),
        cylinder_keepout_center_x_m=_finite_number(
            table["cylinder_keepout_center_x_m"],
            f"{location}.cylinder_keepout_center_x_m",
        ),
        cylinder_keepout_center_y_m=_finite_number(
            table["cylinder_keepout_center_y_m"],
            f"{location}.cylinder_keepout_center_y_m",
        ),
        cylinder_keepout_radius_m=_finite_number(
            table["cylinder_keepout_radius_m"],
            f"{location}.cylinder_keepout_radius_m",
            positive=True,
        ),
        cylinder_keepout_z_min_m=_finite_number(
            table["cylinder_keepout_z_min_m"],
            f"{location}.cylinder_keepout_z_min_m",
        ),
        cylinder_keepout_z_max_m=_finite_number(
            table["cylinder_keepout_z_max_m"],
            f"{location}.cylinder_keepout_z_max_m",
        ),
        cylinder_keepout_clearance_m=_finite_number(
            table["cylinder_keepout_clearance_m"],
            f"{location}.cylinder_keepout_clearance_m",
            nonnegative=True,
        ),
        cylinder_waypoint_tolerance_m=_finite_number(
            table["cylinder_waypoint_tolerance_m"],
            f"{location}.cylinder_waypoint_tolerance_m",
            positive=True,
        ),
    )
    if (
        config.cylinder_keepout_z_max_m
        <= config.cylinder_keepout_z_min_m
    ):
        raise ValueError(
            f"{location}.cylinder_keepout_z_max_m must be greater than "
            "cylinder_keepout_z_min_m"
        )
    return config


def _parse_human_safety(table):
    location = "human_safety"
    table = _require_table(table, location)
    _require_exact_keys(table, _HUMAN_SAFETY_KEYS, location)
    config = HumanSafetyConfig(
        enabled=_boolean(table["enabled"], f"{location}.enabled"),
        center_xy_torso_m=_vector(
            table["center_xy_torso_m"],
            2,
            f"{location}.center_xy_torso_m",
        ),
        radius_m=_finite_number(
            table["radius_m"], f"{location}.radius_m", positive=True
        ),
        z_min_torso_m=_finite_number(
            table["z_min_torso_m"], f"{location}.z_min_torso_m"
        ),
        z_max_torso_m=_finite_number(
            table["z_max_torso_m"], f"{location}.z_max_torso_m"
        ),
        clearance_m=_finite_number(
            table["clearance_m"],
            f"{location}.clearance_m",
            nonnegative=True,
        ),
        control_margin_m=_finite_number(
            table["control_margin_m"],
            f"{location}.control_margin_m",
            nonnegative=True,
        ),
        activation_distance_m=_finite_number(
            table["activation_distance_m"],
            f"{location}.activation_distance_m",
            positive=True,
        ),
        recovery_gain_s_inv=_finite_number(
            table["recovery_gain_s_inv"],
            f"{location}.recovery_gain_s_inv",
            positive=True,
        ),
        approach_velocity_damping=_finite_number(
            table["approach_velocity_damping"],
            f"{location}.approach_velocity_damping",
            nonnegative=True,
        ),
        projection_iterations=_positive_integer(
            table["projection_iterations"],
            f"{location}.projection_iterations",
        ),
        constraint_tolerance_m_s=_finite_number(
            table["constraint_tolerance_m_s"],
            f"{location}.constraint_tolerance_m_s",
            positive=True,
        ),
    )
    if config.z_max_torso_m <= config.z_min_torso_m:
        raise ValueError(
            f"{location}.z_max_torso_m must be greater than "
            "z_min_torso_m"
        )
    return config


def _parse_planning(table):
    location = "planning"
    table = _require_table(table, location)
    _require_exact_keys(table, _PLANNING_KEYS, location)
    return PlanningConfig(
        enabled=_boolean(table["enabled"], f"{location}.enabled"),
        arm=_choice(table["arm"], (*ARMS, "both"), f"{location}.arm"),
        waypoint_count=_positive_integer(
            table["waypoint_count"], f"{location}.waypoint_count"
        ),
        dense_samples=_positive_integer(
            table["dense_samples"], f"{location}.dense_samples"
        ),
        clearance_margin_m=_finite_number(
            table["clearance_margin_m"],
            f"{location}.clearance_margin_m",
            nonnegative=True,
        ),
        smoothness_weight=_finite_number(
            table["smoothness_weight"],
            f"{location}.smoothness_weight",
            positive=True,
        ),
        obstacle_weight=_finite_number(
            table["obstacle_weight"],
            f"{location}.obstacle_weight",
            positive=True,
        ),
        max_iterations=_positive_integer(
            table["max_iterations"], f"{location}.max_iterations"
        ),
        tool_radius_m=_finite_number(
            table["tool_radius_m"],
            f"{location}.tool_radius_m",
            nonnegative=True,
        ),
        deviation_weight=_finite_number(
            table["deviation_weight"],
            f"{location}.deviation_weight",
            positive=True,
        ),
        reach_allowance_m=_finite_number(
            table["reach_allowance_m"],
            f"{location}.reach_allowance_m",
            positive=True,
        ),
        lead_compensation_enabled=_boolean(
            table["lead_compensation_enabled"],
            f"{location}.lead_compensation_enabled",
        ),
        replan_clearance_trigger_m=_finite_number(
            table["replan_clearance_trigger_m"],
            f"{location}.replan_clearance_trigger_m",
            nonnegative=True,
        ),
        include_floor=_boolean(
            table["include_floor"], f"{location}.include_floor"
        ),
        floor_height_world_m=_finite_number(
            table["floor_height_world_m"],
            f"{location}.floor_height_world_m",
        ),
        include_torso_box=_boolean(
            table["include_torso_box"], f"{location}.include_torso_box"
        ),
        torso_box_half_extent_m=_vector(
            table["torso_box_half_extent_m"],
            3,
            f"{location}.torso_box_half_extent_m",
            positive=True,
        ),
        max_linear_speed_m_s=_finite_number(
            table["max_linear_speed_m_s"],
            f"{location}.max_linear_speed_m_s",
            positive=True,
        ),
        max_linear_acceleration_m_s2=_finite_number(
            table["max_linear_acceleration_m_s2"],
            f"{location}.max_linear_acceleration_m_s2",
            positive=True,
        ),
        max_angular_speed_rad_s=_finite_number(
            table["max_angular_speed_rad_s"],
            f"{location}.max_angular_speed_rad_s",
            positive=True,
        ),
        max_angular_acceleration_rad_s2=_finite_number(
            table["max_angular_acceleration_rad_s2"],
            f"{location}.max_angular_acceleration_rad_s2",
            positive=True,
        ),
    )


def _parse_target(table, side):
    location = f"targets.{side}"
    table = _require_table(table, location)
    actual = set(table)
    allowed = _TARGET_KEYS | {"trajectory"}
    missing = sorted(_TARGET_KEYS - actual)
    extra = sorted(actual - allowed)
    if missing or extra:
        raise ValueError(
            f"{location} keys differ; missing={missing}, extra={extra}"
        )
    frame = table["reference_frame"]
    if frame not in TARGET_FRAMES:
        raise ValueError(
            f"{location}.reference_frame must be one of {TARGET_FRAMES}"
        )
    return TargetConfig(
        reference_frame=frame,
        position_m=_vector(table["position_m"], 3, f"{location}.position_m"),
        rpy_rad=_vector(table["rpy_rad"], 3, f"{location}.rpy_rad"),
        trajectory=(
            None
            if "trajectory" not in table
            else _parse_target_trajectory(
                table["trajectory"], side
            )
        ),
    )


def _vector_norm(vector):
    return math.sqrt(sum(component * component for component in vector))


def _parse_trajectory_orientation(table, trajectory_location):
    location = f"{trajectory_location}.orientation"
    table = _require_table(table, location)
    policy = _choice(
        table.get("policy"),
        TRAJECTORY_ORIENTATION_POLICIES,
        f"{location}.policy",
    )
    expected_keys = (
        _FIXED_LOOK_AT_KEYS
        if policy == "look_at_fixed_world_point"
        else _SIM_OBJECT_LOOK_AT_KEYS
    )
    _require_exact_keys(table, expected_keys, location)
    tool_forward = _vector(
        table["tool_forward_axis"],
        3,
        f"{location}.tool_forward_axis",
    )
    tool_up = _vector(
        table["tool_up_axis"], 3, f"{location}.tool_up_axis"
    )
    world_up = _vector(
        table["world_up_direction"],
        3,
        f"{location}.world_up_direction",
    )
    forward_norm = _vector_norm(tool_forward)
    tool_up_norm = _vector_norm(tool_up)
    world_up_norm = _vector_norm(world_up)
    if forward_norm < 1e-9:
        raise ValueError(f"{location}.tool_forward_axis must be non-zero")
    if tool_up_norm < 1e-9:
        raise ValueError(f"{location}.tool_up_axis must be non-zero")
    if world_up_norm < 1e-9:
        raise ValueError(f"{location}.world_up_direction must be non-zero")
    cross = (
        tool_up[1] * tool_forward[2]
        - tool_up[2] * tool_forward[1],
        tool_up[2] * tool_forward[0]
        - tool_up[0] * tool_forward[2],
        tool_up[0] * tool_forward[1]
        - tool_up[1] * tool_forward[0],
    )
    if _vector_norm(cross) < 1e-8 * forward_norm * tool_up_norm:
        raise ValueError(
            f"{location}.tool_forward_axis and tool_up_axis "
            "must not be collinear"
        )
    return TrajectoryOrientationConfig(
        policy=policy,
        object_position_world_m=(
            _vector(
                table["object_position_world_m"],
                3,
                f"{location}.object_position_world_m",
            )
            if policy == "look_at_fixed_world_point"
            else None
        ),
        tool_forward_axis=tool_forward,
        tool_up_axis=tool_up,
        world_up_direction=world_up,
        object_body=(
            None
            if policy == "look_at_fixed_world_point"
            else _non_empty_string(
                table["object_body"], f"{location}.object_body"
            )
        ),
    )


def _parse_target_trajectory(table, side):
    location = f"targets.{side}.trajectory"
    table = _require_table(table, location)
    if "shape" in table:
        return _parse_legacy_target_trajectory(table, location)
    actual = set(table)
    missing = sorted(_TRAJECTORY_REQUIRED_KEYS - actual)
    extra = sorted(actual - _TRAJECTORY_ALLOWED_KEYS)
    if missing or extra:
        raise ValueError(
            f"{location} keys differ; missing={missing}, extra={extra}"
        )
    constraints_table = _require_table(
        table.get("constraints", {}), f"{location}.constraints"
    )
    constraint_extra = sorted(
        set(constraints_table) - _TRAJECTORY_CONSTRAINT_KEYS
    )
    if constraint_extra:
        raise ValueError(
            f"{location}.constraints has unknown keys: {constraint_extra}"
        )
    constraints = TrajectoryConstraintsConfig(
        **{
            key: _optional_positive(
                constraints_table, key, f"{location}.constraints"
            )
            for key in _TRAJECTORY_CONSTRAINT_KEYS
        }
    )
    raw_segments = table["segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"{location}.segments must be a non-empty array")
    segments = tuple(
        _parse_trajectory_segment(item, location, index)
        for index, item in enumerate(raw_segments)
    )
    orientation = (
        None
        if "orientation" not in table
        else _parse_trajectory_orientation(
            table["orientation"], location
        )
    )
    reference_frame = _choice(
        table["reference_frame"],
        TARGET_FRAMES,
        f"{location}.reference_frame",
    )
    if orientation is not None:
        if reference_frame != "world":
            raise ValueError(
                f"{location}.orientation requires "
                'reference_frame = "world"'
            )
        if any(
            segment.end_rpy_rad is not None
            or segment.rpy_rad is not None
            for segment in segments
        ):
            raise ValueError(
                f"{location}.orientation cannot be combined with "
                "per-segment end_rpy_rad or rpy_rad"
            )
    return TargetTrajectoryConfig(
        reference_frame=reference_frame,
        start=_choice(
            table["start"],
            TRAJECTORY_STARTS,
            f"{location}.start",
        ),
        loop=_boolean(table["loop"], f"{location}.loop"),
        open_live_path_plot=_boolean(
            table["open_live_path_plot"],
            f"{location}.open_live_path_plot",
        ),
        constraints=constraints,
        segments=segments,
        orientation=orientation,
    )


def _parse_legacy_target_trajectory(table, location):
    """Translate the original configured demo into the general segment model."""
    _require_exact_keys(table, _LEGACY_TRAJECTORY_KEYS, location)
    _choice(
        table["shape"],
        ("measured_start_displacement",),
        f"{location}.shape",
    )
    _choice(
        table["orientation_policy"],
        ("hold_initial",),
        f"{location}.orientation_policy",
    )
    displacement = _vector(
        table["displacement_m"], 3, f"{location}.displacement_m"
    )
    if all(value == 0.0 for value in displacement):
        raise ValueError(f"{location}.displacement_m must be non-zero")
    duration = _finite_number(
        table["leg_duration_s"],
        f"{location}.leg_duration_s",
        positive=True,
    )
    repeat = _boolean(
        table["repeat_out_and_back"],
        f"{location}.repeat_out_and_back",
    )
    outward = TrajectorySegmentConfig(
        type="line",
        duration_s=duration,
        displacement_m=displacement,
    )
    segments = (outward,)
    if repeat:
        segments += (
            TrajectorySegmentConfig(
                type="line",
                duration_s=duration,
                displacement_m=tuple(-value for value in displacement),
            ),
        )
    return TargetTrajectoryConfig(
        reference_frame=_choice(
            table["reference_frame"],
            TARGET_FRAMES,
            f"{location}.reference_frame",
        ),
        start="measured",
        loop=repeat,
        open_live_path_plot=_boolean(
            table["open_live_path_plot"],
            f"{location}.open_live_path_plot",
        ),
        constraints=TrajectoryConstraintsConfig(
            None, None, None, None
        ),
        segments=segments,
    )


def _parse_trajectory_segment(table, trajectory_location, index):
    location = f"{trajectory_location}.segments[{index}]"
    table = _require_table(table, location)
    if "type" not in table:
        raise ValueError(f"{location}.type is required")
    kind = _choice(
        table["type"], TRAJECTORY_SEGMENT_TYPES, f"{location}.type"
    )
    if kind == "hold":
        _require_exact_keys(table, {"type", "duration_s"}, location)
        return TrajectorySegmentConfig(
            type=kind,
            duration_s=_finite_number(
                table["duration_s"],
                f"{location}.duration_s",
                positive=True,
            ),
        )
    if kind == "line":
        allowed = {
            "type",
            "duration_s",
            "displacement_m",
            "end_position_m",
            "end_rpy_rad",
        }
        extra = sorted(set(table) - allowed)
        positions = {
            key for key in ("displacement_m", "end_position_m")
            if key in table
        }
        if extra or len(positions) != 1:
            raise ValueError(
                f"{location} line requires exactly one of "
                "displacement_m/end_position_m and no unknown keys; "
                f"extra={extra}"
            )
        displacement = (
            None
            if "displacement_m" not in table
            else _vector(
                table["displacement_m"],
                3,
                f"{location}.displacement_m",
            )
        )
        if (
            displacement is not None
            and all(value == 0.0 for value in displacement)
            and "end_rpy_rad" not in table
        ):
            raise ValueError(
                f"{location} zero displacement is a hold, not a line"
            )
        return TrajectorySegmentConfig(
            type=kind,
            duration_s=_optional_positive(table, "duration_s", location),
            displacement_m=displacement,
            end_position_m=(
                None
                if "end_position_m" not in table
                else _vector(
                    table["end_position_m"],
                    3,
                    f"{location}.end_position_m",
                )
            ),
            end_rpy_rad=(
                None
                if "end_rpy_rad" not in table
                else _vector(
                    table["end_rpy_rad"],
                    3,
                    f"{location}.end_rpy_rad",
                )
            ),
        )
    if kind == "waypoints":
        allowed = {
            "type",
            "durations_s",
            "offsets_m",
            "positions_m",
            "rpy_rad",
        }
        extra = sorted(set(table) - allowed)
        positions = {
            key for key in ("offsets_m", "positions_m") if key in table
        }
        if extra or len(positions) != 1:
            raise ValueError(
                f"{location} waypoints requires exactly one of "
                "offsets_m/positions_m and no unknown keys; extra={extra}"
            )
        points_key = next(iter(positions))
        points = _vectors(
            table[points_key],
            3,
            f"{location}.{points_key}",
            minimum_length=2,
        )
        durations = (
            None
            if "durations_s" not in table
            else tuple(
                _finite_number(
                    item,
                    f"{location}.durations_s[{duration_index}]",
                    positive=True,
                )
                for duration_index, item in enumerate(
                    table["durations_s"]
                )
            )
        )
        if durations is not None and len(durations) != len(points) - 1:
            raise ValueError(
                f"{location}.durations_s must have one entry per leg"
            )
        orientations = (
            None
            if "rpy_rad" not in table
            else _vectors(
                table["rpy_rad"],
                3,
                f"{location}.rpy_rad",
                minimum_length=2,
            )
        )
        if orientations is not None and len(orientations) != len(points):
            raise ValueError(
                f"{location}.rpy_rad must match the waypoint count"
            )
        return TrajectorySegmentConfig(
            type=kind,
            durations_s=durations,
            offsets_m=points if points_key == "offsets_m" else None,
            positions_m=points if points_key == "positions_m" else None,
            rpy_rad=orientations,
        )

    allowed = {
        "type",
        "duration_s",
        "radius_m",
        "plane",
        "normal",
        "start_direction",
        "revolutions",
        "clockwise",
        "end_rpy_rad",
    }
    required = {"type", "radius_m", "revolutions", "clockwise"}
    actual = set(table)
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    if missing or extra:
        raise ValueError(
            f"{location} circle keys differ; missing={missing}, extra={extra}"
        )
    has_plane = "plane" in table
    has_normal = "normal" in table
    has_start_direction = "start_direction" in table
    if has_plane and (has_normal or has_start_direction):
        raise ValueError(
            f"{location} circle requires either plane or "
            "normal/start_direction, not both"
        )
    if not has_plane and not (has_normal and has_start_direction):
        raise ValueError(
            f"{location} circle requires plane or both "
            "normal and start_direction"
        )
    if has_plane:
        plane = _choice(
            table["plane"],
            tuple(_CIRCLE_PLANE_DIRECTIONS),
            f"{location}.plane",
        )
        normal, start_direction = _CIRCLE_PLANE_DIRECTIONS[plane]
    else:
        normal = _vector(table["normal"], 3, f"{location}.normal")
        start_direction = _vector(
            table["start_direction"],
            3,
            f"{location}.start_direction",
        )
    return TrajectorySegmentConfig(
        type=kind,
        duration_s=_optional_positive(table, "duration_s", location),
        radius_m=_finite_number(
            table["radius_m"], f"{location}.radius_m", positive=True
        ),
        normal=normal,
        start_direction=start_direction,
        revolutions=_positive_integer(
            table["revolutions"], f"{location}.revolutions"
        ),
        clockwise=_boolean(
            table["clockwise"], f"{location}.clockwise"
        ),
        end_rpy_rad=(
            None
            if "end_rpy_rad" not in table
            else _vector(
                table["end_rpy_rad"], 3, f"{location}.end_rpy_rad"
            )
        ),
    )


def _parse_simulation(table):
    location = "simulation"
    table = _require_table(table, location)
    actual = set(table)
    missing = sorted(_SIMULATION_REQUIRED_KEYS - actual)
    extra = sorted(actual - _SIMULATION_ALLOWED_KEYS)
    if missing or extra:
        raise ValueError(
            f"{location} keys differ; missing={missing}, extra={extra}"
        )
    positions = _require_table(
        table["initial_joint_position_rad"],
        "simulation.initial_joint_position_rad",
    )
    extra = sorted(set(positions) - set(ARMS))
    if extra:
        raise ValueError(
            "simulation.initial_joint_position_rad has unknown arms: "
            f"{extra}"
        )
    values = {
        side: (
            None
            if side not in positions
            else _vector(
                positions[side],
                7,
                f"simulation.initial_joint_position_rad.{side}",
            )
        )
        for side in ARMS
    }
    object_motion = None
    if "look_at_object_motion" in table:
        motion_location = f"{location}.look_at_object_motion"
        motion = _require_table(
            table["look_at_object_motion"], motion_location
        )
        _require_exact_keys(
            motion, _LOOK_AT_OBJECT_MOTION_KEYS, motion_location
        )
        amplitude = _vector(
            motion["linear_amplitude_m"],
            3,
            f"{motion_location}.linear_amplitude_m",
        )
        if _vector_norm(amplitude) < 1e-12:
            raise ValueError(
                f"{motion_location}.linear_amplitude_m must be non-zero"
            )
        object_motion = LookAtObjectMotionConfig(
            body_name=_non_empty_string(
                motion["body_name"], f"{motion_location}.body_name"
            ),
            home_position_world_m=_vector(
                motion["home_position_world_m"],
                3,
                f"{motion_location}.home_position_world_m",
            ),
            linear_amplitude_m=amplitude,
            linear_frequency_hz=_finite_number(
                motion["linear_frequency_hz"],
                f"{motion_location}.linear_frequency_hz",
                positive=True,
            ),
        )
    return SimulationConfig(
        right_initial_joint_position_rad=values["right"],
        left_initial_joint_position_rad=values["left"],
        look_at_object_motion=object_motion,
    )


def load_config(path=DEFAULT_CONFIG_PATH):
    path = Path(path).resolve()
    raw = path.read_bytes()
    parsed = tomllib.loads(raw.decode("utf-8"))
    _require_exact_keys(parsed, _ROOT_KEYS, "root")

    run = _require_table(parsed["run"], "run")
    _require_exact_keys(run, _RUN_KEYS, "run")

    controller = _require_table(parsed["controller"], "controller")
    _require_exact_keys(controller, _CONTROLLER_KEYS, "controller")

    limits = _require_table(parsed["limits"], "limits")
    _require_exact_keys(limits, _LIMIT_KEYS, "limits")

    cylinder_keepout = _parse_cylinder_keepout(parsed["cylinder_keepout"])
    human_safety = _parse_human_safety(parsed["human_safety"])
    planning = _parse_planning(parsed["planning"])

    targets = _require_table(parsed["targets"], "targets")
    _require_exact_keys(targets, set(ARMS), "targets")

    return ProjectConfig(
        run=RunConfig(
            nominal_dt_s=_finite_number(
                run["nominal_dt_s"], "run.nominal_dt_s", positive=True
            ),
            arm=_choice(run["arm"], (*ARMS, "both"), "run.arm"),
        ),
        reactive_pose=_parse_reactive_pose(controller["reactive_pose"]),
        limits=LimitConfig(
            joint_velocity_rad_s=_vector(
                limits["joint_velocity_rad_s"],
                7,
                "limits.joint_velocity_rad_s",
                positive=True,
            ),
            position_lead_rad=_finite_number(
                limits["position_lead_rad"],
                "limits.position_lead_rad",
                positive=True,
            ),
        ),
        cylinder_keepout=cylinder_keepout,
        human_safety=human_safety,
        planning=planning,
        right_target=_parse_target(targets["right"], "right"),
        left_target=_parse_target(targets["left"], "left"),
        simulation=_parse_simulation(parsed["simulation"]),
        source_path=str(path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def control_with_legacy_overrides(control, overrides):
    """Return a new immutable controller config for offline sweep episodes."""
    if overrides is None:
        return control
    names = {
        "KP_POS": "kp_position_s_inv",
        "KP_ROT": "kp_rotation_s_inv",
        "KD_POS": "kd_position",
        "KD_ROT": "kd_rotation",
        "K_NULL": "null_gain_s_inv",
        "DAMPING": "dls_damping",
    }
    unknown = set(overrides) - set(names)
    if unknown:
        raise ValueError(f"unknown gain override: {sorted(unknown)!r}")
    values = {
        names[name]: _finite_number(
            value,
            name,
            positive=(name == "DAMPING"),
            nonnegative=(name != "DAMPING"),
        )
        for name, value in overrides.items()
    }
    return replace(control, **values)


def legacy_gain_dict(control):
    return {
        "KP_POS": control.kp_position_s_inv,
        "KP_ROT": control.kp_rotation_s_inv,
        "KD_POS": control.kd_position,
        "KD_ROT": control.kd_rotation,
        "K_NULL": control.null_gain_s_inv,
        "DAMPING": control.dls_damping,
    }


def effective_config_dict(config):
    return {
        "run": asdict(config.run),
        "controller": {
            "reactive_pose": asdict(config.reactive_pose),
        },
        "limits": asdict(config.limits),
        "cylinder_keepout": asdict(config.cylinder_keepout),
        "human_safety": asdict(config.human_safety),
        "planning": asdict(config.planning),
        "targets": {
            "right": asdict(config.right_target),
            "left": asdict(config.left_target),
        },
        "simulation": asdict(config.simulation),
    }


def effective_config_json(config):
    return json.dumps(
        effective_config_dict(config),
        indent=2,
        sort_keys=True,
    )


def print_effective_config(config):
    print("Effective control configuration:")
    print(effective_config_json(config))
    print(f"config_sha256={config.source_sha256}")


CONFIG = load_config()
