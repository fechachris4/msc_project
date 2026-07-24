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


@dataclass(frozen=True)
class LimitConfig:
    joint_velocity_rad_s: tuple[float, ...]
    position_lead_rad: float


@dataclass(frozen=True)
class CylinderKeepoutConfig:
    """End-effector keep-out cylinder, one-to-one with the C++ EffectiveConfig.

    Field names match ``basic_control/src/app/Options.h`` exactly so the two
    schemas stay comparable. Lengths are metres in the arm base frame; the
    cylinder axis is that frame's +z.
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
class TargetTrajectoryConfig:
    reference_frame: str
    start: str
    loop: bool
    open_live_path_plot: bool
    constraints: TrajectoryConstraintsConfig
    segments: tuple[TrajectorySegmentConfig, ...]


@dataclass(frozen=True)
class TargetConfig:
    reference_frame: str
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]
    trajectory: TargetTrajectoryConfig | None


@dataclass(frozen=True)
class SimulationConfig:
    right_initial_joint_position_rad: tuple[float, ...] | None
    left_initial_joint_position_rad: tuple[float, ...] | None

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
    "targets",
    "simulation",
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
_TARGET_KEYS = {"reference_frame", "position_m", "rpy_rad"}
_TRAJECTORY_REQUIRED_KEYS = {
    "reference_frame",
    "start",
    "loop",
    "open_live_path_plot",
    "segments",
}
_TRAJECTORY_ALLOWED_KEYS = _TRAJECTORY_REQUIRED_KEYS | {"constraints"}
_TRAJECTORY_CONSTRAINT_KEYS = {
    "max_linear_speed_m_s",
    "max_linear_acceleration_m_s2",
    "max_angular_speed_rad_s",
    "max_angular_acceleration_rad_s2",
}
_LEGACY_TRAJECTORY_KEYS = {
    "shape",
    "reference_frame",
    "displacement_m",
    "leg_duration_s",
    "repeat_out_and_back",
    "orientation_policy",
    "open_live_path_plot",
}
_SIMULATION_KEYS = {"initial_joint_position_rad"}


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
    return TargetTrajectoryConfig(
        reference_frame=_choice(
            table["reference_frame"],
            TARGET_FRAMES,
            f"{location}.reference_frame",
        ),
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
        "normal",
        "start_direction",
        "revolutions",
        "clockwise",
        "end_rpy_rad",
    }
    required = {
        "type",
        "radius_m",
        "normal",
        "start_direction",
        "revolutions",
        "clockwise",
    }
    actual = set(table)
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    if missing or extra:
        raise ValueError(
            f"{location} circle keys differ; missing={missing}, extra={extra}"
        )
    return TrajectorySegmentConfig(
        type=kind,
        duration_s=_optional_positive(table, "duration_s", location),
        radius_m=_finite_number(
            table["radius_m"], f"{location}.radius_m", positive=True
        ),
        normal=_vector(table["normal"], 3, f"{location}.normal"),
        start_direction=_vector(
            table["start_direction"],
            3,
            f"{location}.start_direction",
        ),
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
    table = _require_table(table, "simulation")
    _require_exact_keys(table, _SIMULATION_KEYS, "simulation")
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
    return SimulationConfig(
        right_initial_joint_position_rad=values["right"],
        left_initial_joint_position_rad=values["left"],
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
