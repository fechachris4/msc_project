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


@dataclass(frozen=True)
class RunConfig:
    nominal_dt_s: float


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
class TargetConfig:
    reference_frame: str
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]


@dataclass(frozen=True)
class ProjectConfig:
    run: RunConfig
    reactive_pose: ReactivePoseConfig
    limits: LimitConfig
    right_target: TargetConfig
    left_target: TargetConfig
    source_path: str
    source_sha256: str

    def target(self, side):
        if side == "right":
            return self.right_target
        if side == "left":
            return self.left_target
        raise ValueError(f"unknown arm: {side!r}")


_ROOT_KEYS = {"run", "controller", "limits", "targets"}
_RUN_KEYS = {"nominal_dt_s"}
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
_TARGET_KEYS = {"reference_frame", "position_m", "rpy_rad"}


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


def _vector(value, size, location, *, positive=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{location} must contain exactly {size} numbers")
    return tuple(
        _finite_number(item, f"{location}[{index}]", positive=positive)
        for index, item in enumerate(value)
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


def _parse_target(table, side):
    location = f"targets.{side}"
    table = _require_table(table, location)
    _require_exact_keys(table, _TARGET_KEYS, location)
    frame = table["reference_frame"]
    if frame not in TARGET_FRAMES:
        raise ValueError(
            f"{location}.reference_frame must be one of {TARGET_FRAMES}"
        )
    return TargetConfig(
        reference_frame=frame,
        position_m=_vector(table["position_m"], 3, f"{location}.position_m"),
        rpy_rad=_vector(table["rpy_rad"], 3, f"{location}.rpy_rad"),
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

    targets = _require_table(parsed["targets"], "targets")
    _require_exact_keys(targets, set(ARMS), "targets")

    return ProjectConfig(
        run=RunConfig(
            nominal_dt_s=_finite_number(
                run["nominal_dt_s"], "run.nominal_dt_s", positive=True
            )
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
        right_target=_parse_target(targets["right"], "right"),
        left_target=_parse_target(targets["left"], "left"),
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
        "targets": {
            "right": asdict(config.right_target),
            "left": asdict(config.left_target),
        },
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
