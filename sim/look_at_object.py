"""Scripted world-frame mocap object for look-at orientation experiments."""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from controller.look_at import WorldPointKinematics
from runtime_config import LookAtObjectMotionConfig


_TIME_TOLERANCE_S = 1e-12


def _elapsed_time(value):
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(
            "elapsed_time_s must be finite and non-negative"
        )
    return result


def sinusoidal_point_kinematics(config, elapsed_time_s):
    """Configured point position, velocity, and acceleration in world."""
    if not isinstance(config, LookAtObjectMotionConfig):
        raise TypeError("config must be LookAtObjectMotionConfig")
    elapsed = _elapsed_time(elapsed_time_s)
    home = np.asarray(config.home_position_world_m, dtype=float)
    amplitude = np.asarray(config.linear_amplitude_m, dtype=float)
    angular_frequency = 2.0 * np.pi * config.linear_frequency_hz
    phase = angular_frequency * elapsed
    return WorldPointKinematics(
        home + amplitude * np.sin(phase),
        amplitude * angular_frequency * np.cos(phase),
        -amplitude * angular_frequency**2 * np.sin(phase),
    )


@dataclass(slots=True)
class SimulatedMocapPointSource:
    """Drive and sample one named MuJoCo mocap body once per visible cycle."""

    backend: object
    config: LookAtObjectMotionConfig
    _mocap_index: int = field(init=False, repr=False)
    _applied_elapsed_time_s: float | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self):
        if not isinstance(self.config, LookAtObjectMotionConfig):
            raise TypeError("config must be LookAtObjectMotionConfig")
        if not hasattr(self.backend, "model") or not hasattr(
            self.backend, "data"
        ):
            raise TypeError("backend must expose MuJoCo model and data")
        body_id = mujoco.mj_name2id(
            self.backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.config.body_name,
        )
        if body_id < 0:
            raise ValueError(
                f"look-at object body {self.config.body_name!r} "
                "is not in the MuJoCo scene"
            )
        mocap_index = int(
            self.backend.model.body_mocapid[body_id]
        )
        if mocap_index < 0:
            raise ValueError(
                f"look-at object body {self.config.body_name!r} "
                "must be a mocap body"
            )
        self._mocap_index = mocap_index

    @property
    def body_name(self):
        return self.config.body_name

    def apply(self, elapsed_time_s):
        """Move the mocap object for the current visible control cycle."""
        elapsed = _elapsed_time(elapsed_time_s)
        kinematics = sinusoidal_point_kinematics(
            self.config, elapsed
        )
        self.backend.data.mocap_pos[self._mocap_index] = (
            kinematics.position_world_m
        )
        self._applied_elapsed_time_s = elapsed

    def sample_kinematics(self, elapsed_time_s):
        """Read current simulator position and attach analytic derivatives."""
        elapsed = _elapsed_time(elapsed_time_s)
        if (
            self._applied_elapsed_time_s is None
            or abs(self._applied_elapsed_time_s - elapsed)
            > _TIME_TOLERANCE_S
        ):
            raise RuntimeError(
                "look-at object must be updated for the current "
                "visible cycle before target sampling"
            )
        analytic = sinusoidal_point_kinematics(
            self.config, elapsed
        )
        return WorldPointKinematics(
            self.backend.data.mocap_pos[self._mocap_index],
            analytic.velocity_world_m_s,
            analytic.acceleration_world_m_s2,
        )
