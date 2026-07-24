"""Stateful joint-velocity limiting and position-command integration."""

from dataclasses import dataclass

import numpy as np


def _array(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite shape-(7,) array")
    copied = result.copy()
    return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(
        copied.shape
    )


@dataclass(frozen=True, slots=True)
class PositionActuationLimits:
    velocity_rad_s: np.ndarray
    lead_rad: float
    lower_position_rad: np.ndarray
    upper_position_rad: np.ndarray

    def __post_init__(self):
        velocity = _array(self.velocity_rad_s, "velocity_rad_s")
        if np.any(velocity <= 0.0):
            raise ValueError("velocity_rad_s must be positive")
        lead = float(self.lead_rad)
        if not np.isfinite(lead) or lead <= 0.0:
            raise ValueError("lead_rad must be finite and positive")
        lower = np.asarray(self.lower_position_rad, dtype=float)
        upper = np.asarray(self.upper_position_rad, dtype=float)
        if lower.shape != (7,) or upper.shape != (7,):
            raise ValueError("position bounds must be shape-(7,) arrays")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("position bounds must not contain NaN")
        if np.any(lower > upper):
            raise ValueError("lower_position_rad must not exceed upper")
        lower = lower.copy()
        upper = upper.copy()
        lower = np.frombuffer(lower.tobytes(), dtype=lower.dtype)
        upper = np.frombuffer(upper.tobytes(), dtype=upper.dtype)
        object.__setattr__(self, "velocity_rad_s", velocity)
        object.__setattr__(self, "lead_rad", lead)
        object.__setattr__(self, "lower_position_rad", lower)
        object.__setattr__(self, "upper_position_rad", upper)


@dataclass(frozen=True, slots=True)
class PositionActuation:
    qdot_speed_clipped: np.ndarray
    qdot_effective: np.ndarray
    command_before_rad: np.ndarray
    command_after_rad: np.ndarray
    speed_saturated: np.ndarray
    lead_clamped: np.ndarray
    range_clamped: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name)).copy()
            value = np.frombuffer(
                value.tobytes(), dtype=value.dtype
            ).reshape(value.shape)
            object.__setattr__(self, name, value)


class PositionIntegrator:
    """Persistent commanded position; reset only by reconstruction."""

    def __init__(self, measured_position_rad, limits):
        if not isinstance(limits, PositionActuationLimits):
            raise TypeError("limits must be PositionActuationLimits")
        self._limits = limits
        self._command_rad = _array(
            measured_position_rad, "measured_position_rad").copy()

    @property
    def command_rad(self):
        return _array(self._command_rad, "command_rad")

    def step(self, measured_position_rad, requested_velocity_rad_s, dt_s):
        measured = _array(measured_position_rad, "measured_position_rad")
        requested = _array(
            requested_velocity_rad_s, "requested_velocity_rad_s")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        qdot_speed_clipped = np.clip(
            requested,
            -self._limits.velocity_rad_s,
            self._limits.velocity_rad_s,
        )
        speed_saturated = requested != qdot_speed_clipped
        command_before = self._command_rad.copy()
        integrated = command_before + qdot_speed_clipped * dt
        lead_limited = np.clip(
            integrated,
            measured - self._limits.lead_rad,
            measured + self._limits.lead_rad,
        )
        lead_clamped = integrated != lead_limited
        command_after = np.clip(
            lead_limited,
            self._limits.lower_position_rad,
            self._limits.upper_position_rad,
        )
        range_clamped = lead_limited != command_after
        self._command_rad = command_after.copy()
        return PositionActuation(
            qdot_speed_clipped=qdot_speed_clipped,
            qdot_effective=(command_after - command_before) / dt,
            command_before_rad=command_before,
            command_after_rad=command_after,
            speed_saturated=speed_saturated,
            lead_clamped=lead_clamped,
            range_clamped=range_clamped,
        )
