"""Reference conditioning for the reactive controller's known tracking lag.

This is a property of the CONTROLLER, not of any particular target source:
the control law is ``task_twist = Kp*pose_error + Kd*(twist_ref -
twist_measured)`` (``controller/reactive_controller.py:142-157``).  The Kd
term multiplies velocity ERROR, which vanishes in steady state, so ANY
moving reference is tracked with a systematic lag.  Closing the
velocity-resolved loop gives

    tau * edot + e = rdot / Kp,      tau = (1 + Kd) / Kp

so the steady-state lag is ``rdot / Kp`` and is independent of Kd.

The prefilter is derived against what the conditioned source delivers.  It
emits the leaded POSE but passes the reference twist through unchanged,
because the controller's Kd term wants the true reference velocity.
Substituting that into the control law with ``e = p - x``:

    (1 + Kd) * xdot = Kp * (p + L - x) + Kd * pdot
    =>  ((1 + Kd) / Kp) * edot + e = pdot/Kp - L

so ``L = pdot / Kp`` drives the deviation from the reference to zero
exactly — first order is not an approximation here, it is the exact
inverse.  An acceleration term would be injected error, not a refinement:
with the shipped gains it was measured making tracking about 30x worse
(10.5 mm instead of 0.34 mm on a 3 s path).

This is a prefilter on the reference, not a change to the controller; it
is switchable so the uncompensated reactive baseline remains measurable.
"""

from dataclasses import dataclass

import numpy as np

from controller.trajectory import (
    KinematicTargetSample,
    TargetSource,
    TrajectoryRateBounds,
    _kinematic_sample,
    _rotation_from_vector,
)
from controller.state import FramedTarget, Pose


@dataclass(frozen=True, slots=True)
class LeadCompensation:
    """Inverse-model prefilter for the reactive controller's known lag."""

    kp_position_s_inv: float
    kd_position: float
    kp_rotation_s_inv: float
    kd_rotation: float
    enabled: bool = True

    def __post_init__(self):
        for name in (
            "kp_position_s_inv",
            "kd_position",
            "kp_rotation_s_inv",
            "kd_rotation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.enabled and (
            self.kp_position_s_inv <= 0.0 or self.kp_rotation_s_inv <= 0.0
        ):
            raise ValueError(
                "lead compensation needs positive position and rotation gains"
            )
        object.__setattr__(self, "enabled", bool(self.enabled))

    @classmethod
    def from_config(cls, reactive_pose_config, enabled=True):
        return cls(
            reactive_pose_config.kp_position_s_inv,
            reactive_pose_config.kd_position
            if reactive_pose_config.velocity_enabled else 0.0,
            reactive_pose_config.kp_rotation_s_inv,
            reactive_pose_config.kd_rotation
            if reactive_pose_config.velocity_enabled else 0.0,
            enabled,
        )

    def position_lead_m(self, linear_velocity, linear_acceleration=None):
        """Exact first-order inverse; see the derivation in the module docstring.

        ``linear_acceleration`` is accepted and ignored: because ``apply``
        delivers the reference twist unchanged, the closed loop cancels
        exactly at ``pdot / Kp`` and any acceleration term is injected
        error, not a correction.
        """
        if not self.enabled:
            return np.zeros(3)
        return (
            np.asarray(linear_velocity, dtype=float)
            / self.kp_position_s_inv
        )

    def rotation_lead_rad(self, angular_velocity, angular_acceleration=None):
        """Spatial rotation lead; the acceleration argument is ignored."""
        if not self.enabled:
            return np.zeros(3)
        return (
            np.asarray(angular_velocity, dtype=float)
            / self.kp_rotation_s_inv
        )

    def apply(self, sample):
        """Return the prefiltered target; twist is deliberately unchanged.

        The controller's Kd term wants the TRUE reference velocity, so
        only the pose carries the lead.
        """
        if not isinstance(sample, KinematicTargetSample):
            raise TypeError("sample must be a KinematicTargetSample")
        if not self.enabled:
            return sample.target
        target = sample.target
        position = target.pose.position_m + self.position_lead_m(
            target.twist.linear_m_s, sample.linear_acceleration_m_s2
        )
        rotation_lead = self.rotation_lead_rad(
            target.twist.angular_rad_s, sample.angular_acceleration_rad_s2
        )
        rotation = _rotation_from_vector(rotation_lead) @ target.pose.rotation
        return FramedTarget(
            target.reference_frame, Pose(position, rotation), target.twist
        )


@dataclass(frozen=True, slots=True)
class LeadCompensatedSource:
    """One arm's conditioned reference: the inner source plus its lead.

    No ``sample_kinematics`` is defined on purpose.  Defining one that
    returned the UNCOMPENSATED reference would make every consumer going
    through ``controller.trajectory._kinematic_sample`` (TargetProgram's
    C2 boundary check, PeriodicTargetSource's closure check) validate a
    curve that is not the one delivered.  Omitting it makes those helpers
    fall back to ``sample()``, so both boundaries agree by construction.
    """

    source: TargetSource
    lead: LeadCompensation

    def __post_init__(self):
        if not isinstance(self.source, TargetSource):
            raise TypeError("source must satisfy TargetSource")
        if not isinstance(self.lead, LeadCompensation):
            raise TypeError("lead must be a LeadCompensation")

    @property
    def reference_frame(self):
        return self.source.reference_frame

    @property
    def duration_s(self):
        return getattr(self.source, "duration_s", 0.0)

    def planned_sample(self, elapsed_time_s):
        """The reference BEFORE lead compensation, for analysis and plots."""
        return _kinematic_sample(self.source, elapsed_time_s)

    def sample(self, elapsed_time_s):
        return self.lead.apply(self.planned_sample(elapsed_time_s))

    def maximum_rates(self):
        # A held inner source has no rate bounds; mirror the defensive
        # pattern used by ``duration_s`` above rather than raising
        # AttributeError through TargetProgram's hasattr guard.
        rates = getattr(self.source, "maximum_rates", None)
        if rates is None:
            return TrajectoryRateBounds(0.0, 0.0, 0.0, 0.0)
        return rates()
