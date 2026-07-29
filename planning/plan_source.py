"""Deliver an optimised path through the existing target-source seam.

This is the only file that touches the controller boundary, and it does
so without modifying it: ``PlannedDualArmSource`` satisfies the existing
``DualArmTargetSource`` protocol (``controller/trajectory.py:378-383``),
so ``ReactivePositionRunner`` accepts it as its ``source_targets``
argument with no change to any protected file.

Two responsibilities live here because they cannot live in the
controller.

PHASE OWNERSHIP.  The Runner's ``elapsed_time_s`` origin is frozen in
``start()`` and can never be reset (``controller/runner.py:142,244``).
A source that owned no phase would, after a replan, be sampled at the
OLD elapsed time and clamp straight to its final waypoint.  This source
therefore treats the Runner's elapsed time as a monotonic tick and maps
it onto its own plan clock, so a replan is continuous by construction.

LAG COMPENSATION.  The control law is
``task_twist = Kp*pose_error + Kd*(twist_ref - twist_measured)``
(``controller/reactive_controller.py:142-157``).  The Kd term multiplies
velocity ERROR, which vanishes in steady state, so a moving reference is
tracked with a systematic lag.  Closing the velocity-resolved loop gives

    tau * edot + e = rdot / Kp,      tau = (1 + Kd) / Kp

so the steady-state lag is ``rdot / Kp`` and is independent of Kd.

The prefilter is derived against what this source actually delivers.  It
emits the leaded POSE but passes the PLANNED twist ``pdot`` through
unchanged, because the controller's Kd term wants the true reference
velocity.  Substituting that into the control law with ``e = p - x``:

    (1 + Kd) * xdot = Kp * (p + L - x) + Kd * pdot
    =>  ((1 + Kd) / Kp) * edot + e = pdot/Kp - L

so ``L = pdot / Kp`` drives the deviation from the PLAN to zero
exactly — first order is not an approximation here, it is the exact
inverse.  An acceleration term would be injected error, not a
refinement: with the shipped gains it was measured making tracking about
30x worse (10.5 mm instead of 0.34 mm on a 3 s path).

This is a prefilter on the reference, not a change to the controller; it
is switchable so the uncompensated reactive baseline remains measurable.
"""

from dataclasses import dataclass

import numpy as np

from controller.state import (
    DualArmFramedTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)
from controller.trajectory import (
    KinematicTargetSample,
    TargetSource,
    TrajectoryRateBounds,
    _kinematic_sample,
    _rotation_from_vector,
)


def _phase_origin(value):
    """The Runner's clock is non-negative, so a bad origin is a caller bug.

    ``max(0.0, nan)`` returns 0.0 in Python, so an unchecked NaN origin
    would silently pin the plan at t=0 forever instead of failing.
    """
    origin = float(value)
    if not np.isfinite(origin) or origin < 0.0:
        raise ValueError("phase origin must be finite and non-negative")
    return origin


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
        delivers the PLANNED twist unchanged, the closed loop cancels
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


@dataclass(slots=True)
class PlannedArmPath:
    """One arm's plan, owning its own phase on the Runner's clock."""

    source: TargetSource
    lead: LeadCompensation
    _phase_origin_s: float = 0.0

    def __post_init__(self):
        if not isinstance(self.source, TargetSource):
            raise TypeError("source must satisfy TargetSource")
        if not isinstance(self.lead, LeadCompensation):
            raise TypeError("lead must be a LeadCompensation")
        self._phase_origin_s = _phase_origin(self._phase_origin_s)

    @property
    def reference_frame(self):
        return self.source.reference_frame

    @property
    def phase_origin_s(self):
        return self._phase_origin_s

    def plan_time_s(self, elapsed_time_s):
        return max(0.0, float(elapsed_time_s) - self._phase_origin_s)

    def replace(self, source, elapsed_time_s):
        """Adopt a new plan whose local time starts now."""
        if not isinstance(source, TargetSource):
            raise TypeError("source must satisfy TargetSource")
        self.source = source
        self._phase_origin_s = _phase_origin(elapsed_time_s)

    def planned_sample(self, elapsed_time_s):
        """The plan BEFORE lead compensation, for analysis and plots."""
        return _kinematic_sample(
            self.source, self.plan_time_s(elapsed_time_s)
        )

    # No ``sample_kinematics`` is defined on purpose.  Defining one that
    # returned the UNCOMPENSATED plan would make every consumer going
    # through ``controller.trajectory._kinematic_sample`` (TargetProgram's
    # C2 boundary check, PeriodicTargetSource's closure check) validate a
    # curve that is not the one delivered.  Omitting it makes those
    # helpers fall back to ``sample()``, so both boundaries agree by
    # construction.

    def sample(self, elapsed_time_s):
        return self.lead.apply(self.planned_sample(elapsed_time_s))

    def maximum_rates(self):
        # A held arm wraps a StaticTargetSource, which has no rate bounds;
        # mirror the defensive pattern used by ``duration_s`` below rather
        # than raising AttributeError through TargetProgram's hasattr guard.
        rates = getattr(self.source, "maximum_rates", None)
        if rates is None:
            return TrajectoryRateBounds(0.0, 0.0, 0.0, 0.0)
        return rates()

    @property
    def duration_s(self):
        return getattr(self.source, "duration_s", 0.0)


@dataclass(slots=True)
class PlannedDualArmSource:
    """Two planned arms behind the single Runner sampling boundary."""

    right: PlannedArmPath
    left: PlannedArmPath

    def __post_init__(self):
        for name in ("right", "left"):
            if not isinstance(getattr(self, name), PlannedArmPath):
                raise TypeError(f"{name} must be a PlannedArmPath")

    def for_arm(self, side):
        if side == "right":
            return self.right
        if side == "left":
            return self.left
        raise ValueError(f"unknown arm: {side!r}")

    def sample(self, elapsed_time_s):
        elapsed = float(elapsed_time_s)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_time_s must be finite and non-negative")
        return DualArmFramedTargets(
            right=self.right.sample(elapsed),
            left=self.left.sample(elapsed),
        )

    def planned_targets(self, elapsed_time_s):
        """Uncompensated plan for both arms — the comparison reference."""
        elapsed = float(elapsed_time_s)
        return DualArmFramedTargets(
            right=self.right.planned_sample(elapsed).target,
            left=self.left.planned_sample(elapsed).target,
        )


def hold_source(pose_world, frame=TargetFrame.WORLD):
    """A zero-twist source used for the arm that is not being planned."""
    from controller.trajectory import StaticTargetSource

    return StaticTargetSource(
        FramedTarget(frame, pose_world, Twist.zero())
    )
