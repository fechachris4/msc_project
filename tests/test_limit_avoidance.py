"""Deadband joint-limit avoidance: zero in the working range, linear
inward push inside the activation zone [limit - zone, limit]."""

import numpy as np
import pytest

from controller.reactive_controller import (
    JointLimitAvoidance,
    solve_reactive_velocity,
)
from runtime_config import ReactivePoseConfig


def make_config(**overrides):
    values = dict(
        kp_position_s_inv=1.0,
        kp_rotation_s_inv=1.0,
        kd_position=0.3,
        kd_rotation=0.3,
        null_gain_s_inv=2.0,
        limit_avoid_zone_rad=np.deg2rad(20.0),
        dls_damping=0.1,
        position_enabled=True,
        orientation_enabled=True,
        velocity_enabled=False,
    )
    values.update(overrides)
    return ReactivePoseConfig(**values)


LIMIT = np.deg2rad(np.array([0.0, 126.9, 0.0, 145.0, 0.0, 118.0, 0.0]))
ZONE = np.deg2rad(20.0)
ZERO3 = np.zeros(3)


def full_rank_jacobian():
    rows, cols = np.meshgrid(np.arange(6), np.arange(7), indexing="ij")
    return np.sin(1.0 + 3.0 * rows + 7.0 * cols) + 2.0 * (rows == cols)


def null_part(q):
    solve = solve_reactive_velocity(
        full_rank_jacobian(), ZERO3, ZERO3, ZERO3, ZERO3,
        q, LIMIT, ZONE, 2.0, make_config(),
    )
    return solve.qdot_null_objective, solve.qdot_null_projected


def test_objective_zero_everywhere_inside_zone():
    q = np.deg2rad(np.array([170.0, 60.0, -350.0, 110.0, 40.0, -90.0, 720.0]))
    objective, projected = null_part(q)
    assert np.all(objective == 0.0)
    assert np.all(projected == 0.0)


def test_push_is_inward_on_both_sides():
    q = np.zeros(7)
    q[3] = np.deg2rad(135.0)   # j4, 10 deg into its zone (entry 125)
    objective, _ = null_part(q)
    assert objective[3] == pytest.approx(-2.0 * np.deg2rad(10.0))
    q[3] = np.deg2rad(-135.0)
    objective, _ = null_part(q)
    assert objective[3] == pytest.approx(2.0 * np.deg2rad(10.0))


def test_magnitude_at_limit_is_gain_times_zone():
    q = np.zeros(7)
    q[5] = np.deg2rad(118.0)   # j6 exactly at its software limit
    objective, _ = null_part(q)
    assert objective[5] == pytest.approx(-2.0 * ZONE)


def test_unbounded_joints_never_push():
    q = np.zeros(7)
    q[0] = np.deg2rad(359.0)   # continuous joint, any position
    objective, _ = null_part(q)
    assert objective[0] == 0.0


def test_wrap_uses_nearest_turn():
    q = np.zeros(7)
    q[5] = np.deg2rad(250.0)   # Kortex-style reading; signed = -110, in zone
    objective, _ = null_part(q)
    assert objective[5] == pytest.approx(2.0 * np.deg2rad(12.0))


def test_dataclass_validation():
    with pytest.raises(ValueError):
        JointLimitAvoidance(limit_rad=np.zeros(3), zone_rad=ZONE)
    with pytest.raises(ValueError):
        JointLimitAvoidance(limit_rad=np.zeros(7), zone_rad=0.0)
