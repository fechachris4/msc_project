import dataclasses
import unittest

import mujoco
import numpy as np

from controller.state import Twist

from controller.transforms import rotation_about_axis
from controller import reactive_controller
from controller.position_actuation import PositionIntegrator
from tests.control_test_support import apply_cycle, reconstruct_pipeline


HOME = np.array([
    0.0,
    0.26179939,
    3.14159265,
    -2.26892803,
    0.0,
    0.95993109,
    1.57079633,
])
BASE_TWIST = (
    np.array([0.07, -0.03, 0.02]),
    np.array([0.11, -0.06, 0.04]),
)
EXPECTED_CTRL = np.array([
    0.0015717317045702117,
    0.2614111738319953,
    3.1410046283914332,
    -2.266149465830825,
    0.0024399702942880727,
    0.9574911197057119,
    1.568356359705712,
])
EXPECTED_QDOT_RAW = np.array([
    0.7858658522851059,
    -0.19410808400234925,
    -0.2940108042835781,
    2.6904378270960234,
    2.1473300318227477,
    -3.1003725820379477,
    -1.731501236854941,
])
EXPECTED_QDOT_SPEED_CLIPPED = np.array([
    0.7858658522851059,
    -0.19410808400234925,
    -0.2940108042835781,
    1.3892820845874863,
    1.2199851471440364,
    -1.2199851471440364,
    -1.2199851471440364,
])


def _setup_scene():
    """Right arm at HOME with a known joint velocity and a known pose
    offset to the target — shared by ControlTraceTest and
    ComponentTogglesTest."""
    from controller import frames, servo
    from sim import targets, world

    mujoco.mj_resetData(world.model, world.data)
    for side in world.SIDES:
        world.data.qpos[world.qpos_adrs[side]] = HOME
    world.data.qvel[world.dof_adrs["right"]] = np.array(
        [0.12, -0.08, 0.05, -0.03, 0.02, -0.01, 0.04]
    )
    mujoco.mj_forward(world.model, world.data)

    state = frames.arm_controller_state(
        world.read_state(Twist.zero()),
        "right",
        world.MOUNT_CALIBRATION,
    )
    pos = state.ee_pose_world.position_m
    rot = state.ee_pose_world.rotation
    axis = np.array([0.3, -0.4, 0.5])
    axis /= np.linalg.norm(axis)
    ref_rot = rotation_about_axis(axis, 0.7) @ rot
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, ref_rot.flatten())
    targets.set_target("right", pos + np.array([0.35, -0.22, 0.18]))
    targets.set_target_quat("right", quat)
    return reconstruct_pipeline()


class ControlTraceTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = _setup_scene()

    def tearDown(self):
        from sim import world

        mujoco.mj_resetData(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)

    def test_stage_values_and_ctrl_output_match_original_controller(self):
        from controller import servo
        from sim import world

        dt = world.model.opt.timestep
        traces = apply_cycle(
            self.pipeline, dt, BASE_TWIST, arms=("right",))
        self.assertIsNotNone(traces)
        trace = traces["right"]

        self.assertEqual(set(traces), {"right"})
        self.assertIsInstance(trace, servo.ControlTrace)
        self.assertEqual(
            [field.name for field in dataclasses.fields(trace)],
            [
                "J", "e_pos", "e_rot", "e_v", "e_w",
                "p_twist", "d_twist", "task_twist", "q",
                "qdot_measured", "qdot_raw", "qdot_speed_clipped",
                "qdot_safety_filtered", "qdot_effective",
                "ctrl_before", "ctrl_after",
                "speed_saturated", "lead_clamped", "range_clamped",
            ],
        )
        np.testing.assert_allclose(trace.qdot_raw, EXPECTED_QDOT_RAW,
                                   atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            trace.qdot_speed_clipped,
            np.clip(
                EXPECTED_QDOT_RAW,
                -np.asarray(servo.LIMITS.joint_velocity_rad_s),
                np.asarray(servo.LIMITS.joint_velocity_rad_s),
            ),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            trace.task_twist, trace.p_twist + trace.d_twist,
            atol=1e-12, rtol=0.0,
        )
        np.testing.assert_allclose(
            trace.ctrl_after, EXPECTED_CTRL,
            atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            trace.qdot_effective,
            (trace.ctrl_after - trace.ctrl_before) / dt,
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            trace.qdot_measured,
            np.array([0.12, -0.08, 0.05, -0.03, 0.02, -0.01, 0.04]),
        )

        ctrl_integrated = trace.ctrl_before + trace.qdot_speed_clipped * dt
        ctrl_lead_limited = np.clip(
            ctrl_integrated,
            trace.q - servo.LIMITS.position_lead_rad,
            trace.q + servo.LIMITS.position_lead_rad,
        )
        limits = world.PIPELINE_SETUP.right.actuation_limits
        ctrl_after = np.clip(
            ctrl_lead_limited,
            limits.lower_position_rad,
            limits.upper_position_rad,
        )
        np.testing.assert_array_equal(
            trace.speed_saturated,
            trace.qdot_raw != trace.qdot_speed_clipped,
        )
        np.testing.assert_array_equal(
            trace.lead_clamped, ctrl_integrated != ctrl_lead_limited)
        np.testing.assert_array_equal(
            trace.range_clamped, ctrl_lead_limited != ctrl_after)
        np.testing.assert_allclose(
            world.data.ctrl[world.ctrl_adrs["right"]], EXPECTED_CTRL,
            atol=1e-12,
            rtol=0.0,
        )

    def test_trace_owns_read_only_arrays_and_is_frozen(self):
        from controller import servo
        from sim import world

        traces = apply_cycle(
            self.pipeline,
            world.model.opt.timestep,
            BASE_TWIST,
            arms=("right",),
        )
        self.assertIsNotNone(traces)
        trace = traces["right"]
        snapshots = {
            field.name: getattr(trace, field.name).copy()
            for field in dataclasses.fields(trace)
        }

        for field in dataclasses.fields(trace):
            value = getattr(trace, field.name)
            self.assertFalse(value.flags.writeable, field.name)
            with self.assertRaises(ValueError, msg=field.name):
                value.flat[0] = 0.0

        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.q = np.zeros(7)

        world.data.qpos[:] = 0.0
        world.data.qvel[:] = 0.0
        world.data.ctrl[:] = 0.0
        for name, expected in snapshots.items():
            np.testing.assert_array_equal(getattr(trace, name), expected)

    def test_trace_array_buffers_cannot_be_made_writeable(self):
        from controller import servo
        from sim import world

        trace = apply_cycle(
            self.pipeline,
            world.model.opt.timestep,
            BASE_TWIST,
            arms=("right",),
        )["right"]
        for field in dataclasses.fields(trace):
            with self.assertRaises(ValueError, msg=field.name):
                getattr(trace, field.name).setflags(write=True)

    def test_invalid_dt_fails_before_ctrl_mutation(self):
        from controller import servo
        from sim import world

        ctrl_before = world.data.ctrl.copy()
        for dt in (0.0, -0.1, np.nan, np.inf, -np.inf):
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):
                    apply_cycle(
                        self.pipeline, dt, BASE_TWIST, arms=("right",))
                np.testing.assert_array_equal(world.data.ctrl, ctrl_before)

    def test_pd_twist_terms_follow_gains(self):
        from controller import servo
        from sim import world

        trace = apply_cycle(
            self.pipeline,
            world.model.opt.timestep,
            BASE_TWIST,
            arms=("right",),
        )["right"]
        np.testing.assert_array_equal(
            trace.p_twist,
            np.concatenate(
                [
                    servo.CONTROL.kp_position_s_inv * trace.e_pos,
                    servo.CONTROL.kp_rotation_s_inv * trace.e_rot,
                ]
            ),
        )
        np.testing.assert_array_equal(
            trace.d_twist,
            np.concatenate(
                [
                    servo.CONTROL.kd_position * trace.e_v,
                    servo.CONTROL.kd_rotation * trace.e_w,
                ]
            ),
        )
        np.testing.assert_array_equal(
            trace.task_twist, trace.p_twist + trace.d_twist)

    def test_logged_task_twist_drives_the_solve(self):
        """The trace's task_twist must be the one the DLS solve consumed,
        and apply_ctrl's inlined control law must stay numerically
        identical to the standalone qdot_from_error the analysis
        scripts use."""
        from controller import servo
        from sim import world

        trace = apply_cycle(
            self.pipeline,
            world.model.opt.timestep,
            BASE_TWIST,
            arms=("right",),
        )["right"]

        qdot_task = trace.J.T @ np.linalg.solve(
            trace.J @ trace.J.T + servo.CONTROL.dls_damping**2 * np.eye(6),
            trace.task_twist,
        )
        projector = np.eye(7) - np.linalg.pinv(trace.J) @ trace.J
        centering = world.PIPELINE_SETUP.right.centering
        null_gain = centering.enabled * servo.CONTROL.null_gain_s_inv
        qdot_null = -null_gain * (
            trace.q - centering.midpoint_rad)
        np.testing.assert_allclose(
            trace.qdot_raw, qdot_task + projector @ qdot_null,
            atol=1e-12, rtol=0.0)

        np.testing.assert_allclose(
            trace.qdot_raw,
            reactive_controller.solve_reactive_velocity(
                trace.J, trace.e_pos, trace.e_rot, trace.e_v, trace.e_w,
                trace.q, centering.midpoint_rad, null_gain,
                servo.CONTROL,
            ).qdot_raw,
            atol=1e-12, rtol=0.0)

    def test_lead_clamp_mask_and_effective_rate(self):
        from controller import servo

        trace = apply_cycle(
            self.pipeline, 1.0, BASE_TWIST, arms=("right",))["right"]
        expected_ctrl = np.array([
            0.2,
            0.06769130599765076,
            2.94159265,
            -2.06892803,
            0.2,
            0.75993109,
            1.37079633,
        ])
        expected_lead_clamped = np.array(
            [True, False, True, True, True, True, True]
        )

        np.testing.assert_allclose(
            trace.qdot_speed_clipped, EXPECTED_QDOT_SPEED_CLIPPED,
            atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(trace.ctrl_after, expected_ctrl,
                                   atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(
            trace.lead_clamped, expected_lead_clamped)
        np.testing.assert_array_equal(
            trace.range_clamped, np.zeros(7, dtype=bool))
        np.testing.assert_allclose(
            trace.qdot_effective, expected_ctrl - HOME,
            atol=1e-12, rtol=0.0)

    def test_actuator_range_clamp_mask_and_effective_rate(self):
        from controller import servo
        from sim import world

        dt = world.model.opt.timestep
        q6 = 5
        actuator = world.ctrl_adrs["right"][q6]
        high = world.model.actuator_ctrlrange[actuator, 1]
        limits = world.PIPELINE_SETUP.right.actuation_limits
        seed = HOME.copy()
        seed[q6] = high + 0.05
        measured = HOME.copy()
        measured[q6] = high - 0.05
        integrator = PositionIntegrator(seed, limits)
        actuation = integrator.step(measured, np.zeros(7), dt)
        expected_range_clamped = np.zeros(7, dtype=bool)
        expected_range_clamped[q6] = True

        self.assertEqual(actuation.command_before_rad[q6], high + 0.05)
        self.assertEqual(actuation.command_after_rad[q6], high)
        self.assertAlmostEqual(actuation.qdot_effective[q6], -0.05 / dt,
                               places=12)
        np.testing.assert_array_equal(
            actuation.lead_clamped, np.zeros(7, dtype=bool))
        np.testing.assert_array_equal(
            actuation.range_clamped, expected_range_clamped)

    def test_field_shapes_and_dtypes(self):
        from controller import servo
        from sim import world

        trace = apply_cycle(
            self.pipeline,
            world.model.opt.timestep,
            BASE_TWIST,
            arms=("right",),
        )["right"]
        expected_shapes = {
            "J": (6, 7),
            "e_pos": (3,), "e_rot": (3,), "e_v": (3,), "e_w": (3,),
            "p_twist": (6,), "d_twist": (6,), "task_twist": (6,),
            "q": (7,), "qdot_measured": (7,), "qdot_raw": (7,),
            "qdot_speed_clipped": (7,),
            "qdot_safety_filtered": (7,), "qdot_effective": (7,),
            "ctrl_before": (7,), "ctrl_after": (7,),
            "speed_saturated": (7,), "lead_clamped": (7,),
            "range_clamped": (7,),
        }
        fields = dataclasses.fields(trace)
        self.assertEqual(set(expected_shapes), {field.name for field in fields})
        for field in fields:
            value = getattr(trace, field.name)
            self.assertEqual(value.shape, expected_shapes[field.name], field.name)
            expected_dtype = np.bool_ if field.name.endswith(
                ("saturated", "clamped")) else np.float64
            self.assertEqual(value.dtype, expected_dtype, field.name)


class ComponentTogglesTest(unittest.TestCase):
    """Position, orientation, and velocity feedback must be
    independently disableable: a disabled component contributes exact
    zeros to the commanded twist, the enabled ones are untouched, and
    apply_ctrl's law still matches the standalone qdot_from_error."""

    def setUp(self):
        _setup_scene()
        from controller import servo
        self.control = servo.CONTROL

    def tearDown(self):
        from sim import world

        mujoco.mj_resetData(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)

    def _trace_matching_standalone_law(self):
        from controller import servo
        from sim import world

        pipeline = reconstruct_pipeline(self.control)
        trace = apply_cycle(
            pipeline,
            world.model.opt.timestep,
            BASE_TWIST,
            arms=("right",),
        )["right"]
        centering = world.PIPELINE_SETUP.right.centering
        null_gain = centering.enabled * self.control.null_gain_s_inv
        np.testing.assert_allclose(
            trace.qdot_raw,
            reactive_controller.solve_reactive_velocity(
                trace.J, trace.e_pos, trace.e_rot, trace.e_v, trace.e_w,
                trace.q, centering.midpoint_rad, null_gain,
                self.control,
            ).qdot_raw,
            atol=1e-12, rtol=0.0)
        return trace

    def test_flags_default_enabled(self):
        from controller import servo

        self.assertTrue(servo.CONTROL.position_enabled)
        self.assertTrue(servo.CONTROL.orientation_enabled)
        self.assertTrue(servo.CONTROL.velocity_enabled)

    def test_position_only(self):
        from controller import servo

        self.control = dataclasses.replace(
            servo.CONTROL,
            orientation_enabled=False,
            velocity_enabled=False,
        )
        trace = self._trace_matching_standalone_law()
        np.testing.assert_array_equal(
            trace.p_twist[:3],
            servo.CONTROL.kp_position_s_inv * trace.e_pos)
        np.testing.assert_array_equal(trace.p_twist[3:], np.zeros(3))
        np.testing.assert_array_equal(trace.d_twist, np.zeros(6))

    def test_orientation_only(self):
        from controller import servo

        self.control = dataclasses.replace(
            servo.CONTROL,
            position_enabled=False,
            velocity_enabled=False,
        )
        trace = self._trace_matching_standalone_law()
        np.testing.assert_array_equal(trace.p_twist[:3], np.zeros(3))
        np.testing.assert_array_equal(
            trace.p_twist[3:],
            servo.CONTROL.kp_rotation_s_inv * trace.e_rot)
        np.testing.assert_array_equal(trace.d_twist, np.zeros(6))

    def test_velocity_only(self):
        from controller import servo

        self.control = dataclasses.replace(
            servo.CONTROL,
            position_enabled=False,
            orientation_enabled=False,
        )
        trace = self._trace_matching_standalone_law()
        np.testing.assert_array_equal(trace.p_twist, np.zeros(6))
        np.testing.assert_array_equal(
            trace.d_twist,
            np.concatenate(
                [
                    servo.CONTROL.kd_position * trace.e_v,
                    servo.CONTROL.kd_rotation * trace.e_w,
                ]
            ))


if __name__ == "__main__":
    unittest.main()
