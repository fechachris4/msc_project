"""Viewer run with a scripted periodic 6-axis mount disturbance.

Disturbance model: each mount axis is A*sin(2*pi*f*t), at either a
fundamental frequency f or its sub-harmonic f/2; no net translation. This is
NOT a gait simulation and claims no biomechanical validity: amplitudes and
frequencies are hand-chosen (_SPEED_TABLE) to give reproducible
disturbance levels. Report figures hold the key-1.0 amplitudes fixed and
vary only frequency (tools/disturbance_freq_sweep.py); --speed= is only the
legacy key into that table. --scale=
exaggerates the amplitudes for on-screen visibility only. Frame: torso home
is identity, so x = forward, y = left, z = up.

The viewer overlay is reduced to what a report figure needs: torso, arms
and target markers. The human-safety envelope is drawn only while the
filter is actually constraining an arm (--full-overlay restores everything).

usage: mjpython tools/mount_disturbance.py [right|left|both] [--trajectory-plot]
                                  [--full-overlay] [--speed=V] [--scale=S]
"""

import functools
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from sim import (  # noqa: E402
    cylinder_view,
    human_safety_view,
    motion,
    planning_view,
    world,
)

# Hand-chosen disturbance table, keyed by a legacy selector (not a physical
# speed). x, z and pitch oscillate at the fundamental f; y, roll and yaw at
# the sub-harmonic f/2. Values between keys are linearly interpolated.
DEFAULT_SPEED_M_S = 1.0     # row used in the README: f = 1.8 Hz
AMPLITUDE_SCALE = 1.0           # multiplier on all amplitudes; >1 only for visibility
_SPEED_TABLE = {           # key : (fundamental Hz, x, y, z mm, roll, pitch, yaw deg)
    0.5: (1.4, 4.0, 28.0, 8.0, 1.5, 1.0, 2.0),
    1.0: (1.8, 8.0, 22.0, 20.0, 2.0, 1.3, 3.0),
    1.3: (2.0, 10.0, 25.0, 25.0, 2.0, 1.5, 4.0),
    1.5: (2.1, 12.0, 25.0, 32.0, 2.5, 2.0, 5.0),
    1.8: (2.2, 14.0, 25.0, 40.0, 3.0, 2.5, 6.0),
}
_SPEEDS = np.array(sorted(_SPEED_TABLE))
_ROWS = np.array([_SPEED_TABLE[v] for v in _SPEEDS])


def disturbance_params(scale=AMPLITUDE_SCALE, speed=DEFAULT_SPEED_M_S):
    step_hz, x, y, z, roll, pitch, yaw = (
        np.interp(speed, _SPEEDS, _ROWS[:, i]) for i in range(7))
    stride_hz = 0.5 * step_hz
    return dict(
        linear_amplitude=scale * 1e-3 * np.array([x, y, z]),
        linear_frequency=np.array([step_hz, stride_hz, step_hz]),
        rotational_amplitude=scale * np.radians([roll, pitch, yaw]),
        rotational_frequency=np.array([stride_hz, step_hz, stride_hz]),
    )


def level_label(speed):
    """Report-facing name of a disturbance condition: its frequency only.
    Report figures use the key-1.0 amplitude row (see
    tools/disturbance_freq_sweep.py); other keys also change amplitude."""
    f = disturbance_params(speed=speed)["linear_frequency"][0]
    return f"f = {f:.2g} Hz"


def torso_pose_at(t, scale=AMPLITUDE_SCALE, speed=DEFAULT_SPEED_M_S):
    return motion.torso_pose_at(t, **disturbance_params(scale, speed))


def torso_twist_at(t, scale=AMPLITUDE_SCALE, speed=DEFAULT_SPEED_M_S):
    return motion.torso_twist_at(t, **disturbance_params(scale, speed))


def describe(scale=AMPLITUDE_SCALE, speed=DEFAULT_SPEED_M_S):
    p = disturbance_params(scale, speed)
    return (
        f"Scripted periodic mount disturbance, level {level_label(speed)} "
        f"(amplitude scale {scale:g}):\n"
        f"  linear amp (mm)      {np.round(1e3 * p['linear_amplitude'], 1)}"
        f"  @ {p['linear_frequency']} Hz\n"
        f"  rotational amp (deg) "
        f"{np.round(np.degrees(p['rotational_amplitude']), 2)}"
        f"  @ {p['rotational_frequency']} Hz"
    )


def safety_is_constraining(safety_states, sides):
    for side in sides:
        c = safety_states.for_arm(side).constraints
        if np.any(c.active) or np.any(c.signed_clearance_m < 0.0):
            return True
    return False


def hide_look_at_object():
    """The look-at marker is unused by the disturbance task; make it invisible."""
    geom_id = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_BODY, "look_at_object")
    for g in range(world.model.ngeom):
        if world.model.geom_bodyid[g] == geom_id:
            world.model.geom_rgba[g, 3] = 0.0


def install_clean_overlay():
    """Keep only geometry that belongs in a report frame."""
    hide_look_at_object()
    cylinder_view.draw = lambda scene, keepout, routes=None: 0
    planning_view.draw = lambda user_scn, plans, torso_pose_world: 0
    full_draw = human_safety_view.draw

    def draw_when_active(scene, plant, ctrl, safety, config, sides):
        if safety_is_constraining(safety, sides):
            return full_draw(scene, plant, ctrl, safety, config, sides)
        return 0

    human_safety_view.draw = draw_when_active


def install_mount_disturbance(scale=AMPLITUDE_SCALE, speed=DEFAULT_SPEED_M_S):
    """main() installs motion's zero-amplitude defaults; make the scripted disturbance win."""
    configure = world.backend.configure_torso_driver
    world.backend.configure_torso_driver = lambda pose_at, twist_at: configure(
        functools.partial(torso_pose_at, scale=scale, speed=speed),
        functools.partial(torso_twist_at, scale=scale, speed=speed),
    )


def pop_float_option(argv, name, default):
    for a in list(argv):
        if a.startswith(f"--{name}="):
            argv.remove(a)
            return float(a.split("=", 1)[1])
    return default


if __name__ == "__main__":
    argv = sys.argv[1:]
    scale = pop_float_option(argv, "scale", AMPLITUDE_SCALE)
    speed = pop_float_option(argv, "speed", DEFAULT_SPEED_M_S)
    if "--full-overlay" in argv:
        argv.remove("--full-overlay")
    else:
        install_clean_overlay()
    print(describe(scale, speed))
    install_mount_disturbance(scale, speed)
    main.main(argv)
