"""Reactive-loop reference-tracking bandwidth: EE-target tracking-error
amplitude and phase lag vs. commanded sinusoid frequency. Companion to
analysis/diagnose.py, but a different research question — diagnose.py
investigates a torso-collision failure under base motion; this script
characterizes the closed loop's own tracking bandwidth with the base
held static and one EE target driven sinusoidally instead
(sim/target_motion.py, tests/test_target_motion.py's
TargetTrackingBandwidthTest is the regression tripwire this script
sweeps across frequency).

    python -m analysis.bandwidth_sweep

Outputs: analysis/output/bw_error_amplitude.png,
analysis/output/bw_phase_lag.png, and a summary table on stdout. No
controller changes — every logged quantity comes from servo.pose_error,
the same public function apply_ctrl itself uses.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from controller import desired_pos, frames, servo
from sim import target_motion, world

OUT = Path("analysis/output")

SIDE = "right"
AXIS = 0  # world x: the driven, and only nonzero, component of AMPLITUDE
AMPLITUDE = np.array([0.05, 0.0, 0.0])  # m, world xyz

# Log-spaced sweep from ~0.02 Hz to ~1 Hz: KP_POS ~ a few rad/s puts the
# reactive loop's own bandwidth (KP_POS/2pi Hz) inside this range, so the
# sweep spans "well inside bandwidth, barely attenuated" through "well
# above bandwidth, strongly attenuated".
FREQUENCIES = np.geomspace(0.02, 1.0, 7)  # Hz

SETTLE_SECONDS = 2.0        # static-hold settle before the target moves
N_SETTLE_PERIODS_MIN = 5.0  # first-order settling: >= 5 time constants
N_FIT_PERIODS = 2.0         # steady-state periods used for the sine fit

HOME = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109,
        1.57079633]

C_MEAS = "#D55E00"
C_PRED = "#0072B2"


def _reset_to_home():
    """Base static throughout (no sim.motion calls): arms to their home
    joint config, desired_pos.apply() resolves the static targets,
    target_motion.init_home() captures them as the sinusoid's anchor."""
    mujoco.mj_resetData(world.model, world.data)
    for side in world.SIDES:
        world.data.qpos[frames.qpos_adrs[side]] = HOME
    mujoco.mj_forward(world.model, world.data)
    desired_pos.apply()
    target_motion.init_home()
    servo.init_ctrl()


def _fit_sine(t, signal, w):
    """Least-squares fit signal(t) = a*sin(w t) + b*cos(w t), rewritten
    as amplitude*sin(w t - phase_lag). Returns (amplitude, phase_lag)."""
    basis = np.column_stack([np.sin(w * t), np.cos(w * t)])
    (a, b), *_ = np.linalg.lstsq(basis, signal, rcond=None)
    amplitude = np.hypot(a, b)
    phase_lag = np.arctan2(-b, a)
    return amplitude, phase_lag


def run_one(frequency):
    """Drive SIDE's target sinusoidally along AXIS at `frequency` Hz;
    return (error_amplitude_m, actual_phase_lag_rad).

    error_amplitude: steady-state |target - actual| amplitude on the
    driven axis, fit directly from the logged tracking error.
    actual_phase_lag: phase lag of the EE's own tracked motion behind
    the commanded sine, reconstructed as (commanded - error) so it is
    the standard Bode-style quantity (0 at low frequency, -> 90 deg
    above bandwidth), not the error signal's own (less intuitive) phase.
    """
    _reset_to_home()
    dt = world.model.opt.timestep
    zero_twist = (np.zeros(3), np.zeros(3))

    for _ in range(int(SETTLE_SECONDS / dt)):
        servo.apply_ctrl(dt, zero_twist)
        mujoco.mj_step(world.model, world.data)

    w = 2.0 * np.pi * frequency
    # KP_eff, not raw KP_POS: targets.target_velocity() is pinned zero
    # (no feedforward for a moving target), so KD_POS damps against the
    # EE's own tracking velocity rather than assisting it — see
    # tests/test_target_motion.py's TargetTrackingBandwidthTest
    # docstring for the full derivation.
    kp_eff = servo.KP_POS / (1.0 + servo.KD_POS)
    skip_seconds = max(N_SETTLE_PERIODS_MIN / kp_eff, 1.0 / frequency)
    fit_seconds = N_FIT_PERIODS / frequency

    t_start = world.data.time  # phase 0 at motion start: no teleport
    log_t, log_e = [], []
    n_steps = int((skip_seconds + fit_seconds) / dt)
    for _ in range(n_steps):
        t_rel = world.data.time - t_start
        target_motion.set_target_pose(t_rel, SIDE, linear_amplitude=AMPLITUDE,
                                      linear_frequency=frequency,
                                      rotational_amplitude=np.zeros(3))
        servo.apply_ctrl(dt, zero_twist)
        mujoco.mj_step(world.model, world.data)
        if t_rel >= skip_seconds:
            e_pos, _ = servo.pose_error(SIDE)
            log_t.append(t_rel)
            log_e.append(e_pos[AXIS])

    log_t = np.array(log_t)
    log_e = np.array(log_e)
    ref_perturbation = AMPLITUDE[AXIS] * np.sin(w * log_t)
    actual_perturbation = ref_perturbation - log_e

    error_amp, _ = _fit_sine(log_t, log_e, w)
    _, actual_phase_lag = _fit_sine(log_t, actual_perturbation, w)
    return error_amp, actual_phase_lag


def predicted(frequency):
    """First-order model: actual EE tracks the reference through
    H(s) = KP_eff/(s+KP_eff). Error = ref - actual has magnitude
    |A|*w/sqrt(w^2+KP_eff^2); the actual tracked motion lags the
    reference by atan(w/KP_eff)."""
    kp_eff = servo.KP_POS / (1.0 + servo.KD_POS)
    w = 2.0 * np.pi * frequency
    error_amp = AMPLITUDE[AXIS] * w / np.sqrt(w**2 + kp_eff**2)
    phase_lag = np.arctan2(w, kp_eff)
    return error_amp, phase_lag


def sweep():
    """Run every frequency in FREQUENCIES; return arrays of measured and
    predicted (error_amp_m, phase_lag_rad)."""
    meas_amp = np.empty(len(FREQUENCIES))
    meas_phase = np.empty(len(FREQUENCIES))
    pred_amp = np.empty(len(FREQUENCIES))
    pred_phase = np.empty(len(FREQUENCIES))
    for i, f in enumerate(FREQUENCIES):
        meas_amp[i], meas_phase[i] = run_one(f)
        pred_amp[i], pred_phase[i] = predicted(f)
    return meas_amp, meas_phase, pred_amp, pred_phase


def make_figures(meas_amp, meas_phase, pred_amp, pred_phase):
    OUT.mkdir(parents=True, exist_ok=True)
    kp_eff = servo.KP_POS / (1.0 + servo.KD_POS)
    bandwidth_hz = kp_eff / (2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.loglog(FREQUENCIES, meas_amp * 1000.0, "o", color=C_MEAS,
             label="measured")
    ax.loglog(FREQUENCIES, pred_amp * 1000.0, "-", color=C_PRED,
             label="first-order prediction")
    ax.axvline(bandwidth_hz, color="0.4", linewidth=0.8, linestyle=":")
    ax.annotate(f"KP_eff/2pi = {bandwidth_hz:.2f} Hz",
               (bandwidth_hz, 0.02), xycoords=("data", "axes fraction"),
               fontsize=8, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("commanded frequency [Hz]")
    ax.set_ylabel("tracking error amplitude [mm]")
    ax.set_title(f"{SIDE} arm: EE-target tracking error vs. frequency")
    ax.legend()
    fig.savefig(OUT / "bw_error_amplitude.png", dpi=200)

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.semilogx(FREQUENCIES, np.degrees(meas_phase), "o", color=C_MEAS,
               label="measured")
    ax.semilogx(FREQUENCIES, np.degrees(pred_phase), "-", color=C_PRED,
               label="first-order prediction")
    ax.axvline(bandwidth_hz, color="0.4", linewidth=0.8, linestyle=":")
    ax.axhline(45.0, color="red", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("commanded frequency [Hz]")
    ax.set_ylabel("EE tracking phase lag [deg]")
    ax.set_title(f"{SIDE} arm: EE-target tracking phase lag vs. frequency")
    ax.legend()
    fig.savefig(OUT / "bw_phase_lag.png", dpi=200)

    plt.close("all")


def print_summary(meas_amp, meas_phase, pred_amp, pred_phase):
    kp_eff = servo.KP_POS / (1.0 + servo.KD_POS)
    print(f"KP_POS={servo.KP_POS:.2f}  KD_POS={servo.KD_POS:.2f}  "
         f"KP_eff={kp_eff:.2f} rad/s  ({kp_eff / (2 * np.pi):.3f} Hz)")
    print(f"{'f [Hz]':>8}  {'err meas [mm]':>14}  {'err pred [mm]':>14}  "
         f"{'lag meas [deg]':>15}  {'lag pred [deg]':>15}")
    for f, ea, ep, la, lp in zip(FREQUENCIES, meas_amp, pred_amp,
                                 meas_phase, pred_phase):
        print(f"{f:8.3f}  {ea * 1000:14.2f}  {ep * 1000:14.2f}  "
             f"{np.degrees(la):15.1f}  {np.degrees(lp):15.1f}")


def main():
    meas_amp, meas_phase, pred_amp, pred_phase = sweep()
    make_figures(meas_amp, meas_phase, pred_amp, pred_phase)
    print_summary(meas_amp, meas_phase, pred_amp, pred_phase)
    print(f"\nSaved {OUT}/bw_error_amplitude.png and {OUT}/bw_phase_lag.png.")


if __name__ == "__main__":
    main()
