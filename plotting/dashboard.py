"""Live 5-panel control-loop dashboard: tracking, orientation, saturation
headroom, Jacobian conditioning, and joint/servo margins, for one or both
arms, with gain sliders for live tuning.

    python -m plotting.dashboard right
    python -m plotting.dashboard left
    python -m plotting.dashboard              # both arms, live (default)
    python -m plotting.dashboard both --save 10   # headless, 10 sim-seconds

Base motion follows this module's BASE_SCENARIO (main.py's values). Every
per-tick quantity is recomputed from the same public functions apply_ctrl
itself calls (servo.pose_error, servo.twist_error, servo.qdot_from_error,
frames.jacobian_world) — the analysis/diagnose.py convention: the
dashboard shows exactly what the controller saw, nothing re-derived.

Panels (top to bottom), sharing the time axis:
  1. |e_pos| per side (mm) + |base displacement from home| (mm, grey) —
     the thesis success criterion. Monospace readout: RMS / peak /
     rejection % per side over the current window (rejection =
     1 - peak|e|/peak|base disp|).
  2. |e_rot| per side (deg) — orientation error, plotted nowhere else in
     this repo (only printed in diagnose.py's summary).
  3. Saturation headroom: 100 * max_i|qdot_raw_i|/QDOT_LIMIT_i per side,
     pre-clip, red 100% line (the one physical-limit line on this figure).
  4. sigma_min(J) per side, semilogy, grey lines at DAMPING=0.05 ("DLS
     active"), 0.01, 0.001 (diagnose.py's SV_THRESHOLDS).
  5. Margins (deg): min distance to jnt_range over the limited joints
     (solid) + max|ctrl-q| setpoint lead (dashed), per side; grey line at
     degrees(CTRL_LEAD) (a soft anti-windup bound, not hardware).

Right arm: solid, #D55E00. Left arm: dashed, #0072B2 (Okabe-Ito, the
diagnose.py convention) — except panel 5, where solid/dashed instead
distinguishes margin from lead (colour still carries which side).

Sliders (live mode only): KP_POS, KP_ROT, KD_POS, KD_ROT, DAMPING (log
scale). Each writes straight onto controller.servo's module globals — a
runtime-only mutation; committed gains are untouched. Moving any slider
flushes the ring buffers so stale pre-change data doesn't linger in the
window. K_NULL is excluded: it is baked into servo._K_NULL_VEC at import
time, so a runtime change would not reach the null-space term anyway.
"""

import sys
from collections import deque
from pathlib import Path

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.widgets import Slider

from controller import desired_pos, frames, servo
from plotting.style import C_BASE, C_RIGHT, C_LEFT, SIDE_COLOR as _SIDE_COLOR, \
    SIDE_STYLE as _SIDE_STYLE
from sim import motion, world

# Base-motion scenario for this run — same values main.py runs today.
BASE_SCENARIO = dict(
    linear_amplitude=np.array([0.1, 0.3, 0.0]),   # m, world xyz
    linear_frequency=0.1,                          # Hz
    rotational_amplitude=np.zeros(3),              # rad, rpy
    rotational_frequency=0.2,                      # Hz
)

WINDOW_S = 15.0        # rolling window kept on screen, seconds
REDRAW_EVERY = 25      # sim steps between redraws (LivePlot's convention)
SV_THRESHOLDS = (0.05, 0.01, 0.001)  # diagnose.py's DLS-activity markers
SETTLE_SECONDS = 2.0   # --save only; base_vs_error.py's SETTLE pattern


# --- pure per-tick helpers: importable and testable without a display ------


def headroom_frac(qdot_raw):
    """max_i |qdot_raw_i| / QDOT_LIMIT_i — the pre-clip demand as a
    fraction of each joint's speed limit (can exceed 1: qdot_raw is
    unclipped, this is what apply_ctrl is about to clip away)."""
    return float(np.max(np.abs(qdot_raw) / servo.QDOT_LIMIT))


def limit_margin_deg(q, q_low, q_high, limited):
    """Degrees from q to the nearer bound, minimized over the limited
    joints only (continuous joints have no range to be near — same
    convention as servo._centering / world.jnt_range)."""
    margin = np.minimum(q - q_low, q_high - q)
    return float(np.degrees(np.min(margin[limited])))


def ctrl_lead_deg(ctrl, q):
    """max_i |ctrl_i - q_i|, degrees — the anti-windup setpoint lead
    apply_ctrl clamps to CTRL_LEAD."""
    return float(np.degrees(np.max(np.abs(ctrl - q))))


def sigma_min(J):
    """Smallest singular value of the Jacobian — DLS is materially active
    once this falls near DAMPING (servo.qdot_from_error's lambda)."""
    return float(np.linalg.svd(J, compute_uv=False).min())


# --- gain sliders (ported from HumanSL_scratch/analysis/live.py:35-92) ------

SLIDER_SPECS = (
    ("KP_POS", "KP_pos", 0.0, 10.0, "linear"),
    ("KP_ROT", "KP_rot", 0.0, 10.0, "linear"),
    ("KD_POS", "KD_pos", 0.0, 1.0, "linear"),
    ("KD_ROT", "KD_rot", 0.0, 1.0, "linear"),
    ("DAMPING", "lambda", 1e-3, 1.0, "log"),
)


def add_gain_sliders(fig, on_change=None):
    """One slider per SLIDER_SPECS entry, writing straight onto
    controller.servo's module globals. Reserves the bottom margin itself
    (fig.subplots_adjust), so the caller doesn't need to. on_change, if
    given, runs after every mutation (used here to flush the ring
    buffers so a gain step doesn't leave stale data in the window).
    Returns {key: Slider} — keep the reference alive, or the GC drops
    the callbacks."""
    n = len(SLIDER_SPECS)
    fig.subplots_adjust(bottom=0.03 + n * 0.025 + 0.05)

    sliders = {}
    for i, (key, label, vmin, vmax, scale) in enumerate(SLIDER_SPECS):
        ax = fig.add_axes((0.18, 0.03 + i * 0.025, 0.55, 0.02))
        value = getattr(servo, key)
        if scale == "log":
            slider = Slider(ax, label, np.log10(vmin), np.log10(vmax),
                            valinit=np.log10(value))
            slider.valtext.set_text(f"{value:.3g}")
        else:
            slider = Slider(ax, label, vmin, vmax, valinit=value)

        def _on_changed(val, key=key, scale=scale, slider=slider):
            value = 10.0 ** val if scale == "log" else val
            setattr(servo, key, value)
            if scale == "log":
                slider.valtext.set_text(f"{value:.3g}")
            if on_change:
                on_change()

        slider.on_changed(_on_changed)
        sliders[key] = slider
    return sliders


# --- the dashboard loop ------------------------------------------------


def run(arms, save_seconds=None):
    """Run the dashboard. arms: subset of world.SIDES to drive (via
    servo.apply_ctrl) and plot — an unselected arm holds posture and is
    not shown, same selection semantics as main.py. save_seconds: if
    given, run headless for that many sim-seconds and save
    plots/dashboard.png; otherwise run live with gain sliders until the
    figure window is closed."""
    desired_pos.apply()
    servo.init_ctrl()

    dt = world.model.opt.timestep

    if save_seconds is not None:
        # Static settle before the scenario clock starts (base_vs_error.py's
        # SETTLE pattern): a fixed-length save has no ring buffer to age the
        # ~1.3 m boot transient out, so left in it would dominate peak error
        # and make rejection % meaningless. Live mode skips this — the ring
        # buffer ages the transient out on its own.
        zero_twist = (np.zeros(3), np.zeros(3))
        for _ in range(int(SETTLE_SECONDS / dt)):
            servo.apply_ctrl(dt, zero_twist)
            mujoco.mj_step(world.model, world.data)
    t_start = world.data.time  # phase 0 at motion start: no teleport

    maxlen = max(1, int(WINDOW_S / dt))
    n_steps = int(save_seconds / dt) if save_seconds is not None else None

    jlim = {s: world.jnt_range(s) for s in arms}

    t_buf = deque(maxlen=maxlen)
    base_disp_buf = deque(maxlen=maxlen)
    e_pos_buf = {s: deque(maxlen=maxlen) for s in arms}
    e_rot_buf = {s: deque(maxlen=maxlen) for s in arms}
    headroom_buf = {s: deque(maxlen=maxlen) for s in arms}
    sigma_buf = {s: deque(maxlen=maxlen) for s in arms}
    margin_buf = {s: deque(maxlen=maxlen) for s in arms}
    lead_buf = {s: deque(maxlen=maxlen) for s in arms}

    # --- figure ----------------------------------------------------------
    if save_seconds is None:
        plt.ion()
    fig, axes = plt.subplots(
        5, 1, sharex=True, figsize=(9, 12),
        gridspec_kw={"height_ratios": [3, 2, 2, 2, 1]},
    )
    ax_e, ax_rot, ax_head, ax_sv, ax_marg = axes

    lines = {}
    lines["base"] = ax_e.plot([], [], color=C_BASE, linewidth=1.0,
                              label="|base disp|")[0]
    for s in arms:
        c, ls = _SIDE_COLOR[s], _SIDE_STYLE[s]
        lines["e", s] = ax_e.plot([], [], color=c, linestyle=ls,
                                  label=f"{s} |e_pos|")[0]
        lines["rot", s] = ax_rot.plot([], [], color=c, linestyle=ls,
                                      label=s)[0]
        lines["head", s] = ax_head.plot([], [], color=c, linestyle=ls,
                                        label=s)[0]
        lines["sv", s] = ax_sv.plot([], [], color=c, linestyle=ls,
                                    label=s)[0]
        # Panel 5 overrides the side->linestyle convention: solid/dashed
        # here distinguishes margin from lead, not side (colour still
        # does that) — see module docstring.
        lines["marg", s] = ax_marg.plot([], [], color=c, linestyle="-",
                                        label=f"{s} margin")[0]
        lines["lead", s] = ax_marg.plot([], [], color=c, linestyle="--",
                                        label=f"{s} lead")[0]

    ax_e.axhline(0, color="0.85", linewidth=0.5, zorder=0)
    ax_e.set_ylabel("|e_pos|, |base disp| [mm]")
    readout = ax_e.text(0.99, 0.97, "", transform=ax_e.transAxes,
                        family="monospace", fontsize=8, va="top", ha="right")
    ax_e.legend(loc="upper left", fontsize=8)

    ax_rot.axhline(0, color="0.85", linewidth=0.5, zorder=0)
    ax_rot.set_ylabel("|e_rot| [deg]")

    ax_head.axhline(100.0, color="red", linewidth=0.8)
    ax_head.set_ylabel("headroom [%]")

    ax_sv.set_yscale("log")
    for th in SV_THRESHOLDS:
        ax_sv.axhline(th, color="0.6", linewidth=0.6)
    ax_sv.annotate("DLS active, λ", (0.01, SV_THRESHOLDS[0]),
                  xycoords=("axes fraction", "data"), fontsize=7,
                  color="0.4", va="bottom")
    ax_sv.set_ylabel("σ_min(J)")

    ax_marg.axhline(np.degrees(servo.CTRL_LEAD), color="0.6", linewidth=0.6)
    ax_marg.set_ylabel("margin [deg]")
    ax_marg.set_xlabel("sim time (s)")

    title = fig.suptitle("world-frame pose hold under base sway")

    if save_seconds is None:
        def _flush():
            t_buf.clear()
            base_disp_buf.clear()
            for s in arms:
                e_pos_buf[s].clear()
                e_rot_buf[s].clear()
                headroom_buf[s].clear()
                sigma_buf[s].clear()
                margin_buf[s].clear()
                lead_buf[s].clear()
        add_gain_sliders(fig, on_change=_flush)
    else:
        fig.tight_layout()

    def _redraw():
        t_arr = np.array(t_buf)
        base_arr = np.array(base_disp_buf)
        lines["base"].set_data(t_arr, base_arr)

        stats_lines = []
        worst = None
        for s in arms:
            e_arr = np.array(e_pos_buf[s])
            lines["e", s].set_data(t_arr, e_arr)
            lines["rot", s].set_data(t_arr, e_rot_buf[s])
            lines["head", s].set_data(t_arr, headroom_buf[s])
            lines["sv", s].set_data(t_arr, sigma_buf[s])
            lines["marg", s].set_data(t_arr, margin_buf[s])
            lines["lead", s].set_data(t_arr, lead_buf[s])

            rms = float(np.sqrt(np.mean(e_arr**2))) if e_arr.size else 0.0
            peak = float(e_arr.max()) if e_arr.size else 0.0
            peak_base = float(base_arr.max()) if base_arr.size else 0.0
            rejection = (1.0 - peak / peak_base) * 100.0 \
                if peak_base > 0 else float("nan")
            stats_lines.append(
                f"{s:>5s}  rms{rms:6.1f}  pk{peak:6.1f}  "
                f"rej{rejection:5.0f}%  [mm]"
            )
            if worst is None or rejection < worst[1]:
                worst = (s, rejection, peak)
        readout.set_text("\n".join(stats_lines))
        if worst is not None:
            title.set_text(
                f"World-frame pose hold under base sway — {worst[0]} arm "
                f"rejects {worst[1]:.0f}% of base motion "
                f"(peak |e_pos| {worst[2]:.0f} mm)"
            )

        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        if save_seconds is None:
            plt.pause(0.001)

    # --- main loop (mirrors analysis/diagnose.py's ordering) -------------
    step = 0
    while True:
        if n_steps is not None:
            if step >= n_steps:
                break
        elif not plt.fignum_exists(fig.number):
            break

        t = world.data.time - t_start
        motion.set_torso_pose(t, **BASE_SCENARIO)
        # refresh xpos/xmat so every quantity below sees the torso pose
        # at t, not the previous step's (main.py / diagnose.py pattern)
        mujoco.mj_kinematics(world.model, world.data)
        # set_torso_pose (mocap write) and torso_twist_at (feedforward)
        # must stay a matched pair — same scenario, same instant t.
        base_twist = motion.torso_twist_at(t, **BASE_SCENARIO)

        base_pos, _ = frames.torso_pose()
        base_disp_mm = np.linalg.norm(base_pos - motion.HOME_POS) * 1000.0

        # Pre-control state: exactly what apply_ctrl is about to use.
        # damping=servo.DAMPING is passed explicitly (unlike diagnose.py,
        # which pins its gains) so the DAMPING slider is reflected here
        # too — qdot_from_error's own damping default binds at def time,
        # same reason apply_ctrl now passes it explicitly (servo.py fix).
        tick = {}
        for s in arms:
            J = frames.jacobian_world(s)
            e_pos, e_rot = servo.pose_error(s)
            e_v, e_w = servo.twist_error(s, base_twist, J)
            q = world.data.qpos[frames.qpos_adrs[s]].copy()
            qdot_raw = servo.qdot_from_error(
                J, e_pos, e_rot, e_v, e_w, q,
                servo._Q_MID[s], servo._K_NULL_VEC[s],
                damping=servo.DAMPING)
            tick[s] = dict(J=J, e_pos=e_pos, e_rot=e_rot, q=q,
                           qdot_raw=qdot_raw)

        servo.apply_ctrl(dt, base_twist, arms)

        # Post-control: the setpoints the servos will now chase.
        for s in arms:
            ctrl = world.data.ctrl[world.ctrl_adrs[s]]
            tick[s]["lead_deg"] = ctrl_lead_deg(ctrl, tick[s]["q"])

        mujoco.mj_step(world.model, world.data)

        t_buf.append(t)
        base_disp_buf.append(base_disp_mm)
        for s in arms:
            low, high, limited = jlim[s]
            e_pos_buf[s].append(np.linalg.norm(tick[s]["e_pos"]) * 1000.0)
            e_rot_buf[s].append(np.degrees(np.linalg.norm(tick[s]["e_rot"])))
            headroom_buf[s].append(100.0 * headroom_frac(tick[s]["qdot_raw"]))
            sigma_buf[s].append(sigma_min(tick[s]["J"]))
            margin_buf[s].append(
                limit_margin_deg(tick[s]["q"], low, high, limited))
            lead_buf[s].append(tick[s]["lead_deg"])

        step += 1
        if n_steps is None and step % REDRAW_EVERY == 0:
            _redraw()

    _redraw()
    if n_steps is not None:
        Path("plots").mkdir(parents=True, exist_ok=True)
        path = "plots/dashboard.png"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    return fig


if __name__ == "__main__":
    _args = sys.argv[1:]
    _save_seconds = None
    if "--save" in _args:
        _i = _args.index("--save")
        _save_seconds = float(_args[_i + 1])
        del _args[_i:_i + 2]
    _side_arg = _args[0] if _args else "both"
    assert _side_arg in ("right", "left", "both"), \
        "usage: python -m plotting.dashboard [right|left|both] [--save T]"
    _arms = world.SIDES if _side_arg == "both" else (_side_arg,)

    run(_arms, save_seconds=_save_seconds)
