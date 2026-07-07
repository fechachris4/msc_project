"""Diagnose the right-arm tracking failure under torso roll. Read-only:
the controller is never modified — every logged quantity is recomputed
from sim state with the same public functions apply_ctrl itself calls
(servo.pose_error, frames.jacobian_world, servo.qdot_from_error), so the
log shows exactly what the controller saw each cycle.

    python -m analysis.diagnose

Outputs: analysis/output/diagnosis.npz, 8 figures (PNG), and an event
table on stdout. No fixes, no controller changes — evidence only.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from controller import desired_pos, frames, servo
from sim import motion, targets, world

# Pinned failure scenario (the levers observed to break the right arm);
# pinned here so the diagnosis reproduces even if sim/motion.py changes.
LIN_A = np.array([0.1, 0.0, 0.0])   # m
ROT_A = np.array([0.4, 0.0, 0.0])   # rad (23 deg roll)
FREQ = 0.1                          # Hz, both
SIM_SECONDS = 40.0                  # 4 periods

SV_THRESHOLDS = (0.05, 0.01, 0.001)
OUT = Path("analysis/output")

# Okabe-Ito
C_RIGHT = "#D55E00"
C_LEFT = "#0072B2"
C_XYZ = ("#D55E00", "#009E73", "#0072B2")


def _dof_adrs(side):
    adrs = []
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{i}"
        )
        adrs.append(int(world.model.jnt_dofadr[jnt_id]))
    return adrs


def _jnt_range(side):
    """(low, high) per joint; unlimited (continuous) joints get ±inf."""
    low = np.full(7, -np.inf)
    high = np.full(7, np.inf)
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{i}"
        )
        if world.model.jnt_limited[jnt_id]:
            low[i - 1], high[i - 1] = world.model.jnt_range[jnt_id]
    return low, high


def _arm_body_ids(side):
    ids = set()
    for body_id in range(world.model.nbody):
        name = mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_BODY,
                                 body_id)
        if name and name.startswith(f"{side}_"):
            ids.add(body_id)
    return ids


def _contacts_touching(body_ids):
    n = 0
    for c in world.data.contact[: world.data.ncon]:
        b1 = world.model.geom_bodyid[c.geom1]
        b2 = world.model.geom_bodyid[c.geom2]
        if b1 in body_ids or b2 in body_ids:
            n += 1
    return n


def run():
    """Run the pinned scenario headless; return {side: {key: array}}."""
    mujoco.mj_resetData(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    desired_pos.apply()
    servo.init_ctrl()

    dt = world.model.opt.timestep
    n_steps = int(SIM_SECONDS / dt)
    dofs = {s: _dof_adrs(s) for s in world.SIDES}
    jlims = {s: _jnt_range(s) for s in world.SIDES}
    bodies = {s: _arm_body_ids(s) for s in world.SIDES}

    log = {s: {
        "t": np.empty(n_steps),
        "target_pos": np.empty((n_steps, 3)),
        "ee_pos": np.empty((n_steps, 3)),
        "e_pos": np.empty((n_steps, 3)),
        "e_norm": np.empty(n_steps),
        "e_rot": np.empty((n_steps, 3)),
        "e_rot_norm": np.empty(n_steps),
        "sv": np.empty((n_steps, 6)),
        "min_sv": np.empty(n_steps),
        "cond": np.empty(n_steps),
        "rank": np.empty(n_steps, dtype=int),
        "q": np.empty((n_steps, 7)),
        "qdot_meas": np.empty((n_steps, 7)),
        "qdot_raw": np.empty((n_steps, 7)),
        "qdot_sat": np.empty((n_steps, 7)),
        "jlim_dist": np.empty((n_steps, 7)),
        "ctrl_dist": np.empty((n_steps, 7)),
        "servo_lag": np.empty((n_steps, 7)),
        "v_des": np.empty((n_steps, 6)),
        "v_ach": np.empty((n_steps, 6)),
        "contacts": np.empty(n_steps, dtype=int),
    } for s in world.SIDES}

    for k in range(n_steps):
        t = world.data.time
        motion.set_torso_pose(t, linear_amplitude=LIN_A,
                              linear_frequency=FREQ,
                              rotational_amplitude=ROT_A,
                              rotational_frequency=FREQ)

        # Pre-control state: exactly what apply_ctrl is about to use.
        for s in world.SIDES:
            L = log[s]
            e_pos, e_rot = servo.pose_error(s)
            J = frames.jacobian_world(s)
            sv = np.linalg.svd(J, compute_uv=False)
            qdot_raw = servo.qdot_from_error(J, e_pos, e_rot)
            q = world.data.qpos[frames.qpos_adrs[s]].copy()
            qdot_meas = world.data.qvel[dofs[s]].copy()
            low, high = jlims[s]

            L["t"][k] = t
            L["target_pos"][k] = targets.target_position(s)
            L["ee_pos"][k] = frames.ee_pose(s)[0]
            L["e_pos"][k] = e_pos
            L["e_norm"][k] = np.linalg.norm(e_pos)
            L["e_rot"][k] = e_rot
            L["e_rot_norm"][k] = np.linalg.norm(e_rot)
            L["sv"][k] = sv
            L["min_sv"][k] = sv.min()
            L["cond"][k] = sv.max() / max(sv.min(), 1e-300)
            L["rank"][k] = int(np.sum(sv > 1e-10))
            L["q"][k] = q
            L["qdot_meas"][k] = qdot_meas
            L["qdot_raw"][k] = qdot_raw
            L["qdot_sat"][k] = np.clip(qdot_raw, -servo.QDOT_LIMIT,
                                       servo.QDOT_LIMIT)
            L["jlim_dist"][k] = np.minimum(q - low, high - q)
            L["v_des"][k] = np.concatenate([servo.KP_POS * e_pos,
                                            servo.KP_ROT * e_rot])
            L["v_ach"][k] = J @ qdot_meas

        servo.apply_ctrl(dt)

        # Post-control: the setpoints the servos will now chase.
        for s in world.SIDES:
            ctrl = world.data.ctrl[world.ctrl_adrs[s]].copy()
            lo, hi = servo._BOUNDS[s]
            log[s]["ctrl_dist"][k] = np.minimum(ctrl - lo, hi - ctrl)
            log[s]["servo_lag"][k] = ctrl - log[s]["q"][k]

        mujoco.mj_step(world.model, world.data)

        for s in world.SIDES:
            log[s]["contacts"][k] = _contacts_touching(bodies[s])

    return log


def sv_events(log):
    """{side: {threshold: first crossing time or None}}."""
    ev = {}
    for s in world.SIDES:
        ev[s] = {}
        for th in SV_THRESHOLDS:
            below = np.nonzero(log[s]["min_sv"] < th)[0]
            ev[s][th] = log[s]["t"][below[0]] if below.size else None
    return ev


def _mark_events(ax, ev_side, contact_t=None):
    for th, t in ev_side.items():
        if t is not None:
            ax.axvline(t, color="0.4", linewidth=0.8, linestyle=":")
            ax.annotate(f"σ_min<{th:g}\n@{t:.1f}s", (t, 0.98),
                        xycoords=("data", "axes fraction"),
                        fontsize=7, va="top", ha="left", color="0.25")
    if contact_t is not None:
        ax.axvline(contact_t, color="red", linewidth=1.2, linestyle="-.")
        ax.annotate(f"first torso\ncontact @{contact_t:.1f}s",
                    (contact_t, 0.80), xycoords=("data", "axes fraction"),
                    fontsize=7, va="top", ha="left", color="red")


def make_figures(log, ev):
    OUT.mkdir(parents=True, exist_ok=True)
    r, l = log["right"], log["left"]
    t = r["t"]
    pk_r = r["e_norm"].max() * 1000
    pk_l = l["e_norm"].max() * 1000
    touch = np.nonzero(r["contacts"] > 0)[0]
    contact_t = t[touch[0]] if touch.size else None

    # 1 — does tracking fail gradually or suddenly?
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(t, r["e_norm"] * 1000, color=C_RIGHT, label="right")
    ax.plot(t, l["e_norm"] * 1000, color=C_LEFT, linestyle="--",
            label="left (control)")
    _mark_events(ax, ev["right"], contact_t)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|e_pos| [mm]")
    ax.set_title(f"Right arm peaks at {pk_r:.0f} mm vs left {pk_l:.0f} mm "
                 "— gradual or sudden?")
    ax.legend()
    fig.savefig(OUT / "01_error_norm_vs_time.png", dpi=200)

    # 2 — does failure coincide with a singularity?
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.semilogy(t, r["min_sv"], color=C_RIGHT, label="right")
    ax.semilogy(t, l["min_sv"], color=C_LEFT, linestyle="--",
                label="left (control)")
    for th in SV_THRESHOLDS:
        ax.axhline(th, color="red", linewidth=0.6, alpha=0.5)
    _mark_events(ax, ev["right"], contact_t)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("σ_min(J)")
    ax.set_title(f"Right σ_min floor {r['min_sv'].min():.2e} "
                 f"(left {l['min_sv'].min():.2e}) — singularity at failure?")
    ax.legend()
    fig.savefig(OUT / "02_min_singular_value_vs_time.png", dpi=200)

    # 3 — how ill-conditioned does the controller become?
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.semilogy(t, r["cond"], color=C_RIGHT, label="right")
    ax.semilogy(t, l["cond"], color=C_LEFT, linestyle="--",
                label="left (control)")
    _mark_events(ax, ev["right"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cond(J)")
    ax.set_title(f"Right cond(J) peaks at {r['cond'].max():.1e} "
                 f"(left {l['cond'].max():.1e})")
    ax.legend()
    fig.savefig(OUT / "03_condition_number_vs_time.png", dpi=200)

    # 4 — does any joint reach or camp on a limit?
    for s, cs in (("right", C_RIGHT), ("left", C_LEFT)):
        L = log[s]
        low, high = _jnt_range(s)
        lo_c, hi_c = servo._BOUNDS[s]
        fig, axes = plt.subplots(7, 1, sharex=True, figsize=(9, 12),
                                 layout="constrained")
        for j in range(7):
            ax = axes[j]
            ax.plot(t, L["q"][:, j], color=cs, linewidth=1.0)
            for lim in (low[j], high[j]):
                if np.isfinite(lim):
                    ax.axhline(lim, color="red", linewidth=0.8)
            for lim in (lo_c[j], hi_c[j]):
                if np.isfinite(lim):
                    ax.axhline(lim, color="red", linewidth=0.6,
                               linestyle="--", alpha=0.7)
            ax.set_ylabel(f"q{j + 1} [rad]")
        _mark_events(axes[0], ev[s])
        axes[-1].set_xlabel("time [s]")
        near = np.nonzero(np.min(L["jlim_dist"], axis=0) < 0.05)[0] + 1
        fig.suptitle(f"{s} arm joints vs limits (solid red: jnt_range, "
                     f"dashed red: ctrlrange) — joints within 0.05 rad of a "
                     f"limit: {list(near) if near.size else 'none'}")
        fig.savefig(OUT / f"04_joint_positions_{s}.png", dpi=200)

    # 5 — is velocity saturation preventing recovery?
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(9, 12),
                             layout="constrained")
    sat_frac = np.mean(np.abs(r["qdot_raw"]) > servo.QDOT_LIMIT, axis=0)
    for j in range(7):
        ax = axes[j]
        ax.plot(t, r["qdot_raw"][:, j], color="0.6", linewidth=0.8,
                label="commanded (raw)" if j == 0 else None)
        ax.plot(t, r["qdot_sat"][:, j], color=C_RIGHT, linewidth=1.0,
                label="after saturation" if j == 0 else None)
        ax.axhline(servo.QDOT_LIMIT[j], color="red", linewidth=0.8)
        ax.axhline(-servo.QDOT_LIMIT[j], color="red", linewidth=0.8)
        ax.set_ylabel(f"q̇{j + 1} [rad/s]")
    _mark_events(axes[0], ev["right"])
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Right arm commanded q̇ before/after saturation — "
                 f"saturated fraction per joint: "
                 f"{np.array2string(sat_frac, precision=2)}")
    fig.savefig(OUT / "05_qdot_saturation_right.png", dpi=200)

    # 6 — does the EE diverge in a particular direction?
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    pairs = ((0, 1, "x [m]", "y [m]"), (0, 2, "x [m]", "z [m]"),
             (1, 2, "y [m]", "z [m]"))
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        sc = ax.scatter(r["ee_pos"][:, i], r["ee_pos"][:, j], c=t, s=1,
                        cmap="viridis")
        ax.plot(r["target_pos"][0, i], r["target_pos"][0, j], "*",
                color="black", markersize=12, label="target")
        ax.plot(l["ee_pos"][:, i], l["ee_pos"][:, j], color=C_LEFT,
                linewidth=0.6, linestyle="--", alpha=0.6,
                label="left EE (control)")
        ax.plot(l["target_pos"][0, i], l["target_pos"][0, j], "*",
                color=C_LEFT, markersize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axes[-1], label="time [s]")
    fig.suptitle("Right EE world trajectory (time-colored) vs fixed target "
                 "— directional divergence?")
    fig.savefig(OUT / "06_ee_trajectory_world.png", dpi=200)

    # 7 — is poor conditioning correlated with tracking error?
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    sc = ax.scatter(r["min_sv"], r["e_norm"] * 1000, c=t, s=2,
                    cmap="viridis")
    ax.set_xlabel("σ_min(J)")
    ax.set_ylabel("|e_pos| [mm]")
    rho = np.corrcoef(r["min_sv"], r["e_norm"])[0, 1]
    ax.set_title(f"Right arm: σ_min vs error, corr = {rho:+.2f}")
    fig.colorbar(sc, ax=ax, label="time [s]")
    fig.savefig(OUT / "07_scatter_minsv_vs_error.png", dpi=200)

    # 8 — does tracking degrade as conditioning worsens?
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    sc = ax.scatter(r["cond"], r["e_norm"] * 1000, c=t, s=2, cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel("cond(J)")
    ax.set_ylabel("|e_pos| [mm]")
    rho = np.corrcoef(np.log10(r["cond"]), r["e_norm"])[0, 1]
    ax.set_title(f"Right arm: cond(J) vs error, corr(log cond, e) = "
                 f"{rho:+.2f}")
    fig.colorbar(sc, ax=ax, label="time [s]")
    fig.savefig(OUT / "08_scatter_cond_vs_error.png", dpi=200)

    plt.close("all")


def print_summary(log, ev):
    for s in world.SIDES:
        L = log[s]
        print(f"\n=== {s} arm ===")
        print(f"  |e_pos|  peak {L['e_norm'].max() * 1000:8.1f} mm   "
              f"final {L['e_norm'][-1] * 1000:8.1f} mm")
        print(f"  |e_rot|  peak {np.degrees(L['e_rot_norm'].max()):8.1f} deg  "
              f"final {np.degrees(L['e_rot_norm'][-1]):8.1f} deg")
        print(f"  sigma_min floor {L['min_sv'].min():.3e}   "
              f"cond peak {L['cond'].max():.3e}   "
              f"rank min {L['rank'].min()}")
        print(f"  min dist to jnt_range  {np.nanmin(L['jlim_dist']):.4f} rad "
              f"(joint {int(np.unravel_index(np.nanargmin(L['jlim_dist']), L['jlim_dist'].shape)[1]) + 1})")
        print(f"  min dist to ctrlrange  {np.min(L['ctrl_dist']):.4f} rad "
              f"(joint {int(np.unravel_index(np.argmin(L['ctrl_dist']), L['ctrl_dist'].shape)[1]) + 1})")
        print(f"  qdot saturation: any-joint fraction "
              f"{np.mean(np.any(np.abs(L['qdot_raw']) > servo.QDOT_LIMIT, axis=1)):.2%}")
        print(f"  servo lag |ctrl-q| peak {np.abs(L['servo_lag']).max():.3f} rad")
        print(f"  contacts: max {L['contacts'].max()} "
              f"(steps with any: {np.mean(L['contacts'] > 0):.2%})")
        print(f"  sigma_min crossings: " + ", ".join(
            f"<{th:g}: {f'{ev[s][th]:.1f}s' if ev[s][th] is not None else '—'}"
            for th in SV_THRESHOLDS))


def main():
    log = run()
    ev = sv_events(log)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "diagnosis.npz",
        **{f"{s}_{k}": v for s in world.SIDES for k, v in log[s].items()},
    )
    make_figures(log, ev)
    print_summary(log, ev)
    print(f"\nSaved {OUT}/diagnosis.npz and figures 01–08.")


if __name__ == "__main__":
    main()
