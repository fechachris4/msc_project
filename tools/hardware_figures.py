"""The two hardware figures in the README (media/hw_walking_trial.png,
media/hw_rejection.png), drawn from the figure-data CSVs of my thesis
analysis of the treadmill trials. Participant data are not public, so the
CSVs are not in this repo; the script is here to show how the figures were made.

usage: python tools/hardware_figures.py PATH_TO_FIGURE_CSVS
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt
S = (sys.argv[1] if len(sys.argv) > 1 else "hardware_data") + "/"
OUT = "media/"
LOCKED = dict(color="0.35", linestyle="--", linewidth=1.3)
CTRL = dict(color="#0072B2", linestyle="-", linewidth=1.6)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 200})
def letters(axes, x=-0.08, y=1.02):
    for l, ax in zip("abcdef", axes):
        ax.text(x, y, f"({l})", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")

# 1. one real walking trial
d = pd.read_csv(S + "F2_example_walking_trial_data.csv")
t = d.time_s.values
win = (t >= 10) & (t <= 20)
steady = t >= 5          # the extract starts after gait initiation; centre both traces on it
fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.0), sharex=True, sharey=True, layout="constrained")
for ax, ax_name, lab in zip(axes, "xyz", ["forward [mm]", "left [mm]", "up [mm]"]):
    lk = d[f"locked_{ax_name}_mm"].values; lk = medfilt(lk - lk[steady].mean(), 11)
    ee = d[f"ee_minus_goal_{ax_name}_mm"].values
    ee = ee - ee[steady].mean()
    ax.plot(t[win], lk[win], label="arm locked to the mount (computed)", **LOCKED)
    ax.plot(t[win], ee[win], label="arm under control (measured, Vicon)", **CTRL)
    ax.axhline(0, color="0.8", linewidth=0.6, zorder=0)
    ax.set_ylabel(lab)
axes[0].set_ylim(-110, 110); axes[0].set_yticks([-100, -50, 0, 50, 100])
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
axes[-1].set_xlabel("time in trial [s]")
fig.savefig(OUT + "hw_walking_trial.png"); plt.close(fig)

# 2. rejection by speed, and frequency response (amplitude and phase) at 1.0 m/s
f4 = pd.read_csv(S + "F4_motion_and_attenuation_by_speed_data.csv")
f5 = pd.read_csv(S + "F5_attenuation_vs_frequency_data.csv")
pooled = [p for p in f4.participant.unique() if p != "P1"]   # P1: mount-tracking fault
LATE = dict(color="#009E73", linewidth=1.5)
KD0 = dict(color="0.35", linestyle=":", linewidth=1.4)
LOGGED = dict(color="0.35", linestyle="-.", linewidth=1.2)
fig = plt.figure(figsize=(7.5, 4.4), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[0.72, 1.6])
ax = fig.add_subplot(gs[:, 0])
speeds = [0.5, 1.0, 1.5]
for p in pooled:
    s_ = f4[(f4.participant == p) & f4.speed_m_s.isin(speeds)].sort_values("speed_m_s")
    ax.plot(s_.speed_m_s, 100 * (1 - s_.attenuation_ratio), color="0.75", linewidth=0.9, marker="o", markersize=3)
gm = [100 * (1 - np.exp(np.log(f4[(f4.speed_m_s == v) & f4.participant.isin(pooled)].attenuation_ratio).mean())) for v in speeds]
ax.plot(speeds, gm, color=CTRL["color"], linewidth=2.2, marker="o", markersize=6, label="geometric mean")
ax.set_ylim(0, 100); ax.set_xticks(speeds); ax.set_xlim(0.3, 1.7)
ax.set_xticklabels([f"{v:.1f}\n{g:.0f}%" for v, g in zip(speeds, gm)])
ax.set_xlabel("treadmill speed [m/s]\nand mean removed"); ax.set_ylabel("mount motion removed [%]")
ax.plot([], [], color="0.75", marker="o", markersize=3, linewidth=0.9, label="each participant")
ax.legend(loc="upper right")
g = f5[f5.participant.isin(pooled) & (f5.freq_hz <= 2.0)]
med = g.groupby("freq_hz")[["gain", "phase_deg"]].median()
mod = g.groupby("freq_hz")[["model_logged", "model_kd0", "model_delayed",
                            "phase_logged_deg", "phase_kd0_deg", "phase_delayed_deg"]].median()
axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1], sharex=axb)
axb.plot(med.index, med.gain, color=CTRL["color"], linewidth=2.2, label="measured (median of 6)")
axb.plot(mod.index, mod.model_logged, label="model, logged gains", **LOGGED)
axb.plot(mod.index, mod.model_kd0, label="model, no velocity term", **KD0)
axb.plot(mod.index, mod.model_delayed, label="model, velocity signal late", **LATE)
axb.set_ylim(0, 1.0); axb.set_ylabel("motion left /\nmotion imposed")
axc.plot(med.index, med.phase_deg, color=CTRL["color"], linewidth=2.2)
axc.plot(mod.index, mod.phase_logged_deg, **LOGGED)
axc.plot(mod.index, mod.phase_kd0_deg, **KD0)
axc.plot(mod.index, mod.phase_delayed_deg, **LATE)
axc.set_ylim(0, 95); axc.set_yticks([0, 30, 60, 90]); axc.set_ylabel("phase [deg]")
h, l = axb.get_legend_handles_labels(); axc.legend(h, l, loc="lower left", fontsize=8.5)
axc.set_xscale("log"); axc.set_xticks([0.125, 0.25, 0.5, 1, 2]); axc.set_xticklabels(["0.125", "0.25", "0.5", "1", "2"])
axc.set_xlim(0.12, 2.05); axc.set_xlabel("frequency [Hz], walking at 1.0 m/s")
plt.setp(axb.get_xticklabels(), visible=False)
letters([ax, axb, axc], x=-0.1)
fig.savefig(OUT + "hw_rejection.png"); plt.close(fig)
print("mount motion removed [%]:", [round(g) for g in gm])
