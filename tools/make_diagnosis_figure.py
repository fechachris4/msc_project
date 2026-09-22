"""Two-panel diagnosis figure: error and servo lag against contact windows."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.load("/Users/christian/Projects/Code/msc_project/analysis/output/diagnosis.npz")
t_all = D["right_t"]
rc_all = D["right_contacts"] > 0
onset = t_all[np.argmax(rc_all)]
post = t_all >= onset
print(f"onset {onset:.3f}s  post-onset right occupancy {rc_all[post].mean():.3f}  "
      f"left total {(D['left_contacts'] > 0).mean():.3f}")

keep = t_all >= 3.0
t = t_all[keep]
rc = rc_all[keep]
re_ = D["right_e_norm"][keep] * 1000
le = D["left_e_norm"][keep] * 1000
rl = np.abs(D["right_servo_lag"])[keep].max(axis=1)
ll = np.abs(D["left_servo_lag"])[keep].max(axis=1)
print(f"post-3s mean error right {re_.mean():.0f} mm  left {le.mean():.0f} mm  "
      f"right peak lag {rl.max():.2f} rad  left peak lag {ll.max():.2f} rad")

fig, ax = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True)
ORANGE, BLUE = "#C8451E", "#2A6F8E"

for a, (r, l, ylab, lab) in enumerate([
        (re_, le, "position error [mm]", "|e_pos|"),
        (rl, ll, "servo lag [rad]", "max |ctrl - q|")]):
    axx = ax[a]
    axx.fill_between(t, 0, 1, where=rc, transform=axx.get_xaxis_transform(),
                     color="0.87", lw=0,
                     label="right arm touching torso" if a == 0 else None)
    axx.plot(t, r, color=ORANGE, lw=1.7, label=f"right {lab}")
    axx.plot(t, l, color=BLUE, lw=1.4, ls="--", label=f"left {lab}")
    axx.axvline(onset, color="0.35", lw=1.0, ls=":")
    axx.set_ylabel(ylab)
    axx.legend(loc="upper left", fontsize=8.5, framealpha=0.95, ncol=3)
    axx.spines[["top", "right"]].set_visible(False)
    axx.margins(x=0.01)

ax[0].set_title("Torso roll pushes the right arm into the torso. The left arm never touches.",
                fontsize=10.5)
ax[0].annotate(f"first contact  t = {onset:.2f} s", xy=(onset, 0.06),
               xycoords=("data", "axes fraction"), xytext=(6, 0),
               textcoords="offset points", fontsize=8.5, color="0.25")
ax[1].set_xlabel("sim time [s]   (startup transient before 3 s excluded)")
fig.tight_layout()
out = "/Users/christian/Projects/Code/msc_project/analysis/output/diagnosis_summary.png"
fig.savefig(out, dpi=200)
print("wrote", out)
