# Diagnosis: right-arm tracking failure under torso roll

Scenario (pinned in `analysis/diagnose.py`): 0.1 m x-sway + 0.4 rad
(23°) roll, both at 0.1 Hz, 40 s, real `desired_pos` task points.
Data: `analysis/output/diagnosis.npz`, figures 01–08. Controller
untouched; every quantity recomputed with the controller's own
functions each cycle.

## 1. Observations (facts visible in the plots/data)

1. Both arms start ~1.3 m from target in the model's default zero
   configuration, which is exactly singular (σ_min ≈ 4e-33, rank 3 at
   t = 0). The "σ_min < 0.05/0.01/0.001 @ 0.0 s" annotations are this
   startup, not the failure. Both arms converge by t ≈ 3 s. (Figs 01, 02)
2. From t = 3–10 s both arms track with |e_pos| ≲ 50 mm (right briefly
   ~115 mm near the first negative-roll extreme), oscillating at the
   roll frequency. At t = 10.0 s the right error is 19 mm — fully
   recovered. (Fig 01)
3. At t = 11.19 s (roll +15.6°) the first contact appears:
   `right_half_arm_2_link` ↔ `torso`. After onset the right arm is in
   contact in 77% of steps; the left arm records zero contacts in the
   entire run. (stdout table)
4. After 11.2 s the right error climbs to 200–320 mm on every
   positive-roll half-cycle and briefly recovers to ~20 mm near the
   roll zero crossings (t ≈ 20, 30 s). Post-settle (t > 5 s) mean:
   right 177 mm, left 37 mm; final: 184 mm vs 36 mm. (Fig 01)
5. Servo lag |ctrl − q| peaks at 2.67 rad on the right arm, starting
   ~1 s after contact onset; left stays ≤ 0.12 rad. A position servo
   2.7 rad from its setpoint is being physically prevented from
   following it. (stdout)
6. σ_min and cond(J) traces are statistically identical between arms
   (medians 0.014 vs 0.011; both dip to ~1e-4). The dips occur at
   NEGATIVE roll extremes (t ≈ 7.5, 17.5, 27.5, 37.5 s) on both arms;
   the right error peaks at POSITIVE roll extremes (t ≈ 13, 23, 33 s)
   — opposite phase. Rank never drops below 6 after startup. (Figs 02,
   03, 07, 08)
7. Joint 6 camps at its position/ctrl limit on both arms (distance ≈ 0,
   with ~0.007 rad physical overshoot). (Fig 04)
8. Velocity saturation: some joint saturates in 9.5% of right-arm
   cycles vs 6.8% left, concentrated in the startup transient and the
   contact windows. (Fig 05)

## 2. Evidence (for / against each candidate explanation)

**Torso–arm collision (right only).**
FOR: contact onset (11.19 s) immediately precedes the first sustained
error plateau; 77% contact occupancy thereafter; servo lag 2.7 rad only
on the contacting arm; error releases to ~20 mm exactly when the roll
passes back through zero and contact drops; the contact-free left arm
tracks throughout.
AGAINST: the right error starts rising ~1 s before first touch
(19 → 88 mm between t = 10 and 11 s) — the initial rise is bandwidth
lag as the roll builds, not contact.

**Jacobian singularity / ill-conditioning.**
FOR: both arms run surprisingly close to singularity (σ_min ~1e-2 to
1e-4) — a real background concern for 6-DOF control at these task
poses.
AGAINST as the cause: σ_min is the same for the healthy and failing
arm; its dips are anti-phased with the right arm's error peaks; the
scatter correlations (Figs 07, 08) are dominated by the contact
windows, not by conditioning; rank never drops post-startup.

**Joint-limit / ctrlrange windup (joint 6).**
FOR: joint 6 sits pinned at its limit on both arms — it contributes
steady bias (this is the documented left-arm ~6 mm residual mechanism,
grown to ~36 mm under roll).
AGAINST as the cause: perfectly symmetric between arms; the left arm
with the same camping tracks acceptably.

**Velocity saturation locking out recovery.**
AGAINST: saturation fractions are similar between arms and saturation
episodes follow the error growth (recovery demands high qdot); they do
not precede it.

**Controller/kinematics asymmetry (math bug).**
AGAINST: FK and world Jacobian are validated to 1e-9 against MuJoCo for
BOTH arms at 50 random torso poses (`test_kinematics`); since the
side-keyed refactor the two arms execute literally the same code path.

## 3. Hypotheses (most → least likely)

1. **Geometric interference: positive torso roll sweeps the right upper
   arm into the torso box.** The position servos then fight the contact
   force (servo lag 2.7 rad), the integrator keeps commanding into the
   obstruction, and the EE is held 200–320 mm from its world-fixed
   target until the roll reverses. The left/right asymmetry is
   geometric: the two mounts are mirrored (±0.845 rad roll) but the arm
   chain itself is identical, not mirrored — so the arms' postures are
   not mirror images, and only the right arm's elbow region crosses the
   torso volume under +roll.
2. **Reactive bandwidth lag** — the ~40–60 mm oscillation floor on both
   arms (first-order sensitivity ≈ 0.30 at 0.1 Hz, Kp = 2), and the
   right arm's pre-contact error rise. Expected reactive behavior, not
   a defect; it is the baseline the thesis measures.
3. **Joint-6 limit camping** — secondary, symmetric bias contributor;
   limits the achievable orientation and inflates both arms' floors.
4. **Near-singular operation at negative roll extremes** (σ_min ~1e-4,
   both arms) — a latent risk the DLS damping currently absorbs; not
   implicated in this failure.
5. **Velocity saturation** — symptom of large errors, not a cause.
6. **Kinematics/controller math bug** — effectively excluded by the
   validation suite and the shared code path.

## 4. Recommended next experiment

**Re-run the identical pinned scenario with the
`right_half_arm_2_link` ↔ `torso` contact pair excluded in
`sim/scene.xml`** (the scene already uses contact excludes for the
mocap-welded base — same mechanism, one line). One variable changes;
everything else stays.

- If the right arm's post-11 s error collapses onto the left arm's
  ≤ 50 mm envelope → collision is confirmed as the root cause, and the
  residual gap (if any) isolates the joint-6 contribution.
- If large error persists without contact → the collision is
  secondary and the next suspect (limit camping at +roll) is promoted.

This is a diagnostic intervention, not a fix: the real robot cannot
ignore collisions. If confirmed, the eventual remedies to evaluate are
task-point placement, mount geometry, or collision-aware redundancy
resolution — out of scope here.

### Instrumentation caveats

- Runs start from the model's default zero configuration (singular,
  1.3 m from target). Experiment metrics should either start from a
  home keyframe or exclude t < 3 s; the σ_min threshold annotations at
  0.0 s are startup artifacts.
- Logged quantities are recomputed pre-`apply_ctrl` from identical sim
  state (same functions the controller calls); `data.ctrl`-derived
  quantities are read post-`apply_ctrl`; contacts post-`mj_step`.
