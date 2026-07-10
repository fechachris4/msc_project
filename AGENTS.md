# Project context

## What this is
MSc thesis project (Imperial, MUVE Lab, deadline 1 Sept 2026).
Research question: can predictive control improve SRL performance over
reactive control under human-induced base motion?

Current SL controllers are reactive: they correct only after a disturbance
appears as error. This project builds toward a predictive framework that
anticipates disturbances from motion capture and onboard sensing before
stability is lost.

## Pipeline (in order)
1. CURRENT PHASE: reactive controller in MuJoCo (this repo, Python).
   Goal is the strongest possible reactive baseline. A weak baseline
   invalidates the whole comparison.
2. Port validated core to C++ for the real robot.
3. Hardware: dual Kinova Gen3 on a backpack mount, Vicon motion capture,
   multi-directional treadmill in the MUVE Lab.
4. Final scenario: one SL performs a task while the other maintains
   stability. Human walks or is perturbed on the treadmill.

## Rules that follow from this
- Do not start predictive/MPC work unless explicitly asked. Current work
  is the reactive baseline only.
- Code written now should be portable in concept to C++. No Python-only
  cleverness in the control path (no dynamic dispatch tricks, keep the
  control loop explicit and traceable).
- End-effector references are WORLD-FRAME. The task is world-frame pose
  hold while the base moves. Never convert references to torso-frame.
- This is research code for a solo builder on a deadline. No restructuring,
  no test frameworks, no CI, no abstractions unless asked.

## Success criteria for the current phase
Reactive controller in MuJoCo that holds a world-frame end-effector pose under scripted base motion, with error plots (mean, RMSE, peak) that can go straight into the thesis.

## Units

All internal math, sim state, and control code are SI metres/radians.
Millimetres appear only at human-facing boundaries (prints, plots, logged
metrics) and future Vicon input (mm→m converted once at the interface).
Never inside the math path.

## Verification requirements

- Every computed quantity must be checkable against ground truth (e.g., FK position vs. MuJoCo direct reading). New modules need a comparison test before they're considered working.
- Reproduce last known-good behavior before layering new changes.

## Git

- Commit after every verified change. Small commits, descriptive messages. Each commit = a snapshot that works.
- Never leave uncommitted changes across work sessions.

## Debugging protocol

- When behavior degrades: diff against last working commit FIRST, before tuning gains or parameters.
- A constant bias in tracking error suggests a frame/sign/computation error, not a gain problem — check math before sweeping gains.
