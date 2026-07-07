# CLAUDE.md

## Project

MuJoCo simulation of a mobile-base arm with PID end-effector control.
Goal: working sim → real-robot PID baseline → advanced controllers.

## Architecture rules (non-negotiable)

- Code lives in separate single-purpose modules:
  - `controller.py` — control law only
  - `position_reading.py` — end-effector position (FK: torso + base + arm)
  - `velocity_reading.py` — end-effector velocity (Jacobian + body linear/angular terms)
- One module = one job. Never merge these concerns into one file.

## Change discipline

- NEVER modify more than one module per change/commit.
- Working modules are FROZEN. Do not edit a verified module — add new code alongside it instead.
- Before any change, state what you're changing and what could break.
- If asked to modify a frozen module, push back and propose an alternative that adds code instead.

## Verification requirements

- Every computed quantity must be checkable against ground truth (e.g., FK position vs. MuJoCo direct reading). New modules need a comparison test before they're considered working.
- Reproduce last known-good behavior before layering new changes.

## Math and libraries

- Use standard libraries over hand-written math. Pseudo-inverse: `numpy.linalg.pinv`, never a custom implementation.
- All physical constants (joint limits, gains, noise amplitudes) live in a config file, not hardcoded. Always state units explicitly (rad/s vs deg/s, mm vs cm). Joint velocity limit: ~1.25 rad/s — verify against datasheet before use.

## Git

- Commit after every verified change. Small commits, descriptive messages. Each commit = a snapshot that works.
- Never leave uncommitted changes across work sessions.

## Debugging protocol

- When behavior degrades: diff against last working commit FIRST, before tuning gains or parameters.
- A constant bias in tracking error suggests a frame/sign/computation error, not a gain problem — check math before sweeping gains.
