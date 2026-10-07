# Scope and next steps

What the results in the [README](../README.md) cover, and what I would do next.

- Hardware results come from seven participants (six in the pooled means), mostly one arm at a time. Next: both arms together, randomised speed order, and a trial that changes the velocity estimator.
- Feedforward is validated in simulation; running it on hardware is the natural next test.
- The simulated disturbance is a controlled sum of sinusoids. Replaying recorded gait from the Vicon data is the next step.
- The simulated delay is a pure delay; adding the estimator's noise and filtering would match hardware more closely.
- Joint limits are handled by clamping; joint-limit avoidance would extend the workspace for starts far from the target.
- The simulated arms are position-servoed and rigid, as modelled in MuJoCo.
- This repo establishes the reactive baseline plus feedforward. A predictive controller (MPC) that compensates the measured delay is the planned comparison.
