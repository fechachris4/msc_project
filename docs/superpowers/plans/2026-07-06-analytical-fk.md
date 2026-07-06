# Analytical FK for Gen3 Arms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the circular EE-pose derivation in `controller/kinematics.py` with an analytical `T_K_E(q)` computed from joint angles and `MjModel` constants only.

**Architecture:** A `KinematicChain` class is built once per arm prefix at load time by walking `model.body_parentid` from the EE site's body up to `{prefix}base_link`, caching each body's fixed offset, hinge axis/anchor, and global `qpos` address. `fk(qpos)` composes these with pure NumPy. World-frame EE functions become `T_W_T (mocap, from data) · T_T_K (model constant) · T_K_E(q)`.

**Tech Stack:** Python 3.14, `mujoco`, `numpy`, `unittest`. Spec: `docs/superpowers/specs/2026-07-06-analytical-fk-design.md`.

## Global Constraints

- `KinematicChain.__init__` and `fk()` must never read pose data from `MjData` — only `MjModel` constants and the passed-in `qpos` array.
- Resolve joint qpos indices via `model.jnt_qposadr` per joint; never assume contiguous 0–6 indices (the two attached arms interleave in global `qpos`).
- Tests run from the repo root with the project venv: `source .venv/bin/activate`, then `python -m unittest tests.test_kinematics -v` (the real model is loaded via the relative path `sim/scene.xml`).
- Test tolerance vs MuJoCo: `atol=1e-9` (independent computations; exact 1e-16 agreement would itself indicate circularity).
- Gen3 joints 1/3/5/7 are continuous (`model.jnt_limited == 0`); sample those in `[-π, π]`, the limited ones within `model.jnt_range`.
- Follow existing code style in `controller/kinematics.py`: plain functions, explicit NumPy, no cleverness.

---

### Task 1: Rotation helpers

**Files:**
- Modify: `controller/kinematics.py` (add two helpers after `inverse_transform`, line 20)
- Test: `tests/test_kinematics.py` (append test class; leave existing content alone for now — Task 3 removes it)

**Interfaces:**
- Produces: `rotation_from_quat(quat) -> (3,3) ndarray` — MuJoCo `[w, x, y, z]` order; `rotation_about_axis(axis, angle) -> (3,3) ndarray` — Rodrigues rotation about a unit axis.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kinematics.py`:

```python
class RotationHelpersTest(unittest.TestCase):
    def test_rotation_from_quat_matches_z_rotation(self):
        from controller.kinematics import rotation_from_quat
        # quat [w, x, y, z] for a rotation of 0.6 rad about z
        half = 0.3
        quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        np.testing.assert_allclose(
            rotation_from_quat(quat), rotation_z(0.6), atol=1e-12
        )

    def test_rotation_about_axis_matches_z_rotation(self):
        from controller.kinematics import rotation_about_axis
        np.testing.assert_allclose(
            rotation_about_axis(np.array([0.0, 0.0, 1.0]), 0.6),
            rotation_z(0.6),
            atol=1e-12,
        )

    def test_rotation_about_arbitrary_axis_is_orthonormal(self):
        from controller.kinematics import rotation_about_axis
        axis = np.array([1.0, 2.0, -0.5])
        axis /= np.linalg.norm(axis)
        rot = rotation_about_axis(axis, 1.234)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)
        np.testing.assert_allclose(rot @ axis, axis, atol=1e-12)
```

Note: these tests import `controller.kinematics` directly, so the existing `setUp` module stubbing does not apply to them (they are a separate class with no `setUp`). `controller.kinematics` imports `sim.world`, which loads the real model — that is fine from the repo root.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_kinematics.RotationHelpersTest -v`
Expected: FAIL / ERROR with `ImportError: cannot import name 'rotation_from_quat'`

- [ ] **Step 3: Implement the helpers**

In `controller/kinematics.py`, after `inverse_transform`:

```python
def rotation_from_quat(quat):
    """Rotation matrix from a MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])

def rotation_about_axis(axis, angle):
    """Rodrigues rotation about a unit axis."""
    kx, ky, kz = axis
    cross = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_kinematics.RotationHelpersTest -v`
Expected: 3 tests PASS (`OK`)

- [ ] **Step 5: Commit**

```bash
git add controller/kinematics.py tests/test_kinematics.py
git commit -m "Add quaternion and axis-angle rotation helpers"
```

---

### Task 2: KinematicChain with random-sweep validation

**Files:**
- Modify: `controller/kinematics.py` (add `KinematicChain` class and module-level chains after the helpers)
- Test: `tests/test_kinematics.py` (append test class)

**Interfaces:**
- Consumes: `transform_from_pose`, `rotation_from_quat`, `rotation_about_axis` from Task 1.
- Produces:
  - `KinematicChain(model, prefix)` — builds the chain from `{prefix}base_link` to `{prefix}pinch_site`.
  - `KinematicChain.fk(qpos) -> (4, 4) ndarray` — `T_K_E(q)`, base_link frame to EE site.
  - `KinematicChain.joint_ids -> list[int]` — MuJoCo joint ids of the chain, base to tip (used by tests and later by the controller for limits).
  - `KinematicChain.qpos_adrs -> list[int]` — global `qpos` address per chain joint.
  - Module-level `right_chain` and `left_chain` built from `world.model`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kinematics.py`:

```python
class AnalyticalFKTest(unittest.TestCase):
    """Validate analytical FK against MuJoCo at random configurations.

    Independent computation: KinematicChain reads only MjModel constants
    and qpos; MuJoCo's site_xpos/site_xmat is the reference.
    """

    N_SAMPLES = 50
    ATOL = 1e-9

    @classmethod
    def setUpClass(cls):
        import mujoco
        from sim import world
        cls.mujoco = mujoco
        cls.world = world

    def _random_qpos(self, rng, chain):
        model = self.world.model
        q = {}
        for jnt_id, adr in zip(chain.joint_ids, chain.qpos_adrs):
            if model.jnt_limited[jnt_id]:
                low, high = model.jnt_range[jnt_id]
            else:
                low, high = -np.pi, np.pi
            q[adr] = rng.uniform(low, high)
        return q

    def _check_arm(self, prefix):
        import controller.kinematics as kinematics
        mujoco = self.mujoco
        world = self.world
        chain = kinematics.KinematicChain(world.model, prefix)
        base_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link"
        )
        site_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_SITE, prefix + "pinch_site"
        )
        rng = np.random.default_rng(42)
        data = mujoco.MjData(world.model)
        for _ in range(self.N_SAMPLES):
            for adr, value in self._random_qpos(rng, chain).items():
                data.qpos[adr] = value
            mujoco.mj_kinematics(world.model, data)

            T_K_E = chain.fk(data.qpos)

            base_pos = data.xpos[base_id]
            base_rot = data.xmat[base_id].reshape(3, 3)
            ee_pos_expected = data.site_xpos[site_id]
            ee_rot_expected = data.site_xmat[site_id].reshape(3, 3)

            np.testing.assert_allclose(
                base_rot @ T_K_E[:3, 3] + base_pos,
                ee_pos_expected,
                atol=self.ATOL,
            )
            np.testing.assert_allclose(
                base_rot @ T_K_E[:3, :3],
                ee_rot_expected,
                atol=self.ATOL,
            )

    def test_right_arm_fk_matches_mujoco(self):
        self._check_arm("right_")

    def test_left_arm_fk_matches_mujoco(self):
        self._check_arm("left_")

    def test_chain_has_seven_joints(self):
        import controller.kinematics as kinematics
        for prefix in ("right_", "left_"):
            chain = kinematics.KinematicChain(self.world.model, prefix)
            self.assertEqual(len(chain.joint_ids), 7)
```

The comparison maps `T_K_E` into world via MuJoCo's *base_link* pose only — the EE pose itself is never fed into the FK, so the test is not circular (base_link is fixed to the mocap torso; its pose contains no joint-angle information).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_kinematics.AnalyticalFKTest -v`
Expected: ERROR with `AttributeError: module 'controller.kinematics' has no attribute 'KinematicChain'`

- [ ] **Step 3: Implement KinematicChain**

In `controller/kinematics.py`, add `import mujoco` at the top (alongside `numpy`), then after the rotation helpers:

```python
class KinematicChain:
    """Analytical FK for one arm: {prefix}base_link -> {prefix}pinch_site.

    Built once from MjModel constants (body offsets, joint axes/anchors,
    qpos addresses). fk() reads nothing from MjData — only the passed qpos.
    """

    def __init__(self, model, prefix):
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, prefix + "pinch_site"
        )
        base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link"
        )
        if site_id < 0 or base_id < 0:
            raise ValueError(f"missing site or base_link for prefix {prefix!r}")

        # Walk parents from the site's body up to base_link (exclusive),
        # then reverse to get base-to-tip order.
        body_ids = []
        body_id = model.site_bodyid[site_id]
        while body_id != base_id:
            if body_id == 0:
                raise ValueError(
                    f"{prefix}pinch_site does not descend from {prefix}base_link"
                )
            body_ids.append(body_id)
            body_id = model.body_parentid[body_id]
        body_ids.reverse()

        # Per body: fixed parent offset, plus (axis, anchor, qpos adr) if jointed.
        self.joint_ids = []
        self.qpos_adrs = []
        self._steps = []
        for body_id in body_ids:
            T_fixed = transform_from_pose(
                model.body_pos[body_id].copy(),
                rotation_from_quat(model.body_quat[body_id]),
            )
            if model.body_jntnum[body_id] == 0:
                self._steps.append((T_fixed, None, None, None))
                continue
            if model.body_jntnum[body_id] != 1:
                raise ValueError(f"body {body_id} has multiple joints")
            jnt_id = model.body_jntadr[body_id]
            if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"joint {jnt_id} is not a hinge")
            self.joint_ids.append(jnt_id)
            self.qpos_adrs.append(int(model.jnt_qposadr[jnt_id]))
            self._steps.append((
                T_fixed,
                model.jnt_axis[jnt_id].copy(),
                model.jnt_pos[jnt_id].copy(),
                int(model.jnt_qposadr[jnt_id]),
            ))

        self._T_site = transform_from_pose(
            model.site_pos[site_id].copy(),
            rotation_from_quat(model.site_quat[site_id]),
        )

    def fk(self, qpos):
        """T_K_E(q): EE site pose in the base_link frame, as a 4x4 transform."""
        T = np.eye(4)
        for T_fixed, axis, anchor, adr in self._steps:
            T = T @ T_fixed
            if adr is None:
                continue
            rot = rotation_about_axis(axis, qpos[adr])
            T_joint = np.eye(4)
            T_joint[:3, :3] = rot
            T_joint[:3, 3] = anchor - rot @ anchor
            T = T @ T_joint
        return T @ self._T_site


right_chain = KinematicChain(world.model, "right_")
left_chain = KinematicChain(world.model, "left_")
```

Note the module now needs `world.model`, so change the import at the top from `from sim import world` — it already is that; no change needed. The `anchor` term handles `jnt_pos` (joint anchor offset in the body frame); for the Gen3 it is zero, but composing it keeps the walker correct for any hinge chain.

- [ ] **Step 4: Run tests — expect a conflict with the old stubbed test**

Run: `python -m unittest tests.test_kinematics -v`
Expected: `AnalyticalFKTest` and `RotationHelpersTest` PASS. The pre-existing `KinematicsTest` (which stubs `sim` with a `SimpleNamespace` lacking `model`) will now ERROR on import, because module-level `KinematicChain(world.model, ...)` needs the real model. That test is deleted in Task 3; for this commit, delete only its `setUp`/`tearDown` stubbing dependency by running the two new classes:

`python -m unittest tests.test_kinematics.AnalyticalFKTest tests.test_kinematics.RotationHelpersTest -v`
Expected: all PASS (`OK`)

- [ ] **Step 5: Commit**

```bash
git add controller/kinematics.py tests/test_kinematics.py
git commit -m "Add KinematicChain: analytical FK from MjModel constants"
```

---

### Task 3: Rebuild world-frame EE functions, delete the circular derivation

**Files:**
- Modify: `controller/kinematics.py` (replace `right_ee_positions`, add `left_ee_positions`, add fixed mount transforms)
- Modify: `tests/test_kinematics.py` (delete `KinematicsTest` and its stubbing; add world-frame test)

**Interfaces:**
- Consumes: `right_chain`, `left_chain`, `transform_from_pose`, `rotation_from_quat`, `torso_mocap_position` (existing, kinematics.py:32).
- Produces: `right_ee_positions() -> (pos, rot)` and `left_ee_positions() -> (pos, rot)` — same return shape as before (`pose_from_transform` output), now composed as `T_W_T · T_T_K · T_K_E(q)`. Module constants `T_T_KR`, `T_T_KL` (fixed mounts).

- [ ] **Step 1: Update the tests**

In `tests/test_kinematics.py`:
1. Delete the entire `KinematicsTest` class and the now-unused `importlib`, `sys`, `types` imports (keep `unittest`, `numpy`, `rotation_z`).
2. Append:

```python
class WorldFrameEETest(unittest.TestCase):
    """right/left_ee_positions must match MuJoCo's world EE pose while
    reading the EE pose only on the comparison side."""

    def _check(self, ee_positions, site_id_name):
        import mujoco
        import controller.kinematics as kinematics
        from sim import world

        site_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_SITE, site_id_name
        )
        rng = np.random.default_rng(7)
        for jnt_id in range(world.model.njnt):
            adr = world.model.jnt_qposadr[jnt_id]
            if world.model.jnt_limited[jnt_id]:
                low, high = world.model.jnt_range[jnt_id]
            else:
                low, high = -np.pi, np.pi
            world.data.qpos[adr] = rng.uniform(low, high)
        mujoco.mj_kinematics(world.model, world.data)

        pos, rot = ee_positions()
        np.testing.assert_allclose(
            pos, world.data.site_xpos[site_id], atol=1e-9
        )
        np.testing.assert_allclose(
            rot, world.data.site_xmat[site_id].reshape(3, 3), atol=1e-9
        )

    def test_right_ee_positions_matches_mujoco(self):
        import controller.kinematics as kinematics
        self._check(kinematics.right_ee_positions, "right_pinch_site")

    def test_left_ee_positions_matches_mujoco(self):
        import controller.kinematics as kinematics
        self._check(kinematics.left_ee_positions, "left_pinch_site")
```

This mutates the shared `world.data`; that is acceptable here because `world` is process-global test state and no other test in this file depends on its configuration. (If that ever changes, give this test its own `MjData` and a world-frame function variant that accepts `data`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_kinematics.WorldFrameEETest -v`
Expected: `test_left_...` ERRORs (`no attribute 'left_ee_positions'`); `test_right_...` FAILs or passes-by-circularity — after Step 3 both must pass through the analytical path.

- [ ] **Step 3: Rebuild the world-frame functions**

In `controller/kinematics.py`, replace the entire `right_ee_positions` function (lines 47–66 of the original file) with:

```python
# Fixed mounts: {prefix}base_link pose in the torso frame, from model constants
# (base_link's parent in sim/scene.xml is the torso mocap body).
T_T_KR = transform_from_pose(
    world.model.body_pos[world.kinova_right_base_id].copy(),
    rotation_from_quat(world.model.body_quat[world.kinova_right_base_id]),
)
T_T_KL = transform_from_pose(
    world.model.body_pos[world.kinova_left_base_id].copy(),
    rotation_from_quat(world.model.body_quat[world.kinova_left_base_id]),
)

def right_ee_positions():
    """
    W = world, T = torso mocap body, K = Kinova base_link, E = EE site.

    T_W_E(q, t) = T_W_T(t) · T_T_K · T_K_E(q)

    T_W_T is the commanded torso pose (read from data — an input, not an FK
    output); T_T_K is a fixed mount from model constants; T_K_E(q) is the
    analytical chain FK. No EE pose is read from MuJoCo.
    """
    T_W_T = transform_from_pose(*torso_mocap_position())
    return pose_from_transform(T_W_T @ T_T_KR @ right_chain.fk(world.data.qpos))

def left_ee_positions():
    T_W_T = transform_from_pose(*torso_mocap_position())
    return pose_from_transform(T_W_T @ T_T_KL @ left_chain.fk(world.data.qpos))
```

Keep `direct_right_ee_pose` / `direct_left_ee_pose` — they are the measurement side used by tests and plotting. `kinova_right_base_link_position` / `kinova_left_base_link_position` become unused by this module; leave them (they read `data`, still useful diagnostics) unless nothing else imports them — check with `grep -rn "base_link_position" --include="*.py" .` and delete if unreferenced.

- [ ] **Step 4: Run the full test module**

Run: `python -m unittest tests.test_kinematics -v`
Expected: all tests PASS (`OK`) — RotationHelpersTest, AnalyticalFKTest, WorldFrameEETest.

- [ ] **Step 5: Commit**

```bash
git add controller/kinematics.py tests/test_kinematics.py
git commit -m "Rebuild world-frame EE pose on analytical FK, drop circular derivation"
```

---

### Task 4: Point fk_validation plotting at the analytical FK

**Files:**
- Modify: `plotting/fk_validation.py` (docstring only — `sample()` already calls `right_ee_positions`, which is now analytical)

**Interfaces:**
- Consumes: `kinematics.right_ee_positions` (Task 3).

- [ ] **Step 1: Update the docstring**

Replace lines 1–8 of `plotting/fk_validation.py` with:

```python
"""Live FK validation: compare MuJoCo's measured right EE position with the
analytical FK (KinematicChain, computed from qpos and model constants only),
plotted while the simulation runs (headless — no MuJoCo viewer, so plain
python works):

    python -m plotting.fk_validation

Close the plot window to stop; the figure is saved to plots/fk_validation.png.
The two traces are computed independently — agreement here validates the
analytical forward kinematics, not just frame algebra.
"""
```

No code change: `sample()` at plotting/fk_validation.py:19 already routes through `right_ee_positions()`, which Task 3 rebuilt.

- [ ] **Step 2: Smoke-test headless**

Run:
```bash
python -c "
import mujoco
from sim import world
from plotting import fk_validation
mujoco.mj_step(world.model, world.data)
t, direct, fk = fk_validation.sample()
import numpy as np
err = np.linalg.norm(direct - fk)
print('t=', t, 'err=', err)
assert err < 1e-9, err
"
```
Expected: prints a time and an error around 1e-10 or below, no assertion failure. (Do not expect 1e-16 — that was the circularity signature.)

- [ ] **Step 3: Commit**

```bash
git add plotting/fk_validation.py
git commit -m "fk_validation: document that FK trace is now analytical"
```

---

## Self-review notes

- Spec coverage: KinematicChain from model constants (Task 2), jnt_qposadr by name-free chain walk (Task 2), full-pose 4x4 output (Task 2), world-frame rebuild + circular-read deletion (Task 3), random q-sweep tests both arms (Tasks 2–3), fk_validation update (Task 4). No gaps.
- Tolerance: spec said ~1e-10; plan uses atol=1e-9 for margin against accumulation over 9 frame compositions. Same intent (loose enough to be honest, tight enough to catch axis/sign errors, which produce errors of 1e-2 or worse).
- Type consistency: `fk(qpos) -> 4x4`, `*_ee_positions() -> (pos, rot)` used consistently across tasks.
