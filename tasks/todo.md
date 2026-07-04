- [x] Investigate why `sim/scene.xml` fails around line 6.
- [x] Update line 6 to reference the existing Kinova Gen3 model file.
- [x] Verify the XML is well formed and the referenced model assets exist.

## Review

Root cause: line 6 referenced `/Users/christian/HumanSL/mujoco_menagerie/kinova_gen3/gen3.xml`, which does not exist on this machine. An existing Kinova Gen3 MJCF was found under `/Users/christian/Projects/Code/mujoco-3.9.0/my_project/HumanSL/mujoco_menagerie/kinova_gen3/gen3.xml`.

Changed `sim/scene.xml` line 6 to use the existing absolute path. Verification passed with `xmllint --noout sim/scene.xml`, `xmllint --noout /Users/christian/Projects/Code/mujoco-3.9.0/my_project/HumanSL/mujoco_menagerie/kinova_gen3/gen3.xml`, and MuJoCo `compile sim/scene.xml`.

## Current Debug Plan

- [x] Reproduce `main.py` from the project root with `python3`.
- [x] Check whether the scene path should be root-relative or script-relative.
- [x] Make the smallest fix if the path is wrong.
- [ ] Verify MuJoCo loads the scene.
- [x] Record the result here.

## Current Debug Review

Root cause found: [main.py](/Users/christian/Projects/Code/msc_project/main.py:5) loaded `scene.xml` from the process working directory, but the actual file is [sim/scene.xml](/Users/christian/Projects/Code/msc_project/sim/scene.xml:1). Changed `main.py` to resolve `sim/scene.xml` relative to `main.py`.

Verification passed for `python3 -m py_compile main.py`, `xmllint --noout sim/scene.xml`, and an explicit path existence check. Full MuJoCo runtime verification is blocked in this shell because `python3 main.py` fails at `import mujoco` with `ModuleNotFoundError: No module named 'mujoco'`.

## GitHub Publish Plan

- [x] Check whether this directory is already a git repository.
- [ ] Confirm whether to initialize a new repository here.
- [ ] Confirm GitHub repository name and visibility.
- [ ] Stage the intended project files only.
- [ ] Commit with a clear message.
- [ ] Push to GitHub.
- [ ] Record publish result here.

## GitHub Publish Status

Blocked before implementation: `/Users/christian/Projects/Code/msc_project` is not currently a git repository, and `gh auth status` reports the saved token for `fechachris4` is invalid. Need confirmation to initialize a new repository here, plus a valid GitHub login before pushing.
