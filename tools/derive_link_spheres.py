"""Print conservative LinkSphere records from the MuJoCo collision meshes.

Run from the repository root:

    .venv/bin/python tools/derive_link_spheres.py

The emitted values are reviewed into ``controller/link_spheres.py``.  Runtime
control never imports MuJoCo mesh data, keeping the controller portable.
"""

from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _collision_mesh_vertices_body(model, geom_id):
    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[start:start + count], dtype=float
    )
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(
        rotation, np.asarray(model.geom_quat[geom_id])
    )
    return (
        vertices @ rotation.reshape(3, 3).T
        + np.asarray(model.geom_pos[geom_id])
    )


def _sphere_chain(vertices):
    centroid = np.mean(vertices, axis=0)
    _, _, axes = np.linalg.svd(vertices - centroid, full_matrices=False)
    axis = axes[0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    axial = (vertices - centroid) @ axis
    perpendicular = (
        vertices - centroid - np.outer(axial, axis)
    )
    perpendicular_radius = float(
        np.max(np.linalg.norm(perpendicular, axis=1))
    )
    length = float(np.max(axial) - np.min(axial))
    count = max(1, int(np.ceil(length / (2.0 * perpendicular_radius))))
    spacing = length / count
    radius = float(np.hypot(perpendicular_radius, spacing / 2.0) + 1e-6)
    centers = tuple(
        centroid
        + axis * (np.min(axial) + (index + 0.5) * spacing)
        for index in range(count)
    )
    return centers, radius


def main():
    model = mujoco.MjModel.from_xml_path(
        str(PROJECT_ROOT / "sim" / "scene.xml")
    )
    for geom_id in range(model.ngeom):
        if (
            int(model.geom_contype[geom_id]) == 0
            or int(model.geom_type[geom_id])
            != int(mujoco.mjtGeom.mjGEOM_MESH)
        ):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or ""
        if not body_name.startswith("right_"):
            continue
        frame_name = body_name.removeprefix("right_")
        vertices = _collision_mesh_vertices_body(model, geom_id)
        centers, radius = _sphere_chain(vertices)
        print(
            f"# {frame_name}: mesh vertices={len(vertices)}, "
            f"chain radius={radius:.17g} m"
        )
        for index, center in enumerate(centers):
            mount_exempt = (
                frame_name in ("base_link", "shoulder_link")
                or (
                    frame_name == "half_arm_1_link"
                    and index == len(centers) - 1
                )
            )
            print(
                "LinkSphere("
                f"{frame_name!r}, {f'{frame_name}_{index}'!r}, "
                f"{tuple(float(value) for value in center)!r}, "
                f"{radius!r}, {mount_exempt!r}),"
            )


if __name__ == "__main__":
    main()
