"""Reject physics changes when refreshing the appearance of a saved episode."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from backends.collision_rules import collision_participants


def check_physics(previous, candidate):
    previous, candidate = Path(previous), Path(candidate)
    metadata = [
        json.loads((path / "scene.json").read_text()) for path in (previous, candidate)
    ]
    for key in ("gravity", "timestep_s", "model_options"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Appearance refresh changed physics: {key}")
    for kind in ("body", "joint", "actuator", "tendon", "equality", "site", "sensor"):
        if metadata[0]["names"].get(kind) != metadata[1]["names"].get(kind):
            raise ValueError(f"Appearance refresh changed {kind} ordering")
    options = []
    for path in (previous, candidate):
        option = ET.parse(path / "scene.xml").getroot().find("option")
        options.append(ET.tostring(option) if option is not None else None)
    if options[0] != options[1]:
        raise ValueError("Appearance refresh changed solver options")
    checked = []

    def equal(name, left, right):
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError(f"Appearance refresh changed physics: {name}")
        checked.append(name)

    with np.load(previous / "model.npz") as left_archive, np.load(
        candidate / "model.npz"
    ) as right_archive:
        excluded = {"body_geomadr", "body_geomnum", "body_bvhadr", "body_bvhnum"}
        prefixes = (
            "body_",
            "jnt_",
            "dof_",
            "actuator_",
            "tendon_",
            "wrap_",
            "eq_",
            "exclude_",
            "pair_",
            "sensor_",
            "site_",
            "key_",
            "hfield_",
            "flex_",
            "plugin_",
        )
        fields = {
            name
            for name in left_archive.files
            if name.startswith(prefixes) and name not in excluded
        }
        required = {
            "qpos0",
            "qpos_spring",
            "reset_qpos",
            "reset_qvel",
            "reset_ctrl",
            "reset_eq_active",
        }
        geometry_fields = [
            name for name in left_archive.files if name.startswith("geom_")
        ]
        mesh_fields = {
            "mesh_vert",
            "mesh_vertadr",
            "mesh_vertnum",
            "mesh_face",
            "mesh_faceadr",
            "mesh_facenum",
        }
        needed = fields | required | set(geometry_fields) | mesh_fields
        # NPZ indexing decompresses an entire field. Cache the needed arrays
        # once before examining individual collision meshes; leave textures out.
        left = {name: left_archive[name] for name in needed}
        right = {name: right_archive[name] for name in needed}
        for name in sorted(fields | required):
            equal(name, left[name], right[name])
        colliders = [
            sorted(collision_participants(model)[0]) for model in (left, right)
        ]
        if len(colliders[0]) != len(colliders[1]):
            raise ValueError("Appearance refresh changed collision geometry count")
        old_ids, new_ids = colliders
        ignored = {
            "geom_dataid",
            "geom_matid",
            "geom_rgba",
            "geom_group",
            "geom_bvhadr",
            "geom_bvhnum",
        }
        for name in geometry_fields:
            if name not in ignored:
                equal(name, left[name][old_ids], right[name][new_ids])
        for old_id, new_id in zip(old_ids, new_ids):
            if int(left["geom_type"][old_id]) != 7:
                continue
            old_mesh, new_mesh = int(left["geom_dataid"][old_id]), int(
                right["geom_dataid"][new_id]
            )
            for field, address, count in (
                ("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
                ("mesh_face", "mesh_faceadr", "mesh_facenum"),
            ):
                parts = []
                for model, mesh in ((left, old_mesh), (right, new_mesh)):
                    start, length = int(model[address][mesh]), int(model[count][mesh])
                    parts.append(model[field][start : start + length])
                equal(f"{field}:{old_id}", *parts)
    return {
        "passed": True,
        "checked_fields": checked,
        "collision_geometries": len(old_ids),
        "physics_model_sha256": hashlib.sha256(
            (previous / "model.npz").read_bytes()
        ).hexdigest(),
        "render_model_sha256": hashlib.sha256(
            (candidate / "model.npz").read_bytes()
        ).hexdigest(),
        "scope": "Unchanged recorded robot dynamics and collision geometry; regenerated visuals only, no new physics episode",
    }
