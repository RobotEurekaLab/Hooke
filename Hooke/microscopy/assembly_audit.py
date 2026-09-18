"""Offline surface-clearance audit of actual compiled instrument geometry.

Export uses the ordinary runtime environment. Checking uses optional trimesh
and python-fcl, so CAD inspection dependencies stay outside the application.
This checks one recorded pose, not containment or the complete swept volume.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

GROUPS = ("probe", "gripper", "injector", "holder", "instrument")
PAIR_GROUPS = (("probe", "instrument"), ("gripper", "instrument"), ("injector", "instrument"),
               ("probe", "gripper"), ("probe", "injector"), ("gripper", "injector"),
               ("holder", "instrument"), ("holder", "probe"), ("holder", "gripper"), ("holder", "injector"))


def geometry_group(model, index):
    """Assign the actual body tree, including separate parallel platforms."""
    ancestors = set()
    body = int(model.geom_bodyid[index])
    while body:
        ancestors.add(model.body(body).name)
        body = int(model.body_parentid[body])
    for tool in GROUPS[:-1]:
        if ancestors.intersection(tool + suffix for suffix in ("_mount", "_base", "_platform", "_support")):
            return tool
    return "instrument" if ancestors.intersection(("microscope", "stage_base", "focus")) else ""


def geometry_radii(model):
    """Bound each compiled surface about its body's origin, in metres."""
    radii = np.linalg.norm(model.geom_size, axis=1)
    for index, mesh in enumerate(model.geom_dataid):
        if model.geom_type[index] == 7:
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            radii[index] = np.linalg.norm(model.mesh_vert[start:start + count], axis=1).max()
    return radii + np.linalg.norm(model.geom_pos, axis=1)


def part_mesh(arrays, index):
    import trimesh

    kind, size = int(arrays["geom_type"][index]), arrays["sizes"][index]
    if kind == 7:
        m = int(arrays["dataids"][index])
        va, vn = int(arrays["vertadr"][m]), int(arrays["vertnum"][m])
        fa, fn = int(arrays["faceadr"][m]), int(arrays["facenum"][m])
        return trimesh.Trimesh(arrays["vertices"][va:va+vn],
                              arrays["faces"][fa:fa+fn], process=False)
    if kind == 6:
        return trimesh.creation.box(2*size)
    if kind == 5:
        return trimesh.creation.cylinder(size[0], 2*size[1], sections=48)
    return None


def recorded_native_state(directory, operation):
    """Use the archived model and observed PhysX joints; never integrate MuJoCo."""
    import mujoco

    directory = Path(directory).resolve()
    result = json.loads((directory/"result.json").read_text())
    execution = result.get("execution", {})
    trace = result.get("native_trace")
    if execution.get("backend") == "isaac" and isinstance(trace, dict):
        if (result.get("operation") != operation or not result.get("checks")
                or not all(value is True for value in result["checks"].values()) or trace.get("schema_version") != 1
                or trace.get("backend") != "isaac" or trace.get("physics_engine") != "PhysX"
                or trace.get("file") != "trajectory.npz"):
            raise ValueError("Native audit requires a matching successful live Isaac task")
        runtime, steps = execution["native_runtime"], trace["steps"]
        position_error = float(trace["max_fk_position_error_m"])
        rotation_error = float(trace["max_fk_rotation_error_rad"])
    else:
        if (result.get("backend") != "isaac" or result.get("task") != "microscopy_"+operation
                or result.get("status") != "TASK_SUCCEEDED"):
            raise ValueError("Native audit requires a matching successful Isaac task")
        runtime, steps = result["runtime"], result["steps"]
        position_error = float(result["max_fk_position_error_m"])
        rotation_error = float(result["max_fk_rotation_error_rad"])
    source = directory/"source"
    metadata = json.loads((source/"scene.json").read_text())
    for filename in ("model.npz", "model.mjb"):
        if hashlib.sha256((source/filename).read_bytes()).hexdigest() != metadata["archive_sha256"][filename]:
            raise ValueError("Native source model differs from its archived hash")
    model = mujoco.MjModel.from_binary_path(str(source/"model.mjb"))
    data = mujoco.MjData(model)
    with np.load(source/"model.npz", allow_pickle=False) as snapshot:
        data.qpos[:] = snapshot["reset_qpos"]
    mujoco.mj_kinematics(model, data)
    path = directory/"trajectory.npz"
    if trace is not None and hashlib.sha256(path.read_bytes()).hexdigest() != trace["sha256"]:
        raise ValueError("Observed live native trajectory differs from its archived hash")
    with np.load(path, allow_pickle=False) as trajectory:
        positions, times = trajectory["qpos"], trajectory["time"]
    if (positions.shape != (steps, model.nq) or times.shape != (steps,)
            or not np.isfinite(positions).all() or not np.isfinite(times).all()
            or runtime.get("steps") != steps
            or runtime.get("physics_events_since_load") != steps
            or not np.allclose(np.diff(np.r_[0., times]), model.opt.timestep, rtol=1e-5, atol=1e-9)):
        raise ValueError("Observed native trajectory is incomplete or inconsistent")
    if min(position_error, rotation_error) < 0 or not np.isfinite([position_error, rotation_error]).all():
        raise ValueError("Native kinematic error evidence is invalid")
    # Any local point is displaced at most |x|*angle by a rotation error.
    # Include local primitive offsets and CAD vertices conservatively.
    radius = max(float(np.linalg.norm(model.geom_pos, axis=1).max())
                 +float(np.linalg.norm(model.geom_size, axis=1).max()),
                 float(np.linalg.norm(model.mesh_vert, axis=1).max(initial=0.))
                 +float(np.linalg.norm(model.geom_pos, axis=1).max()))
    error_allowance = 2*(position_error+radius*rotation_error)
    evidence = dict(backend="isaac", physics_engine="PhysX", source_model_sha256=metadata["archive_sha256"]["model.mjb"],
                    seed=result.get("seed", metadata.get("task_info", {}).get("seed", 0)),
                    recorded_task=str(directory), trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    actual_physics_steps=steps, max_fk_position_error_m=position_error,
                    max_fk_rotation_error_rad=rotation_error, native_pair_pose_error_allowance_m=error_allowance)
    return model, data, positions, evidence


def export_geometry(output, operation, execute=False, recorded_native=None):
    import mujoco

    from microscopy.cad_assets import asset_evidence
    from microscopy.tasks import make_task

    if execute and recorded_native is not None:
        raise ValueError("Select source execution or recorded native motion")
    task, recorded_positions = None, None
    provenance = dict(backend="mujoco", physics_engine="MuJoCo")
    if recorded_native is not None:
        model, data, recorded_positions, provenance = recorded_native_state(recorded_native, operation)
    else:
        task = make_task(operation)
        task.reset(0)
        model, data = task.model, task.data
    evidence = asset_evidence(model)
    if evidence["profile"] != "cad" and not any(
            assembly.get("profile") == "parallel-v4" for assembly in evidence.get("assemblies", [])):
        raise ValueError("Assembly audit requires the explicit CAD profile")
    names, groups, excluded = [], [], []
    for i in range(model.ngeom):
        name = model.geom(i).name
        names.append(name)
        group = geometry_group(model, i)
        # Near-field contacts are validated by the task's physics, separately
        # from housing clearance. Non-solid sample visuals are also excluded.
        if (name.endswith(("_tip", "_taper", "_pad")) or name.startswith("cell_")
                or name in ("well_center", "holding_hollow_pipette")):
            if group:
                excluded.append(name)
            group = ""
        groups.append(group)
    for assembly in evidence.get("assemblies", []):
        if assembly.get("profile") == "parallel-v4":
            for binding in assembly["bindings"]:
                for name in binding["geoms"]:
                    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    if index < 0 or groups[index] != assembly["tool"]:
                        raise ValueError(f"CAD part is missing from its clearance group: {name}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"geometry.npz"
    positions = data.geom_xpos.copy()
    rotations = data.geom_xmat.reshape(-1, 3, 3).copy()
    minimum, maximum = positions.copy(), positions.copy()
    # Track each body's pose once, regardless of its number of CAD meshes.
    representatives = {}
    for index, group in enumerate(groups):
        if group:
            representatives.setdefault(int(model.geom_bodyid[index]), index)
    body_columns = {body: column for column, body in enumerate(representatives)}
    segment_map = np.array([body_columns.get(int(b), -1) for b in model.geom_bodyid])
    steps = 0
    from microscopy.rigid_motion import RigidMotionBounds
    body_ids = np.array(list(representatives), dtype=int)
    rigid = RigidMotionBounds(data.xpos[body_ids], data.xmat.reshape(-1, 3, 3)[body_ids])
    # Unclassified geometry has no role in any pair. Map it to column zero
    # only for vectorized storage; every queried mesh must have a real group.
    rigid_columns = np.maximum(segment_map, 0)
    if execute or recorded_native is not None:
        def capture_state():
            nonlocal steps
            np.minimum(minimum, data.geom_xpos, out=minimum)
            np.maximum(maximum, data.geom_xpos, out=maximum)
            rigid.observe(data.xpos[body_ids], data.xmat.reshape(-1, 3, 3)[body_ids])
            steps += 1
        if recorded_positions is not None:
            for qpos in recorded_positions:
                data.qpos[:] = qpos
                mujoco.mj_kinematics(model, data)
                capture_state()
        else:
            original_step = task.manager.step
            def step():
                original_step()
                capture_state()
            task.manager.step = step
            try:
                task.execute()
            finally:
                task.manager.step = original_step
            if not task.check():
                raise ValueError("Failed task cannot establish the intended successful motion envelope")
    np.savez_compressed(path, names=np.asarray(names), groups=np.asarray(groups),
        geom_type=model.geom_type, sizes=model.geom_size, dataids=model.geom_dataid,
        positions=positions, positions_min=minimum, positions_max=maximum, rotations=rotations,
        rigid_geom_body_map=rigid_columns, rigid_geom_local_positions=model.geom_pos,
        rigid_geom_local_rotations=np.array([
            data.xmat.reshape(-1, 3, 3)[body_id].T @ data.geom_xmat.reshape(-1, 3, 3)[index]
            for index, body_id in enumerate(model.geom_bodyid)]),
        rigid_geom_radii=geometry_radii(model), **rigid.arrays(),
        vertices=model.mesh_vert, faces=model.mesh_face,
        vertadr=model.mesh_vertadr, vertnum=model.mesh_vertnum,
        faceadr=model.mesh_faceadr, facenum=model.mesh_facenum)
    (output/"pose.json").write_text(json.dumps(dict(operation=operation, seed=provenance.get("seed", 0),
        pose="Reset state; reference slides centred, gripper open",
        motion_steps=steps, task_success=True if recorded_native is not None else task.check() if execute else None,
        motion_segment_steps=rigid.segment_steps,
        geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        excluded_contact_or_sample_geometries=excluded, assets=evidence, provenance=provenance), indent=2))


def check_geometry(output):
    import trimesh

    output = Path(output)
    pose = json.loads((output/"pose.json").read_text())
    path = output/"geometry.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != pose["geometry_sha256"]:
        raise ValueError("Exported geometry differs from the recorded pose")
    managers = {name: trimesh.collision.CollisionManager() for name in GROUPS}
    counts = {name: 0 for name in GROUPS}
    skipped = []
    envelopes = {name: [] for name in GROUPS}
    with np.load(path, allow_pickle=False) as archive:
        # NpzFile access decompresses on every lookup. Materialize once before
        # traversing hundreds of meshes or thousands of motion intervals.
        arrays = {key: archive[key] for key in archive.files}
        for i, (name, group) in enumerate(zip(arrays["names"], arrays["groups"])):
            if not group:
                continue
            mesh = part_mesh(arrays, i)
            if mesh is None:
                skipped.append(str(name))
                continue
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = arrays["rotations"][i], arrays["positions"][i]
            managers[group].add_object(str(name), mesh, matrix)
            oriented = mesh.vertices@matrix[:3, :3].T
            angular_allowance = (arrays["rigid_geom_radii"][i]
                * arrays["rigid_angle_max"][arrays["rigid_geom_body_map"][i]]
                if "rigid_angle_max" in arrays else 0.)
            envelopes[group].append((str(name), np.array([
                oriented.min(axis=0)+arrays["positions_min"][i]-angular_allowance,
                oriented.max(axis=0)+arrays["positions_max"][i]+angular_allowance])))
            counts[group] += 1
    pairs = {}
    swept = {}
    if not counts["instrument"] or not any(counts[tool] for tool in GROUPS[:-1]):
        raise ValueError("Clearance audit requires instrument surfaces and classified tool surfaces")
    for first, second in PAIR_GROUPS:
        if not counts[first] or not counts[second]:
            continue
        collision, names = managers[first].in_collision_other(managers[second], return_names=True)
        pairs[first+" vs "+second] = dict(surface_intersection=bool(collision),
                                         pairs=sorted([list(pair) for pair in names]))
        if pose["motion_steps"]:
            first_bounds = np.array([item[1] for item in envelopes[first]])
            second_bounds = np.array([item[1] for item in envelopes[second]])
            overlaps = np.all((first_bounds[:, None, 0] <= second_bounds[None, :, 1])
                              & (second_bounds[None, :, 0] <= first_bounds[:, None, 1]), axis=2)
            candidates = [[envelopes[first][i][0], envelopes[second][j][0]]
                          for i, j in zip(*np.nonzero(overlaps))]
            swept[first+" vs "+second] = dict(disjoint_envelopes=not bool(candidates),
                                              candidate_pairs=candidates)
    report = dict(operation=pose["operation"], geometry_sha256=pose["geometry_sha256"],
        method="FCL triangle-surface intersection on actual compiled world poses",
        scope="One reset pose; not containment, same-assembly fits, full travel or swept-volume validation",
        counts=counts, skipped=skipped, pairs=pairs,
        motion_steps=pose["motion_steps"], motion_envelopes=swept,
        provenance=pose.get("provenance", dict(backend="mujoco", physics_engine="MuJoCo")),
        motion_scope="Conservative axis-aligned envelopes over every recorded physics step; overlapping boxes are candidates, not confirmed collisions",
        surface_clear=not any(pair["surface_intersection"] for pair in pairs.values()))
    (output/"report.json").write_text(json.dumps(report, indent=2))
    if pose["motion_steps"]:
        report = resolve_motion_candidates(output, allowance_m=.0001+report["provenance"].get("native_pair_pose_error_allowance_m", 0.))
    return report


def resolve_motion_candidates(output, allowance_m=None):
    """Bound clearance by distance minus translation and angular displacement.

    Each point moves at most the body's translation plus its radius times
    angular displacement. Subtract the two point bounds from surface distance.
    An overlapping envelope is resolved only when the resulting clearance
    exceeds the declared geometric allowance. Older translation-only exports
    remain readable.
    """
    import fcl
    from microscopy.rigid_motion import body_displacement, geom_pose

    output = Path(output)
    path = output/"geometry.npz"
    report = json.loads((output/"report.json").read_text())
    if allowance_m is None:
        allowance_m = .0001+report.get("provenance", {}).get("native_pair_pose_error_allowance_m", 0.)
    if not np.isfinite(allowance_m) or allowance_m <= 0:
        raise ValueError("Motion clearance allowance must be finite and positive")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["geometry_sha256"]:
        raise ValueError("Motion report and geometry do not match")
    objects = {}
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
        indices = {str(name): i for i, name in enumerate(arrays["names"])}
        displacement = np.linalg.norm(np.maximum(abs(arrays["positions_min"]-arrays["positions"]),
                                                abs(arrays["positions_max"]-arrays["positions"])), axis=1)
        rigid = "rigid_body_positions" in arrays
        if rigid:
            displacement = body_displacement(arrays["rigid_body_positions"], arrays["rigid_positions_min"],
                arrays["rigid_positions_max"], arrays["rigid_angle_max"],
                arrays["rigid_geom_body_map"], arrays["rigid_geom_radii"])
            segment_displacements = body_displacement(arrays["rigid_segment_positions"],
                arrays["rigid_segment_min"], arrays["rigid_segment_max"], arrays["rigid_segment_angles"],
                arrays["rigid_geom_body_map"], arrays["rigid_geom_radii"])
        def collision_object(name):
            if name not in objects:
                index = indices[name]
                mesh = part_mesh(arrays, index)
                if mesh is None:
                    raise ValueError("Motion candidate has unsupported geometry")
                bvh = fcl.BVHModel()
                bvh.beginModel(len(mesh.vertices), len(mesh.faces))
                bvh.addSubModel(mesh.vertices, mesh.faces)
                bvh.endModel()
                objects[name] = fcl.CollisionObject(bvh, fcl.Transform(
                    arrays["rotations"][index], arrays["positions"][index]))
            return objects[name]
        for envelope in report["motion_envelopes"].values():
            resolved, unresolved = [], []
            for first, second in envelope["candidate_pairs"]:
                distance = float(fcl.distance(collision_object(first), collision_object(second)))
                bound = float(displacement[indices[first]]+displacement[indices[second]])
                margin = distance-bound
                evidence = dict(pair=[first, second], initial_distance_m=distance,
                                clearance_lower_bound_m=margin)
                evidence["maximum_relative_rigid_displacement_bound_m" if rigid
                         else "maximum_relative_translation_bound_m"] = bound
                if margin <= allowance_m and rigid and len(arrays["rigid_segment_positions"]):
                    margin = float("inf")
                    for segment, (positions, rotations) in enumerate(zip(
                            arrays["rigid_segment_positions"], arrays["rigid_segment_rotations"])):
                        for name in (first, second):
                            index = indices[name]
                            objects[name].setTransform(fcl.Transform(*geom_pose(arrays, index, positions, rotations)))
                        distance_segment = float(fcl.distance(objects[first], objects[second]))
                        bound_segment = float(segment_displacements[segment, indices[first]]
                                              + segment_displacements[segment, indices[second]])
                        margin = min(margin, distance_segment-bound_segment)
                        if margin <= allowance_m:
                            break
                    evidence.update(segmented_clearance_lower_bound_m=margin,
                                    segments=len(arrays["rigid_segment_positions"]), evaluated_segments=segment+1)
                    for name in (first, second):
                        index = indices[name]
                        objects[name].setTransform(fcl.Transform(arrays["rotations"][index], arrays["positions"][index]))
                elif margin <= allowance_m and "segment_positions" in arrays and len(arrays["segment_positions"]):
                    segment_margin = float("inf")
                    for start, lower, upper in zip(arrays["segment_positions"], arrays["segment_min"], arrays["segment_max"]):
                        bound_segment = 0.
                        for name in (first, second):
                            index = indices[name]
                            column = int(arrays["segment_map"][index]) if "segment_map" in arrays else index
                            offset = (arrays["positions"][index]-arrays["positions"][arrays["segment_geoms"][column]]
                                      if "segment_map" in arrays else 0.)
                            objects[name].setTransform(fcl.Transform(arrays["rotations"][index], start[column]+offset))
                            bound_segment += float(np.linalg.norm(np.maximum(abs(lower[column]-start[column]),
                                                                             abs(upper[column]-start[column]))))
                        distance_segment = float(fcl.distance(objects[first], objects[second]))
                        segment_margin = min(segment_margin, distance_segment-bound_segment)
                    evidence.update(segmented_clearance_lower_bound_m=segment_margin,
                                    segments=len(arrays["segment_positions"]))
                    margin = segment_margin
                    # Restore the reference pose before another pair is checked.
                    for name in (first, second):
                        index = indices[name]
                        objects[name].setTransform(fcl.Transform(arrays["rotations"][index], arrays["positions"][index]))
                (resolved if margin > allowance_m else unresolved).append(evidence)
            envelope.update(cleared_by_distance_bound=resolved, unresolved=unresolved)
    report["motion_geometric_allowance_m"] = allowance_m
    report["recorded_motion_mesh_clearance_established"] = bool(report.get("surface_clear", True)
        and report["motion_steps"] and not report["skipped"] and report["motion_envelopes"]
        and all(not envelope["unresolved"] for envelope in report["motion_envelopes"].values()))
    report["motion_scope"] = "Conservative rigid displacement bounds over every recorded physics step; surface distance minus translation and radius-times-angle bounds for the full episode or each recorded segment, with a geometric allowance; excludes unobserved motion, containment, same-assembly fits and independent full-travel motions"
    (output/"report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "check", "resolve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=("push", "pick_place", "injection", "cell_injection", "suction_injection"), default="push")
    parser.add_argument("--execute", action="store_true", help="Record every step of the completed source task")
    parser.add_argument("--recorded-native", type=Path, help="Audit archived successful PhysX joint observations without source physics")
    args = parser.parse_args()
    if args.mode == "export":
        export_geometry(args.output, args.operation, args.execute, args.recorded_native)
        print("Compiled geometry and pose exported")
    else:
        report = check_geometry(args.output) if args.mode == "check" else resolve_motion_candidates(args.output)
        print(json.dumps({key: report[key] for key in ("counts", "pairs", "surface_clear", "skipped")}))


if __name__ == "__main__":
    main()
