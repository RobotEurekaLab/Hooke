"""CPU screening of scene assembly, control travel and target camera access.

This reports sampled mechanical failures and review candidates. It does not
certify collision-free paths, human access, graspability or scientific tasks.
Run against current sources; generated reports belong under ignored temp/.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.catalog import scenes
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.export import _portable_resources
from real_labs.layout_audit import model_obstacles, doorway_obstacle_warnings
from real_labs.runtime import InstrumentSession


_ROBOT_KINDS = {
    "robot_arm",
    "rover",
    "mobile_robot",
    "manchester_gen3_inspection_workspace",
}


def _geom_record(model, index):
    body = int(model.geom_bodyid[index])
    ancestor = body
    free_body = None
    while ancestor:
        first = int(model.body_jntadr[ancestor])
        count = int(model.body_jntnum[ancestor])
        if any(
            model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            for j in range(first, first + count)
        ):
            free_body = model.body(ancestor).name
            break
        ancestor = int(model.body_parentid[ancestor])
    return dict(
        id=int(index),
        name=model.geom(index).name or f"geom_{index}",
        body=model.body(body).name or f"body_{body}",
        free_body=free_body,
    )


def _contacts(session, tolerance):
    return [
        dict(
            geoms=[
                _geom_record(session.model, c.geom1),
                _geom_record(session.model, c.geom2),
            ],
            distance_m=float(c.dist),
        )
        for c in session.data.contact
        if c.dist < -tolerance
    ]


def _merge_contacts(target, contacts, state):
    for contact in contacts:
        key = tuple(sorted(g["id"] for g in contact["geoms"]))
        if key not in target or contact["distance_m"] < target[key]["distance_m"]:
            target[key] = dict(contact, state=state)


def _owner(item, geom):
    return any(
        value == item["id"] or value.startswith(item["id"] + "__")
        for value in (geom["name"], geom["body"])
    )


def _collision_classification(session, contacts, *, dynamic=False):
    robots = [
        item for item in session.equipment.values() if item.get("kind") in _ROBOT_KINDS
    ]
    for contact in contacts:
        if any(geom["free_body"] for geom in contact["geoms"]):
            contact["classification"] = (
                "free_body_contact_requires_compliance_review"
                if dynamic
                else "free_carrier_configuration_requires_dynamic_check"
            )
            continue
        contact["classification"] = (
            "requires_motion_planning"
            if any(_owner(item, geom) for item in robots for geom in contact["geoms"])
            else "requires_geometry_or_motion_review"
        )


def camera_target_screening(model, data, manifest, resolution=(1120, 700)):
    """Check projection and opaque line of sight, not optical image quality.

    Render-only cutaways follow the existing review-view convention. Translucent
    surfaces are omitted from opaque rays; refraction and contrast are untested.
    The sample's bounding-sphere estimate is not proof of a graspable object.
    """
    width, height = resolution
    original_groups = model.geom_group.copy()
    alpha = model.geom_rgba[:, 3].copy()
    for i, material in enumerate(model.geom_matid):
        if material >= 0:
            alpha[i] = min(alpha[i], model.mat_rgba[material, 3])
    hidden_in_cutaway = np.array(
        [
            (model.geom(i).name or "").startswith(
                ("arch_roof_", "arch_cutaway_wall_", "arch_luminaire_")
            )
            for i in range(model.ngeom)
        ]
    )
    records = []
    try:
        for item in manifest["equipment"]:
            if not item.get("sample_site"):
                records.append(dict(equipment=item["id"], status="no_sample_interface"))
                continue
            site = model.site(item["sample_site"]).id
            target = data.site_xpos[site]
            radius = float(np.linalg.norm(item.get("sample_size_m", [0, 0, 0])) / 2)
            views = []
            for camera in range(model.ncam):
                name = model.camera(camera).name
                origin = data.cam_xpos[camera]
                vector = target - origin
                distance = float(np.linalg.norm(vector))
                local = data.cam_xmat[camera].reshape(3, 3).T @ vector
                depth = -float(local[2])
                tangent = float(np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
                framed = (
                    depth > 0
                    and abs(local[0]) <= depth * tangent * width / height + radius
                    and abs(local[1]) <= depth * tangent + radius
                )
                row = dict(camera=name, in_frame=bool(framed))
                if framed and distance > 0:
                    visible = alpha >= 0.95
                    if name not in ("interior", "reference"):
                        visible &= ~hidden_in_cutaway
                    model.geom_group[:] = np.where(visible, 0, 1)
                    hit = np.array([-1], dtype=np.int32)
                    first = mujoco.mj_ray(
                        model,
                        data,
                        origin,
                        vector / distance,
                        np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                        1,
                        -1,
                        hit,
                    )
                    clear = first < 0 or first >= distance - radius - 1e-4
                    row.update(
                        opaque_line_of_sight=bool(clear),
                        estimated_target_diameter_px=height
                        * radius
                        / (depth * tangent),
                        first_hit=None if hit[0] < 0 else _geom_record(model, hit[0]),
                    )
                views.append(row)
            candidates = [
                v["camera"]
                for v in views
                if v.get("opaque_line_of_sight")
                and v["estimated_target_diameter_px"] >= 3
            ]
            records.append(
                dict(
                    equipment=item["id"],
                    sample_site=item["sample_site"],
                    attachment=item.get("sample_attachment"),
                    status=(
                        "candidate_camera_view"
                        if candidates
                        else "requires_camera_review"
                    ),
                    candidate_views=candidates,
                    cameras=views,
                    assumed_resolution=list(resolution),
                )
            )
    finally:
        model.geom_group[:] = original_groups
    return records


def audit_model(
    model, manifest, *, samples=21, pair_samples=5, duration=2.0, run_commands=True
):
    """Audit a fresh data state; never change model ranges or collision masks."""
    if samples < 3 or pair_samples < 2 or not 0 < duration <= 30:
        raise ValueError(
            "Require >=3 axis samples, >=2 pair samples and 0<duration<=30"
        )
    session = InstrumentSession(model, manifest)
    penetration_tolerance = 1e-4
    home = _contacts(session, penetration_tolerance)
    cameras = camera_target_screening(model, session.data, manifest)
    contact_coverage = screen_contact_coverage(model)
    doorway = doorway_obstacle_warnings(manifest, model_obstacles(model, session.data))
    range_contacts, dynamic_contacts, drift = {}, {}, {}
    controls = [(key, c) for key, c in session.controls.items() if c.mode == "position"]
    poses = 1

    def sample(values, mode):
        nonlocal poses
        session.reset()
        for key, value in values.items():
            joint = session.controls[key].joint
            session.data.qpos[model.jnt_qposadr[joint]] = value
        mujoco.mj_forward(model, session.data)
        poses += 1
        _merge_contacts(
            range_contacts,
            _contacts(session, penetration_tolerance),
            dict(mode=mode, values={".".join(k): float(v) for k, v in values.items()}),
        )

    for key, control in controls:
        for value in np.linspace(control.lower, control.upper, samples):
            sample({key: value}, "individual_range")
    for (first, a), (second, b) in combinations(controls, 2):
        if first[0] != second[0]:
            continue
        for x, y in product(
            np.linspace(a.lower, a.upper, pair_samples),
            np.linspace(b.lower, b.upper, pair_samples),
        ):
            sample({first: x, second: y}, "same_instrument_pair")
    commands, physics_warnings = [], []
    carrier_displacement = {}
    if run_commands:
        for key, control in session.controls.items():
            session.reset()
            span = control.upper - control.lower
            for fraction in (0.15, 0.85, 0.25):
                target = control.lower + fraction * span
                session.data.ctrl[control.actuator] = target
                steps = max(1, int(np.ceil(duration / model.opt.timestep)))
                chunk = max(1, int(np.ceil(0.02 / model.opt.timestep)))
                start = float(session.data.time)
                context = dict(equipment=key[0], control=key[1], target=target)
                for offset in range(0, steps, chunk):
                    session.advance(min(chunk, steps - offset))
                    mujoco.mj_forward(model, session.data)
                    if (
                        not np.isfinite(session.data.qpos).all()
                        or not np.isfinite(session.data.qvel).all()
                    ):
                        raise RuntimeError("Nonfinite state during control screening")
                    _merge_contacts(
                        dynamic_contacts,
                        _contacts(session, penetration_tolerance),
                        context,
                    )
                if not np.isclose(
                    session.data.time - start,
                    steps * model.opt.timestep,
                    rtol=1e-8,
                    atol=1e-8,
                ):
                    raise RuntimeError(
                        "Physics reset or time discontinuity during audit"
                    )
                error = session._measured(control) - target
                tolerance = max(1e-9, span * 0.025)
                commands.append(
                    dict(
                        context,
                        error=error,
                        tolerance=tolerance,
                        passed=abs(error) <= tolerance,
                        units=control.units,
                        mode=control.mode,
                    )
                )
                for equipment, item in session.equipment.items():
                    if item.get("sample_attachment") != "supported" or not item.get(
                        "process"
                    ):
                        continue
                    observed = session.process_readiness(equipment)
                    if "sample_not_at_target" in observed["reasons"] and (
                        equipment not in carrier_displacement
                        or observed["sample_distance_m"]
                        > carrier_displacement[equipment]["sample_distance_m"]
                    ):
                        carrier_displacement[equipment] = dict(
                            equipment=equipment,
                            while_commanding=context,
                            **{
                                k: observed[k]
                                for k in (
                                    "sample_distance_m",
                                    "load_tolerance_m",
                                    "sample_body",
                                    "sample_site",
                                    "placement_check",
                                )
                            },
                        )
                for other, held in session.controls.items():
                    if other == key:
                        continue
                    deviation = (
                        session._measured(held) - session.data.ctrl[held.actuator]
                    )
                    limit = max(1e-9, (held.upper - held.lower) * 0.025)
                    if abs(deviation) > limit and (
                        other not in drift
                        or abs(deviation) > abs(drift[other]["error"])
                    ):
                        drift[other] = dict(
                            equipment=other[0],
                            control=other[1],
                            error=float(deviation),
                            tolerance=limit,
                            while_commanding=context,
                        )
                for index, warning in enumerate(session.data.warning):
                    if warning.number:
                        physics_warnings.append(
                            dict(
                                index=index, number=int(warning.number), command=context
                            )
                        )
    session.reset()
    ranges, dynamic = list(range_contacts.values()), list(dynamic_contacts.values())
    _collision_classification(session, ranges)
    _collision_classification(session, dynamic, dynamic=True)
    failed_commands = [r for r in commands if not r["passed"]]
    issues = []
    if session.mechanisms:
        issues.append("coupled_mechanisms_require_task_validation")
    if home:
        issues.append("home_pose_interpenetration")
    issues.extend(sorted({r["classification"] for r in ranges}))
    if any(
        r["classification"] == "free_body_contact_requires_compliance_review"
        for r in dynamic
    ):
        issues.append("free_body_contact_requires_compliance_review")
    if any(
        r["classification"] != "free_body_contact_requires_compliance_review"
        for r in dynamic
    ):
        issues.append("transient_rigid_contact_penetration_requires_review")
    if failed_commands:
        issues.append("tracking_not_settled_within_duration")
    if drift:
        issues.append("uncommanded_control_drift")
    if physics_warnings:
        issues.append("physics_warning")
    if carrier_displacement:
        issues.append("free_carrier_left_loading_region")
    if doorway:
        issues.append("doorway_bounds_require_review")
    if any(r["status"] == "requires_camera_review" for r in cameras):
        issues.append("target_camera_access_requires_review")
    # Append this independent structural screen without suppressing any raw issue.
    issues.extend(contact_coverage["issues"])
    return dict(
        id=manifest["id"],
        status="requires_review" if issues else "no_issue_found_in_sampled_checks",
        issues=issues,
        controls=len(session.controls) + len(session.mechanisms),
        sampled_poses=poses,
        command_screening="performed" if run_commands else "not_run",
        velocity_controls_excluded_from_pose_sweep=[
            ".".join(k) for k, c in session.controls.items() if c.mode == "velocity"
        ],
        coupled_mechanisms_excluded_from_scalar_screening=[
            ".".join(key) for key in session.mechanisms
        ],
        home_pose_penetrations=home,
        range_contacts=ranges,
        commands=commands,
        tracking_failures=failed_commands,
        transient_contacts=dynamic,
        uncommanded_control_drift=list(drift.values()),
        physics_warnings=physics_warnings,
        displaced_free_carriers=list(carrier_displacement.values()),
        doorway_bounds_warnings=doorway,
        target_camera_screening=cameras,
        moving_contact_coverage=contact_coverage,
        parameters=dict(
            axis_samples=samples,
            pair_samples=pair_samples,
            command_duration_s=duration,
            contact_tolerance_m=penetration_tolerance,
            contact_sample_period_s=0.02,
        ),
        scope={
            "collision": "Enabled MuJoCo contacts at sampled poses/times only. Kinematic sweeps leave free carriers at reset coordinates; their contacts require separate dynamic interpretation. No continuous swept-volume or visual-only mesh collision guarantee.",
            "robot": "Joint ranges are retained. Robot self/environment collision configurations require motion planning, not a passing task verdict.",
            "targets": "Authored reset pose only: opaque geometric rays and approximate projected extent at a 3 px screening threshold. Opening/loading poses, optical fidelity, human access, graspability and camera calibration are untested.",
            "tasks": "Mechanical controls only; no autonomous robot task or scientific process validated.",
        },
    )


def audit_scene(identifier, **options):
    root, manifest = build_scene(identifier)
    xml = ET.tostring(root, encoding="unicode")
    # Hash this exact captured build, rather than rebuilding after the sweep:
    # another worker may update a scene while an audit is running.
    portable = deepcopy(root)
    _portable_resources(portable)
    scene_digest = hashlib.sha256(
        ET.tostring(portable, encoding="unicode").encode()
    ).hexdigest()
    model = mujoco.MjModel.from_xml_string(xml)
    report = audit_model(model, manifest, **options)
    report.update(
        scene_sha256=scene_digest,
        compiled_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        manifest_sha256=hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        mujoco_version=mujoco.__version__,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--scene", choices=sorted(scenes()))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-commands", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for identifier in sorted(scenes()) if args.all else [args.scene]:
        try:
            report = audit_scene(identifier, run_commands=not args.skip_commands)
        except Exception as error:
            report = dict(id=identifier, status="audit_error", error=str(error))
        (args.output / f"{identifier}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        records.append(report)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("id", "status", "issues", "error")
                    if k in report
                }
            ),
            flush=True,
        )
    (args.output / "audit.json").write_text(
        json.dumps(
            dict(
                scene_count=len(records),
                status_counts=dict(Counter(r["status"] for r in records)),
                issue_scene_counts=dict(
                    Counter(issue for r in records for issue in r.get("issues", []))
                ),
                control_count=sum(r.get("controls", 0) for r in records),
                sampled_poses=sum(r.get("sampled_poses", 0) for r in records),
                command_count=sum(len(r.get("commands", [])) for r in records),
                autonomous_task_validated=False,
                scientific_process_validated=False,
                scenes=records,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
