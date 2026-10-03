"""Inspect laboratory simulation contracts without starting Isaac or a renderer.

This local report implements an inventory and an explicit list of missing data.
It is not NVIDIA SimReady validation. Native archives reuse Hooke's existing
compiled-MuJoCo bridge format; their creation does not qualify a new backend.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco

from backends.capabilities import preflight as native_preflight


REFERENCE = "https://developer.nvidia.com/blog/how-to-use-ai-agents-to-prepare-3d-scenes-for-simulation/"


def scene_inventory(model, data, manifest, xml):
    """Keep object identity, physical behavior and review cameras distinguishable."""
    bodies = []
    for index in range(model.nbody):
        bodies.append(
            dict(
                id=index,
                name=model.body(index).name,
                parent_id=int(model.body_parentid[index]),
                fixed_to_world=bool(model.body_weldid[index] == 0),
                joint_count=int(model.body_jntnum[index]),
                mass_kg=float(model.body_mass[index]),
                principal_inertia_kg_m2=model.body_inertia[index].tolist(),
                gravity_compensation=float(model.body_gravcomp[index]),
                mass_basis="Authored model values; not measured equipment mass",
            )
        )
    geometry = []
    for index in range(model.ngeom):
        body = int(model.geom_bodyid[index])
        geometry.append(
            dict(
                id=index,
                name=model.geom(index).name,
                body_id=body,
                shape=mujoco.mjtGeom(int(model.geom_type[index])).name,
                collision_enabled=bool(
                    model.geom_contype[index] or model.geom_conaffinity[index]
                ),
                fixed_to_world=bodies[body]["fixed_to_world"],
                material_id=int(model.geom_matid[index]),
                friction=model.geom_friction[index].tolist(),
                contact_masks=[
                    int(model.geom_contype[index]),
                    int(model.geom_conaffinity[index]),
                ],
            )
        )
    materials = [
        dict(
            id=i,
            name=model.mat(i).name,
            rgba=model.mat_rgba[i].tolist(),
            physical_material_class=None,
            property_calibration="unverified",
        )
        for i in range(model.nmat)
    ]
    equipment = []
    for item in manifest["equipment"]:
        # Reused accessories can expose controls without a sample interface.
        # Keep them in the inventory without inventing a target at their origin.
        target = None
        if item.get("sample_site"):
            site = model.site(item["sample_site"]).id
            attachment = item["sample_attachment"]
            target = dict(
                site=item["sample_site"],
                body_id=int(model.site_bodyid[site]),
                position_m=data.site_xpos[site].tolist(),
                attachment=attachment,
                description=item["sample_description"],
                # A carried site is not automatically a graspable item.
                freely_graspable=False if attachment == "clamped" else None,
                free_body=item.get("process", {}).get("sample_body"),
                grasp_validation="not_run",
            )
        equipment.append(
            dict(
                id=item["id"],
                semantic_class=item["kind"],
                role=item["role"],
                controls=item["joints"],
                actuators=item["actuators"],
                target=target,
                target_interface_status="declared" if target else "not_defined",
                sources=item.get("source_urls", item.get("reference_urls", [])),
            )
        )
    cameras = [
        dict(
            id=i,
            name=model.camera(i).name,
            purpose="review_view",
            position_m=data.cam_xpos[i].tolist(),
            rotation=data.cam_xmat[i].reshape(3, 3).tolist(),
            vertical_fov_degrees=float(model.cam_fovy[i]),
            runtime_resolution=None,
            polling_rate_hz=None,
            robot_target_visibility="not_run",
        )
        for i in range(model.ncam)
    ]
    structural = native_preflight(model, xml)
    return dict(
        schema_version=1,
        scene=manifest["id"],
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        workflow_reference=REFERENCE,
        units=dict(length="metre", angle="radian", mass="kilogram", up_axis="Z"),
        gravity_m_s2=model.opt.gravity.tolist(),
        counts=dict(
            bodies=len(bodies),
            geometry=len(geometry),
            materials=len(materials),
            equipment=len(equipment),
            review_cameras=len(cameras),
            mjcf_sensors=int(model.nsensor),
            controls=int(model.nu),
        ),
        bodies=bodies,
        geometry=geometry,
        materials=materials,
        equipment=equipment,
        cameras=cameras,
        native_structural_preflight=structural,
        acceptance=dict(
            structural_native_attempt=structural["can_attempt_native"],
            isaac_runtime_validated=False,
            simready_validated=False,
            robot_task_validated=False,
            scientific_process_validated=False,
        ),
        exports=dict(
            scene_xml="MuJoCo mechanical source",
            laboratory_usdc="Visual snapshot only; USD physics and semantic APIs unverified",
            native_archive="not_requested",
        ),
        pending=[
            dict(
                id="semantic_usd",
                action="Author equipment and target labels into stable USD prims; retain source identifiers.",
            ),
            dict(
                id="physical_materials",
                action="Classify surfaces and justify or calibrate contact/mass parameters for the target task.",
            ),
            dict(
                id="task_sensors",
                action="Define runtime camera resolution, rate, target frames and task-view visibility; review cameras are not validated robot sensors.",
            ),
            dict(
                id="native_runtime",
                action="Convert through the existing SceneBridge and run native contact, joint, reset and task checks.",
            ),
            dict(
                id="usd_profile",
                action="Select and pin an appropriate SimReady profile, inspect actual USD APIs and run the official validator.",
            ),
        ],
    )


def inspect_package(directory, *, native_archive=False):
    """Inspect the exact portable XML, optionally preparing an existing-bridge input."""
    directory = Path(directory).resolve()
    path = directory / "scene.xml"
    xml = path.read_text()
    manifest = json.loads((directory / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["scene_sha256"]:
        raise ValueError("Scene XML no longer matches the package manifest")
    spec = mujoco.MjSpec.from_file(str(path))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report = scene_inventory(model, data, manifest, xml)
    report["inventory_state"] = (
        "Authored reset state, not the gravity-settled rendering snapshot"
    )
    if native_archive:
        from backends.baseline import write_snapshot

        destination = directory / "native_source"
        destination.mkdir(exist_ok=True)
        source = SimpleNamespace(
            model=model,
            data=data,
            spec=spec,
            default_scene=path,
            task=manifest["task"],
            task_info={},
        )
        archive = write_snapshot(source, destination)
        report["exports"]["native_archive"] = dict(
            path="native_source",
            format="Hooke SceneBridge compiled source",
            source_scene_sha256=manifest["scene_sha256"],
            archive_sha256=archive["archive_sha256"],
            isaac_qualification=archive["isaac_qualification"],
        )
    (directory / "simulation_preflight.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene")
    selection.add_argument("--batch", type=Path)
    parser.add_argument("--native-archive", action="store_true")
    args = parser.parse_args()
    ids = (
        [args.scene] if args.scene else json.loads(args.batch.read_text())["scene_ids"]
    )
    for identifier in ids:
        report = inspect_package(
            args.input / identifier, native_archive=args.native_archive
        )
        print(
            json.dumps(
                dict(
                    scene=identifier,
                    counts=report["counts"],
                    acceptance=report["acceptance"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
