"""Build portable MJCF packages, CPU validation, and offline geometry snapshots."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from microscopy.blender_export import export_snapshot
from real_labs.builder import build_scene
from real_labs.catalog import scenes
from real_labs.runtime import InstrumentSession
from real_labs.layout_audit import (
    audit_layout,
    write_floorplan,
    model_obstacles,
    doorway_obstacle_warnings,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _portable_resources(root):
    """Normalize mesh references identically for exported files and freshness checks."""
    resources = []
    for item in root.findall("./asset/*[@file]"):
        source = Path(item.get("file"))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        relative = Path("assets") / (digest[:12] + "_" + source.name)
        item.set("file", relative.as_posix())
        resources.append(
            dict(source=str(source), file=relative.as_posix(), sha256=digest)
        )
    return resources


def current_scene_digest(identifier):
    """Hash today's generated geometry without rendering or writing a package."""
    root, _ = build_scene(identifier)
    _portable_resources(root)
    return hashlib.sha256(ET.tostring(root, encoding="unicode").encode()).hexdigest()


def export_scene(identifier, destination, validate=True):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root, manifest = build_scene(identifier)
    layout = audit_layout(manifest)
    write_json(destination / "layout_audit.json", layout)
    write_floorplan(manifest, destination / "floorplan.svg")
    if not layout["passed"]:
        raise ValueError(f"Authored layout checks failed: {identifier}")
    resources = _portable_resources(root)
    for item in resources:
        target = destination / item["file"]
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(item["source"], target)
    xml = destination / "scene.xml"
    ET.ElementTree(root).write(xml, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(xml))
    session = InstrumentSession(model, manifest)
    # Let free specimen carriers rest on their real collision surfaces before
    # exporting a still. This leaves the authored reset state in MJCF intact.
    mujoco.mj_step(model, session.data, nstep=round(1.0 / model.opt.timestep))
    mujoco.mj_forward(model, session.data)
    specimen_checks = [
        dict(equipment=item["id"], **session.process_readiness(item["id"]))
        for item in manifest["equipment"]
        if item.get("process")
    ]
    if not np.isfinite(session.data.qpos).all():
        raise ValueError(f"Nonfinite specimen settling state: {identifier}")
    obstacles = model_obstacles(model, session.data)
    layout["projected_obstacle_count"] = len(obstacles)
    layout["doorway_obstacle_warnings"] = doorway_obstacle_warnings(manifest, obstacles)
    write_json(destination / "layout_audit.json", layout)
    write_floorplan(manifest, destination / "floorplan.svg", obstacles=obstacles)
    manifest.update(
        resources=resources,
        model=dict(
            geoms=model.ngeom,
            joints=model.njnt,
            actuators=model.nu,
            cameras=model.ncam,
            meshes=model.nmesh,
        ),
        simulator_version=mujoco.__version__,
        visual_snapshot=dict(
            state="gravity-settled specimen carriers",
            simulation_time_s=float(session.data.time),
        ),
        scene_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
    )
    write_json(destination / "manifest.json", manifest)
    snapshot_dir = destination / "snapshot"
    snapshot = export_snapshot(model, session.data, snapshot_dir)
    # Some legacy OBJ files have partial UVs despite using only flat colour.
    # Omit unusable, unreferenced UV channels; textured assets must stay intact.
    textured_materials = {
        m["id"] for m in snapshot["materials"] if any(i >= 0 for i in m["texture_ids"])
    }
    textured_meshes = {
        g["mesh"] for g in snapshot["geoms"] if g["material"] in textured_materials
    }
    omitted = []
    with np.load(snapshot_dir / "geometry.npz", allow_pickle=False) as arrays:
        for mesh in snapshot["meshes"]:
            if mesh["face_texcoords"] and (arrays[mesh["face_texcoords"]] < 0).any():
                if mesh["id"] in textured_meshes:
                    raise ValueError(
                        f"Textured mesh has incomplete UVs: {mesh['name']}"
                    )
                mesh["texcoords"] = mesh["face_texcoords"] = None
                omitted.append(mesh["name"])
    snapshot["unused_incomplete_uv_channels_omitted"] = omitted
    write_json(snapshot_dir / "scene.json", snapshot)
    result = dict(id=identifier, compiled=True, **manifest["model"])
    if validate:
        validation = session.smoke_test()
        validation["scene_sha256"] = manifest["scene_sha256"]
        validation["specimens"] = specimen_checks
        validation["passed"] = validation["passed"] and all(
            check["ready"] for check in specimen_checks
        )
        write_json(destination / "validation.json", validation)
        result["validation"] = validation
    return result


def gallery(destination, records):
    cards = []
    for result in records:
        identifier = result["id"]
        manifest = json.loads((destination / identifier / "manifest.json").read_text())
        equipment = "".join(
            f"<tr><td>{html.escape(e['id'])}</td><td>{html.escape(e['kind'])}</td>"
            f"<td>{len(e['actuators'])}</td><td>{html.escape(e['evidence'])}</td></tr>"
            for e in manifest["equipment"]
        )
        sources = " ".join(
            f'<a href="{html.escape(url, quote=True)}">Reference {i+1}</a>'
            for i, url in enumerate(manifest["source_urls"])
        )
        title = html.escape(manifest["title"])
        fidelity = (
            "Workflow-based design; institutional room layout unverified"
            if manifest["layout_fidelity"] == "workflow_based_design"
            else "Reference-informed layout; room dimensions and placements estimated"
        )
        rationale = "".join(
            f"<li>{html.escape(line)}</li>"
            for line in manifest.get("layout_rationale", [])
        )
        extra_views = "".join(
            f'<a href="{identifier}/{name}.png"><img src="{identifier}/{name}.png" alt="{title}: additional reference-oriented view"></a>'
            for name in manifest["camera_names"]
            if name not in {"overview", "interior", "workstation"}
        )
        detail_evidence = ""
        if manifest.get("reference_detail_evidence"):
            evidence_text = html.escape(
                json.dumps(manifest["reference_detail_evidence"], indent=2)
            )
            detail_evidence = f"<details><summary>Observed details, authored estimates and remaining gaps</summary><pre>{evidence_text}</pre></details>"
        cards.append(
            f"""<article id="{identifier}"><h2>{title}</h2>
<p>{html.escape(manifest['institution'])} · {html.escape(manifest['domain'])}</p>
<p><strong>{fidelity}</strong></p>
<div class="images"><a href="{identifier}/overview.png"><img src="{identifier}/overview.png" alt="{title}: full room"></a>
<a href="{identifier}/interior.png"><img src="{identifier}/interior.png" alt="{title}: human-height interior"></a>
<a href="{identifier}/workstation.png"><img src="{identifier}/workstation.png" alt="{title}: equipment detail"></a>
<a href="{identifier}/floorplan.svg"><img src="{identifier}/floorplan.svg" alt="{title}: estimated dimensioned plan"></a>{extra_views}</div>
<p>{result['geoms']} geometry elements · {result['actuators']} controls · Earth gravity 9.81 m/s²</p>
<p class="links"><a href="{identifier}/scene.xml">MJCF</a><a href="{identifier}/laboratory.blend">Blender</a>
<a href="{identifier}/laboratory.glb">GLB</a><a href="{identifier}/laboratory.usdc">USD (visual)</a>
<a href="{identifier}/manifest.json">Evidence &amp; asset manifest</a><a href="{identifier}/validation.json">Mechanical checks</a><a href="{identifier}/layout_audit.json">Layout checks</a></p>
<details><summary>Layout evidence and assumptions</summary><ul>{rationale}</ul></details>
{detail_evidence}
<details><summary>Equipment and reference evidence</summary><table><tr><th>Equipment</th><th>Type</th><th>Controls</th><th>Evidence</th></tr>{equipment}</table><p>{sources}</p></details></article>"""
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hooke · Research laboratories</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf1f4;color:#182b38;font:16px/1.6 system-ui}
header,main{max-width:1400px;margin:auto;padding:32px}header{padding-bottom:0}h1{font-size:36px;margin-bottom:0}
.eyebrow{letter-spacing:.16em;color:#396e80;font-size:13px}article{background:white;border:1px solid #d8e1e7;border-radius:12px;padding:24px;margin:24px 0}
h2{margin:0}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}.images img{width:100%;aspect-ratio:1.6;object-fit:contain;background:#d9dfe2;border-radius:6px}
a{color:#176a84}.links{display:flex;flex-wrap:wrap;gap:16px}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:8px;border-bottom:1px solid #e0e6ea}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
@media(max-width:700px){header,main{padding:16px}.images{grid-template-columns:1fr}}
</style><header><p class="eyebrow">HOOKE / RESEARCH ENVIRONMENTS</p><h1>Laboratory environments</h1>
<p>Reference-informed room layouts and mechanically operable equipment. Room dimensions and occluded arrangements are estimates; institutions have not validated these models.</p>
<p>Images are CPU-rendered simulation geometry. MJCF contains mechanics; Blender, GLB and USD exports contain visual snapshots. Scientific processes and autonomous robot workflows are not yet validated.</p></header><main>"""
    (destination / "index.html").write_text(page + "\n".join(cards) + "</main></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", choices=sorted(scenes()))
    selection.add_argument("--all", action="store_true")
    selection.add_argument(
        "--batch", type=Path, help="JSON document with a scene_ids list"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    identifiers = sorted(scenes()) if args.all else [args.scene]
    if args.batch:
        identifiers = json.loads(args.batch.read_text())["scene_ids"]
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or not all(isinstance(identifier, str) for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or set(identifiers) - set(scenes())
        ):
            parser.error(
                "Batch scene_ids must be a nonempty list of unique registered scene IDs"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for identifier in identifiers:
        result = export_scene(
            identifier, args.output / identifier, not args.skip_validation
        )
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "validation"}),
            flush=True,
        )
        write_json(args.output / "validation.json", results)
    gallery(args.output, results)
    if any(not result.get("validation", {}).get("passed", True) for result in results):
        raise SystemExit("Mechanical validation failed; see per-scene validation.json")


if __name__ == "__main__":
    main()
