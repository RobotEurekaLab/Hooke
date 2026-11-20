"""Publish a compact website catalogue and portable QS-scope MJCF collection.

Reuses recorded Blender previews; builds current models on the CPU. No renderer,
simulation step, network request or source photograph is included in publication.
All output locations must be supplied explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from university.catalog import catalogue

ROOT = Path(__file__).resolve().parents[2]
GITHUB = "https://github.com/RobotEurekaLab/Hooke/blob/main/"


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def source_files():
    folder = ROOT / "Hooke/real_labs"
    result = {}
    for path in [folder / "catalog.json", *sorted((folder / "scenes").glob("*.json"))]:
        for row in json.loads(path.read_text())["scenes"]:
            result[row["id"]] = path.relative_to(ROOT).as_posix()
    return result


def preview_source(identifier, roots, view):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", view):
        raise ValueError(f"Invalid preview view: {view!r}")
    for root in roots:
        image = root / identifier / f"{view}.png"
        if image.is_file():
            return image
    raise FileNotFoundError(f"No recorded {view} preview for {identifier}")


def selected_previews(identifiers, roots):
    views = json.loads((Path(__file__).parent / "preview_views.json").read_text())
    return {
        identifier: preview_source(identifier, roots, views[identifier])
        for identifier in identifiers
    }


def encode_preview(source, target, workspace):
    from PIL import Image

    with Image.open(source) as original:
        preview = original.convert("RGB")
        preview.thumbnail((800, 500), Image.Resampling.LANCZOS)
        temporary = workspace / target.name
        preview.save(temporary, format="WEBP", quality=82, method=6)
        temporary.replace(target)
        return list(preview.size)


def refresh_previews(preview_roots, media_directory, catalog_path, workspace):
    """Refresh gallery views without rebuilding or changing the model archive."""
    records = json.loads(catalog_path.read_text())
    sources = selected_previews((row["id"] for row in records["scenes"]), preview_roots)
    workspace.mkdir(parents=True, exist_ok=True)
    media_directory.mkdir(parents=True, exist_ok=True)
    audit = []
    for row in records["scenes"]:
        source = sources[row["id"]]
        target = media_directory / f"{row['id']}.webp"
        dimensions = encode_preview(source, target, workspace)
        row["view"] = source.stem
        row["preview_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        audit.append(dict(
            scene=row["id"], view=source.stem, preview_source=str(source.resolve()),
            preview_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            preview_sha256=row["preview_sha256"],
            preview_dimensions=dimensions,
        ))
    temporary_catalog = workspace / catalog_path.name
    temporary_catalog.write_bytes(json_bytes(records))
    temporary_catalog.replace(catalog_path)
    result = dict(scene_count=len(audit), model_archive_changed=False, scenes=audit)
    (workspace / "previews.json").write_bytes(json_bytes(result))
    return result


def portable_scene(identifier):
    """Build current geometry and collect portable resources without local paths."""
    from real_labs.builder import build_scene

    root, manifest = build_scene(identifier)
    files = {"LICENSE": (ROOT / "LICENSE").read_bytes()}
    resources = []
    for item in root.findall("./asset/*[@file]"):
        source = Path(item.get("file")).resolve()
        repository_path = source.relative_to(ROOT).as_posix()
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        name = f"assets/{digest[:12]}_{source.name}"
        item.set("file", name)
        files[name] = content
        license_source = next(
            parent / "LICENSE" for parent in source.parents
            if parent.is_relative_to(ROOT) and (parent / "LICENSE").is_file()
        )
        license_name = (
            "LICENSE" if license_source.parent == ROOT
            else f"licenses/{license_source.parent.name}-LICENSE"
        )
        files[license_name] = license_source.read_bytes()
        resources.append(dict(
            file=name, sha256=digest, source_repository_path=repository_path,
            license=license_name,
        ))
    files["scene.xml"] = ET.tostring(root, encoding="unicode").encode()
    manifest.update(
        resources=resources,
        scene_sha256=hashlib.sha256(files["scene.xml"]).hexdigest(),
        publication_validation="MuJoCo compile only; no new dynamics or scientific validation",
    )
    return files, public_manifest(manifest)


def public_manifest(value):
    """Keep public evidence and hashes without exposing unshipped local paths."""
    if isinstance(value, dict):
        return {key: public_manifest(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_manifest(item) for item in value]
    if isinstance(value, str) and not value.startswith(("http://", "https://")):
        return re.sub(r"\btemp/[^\s;,\])]+", "[local reference archive; not distributed]", value)
    return value


def compile_scene(files, workspace):
    import mujoco

    with tempfile.TemporaryDirectory(prefix="compile-", dir=workspace) as directory:
        directory = Path(directory)
        for name, content in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        return dict(
            compiled=True, simulator_version=mujoco.__version__,
            geoms=model.ngeom, joints=model.njnt, actuators=model.nu,
        )


def archive_file(archive, name, content):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compresslevel=9)


def publish(preview_roots, media_directory, catalog_path, archive_path, workspace):
    directory = catalogue()
    labs = sorted(
        (row for row in directory["labs"] if row["institution"]["in_scope"]),
        key=lambda row: (row["institution"]["name"].casefold(), row["lab_name"].casefold()),
    )
    sources = source_files()
    # Check the complete input set before producing partial public output.
    previews = selected_previews((row["scene_id"] for row in labs), preview_roots)
    for path in (media_directory, catalog_path.parent, archive_path.parent, workspace):
        path.mkdir(parents=True, exist_ok=True)
    records, audit = [], []
    temporary_archive = workspace / "qs100-scenes-mjcf.zip"
    with zipfile.ZipFile(temporary_archive, "w") as archive:
        archive_file(archive, "LICENSE", (ROOT / "LICENSE").read_bytes())
        archive_file(archive, "README.md", (
            "# Hooke QS top-100 laboratory scene collection\n\n"
            f"{len(labs)} reference-informed mechanical scene prototypes at "
            f"{len({lab['institution_id'] for lab in labs})} institutions "
            "in the repository's pinned QS 2027 rank <=100 roster, including ties.\n\n"
            "Each scene directory contains scene.xml, manifest.json and any required assets/. "
            "Extract the entire archive before opening scene.xml with MuJoCo. "
            "All model paths are relative; no Hooke checkout is needed to compile them.\n\n"
            "Models were generated from the published Hooke source and compiled with MuJoCo. "
            "This publication check does not repeat dynamics, robot tasks or scientific validation. "
            "The models are estimated reconstructions, not surveyed or institution-validated twins. "
            "They do not provide validated native Isaac articulations.\n\n"
            "The website uses previously recorded Blender views; these can predate mechanical "
            "refinements in the models. Source photographs, raw logs and Blender intermediates "
            "are not included.\n\n"
            "Source and rebuild instructions: https://github.com/RobotEurekaLab/Hooke/blob/main/"
            "docs/university_labs.md\n"
        ).encode())
        for lab in labs:
            identifier = lab["scene_id"]
            source = previews[identifier]
            files, manifest = portable_scene(identifier)
            compiled = compile_scene(files, workspace)
            manifest["publication_check"] = compiled
            manifest["classification"] = lab
            files["manifest.json"] = json_bytes(manifest)
            for name, content in sorted(files.items()):
                archive_file(archive, f"{identifier}/{name}", content)
            target = media_directory / f"{identifier}.webp"
            dimensions = encode_preview(source, target, workspace)
            render = json.loads((source.parent / "render.json").read_text())
            records.append(dict(
                id=identifier, institution_id=lab["institution_id"],
                school=lab["institution"]["name"], rank=lab["institution"]["rank"],
                subject=lab["subject_label"], lab=lab["lab_name"], title=lab["title"],
                preview=f"../docs/assets/qs100/{identifier}.webp",
                preview_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                source=GITHUB + sources[identifier], reference=lab["source_urls"][0],
                download="../docs/assets/qs100-scenes-mjcf.zip",
                archive_directory=identifier, view=source.stem, model_stage="mechanical_prototype",
            ))
            audit.append(dict(
                scene=identifier, preview_source=str(source.resolve()),
                preview_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                preview_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                preview_dimensions=dimensions,
                preview_model_sha256=render.get("scene_sha256"),
                model_sha256=manifest["scene_sha256"],
                preview_model_matches_download=render.get("scene_sha256") == manifest["scene_sha256"],
                **compiled,
            ))
            print(json.dumps(dict(scene=identifier, compiled=True)), flush=True)
        archive_file(archive, "index.json", json_bytes(dict(scenes=records)))
    if temporary_archive.stat().st_size >= 40_000_000:
        raise ValueError("Archive exceeds the 40 MB curated-publication budget")
    temporary_archive.replace(archive_path)
    result = dict(
        schema_version=1, ranking_edition=directory["ranking"]["edition"],
        institution_count=len({row["institution_id"] for row in records}), scene_count=len(records),
        preview_note="Recorded Blender scene previews; some precede mechanical refinements in the downloadable models.",
        download="../docs/assets/qs100-scenes-mjcf.zip", download_bytes=archive_path.stat().st_size,
        download_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(), scenes=records,
    )
    temporary_catalog = workspace / catalog_path.name
    temporary_catalog.write_bytes(json_bytes(result))
    temporary_catalog.replace(catalog_path)
    report = dict(
        scene_count=len(records), institution_count=result["institution_count"],
        archive_bytes=result["download_bytes"], archive_sha256=result["download_sha256"],
        preview_bytes=sum((media_directory / f"{row['id']}.webp").stat().st_size for row in records),
        scenes=audit,
    )
    (workspace / "publication.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews-from", type=Path, action="append", required=True)
    parser.add_argument("--media-directory", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--previews-only", action="store_true", help="Update existing gallery images and view labels; keep models unchanged")
    args = parser.parse_args()
    if args.previews_only:
        result = refresh_previews(args.previews_from, args.media_directory, args.catalog, args.work_directory)
    else:
        if args.archive is None:
            parser.error("--archive is required unless --previews-only is used")
        result = publish(args.previews_from, args.media_directory, args.catalog, args.archive, args.work_directory)
    print(json.dumps({key: value for key, value in result.items() if key != "scenes"}))


if __name__ == "__main__":
    main()
