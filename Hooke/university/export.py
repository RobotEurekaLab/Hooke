"""Export existing laboratories under institution / subject / lab directories.

Uses the existing MJCF builder and CPU mechanical checks. Optional previews are
copied from prior renders; no rendering service or network request is started.
"""

import argparse
import html
import json
from pathlib import Path
import shutil

from university.catalog import catalogue


def scene_directory(lab):
    group = (
        "universities" if lab["institution"]["kind"] == "university" else "institutes"
    )
    return Path(
        group, lab["institution_id"], lab["subject_id"], lab["lab_id"], lab["scene_id"]
    )


def export_laboratories(destination, labs, previews=None):
    from real_labs.export import export_scene

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    cards = []
    for lab in labs:
        relative = scene_directory(lab)
        directory = destination / relative
        result = export_scene(lab["scene_id"], directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["classification"] = lab
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        record = dict(
            lab,
            directory=relative.as_posix(),
            validation=result["validation"],
            model={key: value for key, value in result.items() if key != "validation"},
        )
        records.append(record)
        if previews is not None:
            source = Path(previews) / lab["scene_id"] / "overview.png"
            if source.is_file():
                shutil.copyfile(source, directory / "overview.png")
        image = (
            f'<img src="{relative.as_posix()}/overview.png" alt="{html.escape(lab["title"], quote=True)}">'
            if (directory / "overview.png").is_file()
            else ""
        )
        scope = (
            "QS research scope"
            if lab["institution"]["in_scope"]
            else "Additional institution"
        )
        cards.append(
            f'<article><h2>{html.escape(lab["institution"]["name"])}</h2>'
            f'<p>{html.escape(lab["subject_label"])} · {html.escape(lab["lab_name"])}</p>'
            f"<p>{scope} · mechanical prototype</p>{image}"
            f'<p><a href="{relative.as_posix()}/scene.xml">MJCF</a> · '
            f'<a href="{relative.as_posix()}/manifest.json">Evidence and classification</a> · '
            f'<a href="{relative.as_posix()}/validation.json">Mechanical checks</a> · '
            f'<a href="{relative.as_posix()}/floorplan.svg">Estimated floor plan</a></p></article>'
        )
        print(
            json.dumps(
                dict(
                    scene=lab["scene_id"],
                    directory=relative.as_posix(),
                    passed=result["validation"]["passed"],
                )
            ),
            flush=True,
        )
    (destination / "index.json").write_text(
        json.dumps(dict(labs=records), indent=2) + "\n"
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hooke · University laboratories</title>
<style>body{max-width:1200px;margin:auto;padding:32px;font:16px/1.6 system-ui;background:#eef2f5;color:#203341}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}
article{padding:20px;background:white;border-radius:12px}h2{font-size:21px}img{width:100%;aspect-ratio:1.6;object-fit:contain}a{color:#17687e}</style>
<h1>University laboratory scenes</h1>
<p>Reference-informed mechanical prototypes. Layouts are not surveyed or institution-validated.
Images, when available, are previously recorded previews. Scientific workflows and Isaac dynamics are not validated by this export.</p><main>"""
    (destination / "index.html").write_text(
        page + "\n".join(cards) + "</main></html>\n"
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Include existing universities and research institutes",
    )
    selection.add_argument("--institution", help="Stable institution ID")
    selection.add_argument("--scene", help="Existing real_labs scene ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--previews-from",
        type=Path,
        help="Optional existing real_labs render directory",
    )
    args = parser.parse_args()
    directory = catalogue()
    labs = directory["labs"] + directory["other_institutions"]
    if args.institution:
        labs = [lab for lab in labs if lab["institution_id"] == args.institution]
    if args.scene:
        labs = [lab for lab in labs if lab["scene_id"] == args.scene]
    if not labs:
        parser.error(
            "No constructed laboratory matches the selection; research candidates are not scenes"
        )
    records = export_laboratories(args.output, labs, args.previews_from)
    if any(not record["validation"]["passed"] for record in records):
        raise SystemExit(
            "Mechanical validation failed; inspect per-scene validation.json"
        )


if __name__ == "__main__":
    main()
