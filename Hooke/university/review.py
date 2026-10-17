"""Assemble an offline screenshot review from actual, hash-matched scene renders.

This creates contact sheets and a local HTML gallery. It never synthesizes an
image, downloads source photographs, renders a model, or starts a server.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

from real_labs.export import current_scene_digest
from real_labs.render_version import render_source_digest
from university.catalog import affiliations


VIEWS = ("overview", "interior", "workstation")
VIEW_LABELS = {"source": "Reference-oriented render", "afm_detail": "AFM detail"}


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def collect_records(package_root, scene_ids):
    """Validate current exported assets before they enter a screenshot review."""
    package_root = Path(package_root).resolve()
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("A review requires a nonempty, unique list of scenes")
    bindings = affiliations()
    records = []
    for identifier in scene_ids:
        directory = package_root / identifier
        manifest = json.loads((directory / "manifest.json").read_text())
        rendering = json.loads((directory / "render.json").read_text())
        validation = json.loads((directory / "validation.json").read_text())
        for asset in ("laboratory.blend", "laboratory.glb", "laboratory.usdc"):
            if (
                not (directory / asset).is_file()
                or (directory / asset).stat().st_size == 0
            ):
                raise ValueError(
                    f"Missing editable or visual asset: {identifier}/{asset}"
                )
        digest = hashlib.sha256((directory / "scene.xml").read_bytes()).hexdigest()
        if digest != current_scene_digest(identifier):
            raise ValueError(f"Scene source changed after export: {identifier}")
        if rendering.get("renderer_source_sha256") != render_source_digest():
            raise ValueError(f"Renderer changed after screenshot export: {identifier}")
        if not (
            digest
            == manifest["scene_sha256"]
            == rendering["scene_sha256"]
            == validation.get("scene_sha256")
        ):
            raise ValueError(f"Stale render or scene manifest: {identifier}")
        if not validation["passed"]:
            raise ValueError(f"Failed mechanical checks: {identifier}")
        missing_views = set(VIEWS) - set(rendering["views"])
        if missing_views:
            raise ValueError(
                f"Incomplete render views: {identifier}: {sorted(missing_views)}"
            )
        images = {}
        for view in dict.fromkeys((*VIEWS, *rendering["views"])):
            if view not in manifest["camera_names"]:
                raise ValueError(f"Unregistered rendered camera: {identifier}/{view}")
            path = directory / (view + ".png")
            with Image.open(path) as image:
                if list(image.size) != rendering["dimensions"]:
                    raise ValueError(f"Render dimensions disagree: {identifier}/{view}")
                if max(ImageStat.Stat(image.convert("RGB")).stddev) < 3:
                    raise ValueError(
                        f"Blank or near-blank screenshot: {identifier}/{view}"
                    )
            images[view] = dict(
                file=path.relative_to(package_root).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        records.append(
            dict(
                id=identifier,
                institution=bindings[identifier]["institution"],
                lab_name=bindings[identifier]["lab_name"],
                title=manifest["title"],
                scene_sha256=digest,
                controls=validation["control_count"],
                source_urls=manifest["source_urls"],
                layout_fidelity=manifest["layout_fidelity"],
                reference_notes=manifest["reference_notes"],
                images=images,
            )
        )
    return records


def contact_sheets(package_root, records, page_size=10):
    """Paginate equal-aspect images so large batches remain readable."""
    if page_size < 2 or page_size > 20 or page_size % 2:
        raise ValueError("Contact sheet page size must be even, from 2 through 20")
    pages = [records[i : i + page_size] for i in range(0, len(records), page_size)]
    files = []
    for view, title in (
        ("overview", "University laboratories · room overviews"),
        ("workstation", "University laboratories · instrument details"),
    ):
        for page_number, page_records in enumerate(pages, 1):
            suffix = "" if len(pages) == 1 else f"_{page_number:02d}"
            filename = f"{view}_summary{suffix}.jpg"
            _contact_sheet(
                package_root,
                page_records,
                view,
                title,
                filename,
                page_number,
                len(pages),
                len(records),
                (page_number - 1) * page_size,
            )
            files.append(dict(view=view, page=page_number, file=filename))
    return files


def _contact_sheet(
    package_root,
    records,
    view,
    title,
    filename,
    page_number,
    page_count,
    total,
    first_index,
):
    # All pages use identical 8:5 tiles and title bands.
    width, height, gap, band, header = 864, 540, 24, 76, 110
    sheet = Image.new(
        "RGB",
        (
            width * 2 + gap * 3,
            header + (height + band + gap) * ((len(records) + 1) // 2),
        ),
        "#edf2f5",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 22), title, font=_font(31), fill="#203341")
    draw.text(
        (gap, 66),
        f"{total} reference-informed scenes | Page {page_number}/{page_count} | Actual Blender CPU renders",
        font=_font(19),
        fill="#486273",
    )
    for index, record in enumerate(records):
        x, y = gap + index % 2 * (width + gap), header + index // 2 * (
            height + band + gap
        )
        with Image.open(package_root / record["images"][view]["file"]) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)
            sheet.paste(
                thumbnail,
                (
                    x + (width - thumbnail.width) // 2,
                    y + (height - thumbnail.height) // 2,
                ),
            )
        draw.rectangle((x, y + height, x + width, y + height + band), fill="white")
        school = record["institution"]["name"]
        text = f"{first_index + index + 1:03d}  {school} · QS {record['institution']['rank_display']}"
        label_font = _font(22)
        if draw.textlength(text, font=label_font) > width - 24:
            label_font = _font(17)
        draw.text((x + 12, y + height + 8), text, font=label_font, fill="#203341")
        lab = record["lab_name"]
        lab_font = _font(18 if len(lab) < 78 else 15)
        draw.text((x + 12, y + height + 43), lab, font=lab_font, fill="#486273")
    sheet.save(package_root / filename, quality=94, subsampling=0)


def assemble(package_root, batch_path):
    package_root = Path(package_root).resolve()
    batch = json.loads(Path(batch_path).read_text())
    records = collect_records(package_root, batch["scene_ids"])
    sheets = contact_sheets(package_root, records)
    cards = []
    for record in records:
        identifier = record["id"]
        images = "".join(
            f'<a href="{record["images"][view]["file"]}"><figure><img loading="lazy" src="{record["images"][view]["file"]}" alt="{html.escape(record["title"], quote=True)} {view}"><figcaption>{view.title()}</figcaption></figure></a>'
            for view in VIEWS
        )
        sources = " ".join(
            f'<a href="{html.escape(url, quote=True)}">Reference {i + 1}</a>'
            for i, url in enumerate(record["source_urls"])
        )
        extra_views = " ".join(
            f'<a href="{details["file"]}">{html.escape(VIEW_LABELS.get(view, view.replace("_", " ").title()))}</a>'
            for view, details in record["images"].items()
            if view not in VIEWS
        )
        additional = f"<p>Additional views: {extra_views}</p>" if extra_views else ""
        cards.append(
            f'<article id="{identifier}"><h2>{html.escape(record["institution"]["name"])} · QS {record["institution"]["rank_display"]}</h2><h3>{html.escape(record["lab_name"])}</h3><div class="views">{images}</div>{additional}<p>{html.escape(record["reference_notes"])}</p><p>{record["controls"]} mechanically checked controls · <a href="{identifier}/scene.xml">MJCF</a> · <a href="{identifier}/laboratory.blend">Blender</a> · <a href="{identifier}/laboratory.glb">GLB</a> · <a href="{identifier}/laboratory.usdc">USD visual snapshot</a> · <a href="{identifier}/validation.json">Checks</a> · <a href="{identifier}/manifest.json">Evidence manifest</a></p><p>{sources}</p></article>'
        )
    sheet_links = " · ".join(
        f'<a href="{sheet["file"]}">{sheet["view"].title()} {sheet["page"]}</a>'
        for sheet in sheets
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hooke · University laboratory review</title>
<style>body{max-width:1600px;margin:auto;padding:28px;background:#edf2f5;color:#203341;font:16px/1.55 system-ui}header{padding:12px 0 24px}article{background:white;border:1px solid #d9e2e8;padding:24px;border-radius:12px;margin-bottom:28px}h2{font-size:25px;margin:0}h3{font-size:19px;color:#486273}a{color:#176b84}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%;aspect-ratio:1.6;object-fit:contain;background:#dae2e8}figcaption{padding:6px 0}p{max-width:1200px}@media(max-width:850px){.views{grid-template-columns:1fr}}</style>
<header><h1>University laboratory scenes</h1><p>Reference-informed reconstructions from public institutional evidence. Dimensions and unseen geometry include authored estimates. Institution-level accuracy and scientific processes remain unvalidated. Joint checks qualify mechanical controls; USD exports are visual snapshots.</p>"""
    page += f'<p>{len(records)} scenes · {sheet_links} · <a href="review_manifest.json">Artifact provenance</a></p></header>'
    (package_root / "review.html").write_text(page + "\n".join(cards) + "</html>\n")
    report = dict(
        batch=batch["id"],
        scene_count=len(records),
        screenshot_count=sum(len(record["images"]) for record in records),
        contact_sheets=sheets,
        review_scope="File/hash and nonblank-image checks; visual source fidelity requires human review",
        records=records,
    )
    (package_root / "review_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.input, args.batch)
    print(
        json.dumps(
            dict(scenes=report["scene_count"], gallery=str(args.input / "review.html"))
        )
    )


if __name__ == "__main__":
    main()
