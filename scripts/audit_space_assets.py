"""Compare exported triangle corners, UVs, normals and pixels to pinned inputs."""

import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from prepare_space_assets import ROOT, SOURCES, collada_parts, digest, fetch, usd_parts


def read_obj(path):
    pools = {"v": [], "vt": [], "vn": []}
    faces = []
    for line in path.read_text().splitlines():
        kind, *values = line.split()
        if kind in pools:
            pools[kind].append([float(value) for value in values])
        elif kind == "f":
            if len(values) != 3:
                raise ValueError("Exported face is not a triangle")
            faces.append(
                [[int(index) - 1 for index in value.split("/")] for value in values]
            )
    faces = np.asarray(faces)
    return [np.asarray(pools[kind])[faces[:, :, i]] for i, kind in enumerate(pools)]


def audit(cache, output):
    sources = json.loads(SOURCES.read_text())["assets"]
    manifest = json.loads((output / "manifest.json").read_text())
    rows = []
    for source, record in zip(sources, manifest["assets"], strict=True):
        if source["id"] != record["id"]:
            raise ValueError("Source/export asset ordering differs")
        path = fetch(source, cache, None, offline=True)
        folder = output / source["id"]
        with tempfile.TemporaryDirectory() as temporary:
            parts = (
                collada_parts(path)
                if path.suffix == ".dae"
                else usd_parts(path, Path(temporary))
            )
            xml = ET.parse(folder / "asset.xml")
            texture_files = {
                item.get("name"): item.get("file")
                for item in xml.findall("./asset/texture")
            }
            materials = xml.findall("./asset/material")
            for part, exported, material in zip(
                parts, record["parts"], materials, strict=True
            ):
                corners = read_obj(folder / f"part_{exported['part']}.obj")
                expected = [
                    part["vertices"][part["faces"]]
                    + record["normalization_translation_m"],
                    part["uv"][part["uv_faces"]],
                    part["normals"][part["normal_faces"]],
                ]
                for actual, original in zip(corners, expected, strict=True):
                    np.testing.assert_allclose(actual, original, rtol=1e-8, atol=1e-8)
                if len(corners[0]) != exported["faces"]:
                    raise ValueError("Exported face count differs from manifest")
                if part["texture"]:
                    converted = output.parent / texture_files[material.get("texture")]
                    with Image.open(part["texture"]) as original, Image.open(
                        converted
                    ) as image:
                        np.testing.assert_array_equal(
                            np.asarray(original.convert("RGB")), np.asarray(image)
                        )
        for name, expected_hash in record["files"].items():
            if digest(folder / name) != expected_hash:
                raise ValueError(f"Exported file digest mismatch: {name}")
        rows.append(
            {
                "id": source["id"],
                "faces": record["source_faces"],
                "parts": len(parts),
                "diffuse_textures": record["diffuse_textures"],
                "passed": True,
            }
        )
    return {
        "kind": "pinned_space_asset_conversion_audit",
        "assets": rows,
        "triangles": sum(row["faces"] for row in rows),
        "passed": True,
        "manifest_sha256": digest(output / "manifest.json"),
        "scope": "Source/export triangle corner positions, UVs, normals, decoded RGB pixels and file hashes. Does not establish PBR, measured physical properties or exact collisions.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "temp/space_asset_research/raw"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "Hooke/assets/space")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    document = audit(args.cache, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document))
