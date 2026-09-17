"""Fetch pinned research assets and convert their geometry to portable MJCF.

Run with the separate environment in space_asset_requirements.txt. The runtime
uses only generated OBJ/PNG/XML files and needs neither COLLADA nor USD parsers.
Source licenses remain separate from the license of this conversion code.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import ssl
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "Hooke/worlds/external_sources.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(source, cache, context, offline):
    folder = cache / source["repository"].replace("/", "__")
    for resource in source["resources"]:
        path = folder / resource["path"]
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            url = (
                "https://raw.githubusercontent.com/"
                f"{source['repository']}/{source['commit']}/{resource['path']}"
            )
            with urllib.request.urlopen(url, context=context, timeout=45) as response:
                data = response.read(100 * 1024 * 1024 + 1)
            if len(data) > 100 * 1024 * 1024:
                raise ValueError("Trial resource exceeds 100 MiB")
            if hashlib.sha256(data).hexdigest() != resource["sha256"]:
                raise ValueError(f"Source digest mismatch: {resource['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if digest(path) != resource["sha256"]:
            raise ValueError(f"Cached source digest mismatch: {path}")
    return folder / source["resources"][0]["path"]


def collada_parts(path):
    import collada

    document = collada.Collada(str(path))
    if document.assetInfo.upaxis != "Z_UP" or document.assetInfo.unitmeter != 1:
        raise ValueError("This pack expects meter-scale Z-up COLLADA")
    parts = []
    for geometry in document.scene.objects("geometry"):
        for primitive in geometry.primitives():
            if not isinstance(primitive, collada.triangleset.BoundTriangleSet):
                raise NotImplementedError("Only source triangles are qualified")
            effect = primitive.material.effect
            diffuse = effect.diffuse
            texture = (
                (path.parent / diffuse.sampler.surface.image.path).resolve()
                if isinstance(diffuse, collada.material.Map)
                else None
            )
            parts.append(
                {
                    # Keep the source shared point pool. Tiny planar material
                    # groups otherwise cannot form a MuJoCo mesh convex hull.
                    "vertices": np.asarray(primitive.vertex, dtype=float),
                    "faces": np.asarray(primitive.vertex_index, dtype=int),
                    "uv": np.asarray(primitive.texcoordset[0], dtype=float),
                    "uv_faces": np.asarray(primitive.texcoord_indexset[0], dtype=int),
                    "normals": np.asarray(primitive.normal, dtype=float),
                    "normal_faces": np.asarray(primitive.normal_index, dtype=int),
                    "texture": texture,
                    "rgba": [1, 1, 1, 1] if texture else list(diffuse),
                    "shininess": 0.2,
                    "material_source": effect.id,
                }
            )
    return parts


def srgb(values):
    values = np.asarray(values)
    return np.where(
        values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055
    )


def usd_parts(path, texture_cache):
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(path))
    if (
        UsdGeom.GetStageUpAxis(stage) != "Z"
        or UsdGeom.GetStageMetersPerUnit(stage) != 1
    ):
        raise ValueError("This pack expects meter-scale Z-up USD")
    transforms = UsdGeom.XformCache()
    parts = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise NotImplementedError("Only source triangles are qualified")
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
        vertices = (np.column_stack([points, np.ones(len(points))]) @ matrix)[:, :3]
        uv_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if not uv_var or uv_var.GetInterpolation() != "faceVarying":
            raise NotImplementedError("Expected face-varying source UV coordinates")
        uv = np.asarray(uv_var.ComputeFlattened(), dtype=float)
        uv_faces = np.arange(len(uv)).reshape(-1, 3)
        if len(uv_faces) != len(faces):
            raise ValueError("Source UV topology does not match the source faces")
        normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=float)
        interpolation = mesh.GetNormalsInterpolation()
        if interpolation == "vertex":
            normal_faces = faces
        elif interpolation == "faceVarying":
            normal_faces = np.arange(len(normals)).reshape(-1, 3)
        else:
            raise NotImplementedError(f"Unsupported source normals: {interpolation}")
        normals = normals @ np.linalg.inv(matrix[:3, :3]).T
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        groups = []
        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        if subsets:
            for subset in subsets:
                if subset.GetElementTypeAttr().Get() != "face":
                    raise NotImplementedError("Expected face material subsets")
                groups.append(
                    (subset.GetPrim(), np.asarray(subset.GetIndicesAttr().Get()))
                )
            assigned = np.concatenate([indices for _, indices in groups])
            if sorted(assigned.tolist()) != list(range(len(faces))):
                raise ValueError(
                    "Material subsets must cover each source face exactly once"
                )
        else:
            groups = [(prim, np.arange(len(faces)))]
        for owner, indices in groups:
            material, _ = UsdShade.MaterialBindingAPI(owner).ComputeBoundMaterial()
            shader = material.ComputeSurfaceSource()[0]
            if shader.GetIdAttr().Get() != "UsdPreviewSurface":
                raise NotImplementedError(
                    "Expected USD Preview Surface source material"
                )
            color_input = shader.GetInput("diffuseColor")
            texture = None
            if color_input.HasConnectedSource():
                texture_shader = UsdShade.Shader(color_input.GetConnectedSource()[0])
                asset = texture_shader.GetInput("file").Get()
                if path.suffix != ".usdz":
                    raise NotImplementedError(
                        "External USD texture references need a separate resource audit"
                    )
                member = str(PurePosixPath(asset.path))
                if member.startswith("/") or ".." in PurePosixPath(member).parts:
                    raise ValueError("Invalid source package texture path")
                texture = texture_cache / Path(member).name
                with zipfile.ZipFile(path) as package:
                    texture.write_bytes(package.read(member))
            color = [1, 1, 1] if texture else srgb(color_input.Get()).tolist()
            roughness = float(shader.GetInput("roughness").Get())
            parts.append(
                {
                    "vertices": vertices,
                    "faces": faces[indices],
                    "uv": uv,
                    "uv_faces": uv_faces[indices],
                    "normals": normals,
                    "normal_faces": normal_faces[indices],
                    "texture": texture,
                    "rgba": [*color, float(shader.GetInput("opacity").Get())],
                    "shininess": 1 - roughness,
                    "material_source": str(material.GetPath()),
                }
            )
    return parts


def numbers(values):
    return " ".join(format(float(value), ".9g") for value in values)


def write_obj(path, part, offset):
    part = part.copy()
    for pool, indices in [
        ("vertices", "faces"),
        ("uv", "uv_faces"),
        ("normals", "normal_faces"),
    ]:
        used, remap = np.unique(part[indices], return_inverse=True)
        referenced = part[pool][used]
        if (
            pool == "vertices"
            and np.linalg.matrix_rank(referenced - referenced[0], tol=1e-9) < 3
        ):
            continue  # Retain the original shared nonplanar point pool.
        part[pool] = referenced
        part[indices] = remap.reshape(part[indices].shape)
    with path.open("w") as stream:
        for kind, rows in [
            ("v", part["vertices"] + offset),
            ("vt", part["uv"]),
            ("vn", part["normals"]),
        ]:
            for row in rows:
                stream.write(f"{kind} {numbers(row)}\n")
        for face, uv, normal in zip(
            part["faces"], part["uv_faces"], part["normal_faces"]
        ):
            corners = [f"{v + 1}/{t + 1}/{n + 1}" for v, t, n in zip(face, uv, normal)]
            stream.write("f " + " ".join(corners) + "\n")


def convert(source, path, output):
    folder = output / source["id"]
    folder.mkdir(parents=True, exist_ok=True)
    parts = collada_parts(path) if path.suffix == ".dae" else usd_parts(path, folder)
    if not parts:
        raise ValueError("No source geometry")
    points = np.concatenate([part["vertices"] for part in parts])
    if not np.isfinite(points).all():
        raise ValueError("Non-finite source geometry")
    lower, upper = points.min(0), points.max(0)
    offset = -(lower + upper) / 2
    offset[2] = -lower[2]
    root = ET.Element("mujoco", model=source["id"])
    ET.SubElement(root, "compiler", assetdir="../..")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="asset_root")
    dimensions = upper - lower
    inertia = (
        0.1
        * np.array(
            [
                dimensions[1] ** 2 + dimensions[2] ** 2,
                dimensions[0] ** 2 + dimensions[2] ** 2,
                dimensions[0] ** 2 + dimensions[1] ** 2,
            ]
        )
        / 12
    )
    ET.SubElement(
        world,
        "inertial",
        pos=f"0 0 {dimensions[2] / 2}",
        mass="0.1",
        diaginertia=numbers(inertia),
    )
    textures = {}
    records = []
    for i, part in enumerate(parts):
        mesh_name, material_name = f"part_{i}", f"material_{i}"
        mesh_file = folder / f"part_{i}.obj"
        write_obj(mesh_file, part, offset)
        ET.SubElement(
            assets,
            "mesh",
            name=mesh_name,
            file=f"space/{source['id']}/{mesh_file.name}",
        )
        material = ET.SubElement(
            assets,
            "material",
            name=material_name,
            rgba=numbers(part["rgba"]),
            shininess=str(part["shininess"]),
        )
        if part["texture"]:
            texture_source = part["texture"]
            key = digest(texture_source)
            if key not in textures:
                from PIL import Image

                image = folder / f"diffuse_{len(textures)}.png"
                with Image.open(texture_source) as original:
                    original.convert("RGB").save(image)
                textures[key] = f"texture_{len(textures)}"
                ET.SubElement(
                    assets,
                    "texture",
                    name=textures[key],
                    type="2d",
                    file=f"space/{source['id']}/{image.name}",
                )
            material.set("texture", textures[key])
        ET.SubElement(
            world,
            "geom",
            name=mesh_name,
            type="mesh",
            mesh=mesh_name,
            material=material_name,
            contype="0",
            conaffinity="0",
            mass="0",
        )
        records.append(
            {
                "part": i,
                "faces": len(part["faces"]),
                "uv_corners": 3 * len(part["uv_faces"]),
                "material_source": part["material_source"],
                "obj_sha256": digest(mesh_file),
            }
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(folder / "asset.xml", encoding="unicode")
    (folder / "NOTICE.md").write_text(
        f"# {source['label']}\n\n{source['credit']}\n\n"
        f"Source: https://github.com/{source['repository']}/tree/{source['commit']}\n\n"
        f"Terms: {source['terms_url']}\n\n{source['usage_note']}\n\n"
        "Changes: geometry transformed to a centered XY origin with bottom at Z=0; "
        "source triangle/material groups, UV coordinates and normals exported to OBJ; "
        "diffuse images converted to lossless RGB PNG; USD linear base colors converted "
        "to sRGB for MuJoCo. Full PBR/metallic response is not reproduced. "
        "Source physical properties are not inferred from appearance. The XML "
        "uses an explicit illustrative 0.1 kg mass and bounding-box inertia, "
        "not measured properties; cabin and handrail remain fixed in the scenes.\n"
    )
    return {
        **source,
        "bounds_original_m": [lower.tolist(), upper.tolist()],
        "normalization_translation_m": offset.tolist(),
        "dimensions_m": (upper - lower).tolist(),
        "source_faces": sum(part["faces"] for part in records),
        "parts": records,
        "diffuse_textures": len(textures),
        "path": f"{source['id']}/asset.xml",
        "physical_parameters": "Illustrative 0.1 kg mass and bounding-box inertia; not measured source properties.",
        "files": {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "temp/space_asset_research/raw"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "Hooke/assets/space")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ca-bundle", type=Path)
    args = parser.parse_args()
    if args.output.name != "space":
        parser.error(
            "--output must name a space directory to retain portable asset paths"
        )
    context = ssl.create_default_context(
        cafile=str(args.ca_bundle) if args.ca_bundle else None
    )
    source_document = json.loads(SOURCES.read_text())
    rows = []
    for source in source_document["assets"]:
        path = fetch(source, args.cache, context, args.offline)
        rows.append(convert(source, path, args.output))
    document = {
        "schema_version": 1,
        "assets": rows,
        "scope": "Pinned visual geometry/diffuse conversion; no full PBR, calibrated properties, gas seal or scientific qualification.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {"converted": len(rows), "faces": sum(row["source_faces"] for row in rows)}
        )
    )


if __name__ == "__main__":
    main()
