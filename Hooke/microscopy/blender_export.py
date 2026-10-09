"""Portable visual snapshot of observed MuJoCo geometry, without Blender.

The export contains geometry and appearance only. Positions are in metres;
rotation matrices map local coordinates into the world. Compiled mesh vertices
already include the compiler's scale and principal-axis transformation, so a
consumer applies only the exported geom pose. No simulation is advanced.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


SUPPORTED_TYPES = {0, 2, 3, 4, 5, 6, 7}


def _store(arrays, name, values):
    arrays[name] = np.array(values, copy=True)
    return name


def _mesh(model, index, arrays):
    prefix = f"mesh_{index}"
    vertex, vertices = int(model.mesh_vertadr[index]), int(model.mesh_vertnum[index])
    face, faces = int(model.mesh_faceadr[index]), int(model.mesh_facenum[index])
    normal, normals = int(model.mesh_normaladr[index]), int(model.mesh_normalnum[index])
    record = dict(id=index, name=model.mesh(index).name,
        vertices=_store(arrays, prefix+"_vertices", model.mesh_vert[vertex:vertex+vertices]),
        faces=_store(arrays, prefix+"_faces", model.mesh_face[face:face+faces]),
        normals=_store(arrays, prefix+"_normals", model.mesh_normal[normal:normal+normals]),
        face_normals=_store(arrays, prefix+"_face_normals", model.mesh_facenormal[face:face+faces]),
        texcoords=None, face_texcoords=None)
    coord, coords = int(model.mesh_texcoordadr[index]), int(model.mesh_texcoordnum[index])
    if coord >= 0 and coords:
        record["texcoords"] = _store(arrays, prefix+"_texcoords", model.mesh_texcoord[coord:coord+coords])
        record["face_texcoords"] = _store(arrays, prefix+"_face_texcoords", model.mesh_facetexcoord[face:face+faces])
    return record


def _material(model, index):
    return dict(id=index, name=model.mat(index).name, rgba=model.mat_rgba[index].tolist(),
        specular=float(model.mat_specular[index]), shininess=float(model.mat_shininess[index]),
        reflectance=float(model.mat_reflectance[index]), metallic=float(model.mat_metallic[index]),
        roughness=float(model.mat_roughness[index]), texture_ids=model.mat_texid[index].tolist(),
        texrepeat=model.mat_texrepeat[index].tolist(), texuniform=bool(model.mat_texuniform[index]),
        emission=float(model.mat_emission[index]))


def _texture(model, index, arrays):
    width, height, channels = (int(getattr(model, "tex_"+key)[index])
                               for key in ("width", "height", "nchannel"))
    start = int(model.tex_adr[index])
    pixels = model.tex_data[start:start+width*height*channels].reshape(height, width, channels)
    return dict(id=index, name=model.tex(index).name, type=int(model.tex_type[index]),
                width=width, height=height, channels=channels,
                array=_store(arrays, f"texture_{index}", pixels))


def export_snapshot(model, data, output, visuals=()):
    """Save actual geom/camera poses and optional renderer-neutral cell shapes."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    geoms, mesh_ids, material_ids = [], set(), set()
    for index in range(model.ngeom):
        material = int(model.geom_matid[index])
        rgba = model.geom_rgba[index]
        if material >= 0 and np.array_equal(rgba, [.5, .5, .5, 1.]):
            rgba = model.mat_rgba[material]
        if model.geom_group[index] >= 3 or rgba[3] <= 0:
            continue
        kind = int(model.geom_type[index])
        if kind not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported snapshot geometry: {model.geom(index).name}, type {kind}")
        mesh = int(model.geom_dataid[index]) if kind == 7 else None
        geoms.append(dict(name=model.geom(index).name or f"geom_{index}", type=kind,
            size=model.geom_size[index].tolist(), pos=data.geom_xpos[index].tolist(),
            mat=data.geom_xmat[index].tolist(), rgba=rgba.tolist(), material=material,
            mesh=mesh, source="model"))
        if mesh is not None:
            mesh_ids.add(mesh)
        if material >= 0:
            material_ids.add(material)
    for index, shape in enumerate(visuals):
        if shape["rgba"][3] <= 0:
            continue
        kind = int(shape["type"])
        if kind not in SUPPORTED_TYPES - {7}:
            raise ValueError(f"Unsupported runtime snapshot geometry type {kind}")
        item = dict(name=f"runtime_{index}_{shape.get('role', 'visual')}", type=kind,
                    size=list(shape["size"]), pos=list(shape["pos"]),
                    mat=np.asarray(shape["mat"]).ravel().tolist(), rgba=list(shape["rgba"]),
                    material=-1, mesh=None, source="runtime")
        if "surface" in shape:
            item["surface"] = shape["surface"]
        geoms.append(item)
    arrays = {}
    materials = [_material(model, index) for index in sorted(material_ids)]
    texture_ids = {index for material in materials for index in material["texture_ids"] if index >= 0}
    scene = dict(schema_version=1, units="m", time_s=float(data.time), arrays="geometry.npz",
        geoms=geoms, meshes=[_mesh(model, index, arrays) for index in sorted(mesh_ids)],
        materials=materials, textures=[_texture(model, index, arrays) for index in sorted(texture_ids)],
        cameras=[dict(name=model.camera(index).name, pos=data.cam_xpos[index].tolist(),
                      mat=data.cam_xmat[index].tolist(), fovy=float(model.cam_fovy[index]))
                 for index in range(model.ncam)],
        lights=[dict(name=model.light(index).name, pos=data.light_xpos[index].tolist(),
                     dir=data.light_xdir[index].tolist(), directional=bool(model.light_directional[index]),
                     diffuse=model.light_diffuse[index].tolist(),
                     specular=model.light_specular[index].tolist(),
                     castshadow=bool(model.light_castshadow[index]))
                for index in range(model.nlight)])
    # Reject nonfinite values rather than producing a snapshot that fails much
    # later in an offline renderer. JSON permits NaN unless explicitly disabled.
    encoded = json.dumps(scene, indent=2, allow_nan=False)+"\n"
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Snapshot geometry contains nonfinite values")
    np.savez_compressed(output / "geometry.npz", **arrays)
    (output / "scene.json").write_text(encoded)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Compiled MuJoCo .mjb")
    parser.add_argument("--state", type=Path, required=True, help="Recorded state .npz")
    parser.add_argument("--visuals", type=Path, help="Recorded runtime visual shapes .json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    with np.load(args.state, allow_pickle=False) as state:
        if "qpos" not in state:
            raise ValueError("Recorded state must include qpos")
        for name in ("qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat", "userdata"):
            if name in state:
                values = state[name]
                if values.shape != getattr(data, name).shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid recorded {name}")
                getattr(data, name)[:] = values
        if "time" in state:
            data.time = float(state["time"])
    mujoco.mj_forward(model, data)
    visuals = json.loads(args.visuals.read_text()) if args.visuals else ()
    scene = export_snapshot(model, data, args.output, visuals)
    sources = dict(model=args.model, state=args.state)
    if args.visuals:
        sources["visuals"] = args.visuals
    provenance = {key: dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for key, path in sources.items()}
    (args.output / "sources.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output.resolve()), geoms=len(scene["geoms"]),
                          meshes=len(scene["meshes"]), materials=len(scene["materials"]))))


if __name__ == "__main__":
    main()
