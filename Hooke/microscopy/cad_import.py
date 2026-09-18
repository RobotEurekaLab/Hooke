"""Offline STEP inspection and tessellation; optional OpenCascade dependency.

Generated meshes belong in an ignored private asset directory. Parsing and
meshing verify geometry; they do not grant a vendor redistribution license or
infer movable assemblies, collision bodies, masses or device precision.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def convert(source, output, deflection_mm=.04):
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Tool
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepLib import BRepLib_ToolTriangulatedShape
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TColStd import TColStd_SequenceOfAsciiString
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    if not np.isfinite(deflection_mm) or not 0 < deflection_mm <= 1:
        raise ValueError("Tessellation deflection must be from 0 to 1 mm")
    source, output = Path(source).resolve(), Path(output).resolve()
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError(f"STEP parser rejected {source.name}")
    lengths, angles, solids = (TColStd_SequenceOfAsciiString() for _ in range(3))
    reader.FileUnits(lengths, angles, solids)
    units = [lengths.Value(i).ToCString() for i in range(1, lengths.Length()+1)]
    if not units:
        raise ValueError("STEP has no declared length unit")
    reader.SetSystemLengthUnit(1.)  # OCCT internal millimetres, independent of source unit.
    if not reader.TransferRoots():
        raise ValueError("STEP contains no transferable shape")
    shape = reader.OneShape()
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds, False, False)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_length_units=units, internal_length_unit="mm", mesh_length_unit="m",
                  deflection_mm=deflection_mm, cad_bounds_mm=list(bounds.Get()), parts=[],
                  inspection_only=True, articulation_verified=False,
                  normals="OpenCascade surface/UV normals with kernel fallback; separate CAD-face vertices preserve hard edges",
                  rights="Vendor/source rights remain separate; private geometry inspection only.")
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    transfer = reader.WS().TransferReader()
    while explorer.More():
        solid = explorer.Current()
        entity = transfer.EntityFromShapeResult(solid, 1)
        if entity is None:
            entity = transfer.EntityFromShapeResult(solid.Located(TopLoc_Location()), 1)
        source_name = (entity.Name().ToCString()
                       if entity is not None and hasattr(entity, "Name") else None)
        if not BRepCheck_Analyzer(solid).IsValid():
            raise ValueError(f"CAD solid {len(report['parts'])} is invalid")
        solid_bounds = Bnd_Box()
        BRepBndLib.AddOptimal_s(solid, solid_bounds, False, False)
        cad_bounds = np.asarray(solid_bounds.Get()).reshape(2, 3)
        BRepMesh_IncrementalMesh(solid, deflection_mm, False, .3, True)
        vertices, faces, normals = [], [], []
        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while face_explorer.More():
            face = TopoDS.Face_s(face_explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is None:
                raise ValueError("A CAD face could not be tessellated")
            BRepLib_ToolTriangulatedShape.ComputeNormals_s(face, triangulation)
            if not triangulation.HasNormals():
                raise ValueError("CAD tessellation has no usable surface normals")
            reversed_face = face.Orientation() == TopAbs_REVERSED
            first = len(vertices)
            transform = location.Transformation()
            for i in range(1, triangulation.NbNodes()+1):
                point = triangulation.Node(i).Transformed(transform)
                vertices.append((point.X(), point.Y(), point.Z()))
                normal = triangulation.Normal(i).Transformed(transform)
                vector = np.array([normal.X(), normal.Y(), normal.Z()])
                normals.append(-vector if reversed_face else vector)
            for i in range(1, triangulation.NbTriangles()+1):
                triangle = [first+index-1 for index in triangulation.Triangle(i).Get()]
                faces.append(triangle[::-1] if reversed_face else triangle)
            face_explorer.Next()
        normals = np.asarray(normals)
        if not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-6):
            raise ValueError("CAD tessellation has invalid unit normals")
        # Welding different CAD faces would smooth machined edges. Keep face
        # boundaries in the appearance mesh; inspect closure on a welded copy.
        vertices, faces = np.asarray(vertices)*1e-3, np.asarray(faces, dtype=int)
        # A spherical UV pole can generate a triangle with two numerically
        # identical points. Remove only collapsed edges at double precision,
        # preserving real short edges and separate CAD-face normals.
        scale = max(float(np.abs(vertices).max()), float(np.ptp(vertices, axis=0).max()))
        triangle = vertices[faces]
        edges = np.linalg.norm(triangle-np.roll(triangle, 1, axis=1), axis=2)
        collapsed = edges.min(axis=1) <= 32*np.finfo(float).eps*scale
        part = trimesh.Trimesh(vertices, faces[~collapsed], vertex_normals=normals, process=False)
        welded = trimesh.Trimesh(part.vertices, part.faces, process=True)
        if not len(part.faces) or not np.isfinite(part.vertices).all():
            raise ValueError("CAD tessellation contains no finite triangles")
        filename = f"part-{len(report['parts']):03d}.obj"
        part.export(output/filename, include_normals=True)
        report["parts"].append(dict(file=filename, source_name=source_name,
                                    mesh_sha256=hashlib.sha256((output/filename).read_bytes()).hexdigest(),
                                    vertices=len(part.vertices), faces=len(part.faces),
                                    cad_bounds_mm=cad_bounds.tolist(), bounds_m=part.bounds.tolist(),
                                    bounds_difference_mm=float(np.max(abs(cad_bounds-part.bounds*1e3))),
                                    watertight=bool(welded.is_watertight),
                                    appearance_watertight=bool(part.is_watertight),
                                    collapsed_tessellation_triangles_removed=int(collapsed.sum()),
                                    normals=len(normals)))
        explorer.Next()
    if not report["parts"]:
        raise ValueError("STEP has no solid parts; shell-only import is unsupported")
    (output/"manifest.json").write_text(json.dumps(report, indent=2))
    return report


def inspect_cylinders(source):
    """Inspect mounting surfaces, including the finite extent of each CAD face.

    An analytic cylinder's origin can lie outside its trimmed face. Installation
    shoulders must therefore be checked against finite face bounds and planes.
    Hole/thread classification and assembly fit remain separate checks.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    source = Path(source).resolve()
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError("STEP interface parser rejected the source")
    reader.SetSystemLengthUnit(1.)
    if not reader.TransferRoots():
        raise ValueError("STEP interface source has no transferable shape")
    surfaces, planes = [], []
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        kind = surface.GetType()
        if kind not in (GeomAbs_Cylinder, GeomAbs_Plane):
            explorer.Next()
            continue
        bounds = Bnd_Box()
        BRepBndLib.AddOptimal_s(face, bounds, False, False)
        properties = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, properties)
        centre = properties.CentreOfMass()
        finite = dict(bounds_mm=list(bounds.Get()), area_mm2=float(properties.Mass()),
                      centre_mm=[centre.X(), centre.Y(), centre.Z()])
        if kind == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            point, axis = cylinder.Location(), cylinder.Axis().Direction()
            surfaces.append(dict(radius_mm=float(cylinder.Radius()),
                                 point_mm=[point.X(), point.Y(), point.Z()],
                                 axis=[axis.X(), axis.Y(), axis.Z()], **finite))
        else:
            plane = surface.Plane()
            point, normal = plane.Location(), plane.Axis().Direction()
            planes.append(dict(point_mm=[point.X(), point.Y(), point.Z()],
                               normal=[normal.X(), normal.Y(), normal.Z()], **finite))
        explorer.Next()
    return dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                units="mm", cylinders=surfaces, planes=planes,
                scope="Analytic axes and finite face bounds; normals are unoriented; classification and fit require separate checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deflection-mm", type=float, default=.04)
    parser.add_argument("--preview-gpu", type=int)
    parser.add_argument("--interfaces", action="store_true", help="Inspect analytic mounting cylinder axes")
    args = parser.parse_args()
    report = convert(args.source, args.output, args.deflection_mm)
    if args.interfaces:
        (args.output/"interfaces.json").write_text(json.dumps(inspect_cylinders(args.source), indent=2))
    if args.preview_gpu is not None:
        preview(args.output, args.preview_gpu)
    print(json.dumps(dict(parts=len(report["parts"]), bounds_mm=report["cad_bounds_mm"],
                          units=report["source_length_units"])))


def preview(output, gpu):
    """Render nominal CAD parts at their source assembly positions, without physics claims."""
    import xml.etree.ElementTree as ET

    import mujoco
    from PIL import Image

    from backends.source_renderer import center_directional_shadows, mujoco_renderer
    from microscopy.geometry import camera, numbers

    output = Path(output).resolve()
    report = json.loads((output/"manifest.json").read_text())
    bounds = np.asarray(report["cad_bounds_mm"]).reshape(2, 3)*1e-3
    centre, extent = bounds.mean(axis=0), float(np.max(bounds[1]-bounds[0]))
    root = ET.Element("mujoco", model="manufacturer_cad_geometry_inspection")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "statistic", center=numbers(centre), extent=str(extent))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient=".4 .4 .4", diffuse=".65 .65 .65")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1=".22 .27 .31",
                  rgb2=".08 .12 .16", width="512", height="3072")
    ET.SubElement(asset, "material", name="cad_metal", rgba=".46 .49 .53 1",
                  specular=".65", shininess=".65")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos=numbers(centre+[0, -extent, extent*2]),
                  dir="0 0 -1", diffuse=".8 .8 .8")
    for i, part in enumerate(report["parts"]):
        name = f"part_{i}"
        ET.SubElement(asset, "mesh", name=name, file=str(output/part["file"]))
        ET.SubElement(world, "geom", name=name, type="mesh", mesh=name, material="cad_metal",
                      contype="0", conaffinity="0")
    camera(world, "cad", centre+np.array([.65, -1.4, .8])*extent, centre, 40)
    xml = ET.tostring(root, encoding="unicode")
    (output/"preview.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco_renderer(model, gpu, width=1280, height=960) as renderer:
        renderer.update_scene(data, camera="cad")
        center_directional_shadows(renderer, model)
        Image.fromarray(renderer.render()).save(output/"cad-preview.png")


if __name__ == "__main__":
    main()
