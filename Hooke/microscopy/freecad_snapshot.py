"""Resolve stored FreeCAD assembly snapshots without regenerating features.

This intentionally supports unscaled absolute App::Links and expanded native
assembly groups. Unsupported links fail explicitly. Saved geometry, joint
semantics, physical properties and CAD reuse rights remain separate evidence.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation


def placement(properties):
    node = properties.get("Placement")
    result = np.eye(4)
    if node is not None:
        values = node.find("PropertyPlacement").attrib
        result[:3, :3] = Rotation.from_quat([float(values[f"Q{i}"]) for i in range(4)]).as_matrix()
        result[:3, 3] = [float(values[f"P{axis}"]) for axis in "xyz"]
    if not np.isfinite(result).all():
        raise ValueError("CAD placement must be finite")
    return result


class SavedAssembly:
    """Hash-bound source documents and their saved physical part instances."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root/"acquisition.json").read_text())
        if not self.manifest.get("success"):
            raise ValueError("CAD acquisition has unresolved source documents")
        self._documents = {}

    def document(self, name):
        if Path(name).name != name or name not in self.manifest["documents"]:
            raise ValueError(f"CAD document is outside the inspected source set: {name}")
        if name not in self._documents:
            path = self.root/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest["documents"][name]["sha256"]:
                raise ValueError(f"CAD document differs from its acquisition: {name}")
            with zipfile.ZipFile(path) as archive:
                xml = ET.fromstring(archive.read("Document.xml"))
            types = {o.get("name"): o.get("type") for o in xml.find("Objects").findall("Object")}
            data = {o.get("name"): {p.get("name"): p for p in o.find("Properties").findall("Property")}
                    for o in xml.find("ObjectData")}
            self._documents[name] = types, data
        return self._documents[name]

    def geometry(self, document, name, chain=()):
        """Resolve a link's source geometry; its placement is overridden."""
        key = document, name
        if key in chain:
            raise ValueError("CAD geometry links contain a cycle")
        types, objects = self.document(document)
        properties = objects[name]
        if types[name] == "App::Link":
            if properties["LinkTransform"].find("Bool").get("value") != "false":
                raise NotImplementedError("Relative CAD links need a separate transform resolver")
            scale = properties["Scale"].find("Float").get("value")
            if float(scale) != 1.:
                raise NotImplementedError("Scaled CAD links are unsupported")
            link = properties["LinkedObject"].find("XLink")
            return self.geometry(link.get("file") or document, link.get("name"), chain+(key,))
        if types[name] in ("App::Part", "Assembly::AssemblyObject"):
            inverse_root = np.linalg.inv(placement(properties))
            return [part["source"] | {"relative_transform": inverse_root@part["transform"]}
                    for part in self.instances(document, name, _chain=chain+(key,))]
        if "Shape" not in properties:
            raise ValueError(f"Physical CAD leaf has no saved geometry: {document}:{name}")
        entry = properties["Shape"].find("Part").get("file")
        with zipfile.ZipFile(self.root/document) as archive:
            payload = archive.read(entry)
        if not payload:
            raise ValueError(f"Physical CAD leaf has an empty shape: {document}:{name}")
        return [dict(document=document, object=name, entry=entry, relative_transform=np.eye(4),
                     object_placement=placement(properties), shape_sha256=hashlib.sha256(payload).hexdigest())]

    def instances(self, document="Assembly_MicroManipulator.FCStd", name="Assembly", _chain=()):
        """Flatten geometric groups; logical group aliases create no copies."""
        parts, visited = [], {}
        types, objects = self.document(document)

        def visit(node, parent, scope):
            properties = objects[node]
            kind = types[node]
            logical = kind == "App::DocumentObjectGroup"
            transform = parent if logical else parent@placement(properties)
            key = scope, node
            if key in visited:
                if not np.allclose(visited[key], transform, rtol=0, atol=1e-10):
                    raise ValueError("CAD group alias has inconsistent placement")
                return
            visited[key] = transform
            if logical or kind in ("Assembly::AssemblyObject", "Assembly::AssemblyLink", "App::Part"):
                children = properties["Group"].find("LinkList")
                for child in children:
                    visit(child.get("value"), transform, scope if logical else scope+(node,))
            elif kind == "Sketcher::SketchObject":
                pass  # Stored construction sketches are not physical part instances.
            elif kind == "App::Link" or "Shape" in properties:
                sources = self.geometry(document, node, _chain)
                for source in sources:
                    suffix = (source["object"],) if len(sources) > 1 else ()
                    parts.append(dict(instance="/".join(scope+(node,)+suffix), object=node,
                                      transform=transform@source["relative_transform"], source=source))
            elif kind not in ("App::Origin", "App::Point", "App::Line", "App::Plane",
                              "Assembly::JointGroup", "App::FeaturePython", "Part::FeaturePython"):
                raise NotImplementedError(f"Unhandled CAD object type: {kind}")

        visit(name, np.eye(4), ())
        return parts

    def payload(self, source):
        with zipfile.ZipFile(self.root/source["document"]) as archive:
            return archive.read(source["entry"])


def convert(source_root, output, deflection_mm=.04):
    """Tessellate saved physical geometry once; keep each instance transform."""
    import tempfile

    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Shape

    from microscopy.cad_import import convert as convert_step

    assembly = SavedAssembly(source_root)
    instances = assembly.instances()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shapes = {}
    for part in instances:
        source = part["source"]
        digest = source["shape_sha256"]
        if digest in shapes:
            continue
        with tempfile.NamedTemporaryFile(suffix=".brep") as temporary:
            temporary.write(assembly.payload(source))
            temporary.flush()
            shape = TopoDS_Shape()
            if not BRepTools.Read_s(shape, temporary.name, BRep_Builder()):
                raise ValueError("CAD stored shape could not be read")
        location = shape.Location().Transformation()
        matrix = np.eye(4)
        matrix[:3] = [[location.Value(i+1, j+1) for j in range(4)] for i in range(3)]
        if not np.allclose(matrix, source["object_placement"], rtol=0, atol=1e-8):
            raise ValueError(f"Stored shape placement differs from XML: {source['document']}:{source['object']}")
        # App::Link overrides the source object's placement. Strip that saved
        # outer location before adding the explicit physical-instance pose.
        shape = shape.Located(TopLoc_Location())
        writer = STEPControl_Writer()
        if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
            raise ValueError("CAD snapshot STEP transfer failed")
        step = output/(digest+".step")
        if writer.Write(str(step)) != IFSelect_RetDone:
            raise ValueError("CAD snapshot STEP write failed")
        shapes[digest] = convert_step(step, output/digest, deflection_mm)
    license_source = assembly.root/"LICENSE"
    license_hash = assembly.manifest["documents"]["LICENSE"]["sha256"]
    if hashlib.sha256(license_source.read_bytes()).hexdigest() != license_hash:
        raise ValueError("CAD license differs from the acquired source")
    (output/"LICENSE").write_bytes(license_source.read_bytes())
    report = dict(success=True, source_revision=assembly.manifest["revision"],
                  source_root=str(assembly.root), instances=instances, shapes=shapes,
                  shape_coordinate_unit="mm", mesh_length_unit="m",
                  license_sha256=license_hash, geometry_only=True,
                  scope="Saved upstream physical snapshots and resolved absolute placements; no feature regeneration, articulation or measured hardware precision")
    (output/"assembly.json").write_text(json.dumps(report, indent=2, default=lambda x: x.tolist()))
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Hash-bound CAD acquisition directory")
    parser.add_argument("--output", type=Path, required=True, help="New ignored conversion directory")
    parser.add_argument("--deflection-mm", type=float, default=.04)
    args = parser.parse_args()
    report = convert(args.source, args.output, args.deflection_mm)
    print(json.dumps(dict(success=report["success"], instances=len(report["instances"]),
                          shapes=len(report["shapes"]), mesh_length_unit=report["mesh_length_unit"])))


if __name__ == "__main__":
    main()
