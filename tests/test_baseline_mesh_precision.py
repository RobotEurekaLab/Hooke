"""Native source archives preserve authored inline mesh geometry."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from backends.baseline import write_snapshot


class BaselineMeshPrecisionTests(unittest.TestCase):
    def test_thin_and_symmetric_meshes_keep_their_compiled_frames(self):
        # A sloped 24 mm sheet at metre-scale offsets: six-digit serialization
        # changes its thickness enough to fail the native geometry contract.
        sheet = np.array(
            [
                [-1.7957766728644642, -0.65, 0.5587677469799579],
                [-0.5457766728644642, -0.24, 1.0287677469799579],
                [-0.5457766728644642, 0.24, 1.0287677469799579],
                [-1.7957766728644642, 0.65, 0.5587677469799579],
                [-1.804223327135536, -0.65, 0.5812322530200422],
                [-0.5542233271355359, -0.24, 1.0512322530200422],
                [-0.5542233271355359, 0.24, 1.0512322530200422],
                [-1.804223327135536, 0.65, 0.5812322530200422],
            ]
        )
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        cylinder = np.array(
            [
                [-0.42 + 0.393 * np.cos(a), -0.1 + 0.393 * np.sin(a), z]
                for z in (0.672, 1.512)
                for a in angles
            ]
        )
        root = ET.Element("mujoco")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        for name, vertices in (("thin_sheet", sheet), ("round_column", cylinder)):
            ET.SubElement(
                asset,
                "mesh",
                name=name,
                vertex=" ".join(map(str, vertices.ravel())),
            )
            ET.SubElement(world, "geom", name=name, type="mesh", mesh=name)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source = path / "original.xml"
            ET.ElementTree(root).write(source, encoding="unicode")
            spec = mujoco.MjSpec.from_file(str(source))
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            write_snapshot(
                SimpleNamespace(
                    spec=spec,
                    model=model,
                    data=data,
                    default_scene=source,
                    task="dry_geometry_inspection",
                    task_info={},
                ),
                path,
            )
            restored = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
            for field in ("mesh_pos", "mesh_quat", "geom_size", "geom_pos", "geom_quat"):
                np.testing.assert_allclose(
                    getattr(restored, field),
                    getattr(model, field),
                    rtol=0,
                    atol=1e-12,
                    err_msg=field,
                )
            saved = ET.parse(path / "scene.xml")
            for mesh in spec.meshes:
                element = saved.find(f'./asset/mesh[@name="{mesh.name}"]')
                vertices = np.fromstring(element.get("vertex"), sep=" ")
                np.testing.assert_array_equal(vertices, mesh.uservert)


if __name__ == "__main__":
    unittest.main()
