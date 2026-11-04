"""Ground datum, physical support and portable NASA package regression checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.export import export_scene
from real_labs.layout_audit import audit_layout
from real_labs.nasa_rover_task import NasaRoverTask, direct_gap
from real_labs.runtime import InstrumentSession


class NasaRoverExportTests(unittest.TestCase):
    def assert_ground_support(self, model, data):
        floor = model.geom("arch_floor").id
        for side in ("left", "right"):
            for index in (1, 2, 3):
                body = model.body(f"rover__body_wheel_{side}_{index}").id
                tyres = [
                    geom
                    for geom in np.flatnonzero(model.geom_bodyid == body)
                    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER
                    and model.geom_size[geom, 0] == 0.14
                ]
                self.assertEqual(len(tyres), 1)
                self.assertAlmostEqual(direct_gap(model, data, floor, tyres[0]), 0)

    def test_nonnegative_datum_preserves_world_support_and_mass_center(self):
        root, manifest = build_scene("nasa_planetary")
        audit = audit_layout(manifest)
        self.assertTrue(audit["passed"], audit)
        rover = next(item for item in manifest["equipment"] if item["id"] == "rover")
        self.assertEqual(rover["pos"], [-5, 2.5, 0])
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        session = InstrumentSession(model, manifest)
        data = session.data
        np.testing.assert_allclose(data.xipos[model.body("rover").id], [-5, 2.5, 0.242])
        np.testing.assert_allclose(
            model.body_inertia[model.body("rover").id], [0.9, 1.2, 1.5]
        )
        np.testing.assert_allclose(
            data.site_xpos[model.site("rover__site_sample_cache_1").id],
            [-5.26, 2.5, 0.489],
        )
        self.assert_ground_support(model, data)
        initial = data.qpos.copy()
        session.advance(400)
        session.reset()
        np.testing.assert_array_equal(data.qpos, initial)
        self.assertEqual(data.time, 0)
        self.assert_ground_support(model, data)

    def test_canonical_package_exports_and_reloads_with_bound_task(self):
        with TemporaryDirectory(prefix="hooke_nasa_export_") as directory:
            result = export_scene("nasa_planetary", directory, validate=False)
            self.assertTrue(result["compiled"])
            package = Path(directory)
            audit = json.loads((package / "layout_audit.json").read_text())
            self.assertTrue(audit["passed"], audit)
            manifest = json.loads((package / "manifest.json").read_text())
            model = mujoco.MjModel.from_xml_path(str(package / "scene.xml"))
            session = InstrumentSession(model, manifest)
            self.assert_ground_support(model, session.data)
            task = NasaRoverTask(session, record=False)
            self.assertEqual(task.binding["scene_sha256"], manifest["scene_sha256"])
            self.assertEqual(session.data.time, 0)
            self.assertEqual(len(session.controls), 24)


if __name__ == "__main__":
    unittest.main()
