"""Furniture detail must preserve useful supports and bounded scene footprints."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.architecture import build_bench
from real_labs.bench_details import optical_grid_mesh
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from raw_control_assertions import assert_raw_diagnostics


def fixture(style, width=1.8, depth=1.2, height=0.85):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="5 5 .05")
    item = dict(id="bench", style=style, size=[width, depth, height], pos=[0, 0, 0])
    record = build_bench(world, item, [0.5, 0.6, 0.6, 1], asset=asset)
    return root, record


class BenchGeometryTests(unittest.TestCase):
    def test_frame_brackets_attach_outside_the_tube_without_coplanar_outer_faces(self):
        root, _ = fixture("open_frame")
        for i in range(2):
            for j in range(2):
                name = f"bench_leg_{i}_{j}"
                leg = root.find(f".//geom[@name='{name}']")
                position = np.fromstring(leg.get("pos"), sep=" ")
                extent = np.fromstring(leg.get("size"), sep=" ")
                for axis in (0, 1):
                    bracket = root.find(f".//geom[@name='{name}_steel_bracket_{axis}']")
                    center = np.fromstring(bracket.get("pos"), sep=" ")[axis]
                    half = np.fromstring(bracket.get("size"), sep=" ")[axis]
                    if axis == 0:
                        self.assertAlmostEqual(
                            center - half, position[axis] + extent[axis]
                        )
                        self.assertGreater(
                            center + half, position[axis] + extent[axis] + 0.005
                        )
                    else:
                        self.assertAlmostEqual(
                            center + half, position[axis] - extent[axis]
                        )
                        self.assertLess(
                            center - half, position[axis] - extent[axis] - 0.005
                        )

    def test_cabinet_doors_face_outward_without_coplanar_shell_faces(self):
        root, _ = fixture("cabinet")
        shell = root.find(".//geom[@name='bench_cabinet']")
        shell_front = float(shell.get("pos").split()[1]) - float(
            shell.get("size").split()[1]
        )
        for door in root.findall(".//geom"):
            if not door.get("name", "").startswith("bench_door_"):
                continue
            center, half = float(door.get("pos").split()[1]), float(
                door.get("size").split()[1]
            )
            self.assertLess(center - half, shell_front - 0.01)
            self.assertAlmostEqual(center + half, shell_front)

    def test_supports_stay_inside_the_original_footprint_and_meet_the_floor(self):
        for style in ("optical", "open_frame", "cabinet"):
            with self.subTest(style=style):
                root, record = fixture(style)
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                floor_contacts = 0
                for index in range(1, model.ngeom):
                    if not (model.geom_contype[index] or model.geom_conaffinity[index]):
                        continue
                    rotation = data.geom_xmat[index].reshape(3, 3)
                    center = (
                        data.geom_xpos[index] + rotation @ model.geom_aabb[index, :3]
                    )
                    extent = np.abs(rotation) @ model.geom_aabb[index, 3:]
                    low, high = center - extent, center + extent
                    self.assertTrue(
                        np.all(low[:2] >= [-0.9 - 1e-7, -0.6 - 1e-7]),
                        model.geom(index).name,
                    )
                    self.assertTrue(
                        np.all(high[:2] <= [0.9 + 1e-7, 0.6 + 1e-7]),
                        model.geom(index).name,
                    )
                    self.assertGreaterEqual(low[2], -1e-7)
                    self.assertLessEqual(high[2], 0.85 + 1e-7)
                    floor_contacts += abs(low[2]) < 1e-7
                self.assertEqual(floor_contacts, 1 if style == "cabinet" else 4)
                self.assertEqual(model.njnt, 0)
                self.assertFalse(record["simulated_isolation"])

    def test_working_surface_physically_supports_a_sample_at_the_authored_height(self):
        for style in ("optical", "open_frame", "cabinet"):
            with self.subTest(style=style):
                root, _ = fixture(style)
                body = ET.SubElement(
                    root.find("worldbody"), "body", name="sample", pos="0 0 1.05"
                )
                ET.SubElement(body, "freejoint")
                ET.SubElement(
                    body,
                    "geom",
                    name="sample",
                    type="box",
                    size=".025 .025 .025",
                    mass=".1",
                )
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                data = mujoco.MjData(model)
                mujoco.mj_step(model, data, nstep=1000)
                self.assertAlmostEqual(data.qpos[2], 0.875, delta=0.0005)
                self.assertLess(np.linalg.norm(data.qvel), 0.001)
                self.assertTrue(np.isfinite(data.qpos).all())

    def test_dense_hole_marks_have_metric_pitch_and_no_thread_contacts(self):
        vertex, _, columns, rows = optical_grid_mesh(0.5, 0.5)
        points = np.fromstring(vertex, sep=" ").reshape(-1, 9, 3)
        centers = points[:, 8, :2]
        self.assertEqual((columns, rows), (20, 20))
        np.testing.assert_allclose(np.diff(np.unique(centers[:, 0])), 0.025)
        np.testing.assert_allclose(np.diff(np.unique(centers[:, 1])), 0.025)
        root, record = fixture("optical", width=2.5, depth=4.4)
        grid = record["optical_grid"]
        self.assertEqual(grid["visual_marks"], 17600)
        self.assertEqual(grid["visual_meshes"], 2)
        self.assertEqual(grid["visual_mesh_instances"], 45)
        self.assertFalse(grid["physical_thread_contact"])
        for geom in root.findall(".//geom[@type='mesh']"):
            self.assertEqual(geom.get("contype"), "0")
            self.assertEqual(geom.get("conaffinity"), "0")

    def test_representative_instruments_still_complete_their_mechanical_checks(self):
        for identifier in (
            "indiana_microfluidics",
            "rochester_genomics",
            "manchester_collaborative_robotics",
        ):
            with self.subTest(scene=identifier):
                root, manifest = build_scene(identifier)
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                session = InstrumentSession(model, manifest)
                result = session.smoke_test()
                assert_raw_diagnostics(self, session, result)
                self.assertEqual(
                    len(manifest["furnishing_assumptions"]), len(manifest["benches"])
                )


if __name__ == "__main__":
    unittest.main()
