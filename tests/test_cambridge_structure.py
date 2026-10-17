"""Physical topology of the source-informed underground chamber and gallery."""

import math
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


class CambridgeStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("cambridge_schofield_turner")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def ray(self, origin, direction):
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            np.asarray(origin, dtype=float),
            np.asarray(direction, dtype=float),
            None,
            1,
            -1,
            hit,
        )
        name = self.model.geom(int(hit[0])).name if hit[0] >= 0 else None
        return distance, name

    def gap(self, first, second):
        return mujoco.mj_geomDistance(
            self.model,
            self.data,
            self.model.geom(first).id,
            self.model.geom(second).id,
            0.1,
            None,
        )

    def test_pit_is_physically_enclosed_by_the_documented_polygonal_topology(self):
        for i in range(24):
            angle = math.tau * i / 24
            # The continuous shaft now reaches the old ray origin at z=2.4.
            # Inspect the actual enclosure above it and below the luminaires.
            distance, name = self.ray(
                (0, 0, 2.55), (math.cos(angle), math.sin(angle), 0)
            )
            self.assertIn("wall_turner_sector", name)
            self.assertAlmostEqual(distance, 7, places=6)
        self.assertFalse(
            any(
                g.get("name") in ("arch_wall_back_last", "arch_wall_left_last")
                for g in self.root.findall("./worldbody/geom")
            )
        )

    def test_grate_has_real_openings_into_supported_lower_service_space(self):
        distance, name = self.ray((0.05, -4.4, 0.5), (0, 0, -1))
        self.assertEqual(name, "turner_drive_floor")
        self.assertAlmostEqual(distance, 1.7, places=6)
        distance, name = self.ray((0, -4.4, 0.5), (0, 0, -1))
        self.assertEqual(name, "turner_service_grate_12")
        self.assertAlmostEqual(distance, 0.455, places=6)
        for y in (-5.25, -3.05):
            self.assertLessEqual(
                self.gap("turner_service_grate_12", f"turner_grate_frame_y_{y}"), 1e-6
            )

    def test_upper_gallery_hatch_is_open_and_ladder_is_anchored(self):
        distance, name = self.ray((0, 3.7, 3.6), (0, 0, -1))
        self.assertTrue(name.startswith("arch_floor_turner"), name)
        self.assertAlmostEqual(distance, 3.6, places=6)
        distance, name = self.ray((0.9, 3.7, 3.6), (0, 0, -1))
        self.assertTrue(name.startswith("turner_gallery_deck"), name)
        self.assertAlmostEqual(distance, 0.3, places=6)
        for x in (-0.32, 0.32):
            self.assertLessEqual(
                self.gap(
                    f"turner_access_ladder_side_{x}", f"turner_ladder_top_anchor_{x}"
                ),
                1e-6,
            )
            self.assertLessEqual(
                self.gap(f"turner_ladder_top_anchor_{x}", "turner_gallery_deck_3"),
                1e-6,
            )

    def test_ceiling_coffers_and_light_mounts_have_actual_support(self):
        walls = [
            i
            for i in range(self.model.ngeom)
            if "wall_turner_sector" in self.model.geom(i).name
        ]
        for y in (-4.6, -2.3, 0, 2.3, 4.6):
            coffer = self.model.geom(f"arch_roof_turner_coffer_{y}").id
            connected = [
                g
                for g in walls
                if mujoco.mj_geomDistance(self.model, self.data, coffer, g, 0.1, None)
                <= 1e-6
            ]
            self.assertGreaterEqual(len(connected), 2)
            self.assertLessEqual(
                self.gap(
                    f"arch_luminaire_turner_{y}", f"arch_roof_turner_light_mount_{y}"
                ),
                1e-6,
            )
            self.assertLessEqual(
                self.gap(
                    f"arch_roof_turner_coffer_{y}", f"arch_roof_turner_light_mount_{y}"
                ),
                1e-6,
            )

    def test_gallery_tread_has_a_continuous_bearing_path_to_concrete(self):
        for side, label in ((-1, "left"), (1, "right")):
            ledger = f"turner_gallery_ledger_{side}"
            joist = f"turner_gallery_edge_joist_{side}"
            self.assertLessEqual(self.gap(ledger, f"arch_roof_turner_{label}"), 1e-6)
            self.assertLessEqual(self.gap(joist, ledger), 1e-6)
            self.assertLessEqual(
                self.gap(joist, f"turner_gallery_deck_{0 if side < 0 else 1}"),
                1e-6,
            )
            for y in (-7.7, -4.6, -2.3, 0, 2.3, 3.045, 4.355, 4.6, 7.7):
                cross = f"turner_gallery_cross_joist_{y}"
                self.assertLessEqual(self.gap(cross, ledger), 1e-6)
                self.assertLessEqual(self.gap(cross, joist), 1e-6)
            for y in (3.045, 4.355):
                self.assertLessEqual(
                    self.gap(
                        f"turner_gallery_hatch_joist_{side}",
                        f"turner_gallery_cross_joist_{y}",
                    ),
                    1e-6,
                )
        for mesh in self.root.findall("./asset/mesh"):
            if mesh.get("name", "").startswith("turner_gallery_deck"):
                z = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)[:, 2]
                self.assertAlmostEqual(float(np.ptp(z)), 0.010, places=6)

    def test_pit_roof_is_continuous_between_coffers(self):
        for x in (-5, -3, -1.5, 1.5, 3, 5):
            for y in (-1.7, -0.7, 0.7, 1.7):
                distance, name = self.ray((x, y, 2.5), (0, 0, 1))
                self.assertIn(name, ("arch_roof_turner_left", "arch_roof_turner_right"))
                self.assertAlmostEqual(distance, 0.5, places=6)

    def test_custom_meshes_are_closed_positive_volume_and_resource_bound(self):
        assets = {a.get("name"): a for a in self.root.findall("./asset/mesh")}
        for name, mesh in assets.items():
            if not name.startswith(("arch_", "turner_gallery_deck")):
                continue
            vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
            faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
            counts = {}
            for a, b, c in faces:
                for edge in ((a, b), (b, c), (c, a)):
                    edge = tuple(sorted(edge))
                    counts[edge] = counts.get(edge, 0) + 1
            self.assertEqual(set(counts.values()), {2}, name)
            volume = (
                np.einsum(
                    "ij,ij->i",
                    vertices[faces[:, 0]],
                    np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
                ).sum()
                / 6
            )
            self.assertGreater(volume, 0, name)
        for geom in self.root.findall("./worldbody/geom"):
            if geom.get("mesh"):
                self.assertIn(geom.get("mesh"), assets)

    def test_gallery_clear_width_and_estimated_dimension_semantics(self):
        left = self.model.geom("turner_gallery_post_-1_0")
        right = self.model.geom("turner_gallery_post_1_0")
        self.assertAlmostEqual(
            right.pos[0] - right.size[0] - left.pos[0] - left.size[0], 2
        )
        evidence = self.manifest["reference_detail_evidence"]["chamber_structure"]
        self.assertEqual(evidence["published_dimensions"]["shell_diameter_m"], 14)
        self.assertTrue(
            any("datum unresolved" in x for x in evidence["authored_assumptions"])
        )
        self.assertTrue(any("high-speed dynamics" in x for x in evidence["unfinished"]))
        structure = self.manifest["reference_detail_evidence"]["turner_structure"]
        self.assertTrue(any("rigidly parked swing-platform" in x for x in structure["implemented"]))
        self.assertIn("No high-g or swing dynamics", structure["operation_scope"])
        self.assertEqual(self.model.nu, 2)


if __name__ == "__main__":
    unittest.main()
