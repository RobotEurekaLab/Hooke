"""Visible optical apertures and supported retaining pads on the Yale surrogate."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.usability import camera_target_screening


class YaleOpticalAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("yale_becton_cleanroom_metrology")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def geom(self, key):
        return self.model.geom("wafer_inspection__yale_" + key).id

    def distance(self, first, second):
        return mujoco.mj_geomDistance(
            self.model, self.data, self.geom(first), self.geom(second), 1, None
        )

    def test_rays_through_open_eyepieces_reach_recessed_lens_faces(self):
        for side in (-1, 1):
            lens = self.geom(f"eyepiece_lens_{side}")
            cuff = self.geom(f"eyepiece_cuff_{side}")
            axis = self.data.geom_xmat[lens].reshape(3, 3)[:, 2]
            center = self.data.geom_xpos[lens]
            # This starts outside the sleeve and aims through its real mouth.
            origin = center + axis * 0.040
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                -axis,
                np.ones(6, dtype=np.uint8),
                True,
                -1,
                hit,
            )
            self.assertEqual(hit[0], lens)
            self.assertAlmostEqual(distance, 0.0385, places=7)
            # Compiled mesh positions include the mesh inertia-frame offset;
            # project all actual cuff vertices onto the lens axis.
            mesh = self.model.geom_dataid[cuff]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start : start + count]
            rotation = self.data.geom_xmat[cuff].reshape(3, 3)
            points = vertices @ rotation.T + self.data.geom_xpos[cuff]
            mouth = np.max((points - center) @ axis)
            self.assertAlmostEqual(
                mouth - self.model.geom_size[lens, 1], 0.0045, places=7
            )

    def test_contact_walls_leave_the_mouth_open_but_block_a_rim_probe(self):
        root = deepcopy(self.root)
        for side in (-1, 1):
            lens = self.geom(f"eyepiece_lens_{side}")
            frame = self.data.geom_xmat[lens].reshape(3, 3)
            # The lens front is 4.5 mm behind the rim. A 1 mm sphere fits in
            # the mouth without touching it; another sphere intersects a wall.
            center = self.data.geom_xpos[lens] + frame[:, 2] * 0.0055
            for kind, offset in (("open", 0), ("rim", 0.023)):
                position = center + offset * frame[:, 0]
                body = ET.SubElement(
                    root.find("worldbody"),
                    "body",
                    name=f"probe_{side}_{kind}",
                    pos=" ".join(map(str, position)),
                )
                ET.SubElement(body, "freejoint")
                ET.SubElement(
                    body,
                    "geom",
                    name=f"probe_geom_{side}_{kind}",
                    type="sphere",
                    size=".001",
                    mass=".001",
                    contype="1",
                    conaffinity="1",
                )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for side in (-1, 1):
            open_probe = model.geom(f"probe_geom_{side}_open").id
            self.assertFalse(
                any(
                    open_probe in (c.geom1, c.geom2) and c.dist < -1e-6
                    for c in data.contact
                )
            )
            rim_probe = model.geom(f"probe_geom_{side}_rim").id
            self.assertTrue(
                any(
                    rim_probe in (c.geom1, c.geom2) and c.dist < -1e-4
                    for c in data.contact
                )
            )

    def test_castings_tubes_and_retaining_pads_have_continuous_support(self):
        pairs = [
            ("head_support", "main_casting"),
            ("main_casting", "lower_barrel"),
            ("main_casting", "binocular_housing"),
        ]
        for side in (-1, 1):
            pairs.extend(
                (
                    ("binocular_housing", f"eyepiece_stem_{side}"),
                    ("main_casting", f"knob_spindle_{side}"),
                    (f"knob_spindle_{side}", f"zoom_knob_{side}"),
                    (f"clamp_block_{side}", f"clamp_finger_{side}"),
                    (f"clamp_finger_{side}", f"retaining_pad_{side}"),
                    (f"retaining_pad_{side}", "wafer_coupon"),
                )
            )
            for part in (f"eyepiece_stem_{side}", f"eyepiece_lens_{side}"):
                self.assertLessEqual(
                    min(
                        self.distance(part, f"eyepiece_cuff_{side}_contact_{i}")
                        for i in range(32)
                    ),
                    1e-6,
                )
        for first, second in pairs:
            with self.subTest(first=first, second=second):
                self.assertTrue(self.model.geom_contype[self.geom(first)])
                self.assertTrue(self.model.geom_contype[self.geom(second)])
                self.assertLessEqual(self.distance(first, second), 1e-6)
        for part in ("lower_barrel", "objective_lens"):
            self.assertLessEqual(
                min(
                    self.distance(part, f"objective_collar_contact_{i}")
                    for i in range(32)
                ),
                1e-6,
            )

    def test_all_existing_camera_views_retain_coupon_access(self):
        rows = camera_target_screening(self.model, self.data, self.manifest)
        row = next(r for r in rows if r["equipment"] == "wafer_inspection")
        self.assertEqual(
            {r["camera"] for r in row["cameras"]},
            {"overview", "workstation", "interior", "source"},
        )
        for view in row["cameras"]:
            with self.subTest(camera=view["camera"]):
                self.assertTrue(view["in_frame"])
                self.assertTrue(view["opaque_line_of_sight"])
                self.assertGreater(view["estimated_target_diameter_px"], 10)


if __name__ == "__main__":
    unittest.main()
