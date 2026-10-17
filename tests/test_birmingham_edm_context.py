"""Fixed photo context stays supported and clear of the authored XY task."""

import itertools
import math
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


class EDMContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, _ = build_scene("birmingham_advanced_micro_edm")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def test_all_added_solids_connect_to_existing_fixed_support(self):
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        names = [m.geom(i).name or "" for i in range(m.ngeom)]
        new = {i for i, name in enumerate(names) if name.startswith("edm_source_")}
        reached = {
            i
            for i in range(m.ngeom)
            if i not in new
            and m.geom_contype[i]
            and m.body_weldid[m.geom_bodyid[i]] == 0
        }
        self.assertGreater(len(new), 100)
        for i in new:
            self.assertTrue(m.geom_contype[i] and m.geom_conaffinity[i], names[i])
            self.assertEqual(m.body_weldid[m.geom_bodyid[i]], 0)
        pending = set(new)
        while pending:
            connected = {
                i
                for i in pending
                if any(
                    mujoco.mj_geomDistance(m, d, i, j, 1e-7, None) <= 1e-8
                    for j in reached
                )
            }
            if not connected:
                break
            reached.update(connected)
            pending.difference_update(connected)
        self.assertFalse(pending, [names[i] for i in pending])

    def test_context_clears_entire_continuous_slider_envelope(self):
        m = self.model
        d = mujoco.MjData(m)
        new = np.array(
            [
                i
                for i in range(m.ngeom)
                if (m.geom(i).name or "").startswith("edm_source_")
            ]
        )
        moving = np.flatnonzero(m.body_weldid[m.geom_bodyid] != 0)
        self.assertTrue(np.all(m.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE))
        lows, highs = [], []
        for pose in itertools.product(*m.jnt_range):
            d.qpos[:] = pose
            mujoco.mj_forward(m, d)
            rotation = d.geom_xmat.reshape(-1, 3, 3)
            centre = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
            half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
            lows.append(centre - half)
            highs.append(centre + half)
        low, high = np.min(lows, axis=0), np.max(highs, axis=0)
        # Affine translations with fixed orientation: four corners bound every
        # intermediate pose, including visual and native-filtered moving geoms.
        gaps = np.linalg.norm(
            np.maximum(
                np.maximum(
                    low[new, None] - high[moving], low[moving] - high[new, None]
                ),
                0,
            ),
            axis=2,
        )
        self.assertGreater(float(gaps.min()), 0.15)

    def test_physical_probe_passes_guard_hole_but_is_stopped_by_strip(self):
        normal = np.array([-math.sin(1.08), math.cos(1.08), 0.0])
        tangent = np.array([math.cos(1.08), math.sin(1.08), 0.0])
        for x, should_pass in ((-0.310, True), (-0.315, False)):
            with self.subTest(aperture=should_pass):
                root = ET.fromstring(ET.tostring(self.root))
                root.find("option").set("gravity", "0 0 0")
                start = np.array([0.40, -1.26, 0.835]) + x * tangent - 0.04 * normal
                body = ET.SubElement(
                    root.find("worldbody"),
                    "body",
                    name="context_probe",
                    pos=" ".join(map(str, start)),
                )
                ET.SubElement(body, "freejoint", name="context_probe_free")
                ET.SubElement(
                    body,
                    "geom",
                    name="context_probe_ball",
                    type="sphere",
                    size=".002",
                    density="1000",
                    contype="1",
                    conaffinity="1",
                )
                m = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                d = mujoco.MjData(m)
                joint = m.joint("context_probe_free").id
                va, qa = m.jnt_dofadr[joint], m.jnt_qposadr[joint]
                d.qvel[va : va + 3] = 0.15 * normal
                touched = set()
                probe = m.geom("context_probe_ball").id
                for _ in range(round(0.5 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                    for contact in d.contact:
                        if probe in (contact.geom1, contact.geom2):
                            other = (
                                contact.geom2
                                if contact.geom1 == probe
                                else contact.geom1
                            )
                            touched.add(m.geom(other).name)
                travel = float(np.dot(d.qpos[qa : qa + 3] - start, normal))
                self.assertFalse(any(d.warning.number))
                if should_pass:
                    self.assertGreater(travel, 0.070)
                    self.assertFalse(touched)
                else:
                    self.assertLess(travel, 0.036)
                    self.assertIn("edm_source_guard_vertical_0", touched)


if __name__ == "__main__":
    unittest.main()
