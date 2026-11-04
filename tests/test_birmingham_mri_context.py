"""Passive source-context additions remain supported outside table travel."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


def is_context(name):
    return name.startswith(("scanner__mri_access_service_", "europe3_mri_source_"))


class MRIContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, _ = build_scene("birmingham_chbh_mri")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def test_new_accessories_and_services_have_connected_enabled_support(self):
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        names = [m.geom(i).name or "" for i in range(m.ngeom)]
        new = {i for i, name in enumerate(names) if is_context(name)}
        self.assertTrue(new)
        self.assertFalse(any("stored_phantom_pad" in name for name in names))
        anchors = {
            i
            for i, name in enumerate(names)
            if name.startswith("europe3_mri_storage_shelf_")
            or name == "scanner__mri_access_table_foot"
        }
        for i in new:
            self.assertTrue(m.geom_contype[i] and m.geom_conaffinity[i], names[i])
            self.assertEqual(m.body_weldid[m.geom_bodyid[i]], 0)
        reached, pending = set(anchors), set(new)
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

    def test_fixed_additions_clear_complete_two_slider_envelope(self):
        m = self.model
        d = mujoco.MjData(m)
        moving = np.flatnonzero(m.body_weldid[m.geom_bodyid] != 0)
        additions = np.array(
            [i for i in range(m.ngeom) if is_context(m.geom(i).name or "")]
        )
        minima, maxima = [], []
        for pose in itertools.product(*(m.jnt_range[j] for j in range(m.njnt))):
            self.assertTrue(np.all(m.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE))
            d.qpos[:] = pose
            mujoco.mj_forward(m, d)
            r = d.geom_xmat.reshape(-1, 3, 3)
            c = d.geom_xpos + np.einsum("nij,nj->ni", r, m.geom_aabb[:, :3])
            h = np.einsum("nij,nj->ni", abs(r), m.geom_aabb[:, 3:])
            minima.append(c - h)
            maxima.append(c + h)
        low, high = np.min(minima, axis=0), np.max(maxima, axis=0)
        # Fixed-orientation slides have affine positions; corner bounds contain
        # every pose, including native parent-filtered and visual-only shapes.
        gap = np.maximum(
            np.maximum(
                low[additions, None] - high[moving], low[moving] - high[additions, None]
            ),
            0,
        )
        minimum = np.linalg.norm(gap, axis=2).min()
        self.assertGreater(minimum, 0.20)

    def test_white_accessory_has_a_real_unfilled_opening(self):
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ring = [
            i
            for i in range(m.ngeom)
            if "source_open_white_accessory" in (m.geom(i).name or "")
        ]
        self.assertEqual(len(ring), 96)
        self.assertTrue(
            all(m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE for i in ring)
        )
        # The centreline segment in each layer stays 60 mm from its axis; tube
        # radius 26 mm leaves a physical opening, not a dark painted cap.
        for i in ring:
            radius = np.linalg.norm(d.geom_xpos[i, 1:] - [0.49, 1.272])
            self.assertGreater(radius - m.geom_size[i, 0], 0.033)


if __name__ == "__main__":
    unittest.main()
