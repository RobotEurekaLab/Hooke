"""Physical guide openings, parent-filtered clearance and measured GCF motion."""

import copy
from itertools import product
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


class HkustProbeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("hkust_geotechnical_centrifuge")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.session = InstrumentSession(cls.model, cls.manifest)
        m = cls.model
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        changed = {i for i in enabled if "__gcf_probe_" in m.geom(i).name}
        cls.pairs = np.array(
            [
                (a, b)
                for a in sorted(changed)
                for b in enabled
                if (b not in changed or a < b)
                and m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            ],
            dtype=int,
        )

    def setUp(self):
        self.session.reset()

    def _set_pose(self, pose):
        for semantic, value in pose.items():
            c = self.session.controls["gcf_model_testbed", semantic]
            self.session.data.qpos[self.model.jnt_qposadr[c.joint]] = value
        mujoco.mj_forward(self.model, self.session.data)

    def _check_pairs(self, *, unculled=False):
        m, d, pairs = self.model, self.session.data, self.pairs
        if not unculled:
            rotation = d.geom_xmat.reshape(-1, 3, 3)
            center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
            half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
            a, b = pairs.T
            near = (
                np.linalg.norm(
                    np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0),
                    axis=1,
                )
                < 0.002
            )
            pairs = pairs[near]
        for a, b in pairs:
            gap = mujoco.mj_geomDistance(m, d, int(a), int(b), 0.002, None)
            self.assertGreaterEqual(
                gap, -1e-7, (m.geom(int(a)).name, m.geom(int(b)).name, gap)
            )

    def test_new_contacts_clear_full_combined_travel_including_parent_pairs(self):
        m = self.model
        height = m.body("gcf_model_testbed__body_probe_height").id
        trolley = m.body("gcf_model_testbed__body_carriage_x").id
        self.assertTrue(
            any(
                {m.geom_bodyid[a], m.geom_bodyid[b]} == {height, trolley}
                for a, b in self.pairs
            )
        )
        self.assertGreater(
            np.count_nonzero(
                (m.geom_bodyid == height) & (m.geom_contype | m.geom_conaffinity)
            ),
            0,
        )
        for bridge, carriage, height, angle in product(
            (-0.48, 0, 0.48),
            (-0.76, 0, 0.76),
            np.linspace(-0.04, 0.09, 5),
            np.linspace(-0.7, 0.7, 5),
        ):
            self._set_pose(
                dict(
                    bridge_y=bridge,
                    carriage_x=carriage,
                    probe_height=height,
                    probe_rotation=angle,
                )
            )
            self._check_pairs()
        # Independently query every pair at off-grid poses: this catches an
        # erroneous near-pair filter, including hidden parent-body overlaps.
        for height, angle in ((-0.033, -0.53), (0.027, 0.19), (0.081, 0.61)):
            self._set_pose(
                dict(
                    bridge_y=0.193,
                    carriage_x=-0.327,
                    probe_height=height,
                    probe_rotation=angle,
                )
            )
            self._check_pairs(unculled=True)

    def test_real_bearing_gaps_admit_probes_and_solid_walls_block_them(self):
        root = copy.deepcopy(self.root)
        m, d = self.model, self.session.data
        stage = m.body("gcf_model_testbed__body_probe_height").id
        frame, origin = d.xmat[stage].reshape(3, 3), d.xpos[stage]
        # 0.4 mm spheres fit inside the actual 0.6 mm radial running gaps,
        # with the physical rail/shaft still present inside each bearing.
        positions = ((-0.065 + 0.0083, 0, 0), (0.0143, 0, 0.02))
        for i, local in enumerate(positions):
            body = ET.SubElement(
                root.find("worldbody"),
                "body",
                name=f"gap_probe_{i}",
                pos=" ".join(map(str, origin + frame @ local)),
            )
            ET.SubElement(body, "freejoint", name=f"gap_joint_{i}")
            ET.SubElement(
                body,
                "geom",
                name=f"gap_geom_{i}",
                type="sphere",
                size="0.0002",
                mass="0.001",
                contype="1",
                conaffinity="1",
            )
        pm = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        pd = mujoco.MjData(pm)
        mujoco.mj_forward(pm, pd)
        probes = {pm.geom(f"gap_geom_{i}").id for i in range(2)}
        self.assertFalse(
            any(
                (c.geom1 in probes or c.geom2 in probes) and c.dist < -1e-8
                for c in pd.contact
            )
        )
        for i, local in enumerate(((-0.065 + 0.012, 0, 0), (0.025, 0, 0.02))):
            joint = pm.joint(f"gap_joint_{i}").id
            address = pm.jnt_qposadr[joint]
            pd.qpos[address : address + 3] = origin + frame @ local
        mujoco.mj_forward(pm, pd)
        for probe in probes:
            self.assertTrue(
                any(
                    probe in (c.geom1, c.geom2) and c.dist < -0.0001 for c in pd.contact
                )
            )

    def test_rails_engage_bearings_and_rotor_collar_has_axial_support(self):
        m, d = self.model, self.session.data
        prefix = "gcf_model_testbed__gcf_probe_"
        for h in np.linspace(-0.04, 0.09, 31):
            self._set_pose(dict(probe_height=h))
            stage = m.body("gcf_model_testbed__body_probe_height").id
            for side in (-1, 1):
                rail = m.geom(prefix + f"rail_{side}").id
                self.assertLess(d.geom_xpos[rail, 2] - 0.24, d.xpos[stage, 2] - 0.038)
                self.assertGreater(
                    d.geom_xpos[rail, 2] + 0.24, d.xpos[stage, 2] + 0.038
                )
            collar = m.geom(prefix + "retaining_collar").id
            self.assertAlmostEqual(
                d.geom_xpos[collar, 2] - 0.005, d.xpos[stage, 2] + 0.055, places=12
            )

    def test_native_servos_move_reverse_return_and_keep_other_axes_held(self):
        s, m = self.session, self.model
        for name in ("probe_height", "probe_rotation", "carriage_x", "bridge_y"):
            c = s.controls["gcf_model_testbed", name]
            inactive = [
                int(m.jnt_qposadr[x.joint])
                for key, x in s.controls.items()
                if key != ("gcf_model_testbed", name)
            ]
            held = s.data.qpos[inactive].copy()
            for fraction in (0.10, 0.90, 0.20):
                target = c.lower + fraction * (c.upper - c.lower)
                report = s.command("gcf_model_testbed", name, target, duration=2)
                self.assertLess(abs(report["error"]), 0.001)
                np.testing.assert_allclose(s.data.qpos[inactive], held, atol=1e-7)
                self.assertFalse(np.any(s.data.warning.number))
                self._check_pairs()
            report = s.command("gcf_model_testbed", name, 0, duration=2)
            self.assertLess(abs(report["error"]), 1e-6)
            self._check_pairs()


if __name__ == "__main__":
    unittest.main()
