"""Connected reference fixture, preserved drive contract and actual inspection."""

from collections import Counter
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.whittle_probe_geometry import PREFIX
from real_labs.whittle_probe_motion import (
    ReferenceInspection,
    SCENE,
    EQUIPMENT,
    inspect_whittle_reference,
)


def session():
    root, manifest = build_scene(SCENE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def gap(s, first, second):
    return float(
        mujoco.mj_geomDistance(
            s.model, s.data, s.model.geom(first).id, s.model.geom(second).id, 1, None
        )
    )


class WhittleProbeTests(unittest.TestCase):
    def test_original_frames_ranges_and_lumped_inertia(self):
        s = session()
        m = s.model
        for key, position, limits, mass, gain, force in (
            ("probe_y", [2.14, 0.60, 1.46], [-0.23, 0.23], 1, 400, 800),
            ("probe_z", [0, 0, 0.27], [-0.05, 0.06], 0.25, 200, 300),
        ):
            c = s.controls[EQUIPMENT, key]
            body = m.jnt_bodyid[c.joint]
            np.testing.assert_array_equal(m.body_pos[body], position)
            np.testing.assert_array_equal(m.jnt_range[c.joint], limits)
            self.assertEqual(m.body_mass[body], mass)
            np.testing.assert_array_equal(m.body_inertia[body], [mass * 0.006] * 3)
            members = np.flatnonzero(m.body_weldid == m.body_weldid[body])
            self.assertEqual(float(m.body_mass[members].sum()), mass)
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], gain)
            np.testing.assert_array_equal(
                m.actuator_forcerange[c.actuator], [-force, force]
            )
        np.testing.assert_array_equal(
            m.geom(PREFIX + "reference").size, [0.045, 0.035, 0.04]
        )
        np.testing.assert_array_equal(
            m.site(EQUIPMENT + "__site_dry_reference").pos, [0, 0, 0.07]
        )
        np.testing.assert_array_equal(m.geom(PREFIX + "needle").size[:2], [0.004, 0.05])

    def test_connected_assemblies_and_fixed_load_path(self):
        s = session()
        m = s.model
        for first, second in (
            ("reference", "reference_pad"),
            ("reference_pad", "carriage"),
            ("head", "needle"),
            ("column", "carriage"),
        ):
            self.assertLessEqual(gap(s, PREFIX + first, PREFIX + second), 1e-8)
        for i, j in product(range(2), repeat=2):
            for first, second in (
                (f"leg_{i}_{j}", "bed"),
                (f"rail_y_mount_{i}_{j}", "bed"),
                (f"rail_y_mount_{i}_{j}", f"rail_y_{i}"),
            ):
                self.assertLessEqual(gap(s, PREFIX + first, PREFIX + second), 1e-8)
            self.assertLessEqual(gap(s, PREFIX + f"leg_{i}_{j}", "arch_floor"), 1e-8)
        # Parent filtering must not conceal disconnected subparts of either
        # moving rigid assembly. Same-weld material interfaces may overlap.
        for c in s.controls.values():
            weld = m.body_weldid[m.jnt_bodyid[c.joint]]
            ids = set(
                np.flatnonzero(
                    (m.geom_contype | m.geom_conaffinity)
                    & (m.body_weldid[m.geom_bodyid] == weld)
                )
            )
            reached = {ids.pop()}
            while ids:
                linked = {
                    g
                    for g in ids
                    if any(
                        mujoco.mj_geomDistance(m, s.data, int(g), int(h), 0.00001, None)
                        <= 1e-8
                        for h in reached
                    )
                }
                self.assertTrue(linked, [m.geom(int(g)).name for g in ids])
                ids -= linked
                reached |= linked

    def test_open_shells_are_closed_surfaces_with_real_bores(self):
        root, _ = build_scene(SCENE)
        meshes = [
            m
            for m in root.findall("./asset/mesh")
            if m.get("name", "").startswith(PREFIX)
        ]
        # Six open bearings and six casing shells remain part of the probe rig.
        # The later context layer adds inlet flanges and the exposed band ends.
        probe = {f"bearing_y_{i}_{j}" for i, j in product(range(2), repeat=2)}
        probe |= {f"bearing_z_{i}" for i in range(2)}
        probe |= {f"casing_{i}" for i in range(6)}
        context = {"context_inlet_flange", "context_inlet_reducer"}
        context |= {
            f"context_band_{i}_{end}"
            for i, end in product(range(6), range(2))
            if (i, end) != (0, 0)
        }
        self.assertCountEqual(
            [mesh.get("name") for mesh in meshes],
            [PREFIX + name + "_shell_mesh" for name in probe | context],
        )
        for mesh in meshes:
            vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
            faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
            # The shared annular helper repeats seam vertices. Weld coincident
            # positions for topology, without changing actual simulation meshes.
            _, inverse = np.unique(np.round(vertices, 9), axis=0, return_inverse=True)
            welded = inverse[faces]
            edges = Counter(
                tuple(sorted(e)) for f in welded for e in zip(f, np.roll(f, -1))
            )
            self.assertEqual(set(edges.values()), {2}, mesh.get("name"))
            volume = (
                np.einsum(
                    "ij,ij->i",
                    vertices[faces[:, 0]],
                    np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
                ).sum()
                / 6
            )
            self.assertGreater(volume, 0, mesh.get("name"))
        s = session()
        info = s.equipment[EQUIPMENT]
        self.assertEqual(
            info["casing_presentation"]["enabled_contact_cells_preserved"], 288
        )

    def test_coupled_full_travel_and_unculled_witnesses(self):
        s = session()
        task = ReferenceInspection(s)
        m, d = s.model, s.data
        for y, z in product(np.linspace(-0.23, 0.23, 7), np.linspace(-0.05, 0.06, 7)):
            d.qpos[:] = [y, z]
            mujoco.mj_forward(m, d)
            task.check(m, d)
            self.assertFalse(any(c.dist < -1e-8 for c in d.contact))
            # Both complete Y sleeves remain between fixed end supports.
            self.assertLess(abs(y) + 0.10 + 0.05, 0.44 - 0.02)
            self.assertGreater(0.27 + z - 0.04, 0.12)
            self.assertLess(0.27 + z + 0.04, 0.42)
        for y, z in ((0.071, -0.037), (-0.193, 0.051), (0.213, 0.014)):
            d.qpos[:] = [y, z]
            mujoco.mj_forward(m, d)
            direct = min(
                mujoco.mj_geomDistance(m, d, int(a), int(b), 0.002, None)
                for a, b in task.pairs
            )
            task.reset_counts()
            task.check(m, d)
            self.assertAlmostEqual(task.minimum, direct, places=10)
            self.assertGreaterEqual(direct, 0.0005998)

    def test_binding_rejects_before_mutation_and_failure_holds(self):
        for change in ("mask", "head", "state"):
            s = session()
            task = ReferenceInspection(s)
            if change == "mask":
                s.model.geom_contype[s.model.geom(PREFIX + "rail_y_0").id] = 0
            elif change == "head":
                s.model.geom_pos[s.model.geom(PREFIX + "head").id, 0] += 0.01
            else:
                s.data = mujoco.MjData(s.model)
            before = s.data.qpos.copy(), s.data.ctrl.copy(), s.data.time
            with self.assertRaises(ValueError):
                task.run()
            np.testing.assert_array_equal(s.data.qpos, before[0])
            np.testing.assert_array_equal(s.data.ctrl, before[1])
            self.assertEqual(s.data.time, before[2])
        s = session()
        s.data.qfrc_applied[1] = 100
        task = ReferenceInspection(s)
        result = task.run()
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["executed_steps"], result["checked_steps"])
        self.assertIsNone(result["whole_route_clearance_lower_bound_m"])
        self.assertTrue(result["phases"][-1]["hold_requested"])
        self.assertEqual(s.data.qfrc_applied[1], 100)
        self.assertGreater(s.data.time, 0)
        self.assertFalse(result["state_reset"])

    def test_actual_datum_traverse_and_return(self):
        s = session()
        result = inspect_whittle_reference(s)
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["executed_steps"], result["checked_steps"])
        self.assertGreaterEqual(
            result["whole_route_clearance_lower_bound_m"], 0.0005998
        )
        self.assertEqual(len(result["phases"]), 6)
        for phase in result["phases"]:
            self.assertTrue(phase["observation"]["registered"])
        for phase in result["phases"][2:4]:
            self.assertAlmostEqual(phase["observation"]["gap_m"], 0.006, delta=0.00005)
        np.testing.assert_allclose(s.data.qpos, [0, 0], atol=5e-5)


if __name__ == "__main__":
    unittest.main()
