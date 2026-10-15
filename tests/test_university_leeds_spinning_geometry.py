"""Leeds shaft/bearing and carriage support under the original motion contract."""

import itertools
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.yarn_details import RUNNING_CLEARANCE_M


def session():
    root, manifest = build_scene("leeds_litac_yarn_spinning")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def distance(s, first, second):
    model = s.model
    first = model.geom(first).id if isinstance(first, str) else first
    second = model.geom(second).id if isinstance(second, str) else second
    return mujoco.mj_geomDistance(model, s.data, first, second, 10, None)


def named(model, suffix):
    return model.geom("yarn_machine__yarn_" + suffix).id


class LeedsSpinningGeometryTests(unittest.TestCase):
    def test_original_axes_limits_target_scale_and_camera_contract_survive(self):
        s = session()
        self.assertEqual(s.model.njnt, 2)
        self.assertEqual(s.model.nu, 2)
        roller = s.controls["yarn_machine", "feed_roller"]
        carriage = s.controls["yarn_machine", "bobbin_carriage"]
        np.testing.assert_allclose([roller.lower, roller.upper], [-np.pi, np.pi])
        np.testing.assert_allclose([carriage.lower, carriage.upper], [-0.16, 0.16])
        site = s.model.site("yarn_machine__site_inert_yarn_bobbin").id
        np.testing.assert_allclose(s.data.site_xpos[site], [0, -0.185, 0.724])
        self.assertEqual(
            {s.model.camera(i).name for i in range(s.model.ncam)},
            {"overview", "interior", "workstation", "bobbin"},
        )
        self.assertIn(
            "0.5 mm", s.equipment["yarn_machine"]["assembly_adaptations"]["authored"]
        )

    def test_shaft_journals_enter_hollow_bearings_and_brackets_reach_the_frame(self):
        s = session()
        m = s.model
        shaft_body = m.body("yarn_machine__body_feed_roller").id
        shaft = next(
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == shaft_body
            and np.allclose(m.geom_size[g], [0.043, 0.64, 0])
        )
        for side in ("left", "right"):
            journal = named(m, side + "_journal")
            self.assertLess(distance(s, shaft, journal), 0)
            proxies = [
                g
                for g in range(m.ngeom)
                if m.geom(g).name.startswith(
                    f"yarn_machine__yarn_{side}_feed_bearing_proxy_"
                )
            ]
            clearance = min(distance(s, journal, g) for g in proxies)
            self.assertAlmostEqual(clearance, RUNNING_CLEARANCE_M, places=6)
            bracket = named(m, side + "_bearing_bracket")
            self.assertLess(min(distance(s, bracket, g) for g in proxies), 0)
            side_body = m.body(f"yarn_machine__side_{side}").id
            supports = [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == side_body and m.geom_contype[g]
            ]
            self.assertLessEqual(min(distance(s, bracket, g) for g in supports), 0)

    def test_split_guide_bearings_bridge_plate_gap_without_striking_deck(self):
        s = session()
        m = s.model
        body = m.body("yarn_machine__body_bobbin_carriage").id
        plate = next(
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == body
            and np.allclose(m.geom_size[g], [0.18, 0.11, 0.021])
        )
        deck = next(
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith("yarn_machine__")
            and np.allclose(m.geom_size[g], [0.76, 0.43, 0.05])
        )
        rails = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith("yarn_machine__")
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            and np.allclose(m.geom_size[g], [0.012, 0.5, 0])
        ]
        self.assertEqual(len(rails), 2)
        self.assertTrue(all(m.geom_contype[g] == 1 for g in rails))
        for index, xindex in itertools.product(range(2), range(2)):
            prefix = f"yarn_machine__yarn_linear_bearing_{index}_{xindex}_proxy_"
            shoes = [g for g in range(m.ngeom) if m.geom(g).name.startswith(prefix)]
            self.assertLess(min(distance(s, plate, g) for g in shoes), 0)
            self.assertGreater(min(distance(s, deck, g) for g in shoes), 0.01)
            clearance = min(distance(s, rail, shoe) for rail in rails for shoe in shoes)
            self.assertAlmostEqual(clearance, RUNNING_CLEARANCE_M, places=6)

    def test_full_strokes_have_no_interference_including_parent_filtered_pairs(self):
        s = session()
        m = s.model
        controls = list(s.controls.values())
        moving_bodies = {m.jnt_bodyid[c.joint] for c in controls}
        moving = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] in moving_bodies and m.geom_contype[g]
        ]
        fixed = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] not in moving_bodies and m.geom_contype[g]
        ]
        states = [(q, 0) for q in np.linspace(controls[0].lower, controls[0].upper, 65)]
        states += [
            (0, q) for q in np.linspace(controls[1].lower, controls[1].upper, 65)
        ]
        states += list(
            itertools.product(
                np.linspace(controls[0].lower, controls[0].upper, 7),
                np.linspace(controls[1].lower, controls[1].upper, 7),
            )
        )
        for values in states:
            s.reset()
            for c, value in zip(controls, values):
                s.data.qpos[m.jnt_qposadr[c.joint]] = value
            mujoco.mj_forward(m, s.data)
            self.assertFalse([c for c in s.data.contact if c.dist < -1e-5], values)
            # Parent-child collision filtering cannot hide a solid bearing bore.
            for first in moving:
                separation = np.linalg.norm(
                    s.data.geom_xpos[fixed] - s.data.geom_xpos[first], axis=1
                )
                near = np.asarray(fixed)[
                    separation < m.geom_rbound[fixed] + m.geom_rbound[first] + 0.001
                ]
                for second in near:
                    gap = distance(s, first, int(second))
                    self.assertGreaterEqual(
                        gap,
                        -1e-5,
                        (values, m.geom(first).name, m.geom(int(second)).name, gap),
                    )

    def test_reversals_track_actual_axes_without_collision_or_held_axis_drift(self):
        s = session()
        m = s.model
        for key, control in s.controls.items():
            s.reset()
            for fraction in (0.15, 0.85, 0.25, 0, 1, 0.5):
                target = control.lower + fraction * (control.upper - control.lower)
                s.data.ctrl[control.actuator] = target
                for _ in range(round(2 / m.opt.timestep)):
                    mujoco.mj_step(m, s.data)
                    self.assertFalse([c for c in s.data.contact if c.dist < -1e-5], key)
                    self.assertTrue(np.isfinite(s.data.qpos).all())
                self.assertLessEqual(
                    abs(s._measured(control) - target),
                    0.025 * (control.upper - control.lower),
                )
                for other in s.controls.values():
                    if other.actuator != control.actuator:
                        self.assertLess(abs(s._measured(other)), 1e-5)
            self.assertFalse(any(w.number for w in s.data.warning))
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, m.qpos0)
        self.assertEqual(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
