"""Open, supported source-informed Duke assemblies and guarded dry motion."""

from copy import copy
import itertools
import math
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.duke_frame_geometry import PREFIX
from real_labs.duke_frame_motion import GuardedInspection, SCENE, CONTROLS
from real_labs.runtime import InstrumentSession
from real_labs.equipment import add_equipment


def load():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class DukeFrameTests(unittest.TestCase):
    def test_two_named_instances_have_independent_geometry_and_controls(self):
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
        world = ET.SubElement(root, "worldbody")
        records = [
            add_equipment(
                root, world, "duke_830leat_alignment_frame", name, (offset, 0, 0)
            )
            for name, offset in (("duke_first", -2), ("duke_second", 2))
        ]
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.assertEqual(model.nu, 6)
        for name, record in zip(("duke_first", "duke_second"), records):
            self.assertTrue(model.geom(name + "__duke_ram").contype)
            self.assertTrue(
                all(v.startswith(name + "__") for v in record["joints"].values())
            )

    def test_original_controls_and_interpreted_clear_width(self):
        s = load()
        m = s.model
        for name, frame, axis, limits, mass, kp in (
            (CONTROLS[0], (0, 0, 1.47), (0, 0, -1), (0, 0.2), 0.6, 500),
            (CONTROLS[1], (0, 0, 0.934), (0, 0, 1), (-0.6, 0.6), 0.35, 160),
        ):
            c = s.controls["biaxial_frame", name]
            for actual, expected in (
                (m.body_pos[m.jnt_bodyid[c.joint]], frame),
                (m.jnt_axis[c.joint], axis),
                (m.jnt_range[c.joint], limits),
            ):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(m.body_mass[m.jnt_bodyid[c.joint]], mass)
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], kp)
        left, right = [m.geom(PREFIX + f"column_{x}").id for x in (-0.31, 0.31)]
        clear = (
            s.data.geom_xpos[right, 0]
            - s.data.geom_xpos[left, 0]
            - m.geom_size[left, 0]
            - m.geom_size[right, 0]
        )
        self.assertAlmostEqual(clear, 0.55)
        self.assertTrue(m.geom(PREFIX + "ram").contype)
        self.assertTrue(m.geom(PREFIX + "torsion_spindle").contype)

    def test_full_travel_direct_filtered_clearance_and_engagement(self):
        task = GuardedInspection(load())
        m, d = task.model, task.data
        controls = [task.session.controls["biaxial_frame", key] for key in CONTROLS]
        poses = list(itertools.product((0, 0.5, 1), repeat=3))
        for axis in range(3):
            for peer in (0, 1):
                for fraction in np.linspace(0, 1, 21):
                    row = [peer] * 3
                    row[axis] = fraction
                    poses.append(row)
        for pose in poses:
            for c, value in zip(controls, pose):
                d.qpos[m.jnt_qposadr[c.joint]] = c.lower + value * (c.upper - c.lower)
            mujoco.mj_forward(m, d)
            task.check(m, d)
            ram = m.geom(PREFIX + "ram").id
            self.assertLess(d.geom_xpos[ram, 2] - m.geom_size[ram, 1], 1.605)
            self.assertGreater(d.geom_xpos[ram, 2] + m.geom_size[ram, 1], 1.82)
            self.assertGreaterEqual(
                task.observe()["actual_grip_to_coupon_vertical_gap_m"], 0.055 - 1e-12
            )
        self.assertGreaterEqual(task.minimum, 0.00025)

    def test_moving_parts_form_real_connected_groups(self):
        s = load()
        m, d = s.model, s.data
        for key in CONTROLS:
            body = int(m.jnt_bodyid[s.controls["biaxial_frame", key].joint])
            ids = [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == body and m.geom_contype[g]
            ]
            reached = {ids[0]}
            while True:
                new = {
                    g
                    for g in ids
                    if g not in reached
                    and any(
                        mujoco.mj_geomDistance(m, d, g, h, 0.01, None) <= 1e-7
                        for h in reached
                    )
                }
                if not new:
                    break
                reached.update(new)
            self.assertEqual(
                set(ids), reached, [m.geom(g).name for g in set(ids) - reached]
            )
        # Query cap is distinctly larger than the support acceptance tolerance.
        self.assertGreater(
            mujoco.mj_geomDistance(
                m,
                d,
                m.geom(PREFIX + "door_handle").id,
                m.geom(PREFIX + "ram").id,
                0.01,
                None,
            ),
            1e-7,
        )

    def test_shelf_brackets_and_control_stems_touch_real_supports(self):
        s = load()
        m, d = s.model, s.data
        walls = [g for g in range(m.ngeom) if m.geom(g).name.startswith("arch_wall_")]
        for index in range(3):
            for height in (1.50, 2.13):
                anchor = m.geom(f"duke_open_shelf_{index}_anchor_{height}").id
                back = m.geom(f"duke_open_shelf_{index}_back").id
                self.assertLessEqual(
                    mujoco.mj_geomDistance(m, d, anchor, back, 0.1, None), 1e-7
                )
                self.assertLessEqual(
                    min(
                        mujoco.mj_geomDistance(m, d, anchor, w, 0.1, None)
                        for w in walls
                    ),
                    1e-7,
                )
        control = m.geom(PREFIX + "small_controller").id
        for x in (1.08 + 0.085, 1.08 + 0.14):
            knob = m.geom(PREFIX + f"controller_knob_{x}").id
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, knob, control, 0.01, None), 1e-7
            )
            self.assertGreater(
                min(mujoco.mj_geomDistance(m, d, knob, w, 0.01, None) for w in walls),
                1e-7,
            )

    def test_disturbed_held_axis_fails_without_false_full_coverage(self):
        session = load()
        task = GuardedInspection(session)
        joint = session.controls["biaxial_frame", CONTROLS[0]].joint
        session.data.qfrc_applied[session.model.jnt_dofadr[joint]] = 1000
        report = task.move((0, 0, 0.02))
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["reset_performed"])
        self.assertTrue(report["hold_requested"])
        checks = report["direct_clearance"]
        self.assertGreater(report["physics_steps"], checks["checked_steps"])
        self.assertIsNone(checks["whole_leg_clearance_lower_bound_m"])

    def test_subdivision_preserves_legal_endpoint_without_relaxing_limits(self):
        session = load()
        start = (0.12236169158497759, -0.542388708439772, 1.5599255632904756)
        for key, value in zip(CONTROLS, start):
            control = session.controls["biaxial_frame", key]
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
            session.data.ctrl[control.actuator] = value
        mujoco.mj_forward(session.model, session.data)
        task = GuardedInspection(session)
        target = (0.2, 0.6, math.radians(100))
        before = task.state()
        planned = []

        def validate_plan(plan, **kwargs):
            for point in plan.waypoints:
                task.motion._configuration(point)
            planned.append(plan)
            return {"status": "planning_only", "physics_steps": 0}

        with patch.object(task.motion, "execute", side_effect=validate_plan):
            task.move(target)
            self.assertEqual(planned[0].waypoints[-1], target)
            with self.assertRaises(ValueError):
                task.move((target[0], np.nextafter(target[1], np.inf), target[2]))
        self.assertEqual(len(planned), 1)
        self.assertEqual(task.state(), before)

    def test_actual_commands_and_binding_failure(self):
        s = load()
        task = GuardedInspection(s)
        report = task.move((0.015, 0.05, 0.15))
        self.assertEqual(report["status"], "succeeded", report.get("failure"))
        self.assertEqual(
            report["physics_steps"], report["direct_clearance"]["checked_steps"]
        )
        self.assertEqual(task.move((0, 0, 0))["status"], "succeeded")
        for kind in ("foreign", "mask"):
            session = load()
            check = GuardedInspection(session)
            if kind == "foreign":
                session.data = mujoco.MjData(copy(session.model))
            else:
                session.model.geom(PREFIX + "ram").contype = 0
            before = session.data.qpos.copy()
            with self.assertRaises(ValueError):
                check.move((0, 0, 0))
            np.testing.assert_array_equal(before, session.data.qpos)


if __name__ == "__main__":
    unittest.main()
