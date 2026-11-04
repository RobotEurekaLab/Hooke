"""Complete arm interfaces, missed-contact regression and measured task coverage."""

from copy import copy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.motion import PlanningError
from real_labs.robot_arm_motion import ArmMotion
from real_labs.runtime import InstrumentSession


def session(scene="waterloo_robohub"):
    root, manifest = build_scene(scene)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


class CompleteArmTests(unittest.TestCase):
    def test_primary_outer_shapes_enabled_on_both_scales(self):
        for scene, eid, prefix, scale in (
            ("waterloo_robohub", "robot_arm", "", 1),
            ("nasa_planetary", "rover", "arm_", 0.62),
        ):
            with self.subTest(scene=scene):
                s = session(scene)
                motion = ArmMotion(s)
                for key in (
                    "foundation",
                    "yaw_housing",
                    "shoulder_housing",
                    "upper_link",
                    "elbow_housing",
                    "forearm_link",
                    "roll_housing",
                    "pitch_housing",
                    "tool_flange",
                    "palm",
                    "finger_left",
                    "finger_right",
                ):
                    g = s.model.geom(eid + "__arm_contact_" + prefix + key).id
                    self.assertEqual(s.model.geom_contype[g], 1)
                    self.assertEqual(s.model.geom_conaffinity[g], 1)
                contacts = motion.configuration_contacts(motion.measured())
                if scene == "nasa_planetary":
                    # The source-bound pickup cradle starts near a real sample.
                    # It is not an empty-tool pose with 10 mm world clearance.
                    self.assertTrue(contacts)
                    self.assertTrue(all(row["distance_m"] > 0 for row in contacts))
                    self.assertTrue(all(
                        "nasa_task__sample" in row["geoms"]
                        or "nasa_task__pickup_post" in row["geoms"]
                        for row in contacts
                    ))
                    with self.assertRaises(PlanningError):
                        motion.plan_joint_target(motion.measured())
                else:
                    self.assertFalse(contacts)
                self.assertAlmostEqual(
                    s.equipment[eid]["arm_interfaces"][prefix or "main"]["scale"], scale
                )
                for name in motion.names:
                    body = s.model.jnt_bodyid[s.controls[eid, name].joint]
                    own = np.flatnonzero(s.model.geom_bodyid == body)
                    self.assertGreater(
                        sum(bool(s.model.geom_contype[g]) for g in own), 0
                    )

    def test_original_joint_frames_ranges_and_sites_remain(self):
        s = session()
        expected = {
            "base_yaw": ((0, 0, 0.115), (-np.pi, np.pi)),
            "shoulder": ((0, 0, 0.075), (-1.65, 1.65)),
            "elbow": ((0.21, 0, 0.25), (-2.15, 2.15)),
            "wrist_roll": ((0.24, 0, 0.075), (-np.pi, np.pi)),
            "wrist_pitch": ((0.065, 0, 0), (-1.9, 1.9)),
            "tool_yaw": ((0.045, 0, 0), (-np.pi, np.pi)),
        }
        for name, (position, limits) in expected.items():
            c = s.controls["robot_arm", name]
            np.testing.assert_allclose(
                s.model.body_pos[s.model.jnt_bodyid[c.joint]], position
            )
            np.testing.assert_allclose(s.model.jnt_range[c.joint], limits, atol=1e-7)
        tcp = s.equipment["robot_arm"]["sites"]["tcp"]
        np.testing.assert_allclose(
            s.model.site_pos[s.model.site(tcp).id], [0.112, 0, 0]
        )

    def test_foundation_reaches_real_bench(self):
        s = session()
        m, d = s.model, s.data
        foundation = m.geom("robot_arm__arm_contact_foundation").id
        table = m.geom("handoff_station_top").id
        bottom = d.geom_xpos[foundation, 2] - m.geom_size[foundation, 1]
        top = d.geom_xpos[table, 2] + m.geom_size[table, 2]
        self.assertAlmostEqual(bottom - top, 0, delta=1e-9)
        # The ring and spindle do not fill each other's bore. The native
        # parent filter is bypassed by direct distance measurements below.
        motion = ArmMotion(s)
        self.assertTrue(motion.engineered.any())
        self.assertTrue((~motion.engineered).any())

    def test_bearing_running_clearances_over_preserved_travel(self):
        for scene in ("waterloo_robohub", "nasa_planetary"):
            with self.subTest(scene=scene):
                s = session(scene)
                motion = ArmMotion(s)
                for i in range(6):
                    for value in np.linspace(motion.lower[i], motion.upper[i], 9):
                        q = np.zeros(6)
                        q[i] = value
                        motion._pose(q)
                        gaps, _ = motion._distances(motion.model, motion.data)
                        minimum = (
                            gaps[motion.engineered]
                            - motion.thresholds[motion.engineered]
                        )
                        self.assertGreaterEqual(
                            float(min(minimum)), -1e-8, (scene, i, value)
                        )

    def test_added_collar_mass_keeps_original_gravity_compensation(self):
        s = session()
        m, d = s.model, s.data
        tcp = m.site(s.equipment["robot_arm"]["sites"]["tcp"]).id
        start = d.site_xpos[tcp].copy()
        for i in range(m.nbody):
            name = m.body(i).name
            if name.startswith("robot_arm__arm_contact_") and name.endswith("_mount"):
                self.assertEqual(m.body_gravcomp[i], 1)
        mujoco.mj_step(m, d, nstep=round(5 / m.opt.timestep))
        mujoco.mj_forward(m, d)
        self.assertLess(np.linalg.norm(d.site_xpos[tcp] - start), 1e-7)

    def test_each_fixed_arm_assembly_has_a_continuous_support_path(self):
        # Direct queries bypass same-body/parent filters. Every component in
        # each welded link must connect to the other components of that link.
        s = session()
        m, d = s.model, s.data
        groups = {}
        for g in range(m.ngeom):
            if m.geom(g).name.startswith("robot_arm__arm_contact_"):
                groups.setdefault(int(m.body_weldid[m.geom_bodyid[g]]), []).append(g)
        for weld, geoms in groups.items():
            components = [{g} for g in geoms]
            for ai, a in enumerate(geoms):
                for b in geoms[ai + 1 :]:
                    if mujoco.mj_geomDistance(m, d, a, b, 1e-6, None) > 1e-8:
                        continue
                    ia = next(i for i, c in enumerate(components) if a in c)
                    ib = next(i for i, c in enumerate(components) if b in c)
                    if ia != ib:
                        components[ia].update(components[ib])
                        components.pop(ib)
            self.assertEqual(len(components), 1, m.body(weld).name)

    def test_finger_guides_retain_clearance_and_loaded_rover_keeps_sample_obstacles(self):
        for scene, eid, prefix in (
            ("waterloo_robohub", "robot_arm", ""),
            ("nasa_planetary", "rover", "arm_"),
        ):
            s = session(scene)
            motion = ArmMotion(s)
            for side in ("left", "right"):
                c = s.controls[eid, prefix + "gripper_" + side]
                adr = s.model.jnt_qposadr[c.joint]
                for value in np.linspace(c.lower, c.upper, 9):
                    motion.reference_qpos[adr] = value
                    contacts = motion.configuration_contacts(np.zeros(6))
                    gaps, _ = motion._distances(motion.model, motion.data)
                    self.assertTrue(np.all(
                        gaps[motion.engineered] >= motion.thresholds[motion.engineered] - 1e-8
                    ), (scene, side, value))
                    if scene == "waterloo_robohub":
                        self.assertFalse(contacts, (scene, side, value))
                    elif value == c.lower:
                        sample = [row for row in contacts if "nasa_task__sample" in row["geoms"]]
                        self.assertTrue(sample)
                        self.assertLess(min(row["distance_m"] for row in sample), -0.006)
                motion.reference_qpos[adr] = s.data.qpos[adr]

    def test_core_failure_does_not_claim_unobserved_step_clearance(self):
        s = session()
        motion = ArmMotion(s)
        s.data.qfrc_applied[motion.dadr[0]] = 1000
        target = motion.measured()
        target[0] = 0.02
        plan = motion.plan_joint_target(target)
        result = motion.execute(plan)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        coverage = result["direct_checks"]
        self.assertLess(coverage["fully_checked_steps"], result["physics_steps"])
        self.assertFalse(coverage["coverage_complete"])
        self.assertIsNone(coverage["whole_leg_external_clearance_lower_bound_m"])
        self.assertGreater(s.data.time, 0)

    def test_old_36mm_missed_table_pose_is_rejected(self):
        s = session()
        motion = ArmMotion(s)
        q = [
            0,
            0.587688431243934,
            0.7546779217804866,
            -2.0819692901098943,
            0.2172500655479257,
            0,
        ]
        contacts = motion.configuration_contacts(q)
        table = [row for row in contacts if "handoff_station_top" in row["geoms"]]
        self.assertTrue(table, contacts)
        self.assertLess(min(row["distance_m"] for row in table), -0.01)
        before = s.data.qpos.copy(), s.data.ctrl.copy()
        with self.assertRaisesRegex(PlanningError, "goal violates"):
            motion.plan_joint_target(q)
        np.testing.assert_array_equal(s.data.qpos, before[0])
        np.testing.assert_array_equal(s.data.ctrl, before[1])
        self.assertEqual(s.data.time, 0)

    def test_foreign_model_and_changed_geometry_rejected_before_command(self):
        for mode in ("foreign", "mask", "class", "flags"):
            with self.subTest(mode=mode):
                s = session()
                motion = ArmMotion(s)
                if mode == "foreign":
                    s.data = mujoco.MjData(copy(s.model))
                elif mode == "mask":
                    g = s.model.geom("robot_arm__arm_contact_pitch_housing").id
                    s.model.geom_contype[g] = s.model.geom_conaffinity[g] = 0
                elif mode == "class":
                    s.equipment["robot_arm"]["arm_interfaces"]["main"][
                        "engineered_pair_classes"
                    ][0]["minimum_clearance_m"] = 0
                else:
                    s.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
                before = s.data.qpos.copy(), s.data.ctrl.copy()
                with self.assertRaises((ValueError, PlanningError)):
                    motion.plan_joint_target(np.zeros(6))
                np.testing.assert_array_equal(s.data.qpos, before[0])
                np.testing.assert_array_equal(s.data.ctrl, before[1])
                self.assertEqual(s.data.time, 0)

    def test_callback_failure_keeps_measured_state_and_requests_hold(self):
        s = session()
        motion = ArmMotion(s)
        goal = motion.measured()
        goal[0] = 0.02
        plan = motion.plan_joint_target(goal)
        states = []

        def abort(m, d):
            states.append((d.qpos.copy(), d.qvel.copy()))
            raise RuntimeError("intentional_read_only_abort")

        result = motion.execute(plan, step_observer=abort)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        np.testing.assert_array_equal(s.data.qpos, states[-1][0])
        np.testing.assert_array_equal(s.data.qvel, states[-1][1])
        self.assertEqual(result["physics_steps"], 1)
        self.assertEqual(result["direct_checks"]["fully_checked_steps"], 1)
        np.testing.assert_allclose(s.data.ctrl[motion.actuators], motion.measured())


if __name__ == "__main__":
    unittest.main()
