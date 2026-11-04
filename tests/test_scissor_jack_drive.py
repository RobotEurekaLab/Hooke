"""CPU physics and contract checks for the production torque-driven jack."""

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.epfl_jack_geometry import add_scissor_jack, numbers
from real_labs.scissor_jack_drive import ScissorJackDrive


def fixture(*, payload_mass=1.8, tilt=0):
    """Real production link/contact geometry, two hoods and a preceding free body."""
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=".002", integrator="implicitfast",
                  solver="Newton", iterations="80", tolerance="1e-10")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    free = ET.SubElement(world, "body", name="preceding_free_body", pos="4 0 1", gravcomp="1")
    ET.SubElement(free, "freejoint", name="preceding_free_joint")
    ET.SubElement(free, "geom", type="sphere", size=".02", mass=".1")
    metadata = []
    for i, name in enumerate(("left", "right")):
        for material in ("metal", "rubber"):
            ET.SubElement(asset, "material", name=f"{name}__mat_{material}", rgba=".3 .3 .3 1")
        parent = ET.SubElement(world, "body", name=name, pos=numbers((i, 0, .6)),
                               euler=numbers((tilt, 0, .3)))
        datum_name, rim_name = f"{name}_bench_datum", f"{name}_bath_rim"
        ET.SubElement(parent, "site", name=datum_name, pos="0 0 0", size=".001")
        assembly = add_scissor_jack(SimpleNamespace(root=root, name=name), parent, origin=(0, 0, 0))
        payload = ET.SubElement(assembly["deck"], "body", name=f"{name}_thermal_payload", pos=".075 0 .018")
        ET.SubElement(payload, "inertial", mass=str(payload_mass), pos="0 0 .065",
                      diaginertia=numbers((payload_mass*.005, payload_mass*.005, payload_mass*.008)))
        ET.SubElement(payload, "site", name=rim_name, pos="0 0 .140", size=".001")
        metadata.append(dict(kind="scissor_jack_v1", joints=assembly["joints"],
                             equalities=assembly["equalities"], actuator=assembly["actuator"],
                             deck_site=assembly["deck_site"], datum_site=datum_name,
                             observation_site=rim_name, range=[.245, .375],
                             deck_to_observation_offset_m=.140))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, metadata


class ScissorJackDriveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, cls.metadata = fixture()

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.drives = [ScissorJackDrive(self.model, meta) for meta in self.metadata]
        for drive in self.drives:
            drive.reset(self.data)

    def test_two_namespaces_freejoint_motion_and_independent_holds(self):
        left, right = self.drives
        self.assertNotEqual(left.qpos["screw_rotation"], left.dof["screw_rotation"])
        self.assertTrue(set(left.joints.values()).isdisjoint(right.joints.values()))
        qpos, qvel, time = self.data.qpos.copy(), self.data.qvel.copy(), self.data.time
        self.assertAlmostEqual(left.start(self.data, .375, .2), 15)
        np.testing.assert_array_equal(self.data.qpos, qpos)
        np.testing.assert_array_equal(self.data.qvel, qvel)
        self.assertEqual(self.data.time, time)
        held = right.measured(self.data)
        largest_hold_error = 0
        for _ in range(math.ceil(16/self.model.opt.timestep)):
            mujoco.mj_forward(self.model, self.data)
            for drive in self.drives:
                drive.update(self.data)
            mujoco.mj_step(self.model, self.data)
            largest_hold_error = max(largest_hold_error, abs(right.measured(self.data)-held))
        mujoco.mj_forward(self.model, self.data)
        self.assertLess(abs(left.measured(self.data)-.375), .0005)
        self.assertLess(largest_hold_error, .0005)
        self.assertFalse(any(w.number for w in self.data.warning))
        self.assertTrue(np.isfinite(self.data.qpos).all())
        # A short move on the other drive must keep the raised jack held.
        raised = left.measured(self.data)
        duration = right.start(self.data, .253, .05)
        self.assertAlmostEqual(duration, 15*.008/.13)
        for _ in range(math.ceil((duration+1)/self.model.opt.timestep)):
            mujoco.mj_forward(self.model, self.data)
            for drive in self.drives:
                drive.update(self.data)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.assertLess(abs(right.measured(self.data)-.253), .0005)
        self.assertLess(abs(left.measured(self.data)-raised), .0005)
        status = left.inspect(self.data)
        self.assertEqual(status["target"], .375)
        self.assertTrue(status["duration_extended"])
        self.assertFalse(status["saturated"])
        self.assertLess(abs(status["requested_torque_Nm"]), .2)

    def test_reset_observes_height_without_writing_pose_or_control(self):
        drive = self.drives[0]
        before = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy())
        drive.reset(self.data)
        for actual, expected in zip((self.data.qpos, self.data.qvel, self.data.ctrl), before):
            np.testing.assert_array_equal(actual, expected)
        self.assertAlmostEqual(drive.inspect(self.data)["target"], .245)

    def test_reset_clamps_only_small_roundoff(self):
        drive = self.drives[0]
        self.data.site_xpos[drive.observation_site, 2] -= 5e-9
        drive.reset(self.data)
        self.assertEqual(drive.target, drive.lower)
        self.data.site_xpos[drive.observation_site, 2] -= 2e-8
        with self.assertRaisesRegex(ValueError, "Reset bath height"):
            drive.reset(self.data)

    def test_invalid_targets_durations_and_metadata_are_rejected(self):
        drive = self.drives[0]
        for target in (.244, .376, math.nan, math.inf, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                drive.start(self.data, target, 1)
        for duration in (0, -1, 30.01, math.nan, math.inf, True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                drive.start(self.data, .3, duration)
        for key, value in (("range", [.245, math.nan]), ("range", [.245, .4]),
                           ("deck_to_observation_offset_m", math.inf), ("deck_to_observation_offset_m", .15),
                           ("actuator", "missing"), ("kind", "position")):
            meta = copy.deepcopy(self.metadata[0]); meta[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ScissorJackDrive(self.model, meta)
        meta = copy.deepcopy(self.metadata[0]); del meta["joints"]["a1_angle"]
        with self.assertRaises(ValueError):
            ScissorJackDrive(self.model, meta)

    def test_foreign_motor_loop_and_observation_are_rejected(self):
        for key in ("actuator", "observation_site"):
            meta = copy.deepcopy(self.metadata[0]); meta[key] = self.metadata[1][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                ScissorJackDrive(self.model, meta)
        meta = copy.deepcopy(self.metadata[0])
        meta["equalities"]["lower_scissor"] = self.metadata[1]["equalities"]["lower_scissor"]
        with self.assertRaises(ValueError):
            ScissorJackDrive(self.model, meta)

    def test_wrong_motor_and_transmission_parameters_are_rejected(self):
        drive = self.drives[0]
        for array, index, value in ((self.model.actuator_gear, (drive.actuator, 0), 2),
                                    (self.model.actuator_gainprm, (drive.actuator, 0), 10),
                                    (self.model.eq_data, (drive.equalities["screw_transmission"], 1), .001)):
            old = array[index]
            try:
                array[index] = value
                with self.assertRaises(ValueError):
                    ScissorJackDrive(self.model, self.metadata[0])
            finally:
                array[index] = old

    def test_incompatible_link_dimensions_are_rejected(self):
        body = self.drives[0].bodies["b2_angle"]
        old = self.model.body_pos[body].copy()
        try:
            self.model.body_pos[body, 0] += .01
            with self.assertRaises(ValueError):
                ScissorJackDrive(self.model, self.metadata[0])
        finally:
            self.model.body_pos[body] = old

    def test_wrong_base_screw_and_second_loop_anchor_are_rejected(self):
        drive = self.drives[0]
        for array, index, value in (
            (self.model.body_pos, (drive.bodies["a1_angle"], 0), .001),
            (self.model.body_pos, (drive.bodies["screw_rotation"], 2), .1),
            (self.model.body_quat, (drive.bodies["nut_translation"], 0), .99),
            (self.model.eq_data, (drive.equalities["lower_scissor"], 3), 0),
        ):
            old = array[index]
            try:
                array[index] = value
                with self.assertRaises(ValueError):
                    ScissorJackDrive(self.model, self.metadata[0])
            finally:
                array[index] = old

    def test_tilted_bench_is_rejected(self):
        model, _, metadata = fixture(tilt=.02)
        with self.assertRaisesRegex(ValueError, "upright"):
            ScissorJackDrive(model, metadata[0])

    def test_real_overload_reports_saturation_and_caps_torque(self):
        model, data, metadata = fixture(payload_mass=1000)
        drive = ScissorJackDrive(model, metadata[0]); drive.reset(data)
        qpos = data.qpos.copy()
        with self.assertRaisesRegex(RuntimeError, "torque_saturation"):
            drive.update(data)
        status = drive.inspect(data)
        self.assertTrue(status["saturated"])
        self.assertGreater(abs(status["requested_torque_Nm"]), .2)
        self.assertEqual(abs(status["applied_torque_Nm"]), .2)
        self.assertEqual(abs(data.ctrl[drive.actuator]), .2)
        np.testing.assert_array_equal(data.qpos, qpos)

    def test_foreign_control_disabled_loop_and_time_reset_fail(self):
        drive = self.drives[0]
        self.data.ctrl[drive.actuator] = .01
        with self.assertRaisesRegex(RuntimeError, "control_changed"):
            drive.update(self.data)
        self.data.ctrl[drive.actuator] = 0
        equality = drive.equalities["lower_scissor"]
        self.data.eq_active[equality] = False
        with self.assertRaisesRegex(RuntimeError, "equality_disabled"):
            drive.update(self.data)
        self.data.eq_active[equality] = True
        self.data.time = -.01
        with self.assertRaisesRegex(RuntimeError, "time_reversed"):
            drive.update(self.data)

    def test_nonfinite_and_excessive_tracking_error_fail(self):
        drive = self.drives[0]
        self.data.qvel[drive.dof["screw_rotation"]] = math.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            drive.update(self.data)
        self.data.qvel[:] = 0
        self.data.site_xpos[drive.observation_site, 2] += .001
        with self.assertRaisesRegex(RuntimeError, "height_tracking_error"):
            drive.update(self.data)

    def test_loop_drift_and_live_configuration_change_fail(self):
        drive = self.drives[0]
        row = np.flatnonzero((self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                             & (self.data.efc_id == drive.equalities["screw_transmission"]))[0]
        self.data.efc_pos[row] = .0001/drive.jack.METRES_PER_RADIAN
        with self.assertRaisesRegex(RuntimeError, "loop_closure_error"):
            drive.update(self.data)
        self.data.efc_pos[row] = 0
        old = self.model.actuator_ctrlrange[drive.actuator].copy()
        try:
            self.model.actuator_ctrlrange[drive.actuator] = [-1, 1]
            with self.assertRaisesRegex(RuntimeError, "configuration_changed"):
                drive.update(self.data)
        finally:
            self.model.actuator_ctrlrange[drive.actuator] = old


if __name__ == "__main__":
    unittest.main()
