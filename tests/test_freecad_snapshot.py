"""Actual saved assembly links, aliases and independently measured ball datums."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.freecad_snapshot import SavedAssembly
from microscopy.parallel_kinematics import CAD_GEOMETRY, FIRMWARE_GEOMETRY, closure_residuals, forward


class SavedStepperAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("HOOKE_TEST_STEPPER_CAD_ROOT")
        if not root:
            raise unittest.SkipTest("Set HOOKE_TEST_STEPPER_CAD_ROOT to the hash-verified source documents")
        cls.assembly = SavedAssembly(root)
        cls.parts = cls.assembly.instances()

    def test_logical_group_aliases_and_compound_motor_links(self):
        self.assertEqual(len(self.parts), 173)
        self.assertEqual(len({part["instance"] for part in self.parts}), 173)
        self.assertEqual(sum(p["object"] == "Screw021" for p in self.parts), 1)
        motors = [p for p in self.parts if p["source"]["document"] == "StepperMotorNema17.FCStd"]
        self.assertEqual(len(motors), 33)  # Seven bodies and four fasteners per motor.
        self.assertFalse(any(p["source"]["object"].startswith("Sketch") for p in motors))
        for part in self.parts:
            np.testing.assert_allclose(part["transform"][:3, :3].T@part["transform"][:3, :3], np.eye(3), atol=1e-12)

    def test_cad_ball_pairs_independently_close_all_six_rods(self):
        balls = [p for p in self.parts if p["source"]["document"] == "JointBall.FCStd"]
        self.assertEqual(len(balls), 12)
        for index, leg in enumerate(("Actuator", "Actuator001", "Actuator002")):
            points = np.array([p["transform"][:3, 3] for p in balls if p["instance"].split("/")[1] == leg])*1e-3
            # Saved ball geometry has a kernel-verified spherical centre at local zero.
            rotor, platform = points[:2], points[2:]
            np.testing.assert_allclose(np.linalg.norm(rotor[:, None]-platform, axis=2).min(axis=1), .0725, rtol=0, atol=1e-8)
            measured = platform.mean(axis=0)-.0475
            np.testing.assert_allclose(measured, CAD_GEOMETRY.platform_attachments_m[index], rtol=0, atol=1e-8)

    def test_firmware_and_cad_datums_remain_distinct(self):
        angles = np.deg2rad([42., 42., 42.])
        np.testing.assert_allclose(forward(angles, CAD_GEOMETRY), 0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(closure_residuals(angles, [0, 0, 0], CAD_GEOMETRY), 0, rtol=0, atol=1e-12)
        self.assertGreater(np.max(abs(forward(angles, FIRMWARE_GEOMETRY))), 40e-6)


if __name__ == "__main__":
    unittest.main()
