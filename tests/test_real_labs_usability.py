"""Usability screening exposes collision configurations without erasing ranges."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.export import current_scene_digest
from real_labs.usability import audit_model, audit_scene, camera_target_screening


def fixture(obstruction=None, kind="inspection_cart", free_obstruction=False):
    obstacle = (
        ""
        if obstruction is None
        else (
            f'<geom name="obstacle" type="sphere" size=".04" pos="{obstruction} 0 .2"/>'
        )
    )
    if obstruction is not None and free_obstruction:
        obstacle = (
            f'<body name="free_obstacle" pos="{obstruction} 0 .2"><freejoint/>'
            '<geom name="obstacle" type="sphere" size=".04" mass=".005"/></body>'
        )
    xml = f"""<mujoco><option gravity="0 0 0"/>
      <worldbody>{obstacle}
      <camera name="workstation" pos="1.5 0 .2" xyaxes="0 1 0 0 0 1"/>
      <body name="cart" pos="0 0 .2"><joint name="slide" type="slide"
        axis="1 0 0" range="0 1" damping="2"/>
        <geom name="cart__sample" type="sphere" size=".04" mass=".1"/>
        <site name="target" size=".001"/>
      </body></worldbody><actuator><position name="drive" joint="slide"
        kp="50" ctrlrange="0 1"/></actuator></mujoco>"""
    manifest = dict(
        id="test",
        room=dict(size=[4, 4, 2]),
        equipment=[
            dict(
                id="cart",
                kind=kind,
                joints={"slide": "slide"},
                actuators={"slide": "drive"},
                sample_site="target",
                sample_size_m=[0.08, 0.08, 0.08],
                sample_attachment="clamped",
            )
        ],
    )
    return mujoco.MjModel.from_xml_string(xml), manifest


class UsabilityScreeningTests(unittest.TestCase):
    def test_audit_hash_matches_the_exact_portable_scene_format(self):
        result = audit_scene("yonsei_mics_rf_probe", run_commands=False)
        self.assertEqual(
            result["scene_sha256"], current_scene_digest("yonsei_mics_rf_probe")
        )
        self.assertEqual(len(result["manifest_sha256"]), 64)

    def test_midstroke_obstacle_is_not_hidden_by_clear_endpoints(self):
        model, manifest = fixture(0.5)
        result = audit_model(model, manifest, run_commands=False)
        self.assertFalse(result["home_pose_penetrations"])
        self.assertTrue(result["range_contacts"])
        self.assertIn("requires_geometry_or_motion_review", result["issues"])
        self.assertEqual(result["sampled_poses"], 22)
        self.assertEqual(result["command_screening"], "not_run")
        np.testing.assert_allclose(model.jnt_range[0], [0, 1])

    def test_robot_collision_requires_planning_without_shortening_joint_range(self):
        model, manifest = fixture(0.5, kind="robot_arm")
        result = audit_model(model, manifest, run_commands=False)
        self.assertIn("requires_motion_planning", result["issues"])
        self.assertEqual(result["status"], "requires_review")
        np.testing.assert_allclose(model.jnt_range[0], [0, 1])

    def test_home_penetration_is_an_assembly_issue_even_for_a_robot(self):
        model, manifest = fixture(0, kind="robot_arm")
        result = audit_model(model, manifest, run_commands=False)
        self.assertIn("home_pose_interpenetration", result["issues"])

    def test_frozen_free_body_contact_requires_dynamic_interpretation(self):
        model, manifest = fixture(0.5, free_obstruction=True)
        result = audit_model(model, manifest, run_commands=False)
        self.assertEqual(result["status"], "requires_review")
        self.assertTrue(result["range_contacts"])
        self.assertTrue(
            all(
                contact["classification"]
                == "free_carrier_configuration_requires_dynamic_check"
                for contact in result["range_contacts"]
            )
        )
        self.assertNotIn("requires_geometry_or_motion_review", result["issues"])

    def test_target_rays_distinguish_occlusion_and_restore_model_groups(self):
        for obstruction, expected in (
            (None, "candidate_camera_view"),
            (0.5, "requires_camera_review"),
        ):
            model, manifest = fixture(obstruction)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            groups = model.geom_group.copy()
            result = camera_target_screening(model, data, manifest)
            self.assertEqual(result[0]["status"], expected)
            np.testing.assert_array_equal(model.geom_group, groups)

    def test_dynamic_failure_is_measured_and_reported(self):
        model, manifest = fixture(0.5)
        result = audit_model(model, manifest)
        self.assertEqual(len(result["commands"]), 3)
        self.assertTrue(result["tracking_failures"])
        self.assertIn("tracking_not_settled_within_duration", result["issues"])
        self.assertTrue(result["transient_contacts"])

    def test_free_carrier_loss_is_reported_after_actual_motion(self):
        model, manifest = fixture()
        # Use a separate physical setup: the stage can move out from under an
        # unsecured carrier, which then drops to the floor under gravity.
        root = ET.fromstring(
            """<mujoco><option gravity="0 0 -9.81"/>
          <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
            <body name="cart" pos="0 0 .2">
              <joint name="slide" type="slide" axis="1 0 0" range="0 1" damping="2"/>
              <geom name="cart__deck" type="box" size=".04 .04 .02" mass=".1"/>
              <site name="target" pos="0 0 .05"/>
            </body><body name="carrier" pos="0 0 .25"><freejoint/>
              <geom name="carrier_geom" type="box" size=".01 .01 .01" mass=".005"/>
            </body>
          </worldbody><actuator><position name="drive" joint="slide" kp="50" ctrlrange="0 1"/></actuator>
        </mujoco>"""
        )
        item = manifest["equipment"][0]
        item.update(
            sample_attachment="supported",
            process=dict(
                sample_site="target",
                sample_body="carrier",
                load_tolerance=0.04,
            ),
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        result = audit_model(model, manifest)
        self.assertIn("free_carrier_left_loading_region", result["issues"])
        lost = result["displaced_free_carriers"][0]
        self.assertEqual(lost["equipment"], "cart")
        self.assertGreater(lost["sample_distance_m"], lost["load_tolerance_m"])
        self.assertEqual(lost["while_commanding"]["control"], "slide")


if __name__ == "__main__":
    unittest.main()
