"""Mounted-view registration retains physical attachment and rejects false metadata."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.cameras import mounted_camera_records


def fixture():
    root = ET.fromstring(
        """<mujoco><worldbody>
      <camera name="overview" pos="2 -2 2"/>
      <body name="rig__carriage" pos="0 0 .5">
        <joint name="travel" type="slide" axis="1 0 0" range="-.5 .5"/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
        <site name="rig__datum" pos="0 0 .12"/>
        <camera name="rig__sensor" pos="0 0 .12" quat="1 0 0 0" fovy="45"/>
      </body></worldbody></mujoco>"""
    )
    metadata = [
        dict(
            id="rig",
            mounted_cameras=[
                dict(
                    name="rig__sensor",
                    body="rig__carriage",
                    site="rig__datum",
                    local_position_m=[0, 0, 0.12],
                    local_quaternion_wxyz=[1, 0, 0, 0],
                    fovy_deg=45,
                    purpose="dry target inspection",
                )
            ],
        )
    ]
    return root, metadata


class MountedCameraTests(unittest.TestCase):
    def test_registered_camera_moves_with_carriage_and_preserves_fixed_view(self):
        root, metadata = fixture()
        before, declarations = ET.tostring(root), deepcopy(metadata)
        records = mounted_camera_records(root.find("worldbody"), metadata, ["overview"])
        self.assertEqual([r["name"] for r in records], ["rig__sensor"])
        self.assertEqual(ET.tostring(root), before)
        self.assertEqual(metadata, declarations)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        fixed, mounted = model.camera("overview").id, model.camera("rig__sensor").id
        initial = data.cam_xpos.copy()
        data.qpos[0] = 0.3
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.cam_xpos[mounted] - initial[mounted], [0.3, 0, 0]
        )
        np.testing.assert_array_equal(data.cam_xpos[fixed], initial[fixed])
        self.assertEqual(model.cam_bodyid[mounted], model.body("rig__carriage").id)
        self.assertEqual(model.ncam, 2)

    def test_wrong_mount_datum_pose_or_projection_is_rejected_without_mutation(self):
        changes = [
            dict(body="other__carriage"),
            dict(site="missing"),
            dict(local_position_m=[0, 0, 0.13]),
            dict(fovy_deg=30),
            dict(local_quaternion_wxyz=[2, 0, 0, 0]),
            dict(local_position_m=[float("nan"), 0, 0]),
        ]
        for change in changes:
            root, metadata = fixture()
            metadata[0]["mounted_cameras"][0].update(change)
            before = ET.tostring(root)
            with self.subTest(change=change), self.assertRaises(ValueError):
                mounted_camera_records(root.find("worldbody"), metadata, ["overview"])
            self.assertEqual(ET.tostring(root), before)

    def test_names_cannot_shadow_room_views_or_another_declared_sensor(self):
        root, metadata = fixture()
        with self.assertRaises(ValueError):
            mounted_camera_records(root.find("worldbody"), metadata, ["rig__sensor"])
        metadata[0]["mounted_cameras"] *= 2
        with self.assertRaises(ValueError):
            mounted_camera_records(root.find("worldbody"), metadata, ["overview"])

    def test_camera_reparenting_or_tracking_mode_cannot_fake_a_physical_mount(self):
        for alteration in ("reparent", "track", "wrong_site"):
            root, metadata = fixture()
            world = root.find("worldbody")
            body = world.find("body")
            camera = body.find("camera")
            if alteration == "reparent":
                body.remove(camera)
                world.append(camera)
            elif alteration == "track":
                camera.set("mode", "track")
            else:
                site = body.find("site")
                body.remove(site)
                world.append(site)
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                mounted_camera_records(world, metadata, ["overview"])

    def test_scenes_without_declared_mounts_keep_their_existing_view_contract(self):
        root, _ = fixture()
        self.assertEqual(
            mounted_camera_records(root.find("worldbody"), [], ["overview"]), []
        )

    def test_invalid_xml_projection_cannot_pass_a_valid_metadata_check(self):
        for change in (
            dict(fovy="nan"),
            dict(fovy="inf"),
            dict(orthographic="true"),
            dict(focal=".02 .02", sensorsize=".03 .03"),
        ):
            root, metadata = fixture()
            root.find(".//body/camera").attrib.update(change)
            before = ET.tostring(root)
            with self.subTest(change=change), self.assertRaises(ValueError):
                mounted_camera_records(root.find("worldbody"), metadata, ["overview"])
            self.assertEqual(ET.tostring(root), before)


if __name__ == "__main__":
    unittest.main()
