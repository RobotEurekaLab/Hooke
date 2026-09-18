"""Backend isolation, reset exports and live observation provenance contracts."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.session import ExperimentSession
from webui.microscopy_api import bp


class NativeTransport:
    """Small RPC fixture; it does not claim to simulate PhysX."""
    def __init__(self):
        self.loads = []
        self.closed = False

    def call(self, op, **kwargs):
        if op == "load":
            self.loads.append(kwargs)
            self.arrays = dict(np.load(Path(kwargs["source"]) / "model.npz"))
            self.state = dict(qpos=self.arrays["reset_qpos"].tolist(),
                              qvel=self.arrays["reset_qvel"].tolist(), time=0., contacts=[])
            self.frames = 0
            self.output = Path(kwargs["output"])
            metadata = json.loads((Path(kwargs["source"]) / "scene.json").read_text())
            self.mapping = metadata["task_info"]["camera_mapping"]
            self.camera_names = metadata["names"]["camera"]
            return dict(state=self.state)
        if op == "render":
            for name in dict.fromkeys(self.mapping.values()):
                i = self.camera_names.index(name)
                folder = self.output / f"camera_{i}"
                folder.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (20+i*50, 40, 60)).save(folder / f"{self.frames:05d}.png")
            self.frames += 1
            return dict(frames=self.frames)
        if op == "info":
            return dict(steps=0, frames=self.frames, physics_events_since_load=0)
        raise AssertionError(f"Unexpected native RPC: {op}")

    def close(self):
        self.closed = True


class NativeSessionContracts(unittest.TestCase):
    def test_live_views_use_native_model_camera_ids(self):
        native = NativeTransport()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "HOOKE_MICROSCOPY_ASSETS": "reference", "HOOKE_MICROSCOPY_STAND": "te2000-s-reference",
            "HOOKE_MICROSCOPY_STAGE": "reference", "HOOKE_MICROSCOPY_OPTICS": "estimated",
        }), patch("backends.worker_client.IsaacWorker", return_value=native):
            session = ExperimentSession(directory, "isaac", 0, 1)
            try:
                session.command(dict(command="reset", operation="cell_injection", seed=0))
                self.assertEqual(native.camera_names, ["objective_detail", "workstation_overview",
                                                      "instrument_closeup", "detection_path_detail", "cell_detail"])
                for view, color in {"objective": 20, "overview": 70, "closeup": 120, "detection": 170, "sample": 220}.items():
                    with Image.open(Path(directory) / (view+".png")) as image:
                        self.assertEqual(image.getpixel((0, 0)), (color, 40, 60))
            finally:
                session.close()

    def test_objective_view_does_not_reuse_an_image_from_another_scene(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        with tempfile.TemporaryDirectory() as directory, patch("webui.microscopy_api.LIVE", Path(directory)):
            folder = Path(directory)
            Image.new("RGB", (8, 8)).save(folder/"objective.png")
            (folder/"state.json").write_text(json.dumps(dict(phase="ready", volume_unit="nL")))
            self.assertEqual(app.test_client().get("/api/microscopy/image/objective").status_code, 404)
            (folder/"state.json").write_text(json.dumps(dict(phase="ready", volume_unit="pL",
                calibration=dict(reference_objective=dict(profile="mrh08430-drawing-reference")))))
            self.assertEqual(app.test_client().get("/api/microscopy/image/objective").status_code, 200)
            Image.new("RGB", (8, 8)).save(folder/"detection.png")
            self.assertEqual(app.test_client().get("/api/microscopy/image/detection").status_code, 404)
            (folder/"state.json").write_text(json.dumps(dict(phase="ready", volume_unit="pL",
                calibration=dict(reference_detection_path=dict(profile="original-cfi-collection-reference")))))
            self.assertEqual(app.test_client().get("/api/microscopy/image/detection").status_code, 200)

    def test_reset_reexports_static_cell_geometry_and_never_uses_source_rgb(self):
        native = NativeTransport()
        original_step = mujoco.mj_step
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "HOOKE_MICROSCOPY_ASSETS": "reference", "HOOKE_MICROSCOPY_STAND": "auto",
            "HOOKE_MICROSCOPY_OPTICS": "estimated",
        }), patch("backends.worker_client.IsaacWorker", return_value=native) as worker, patch(
            "backends.source_renderer.mujoco_renderer", side_effect=AssertionError("Source RGB in native session")
        ):
            session = ExperimentSession(directory, "isaac", 0, 1)
            try:
                first = session.command(dict(command="reset", operation="cell_injection", seed=0))
                first_position = session.task.model.geom("cell_0_shell").pos.copy()
                second = session.command(dict(command="reset", operation="cell_injection", seed=9))
                current = session.task.model.geom("cell_0_shell")
                self.assertGreater(np.linalg.norm(first_position-current.pos), 1e-7)
                np.testing.assert_allclose(native.arrays["geom_pos"][current.id], current.pos)
                self.assertEqual(len(native.loads), 2)
                worker.assert_called_once()
                self.assertEqual(second["state"]["generation"], first["state"]["generation"]+1)
                self.assertEqual(second["state"]["physics_engine"], "PhysX")
                self.assertEqual(second["state"]["microscope_image_source"], "synthetic_object_space")
                with Image.open(Path(directory) / "overview.png") as image:
                    self.assertEqual(image.getpixel((0, 0)), (20, 40, 60))
                with Image.open(Path(directory) / "sample.png") as image:
                    self.assertEqual(image.getpixel((0, 0)), (120, 40, 60))
                qpos = session.task.data.qpos.copy()
                for request in (dict(command="reset", seed=-1), dict(command="step", pressure_pa=5000,
                                actuators={"injector_x": 1.})):
                    with self.assertRaises(ValueError):
                        session.command(request)
                self.assertEqual(len(native.loads), 2)
                np.testing.assert_array_equal(session.task.data.qpos, qpos)
                self.assertEqual(session.task.data.userdata[0], 0.)
            finally:
                session.close()
        self.assertIs(mujoco.mj_step, original_step)
        self.assertTrue(native.closed)


class BackendRouteContracts(unittest.TestCase):
    def test_fluorescence_channel_requires_current_cell_state_and_its_own_frame(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        with tempfile.TemporaryDirectory() as directory, patch("webui.microscopy_api.LIVE", Path(directory)):
            folder = Path(directory)
            Image.new("RGB", (8, 8), "red").save(folder/"microscope.png")
            Image.new("RGB", (8, 8), "blue").save(folder/"microscope_fluorescence.png")
            state_path = folder/"state.json"
            state_path.write_text(json.dumps(dict(phase="ready", volume_unit="nL")))
            client = app.test_client()
            url = "/api/microscopy/image/microscope"
            self.assertEqual(client.get(url+"?channel=unknown").status_code, 400)
            self.assertEqual(client.get(url+"?channel=fluorescence").status_code, 404)
            state_path.write_text(json.dumps(dict(phase="ready", volume_unit="pL", calibration={
                "display_channels": ["phase_contrast", "fluorescence"]})))
            with client.get(url) as phase, client.get(url+"?channel=fluorescence") as fluorescence:
                self.assertEqual(phase.status_code, 200)
                self.assertEqual(fluorescence.status_code, 200)
                self.assertNotEqual(phase.data, fluorescence.data)
            (folder/"microscope_fluorescence.png").unlink()
            self.assertEqual(client.get(url+"?channel=fluorescence").status_code, 404)

    def test_loading_scene_does_not_serve_previous_joint_pose_image(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        with tempfile.TemporaryDirectory() as directory, patch("webui.microscopy_api.LIVE", Path(directory)):
            Image.new("RGB", (32, 32)).save(Path(directory) / "overview.png")
            (Path(directory) / "state.json").write_text(json.dumps(dict(phase="initializing")))
            self.assertEqual(app.test_client().get("/api/microscopy/image/overview").status_code, 404)

    def test_native_requests_cannot_return_source_state_images_or_video(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory, patch("webui.microscopy_api.MEDIA", Path(directory)), patch(
            "webui.microscopy_api.LIVE", Path(directory) / "live"
        ):
            source = Path(directory) / "live"
            source.mkdir()
            (source / "state.json").write_text(json.dumps(dict(phase="ready", backend="mujoco")))
            Image.new("RGB", (32, 32)).save(source / "overview.png")
            self.assertEqual(client.get("/api/microscopy/state").get_json()["backend"], "mujoco")
            self.assertEqual(client.get("/api/microscopy/state?backend=isaac").get_json(), dict(phase="offline"))
            self.assertEqual(client.get("/api/microscopy/image/overview?backend=isaac").status_code, 404)
            self.assertEqual(client.get("/api/microscopy/recording/push?backend=isaac").status_code, 404)
            self.assertEqual(client.get("/api/microscopy/recording/push?format=gif").status_code, 400)
            with patch("webui.microscopy_api.workstation.command") as command:
                self.assertEqual(client.post("/api/microscopy/control", json=dict(command="reset", backend="invalid")).status_code, 400)
                command.assert_not_called()
            self.assertEqual(client.get("/api/microscopy/state?backend=invalid").status_code, 400)


if __name__ == "__main__":
    unittest.main()
