"""Recording publication, backend isolation and HTTP cache/range contracts."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy import BACKENDS, OPERATIONS
from microscopy.recording_catalog import FORMATS, publish_recordings, recording_directory
from webui.microscopy_api import bp


class RecordingContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.media = Path(self.temporary.name)
        self.release = self.media/"recordings"/"new-version"
        app = Flask(__name__)
        app.register_blueprint(bp)
        self.client = app.test_client()
        self.patch = patch("webui.microscopy_api.MEDIA", self.media)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def make_release(self):
        # These byte fixtures test publication/routing, not video decoding.
        for backend in BACKENDS:
            for operation in OPERATIONS:
                for extension in FORMATS:
                    path = self.release/backend/operation/f"demo.{extension}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"new {backend} {operation} {extension}".encode())
                    path.with_name(path.name+".json").write_text(json.dumps(dict(
                        backend=backend, operation=operation,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())))

    def legacy_recording(self):
        path = self.media/"push"/"demo.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"historical source recording")
        return path

    def test_legacy_source_does_not_supply_native_recordings(self):
        path = self.legacy_recording()
        self.assertEqual(recording_directory(self.media, "mujoco"), (self.media, "legacy"))
        with self.client.get("/api/microscopy/recording/push") as response:
            self.assertEqual(response.data, path.read_bytes())
        self.assertEqual(self.client.get("/api/microscopy/recording/push?backend=isaac").status_code, 404)
        response = self.client.get("/api/microscopy/recordings?backend=isaac")
        self.assertEqual(response.get_json()["operations"]["push"], [])
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_complete_release_selects_correct_backend_and_preserves_history(self):
        historical = self.legacy_recording()
        old = historical.read_bytes()
        self.make_release()
        manifest = publish_recordings(self.media, self.release)
        self.assertEqual(len(manifest["video_sha256"]), 20)
        for backend in BACKENDS:
            self.assertEqual(recording_directory(self.media, backend),
                             (self.release/backend, "new-version"))
            catalog = self.client.get(f"/api/microscopy/recordings?backend={backend}").get_json()
            self.assertEqual(catalog["revision"], "new-version")
            self.assertTrue(all(formats == list(FORMATS) for formats in catalog["operations"].values()))
            with self.client.get(f"/api/microscopy/recording/push?backend={backend}") as response:
                self.assertEqual(response.data, f"new {backend} push mp4".encode())
                self.assertIn("max-age=0", response.headers["Cache-Control"])
        self.assertEqual(historical.read_bytes(), old)

    def test_published_video_supports_browser_range_and_conditional_requests(self):
        self.make_release()
        publish_recordings(self.media, self.release)
        url = "/api/microscopy/recording/cell_injection?backend=isaac&format=webm"
        data = (self.release/"isaac"/"cell_injection"/"demo.webm").read_bytes()
        with self.client.get(url, headers={"Range": "bytes=2-8"}) as response:
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.data, data[2:9])
            self.assertEqual(response.headers["Content-Range"], f"bytes 2-8/{len(data)}")
            etag = response.headers["ETag"]
        with self.client.get(url, headers={"If-None-Match": etag}) as response:
            self.assertEqual(response.status_code, 304)

    def test_incomplete_release_leaves_current_manifest_unchanged(self):
        self.make_release()
        publish_recordings(self.media, self.release)
        index = self.media/"recordings"/"current.json"
        previous = index.read_bytes()
        (self.release/"isaac"/"suction_injection"/"demo.webm").unlink()
        with self.assertRaises(OSError):
            publish_recordings(self.media, self.release)
        self.assertEqual(index.read_bytes(), previous)

    def test_changed_encoded_file_cannot_be_published(self):
        self.make_release()
        (self.release/"isaac"/"cell_injection"/"demo.mp4").write_bytes(b"wrong file")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            publish_recordings(self.media, self.release)
        self.assertFalse((self.media/"recordings"/"current.json").exists())

    def test_missing_published_file_does_not_fall_back_to_old_recording(self):
        self.legacy_recording()
        self.make_release()
        publish_recordings(self.media, self.release)
        (self.release/"mujoco"/"push"/"demo.mp4").unlink()
        self.assertEqual(self.client.get("/api/microscopy/recording/push").status_code, 404)

    def test_malformed_or_escaping_manifest_is_rejected(self):
        index = self.media/"recordings"/"current.json"
        index.parent.mkdir()
        for value in ("not JSON", [], dict(schema_version=1, directory="../outside", revision="x"),
                      dict(schema_version=1, directory="/tmp", revision="x")):
            with self.subTest(value=value):
                index.write_text(value if isinstance(value, str) else json.dumps(value))
                self.assertEqual(self.client.get("/api/microscopy/recordings").status_code, 503)
                self.assertEqual(self.client.get("/api/microscopy/recording/push").status_code, 503)

    def test_catalog_rejects_unknown_backend(self):
        self.assertEqual(self.client.get("/api/microscopy/recordings?backend=other").status_code, 400)


if __name__ == "__main__":
    unittest.main()
