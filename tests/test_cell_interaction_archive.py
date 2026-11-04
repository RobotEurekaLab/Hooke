"""Interaction archives preserve observable data and survive failed writes."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.interaction_trace import archive_interaction_trace
from microscopy.session import ExperimentSession
from microscopy.tasks import make_task


class InteractionArchiveTests(unittest.TestCase):
    def source(self, columns):
        return SimpleNamespace(interaction_trace=lambda: columns,
            interaction_trace_metadata=lambda: dict(schema_version=1, rows=2, dropped_rows=0))

    def test_session_archives_actual_steps_without_rendering_or_large_json(self):
        task = make_task("suction_injection")
        task.reset(0)
        with TemporaryDirectory() as folder:
            output = Path(folder)
            session = ExperimentSession(output, "mujoco", 0, 4)
            session.task = task
            session.episode = output / "episode"
            session.episode.mkdir()
            with patch.object(session, "capture"):
                session.command(dict(command="step", seconds=.003))
            result = json.loads((session.episode / "result.json").read_text())
            metadata = result["interaction_trace"]
            path = session.episode / metadata["file"]
            self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with np.load(path, allow_pickle=False) as trace:
                self.assertEqual(len(trace["time_s"]), metadata["rows"])
                np.testing.assert_array_equal(trace["tip_world_m"][-1], task.data.site("injector_tcp").xpos)
                np.testing.assert_array_equal(trace["cell_world_m"][-1], task.mechanics.position_m)
                self.assertTrue(trace["holding_active"].all())
            self.assertNotIn("tip_world_m", metadata)
            self.assertFalse((session.episode / "interaction.tmp.npz").exists())
            session.close()

    def test_invalid_trace_preserves_existing_archive(self):
        columns = dict(time_s=np.array([.001, .002]), force_n=np.array([0., 1e-9]))
        with TemporaryDirectory() as folder:
            output = Path(folder)
            archive_interaction_trace(self.source(columns), output)
            original = (output / "interaction.npz").read_bytes()
            for invalid in (
                    dict(columns, time_s=np.array([.002, .001])),
                    dict(columns, force_n=np.array([np.nan, 0.])),
                    dict(columns, force_n=np.array([0.])),
                    dict(columns, force_n=np.array([object(), object()], dtype=object))):
                with self.subTest(columns=invalid), self.assertRaises(ValueError):
                    archive_interaction_trace(self.source(invalid), output)
                self.assertEqual((output / "interaction.npz").read_bytes(), original)

    def test_failed_write_keeps_previous_archive_and_removes_partial_file(self):
        columns = dict(time_s=np.array([.001, .002]))
        with TemporaryDirectory() as folder:
            output = Path(folder)
            archive_interaction_trace(self.source(columns), output)
            original = (output / "interaction.npz").read_bytes()

            def fail(path, **values):
                Path(path).write_bytes(b"partial archive")
                raise OSError("simulated write failure")

            with patch("microscopy.interaction_trace.np.savez_compressed", side_effect=fail):
                with self.assertRaises(OSError):
                    archive_interaction_trace(self.source(columns), output)
            self.assertEqual((output / "interaction.npz").read_bytes(), original)
            self.assertFalse((output / "interaction.tmp.npz").exists())

    def test_empty_and_unsupported_mechanics_create_no_archive(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            self.assertIsNone(archive_interaction_trace(object(), output))
            source = self.source(dict(time_s=np.array([])))
            source.interaction_trace_metadata = lambda: dict(schema_version=1, rows=0)
            self.assertEqual(archive_interaction_trace(source, output)["rows"], 0)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
