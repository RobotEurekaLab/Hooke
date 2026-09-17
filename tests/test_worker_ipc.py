"""Transport must retain measured physics and reject incomplete frames."""

import io
from pathlib import Path
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.ipc import MAX_MESSAGE, read_message, write_message


class WorkerIPC(unittest.TestCase):
    def message(self):
        return {
            "ok": True,
            "result": {
                "qpos": [-0.0, 1e-300, 1.2345678901234567],
                "body_poses": {2: [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]},
                "contacts": [{"geom1": 1, "geom2": 2, "force": [0.0, 0.0, 9.81]}],
            },
        }

    def test_primitive_roundtrip_preserves_float_bits_and_integer_keys(self):
        stream = io.BytesIO()
        write_message(stream, self.message())
        stream.seek(0)
        actual = read_message(stream)
        self.assertEqual(actual, self.message())
        for actual_value, expected in zip(
            actual["result"]["qpos"], self.message()["result"]["qpos"]
        ):
            self.assertEqual(struct.pack("d", actual_value), struct.pack("d", expected))
        self.assertIn(2, actual["result"]["body_poses"])
        self.assertIsNone(read_message(stream))

    def test_truncated_or_oversized_frames_are_rejected(self):
        for raw, error in [
            (b"\0", EOFError),
            (struct.pack("!I", 3) + b"xx", EOFError),
            (struct.pack("!I", 0), ValueError),
            (struct.pack("!I", MAX_MESSAGE + 1), ValueError),
        ]:
            with self.subTest(raw=raw), self.assertRaises(error):
                read_message(io.BytesIO(raw))
        with self.assertRaises(ValueError):
            read_message(io.BytesIO(b"{}"), "json")

    def test_pinned_native_python_can_decode_and_return_same_frame(self):
        # This starts only Python, never SimulationApp or a GPU process.
        native = Path("/home/amax/data/datacopy/isaacsim/kit/python/bin/python3")
        if not native.is_file():
            self.skipTest("Pinned Isaac Python is not installed on this machine")
        stream = io.BytesIO()
        write_message(stream, self.message())
        code = (
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "from backends.ipc import read_message,write_message;"
            "write_message(sys.stdout.buffer,read_message(sys.stdin.buffer))"
        )
        returned = subprocess.check_output(
            [
                str(native),
                "-c",
                code,
                str(Path(__file__).resolve().parents[1] / "Hooke"),
            ],
            input=stream.getvalue(),
        )
        decoded = read_message(io.BytesIO(returned))
        self.assertEqual(decoded, self.message())
        for actual, expected in zip(
            decoded["result"]["qpos"], self.message()["result"]["qpos"]
        ):
            self.assertEqual(struct.pack("d", actual), struct.pack("d", expected))


if __name__ == "__main__":
    unittest.main()
