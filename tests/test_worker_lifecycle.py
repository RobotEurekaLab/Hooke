"""Native-worker lifecycle contracts using CPU sockets and owned child processes."""

import io
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.ipc import read_message, write_message
from backends.worker_client import (
    IsaacWorker, WorkerCleanupError, WorkerRemoteError, WorkerTimeouts,
    WorkerTransportError,
)
from backends.worker_process import OwnedProcessSession, process_identity


FAST = WorkerTimeouts(startup=.25, load=.5, step=.03, request=.1,
                      shutdown=.025, terminate=.1, kill=1)


def bare_worker(connection=None, transport="binary", cls=IsaacWorker):
    worker = cls.__new__(cls)
    worker.stream = connection.makefile("rwb") if connection else None
    worker.connection = connection
    worker.process = worker.process_session = None
    worker.directory = worker.gpu_lock = worker.log = None
    worker.closed = worker.broken = False
    worker.failure = worker.cleanup_report = None
    worker.timeouts = FAST
    worker.transport = transport
    return worker


def encoded(value, transport="binary"):
    buffer = io.BytesIO()
    write_message(buffer, value, transport)
    return buffer.getvalue()


class WorkerTransportLifecycle(unittest.TestCase):
    def pair(self, transport="binary", cls=IsaacWorker):
        client, peer = socket.socketpair()
        peer.settimeout(1)
        worker = bare_worker(client, transport, cls)
        self.addCleanup(peer.close)

        def cleanup():
            worker.broken = True
            worker.close()
        self.addCleanup(cleanup)
        return worker, peer

    def test_timeout_is_terminal_and_later_calls_write_nothing(self):
        for transport in ("binary", "json"):
            with self.subTest(transport=transport):
                worker, peer = self.pair(transport)
                with self.assertRaises(WorkerTransportError) as failure:
                    worker.call("step")
                self.assertIsInstance(failure.exception.__cause__, TimeoutError)
                first = read_message(io.BytesIO(peer.recv(65536)), transport)
                self.assertEqual(first, {"op": "step"})
                self.assertTrue(worker.closed and worker.broken)
                self.assertIsNone(worker.stream)
                with self.assertRaises(WorkerTransportError):
                    worker.call("info")
                self.assertEqual(peer.recv(65536), b"")
                with self.assertRaises(BrokenPipeError):
                    peer.sendall(encoded({"ok": True, "result": "late"}, transport))

    def test_eof_and_invalid_response_are_terminal(self):
        cases = [b"", struct.pack("!I", 3) + b"xx",
                 struct.pack("!I", 2) + b"xx",
                 encoded({"result": 1}), encoded({"ok": True})]
        for reply in cases:
            with self.subTest(reply=reply):
                worker, peer = self.pair()
                peer.sendall(reply)
                peer.shutdown(socket.SHUT_WR)
                with self.assertRaises(WorkerTransportError):
                    worker.call("info")
                self.assertTrue(worker.closed and worker.broken)

    def test_remote_scene_error_keeps_transport_usable(self):
        worker, peer = self.pair()
        peer.sendall(encoded({"ok": False, "error": "unsupported model"}) +
                     encoded({"ok": True, "result": {"loaded": True}}))
        with self.assertRaisesRegex(WorkerRemoteError, "unsupported model"):
            worker.call("load")
        self.assertFalse(worker.closed or worker.broken)
        self.assertEqual(worker.call("load"), {"loaded": True})

    def test_invalid_local_json_does_not_poison_connection(self):
        worker, peer = self.pair("json")
        with self.assertRaises(ValueError):
            worker.call("step", control=[float("nan")])
        self.assertFalse(worker.broken)
        peer.sendall(encoded({"ok": True, "result": 1}, "json"))
        self.assertEqual(worker.call("info"), 1)
        self.assertEqual(read_message(io.BytesIO(peer.recv(65536)), "json"), {"op": "info"})

    def test_load_and_step_use_independent_budgets(self):
        worker, peer = self.pair()
        errors = []

        def server():
            try:
                with peer.makefile("rwb") as stream:
                    self.assertEqual(read_message(stream)["op"], "load")
                    time.sleep(.08)  # Longer than the step budget, shorter than load.
                    write_message(stream, {"ok": True, "result": "loaded"})
                    self.assertEqual(read_message(stream)["op"], "step")
                    self.assertIsNone(read_message(stream))
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=server)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.assertEqual(worker.call("load"), "loaded")
        began = time.monotonic()
        with self.assertRaisesRegex(WorkerTransportError, "budget 0.03s"):
            worker.call("step")
        self.assertLess(time.monotonic() - began, .5)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_close_does_not_dispatch_subclass_call_or_extend_shutdown_budget(self):
        class Override(IsaacWorker):
            def call(self, *args, **kwargs):
                raise AssertionError("close must bypass a subclass call override")
        worker, peer = self.pair(cls=Override)
        began = time.monotonic()
        worker.close()
        self.assertLess(time.monotonic() - began, .5)
        self.assertEqual(read_message(io.BytesIO(peer.recv(65536))), {"op": "shutdown"})
        worker.close()

    def test_cancellation_cleans_transport_and_preserves_original_exception(self):
        worker, peer = self.pair()
        with patch.object(worker, "_exchange", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                worker.call("step")
        self.assertTrue(worker.closed and worker.broken)
        self.assertEqual(peer.recv(1), b"")

    def test_timeout_configuration_rejects_invalid_values(self):
        for value in (0, -1, float("inf"), float("nan"), True, "5", 3601):
            with self.subTest(value=value), self.assertRaises(ValueError):
                WorkerTimeouts(load=value)
        self.assertEqual(WorkerTimeouts().load, 600)
        self.assertEqual(WorkerTimeouts().shutdown, 5)


TREE = r'''
import ctypes, json, os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1]); role = sys.argv[2]
if role != "wrapper":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ctypes.CDLL(None).prctl(15, b"hooke test kid", 0, 0, 0)
if role == "grandchild":
    os.setpgid(0, 0)
else:
    child_role = "child" if role == "wrapper" else "grandchild"
    subprocess.Popen([sys.executable, __file__, str(root), child_role])
(root / (role + ".json")).write_text(json.dumps({"pid": os.getpid()}))
if role == "wrapper" and (root / "exit_wrapper").exists():
    while not (root / "grandchild.json").exists(): time.sleep(.01)
    sys.exit(0)
while True: time.sleep(10)
'''


class WorkerProcessLifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = self.root / "fixture.py"
        self.script.write_text(TREE)

    def launch(self, *, exit_wrapper=False):
        if exit_wrapper:
            (self.root / "exit_wrapper").touch()
        process = subprocess.Popen([sys.executable, str(self.script), str(self.root), "wrapper"],
                                   start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        session = OwnedProcessSession(process)
        self.addCleanup(session.close, grace=0, terminate=.1, kill=1)
        deadline = time.monotonic() + 3
        while not (self.root / "grandchild.json").exists():
            if time.monotonic() >= deadline:
                self.fail("CPU process fixture did not become ready")
            time.sleep(.01)
        if exit_wrapper:
            process.wait(timeout=1)
        return process, session

    def test_wrapper_exit_does_not_leave_children_or_release_lease_early(self):
        process, session = self.launch(exit_wrapper=True)
        sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
        def stop_sentinel():
            sentinel.terminate()
            sentinel.wait(timeout=2)
        self.addCleanup(stop_sentinel)
        child = json.loads((self.root / "child.json").read_text())["pid"]
        grandchild = json.loads((self.root / "grandchild.json").read_text())["pid"]
        self.assertEqual(process_identity(child).session, process.pid)
        self.assertNotEqual(process_identity(grandchild).group, process.pid)
        worker = bare_worker()
        worker.process, worker.process_session = process, session
        released = []
        worker.gpu_lock = Mock()
        worker.gpu_lock.close.side_effect = lambda: released.append(session.live_members())
        worker.close()
        self.assertEqual(released, [[]])
        self.assertIsNone(worker.process)
        self.assertTrue(worker.cleanup_report["exited"])
        killed = {item["pid"] for item in worker.cleanup_report["signals"] if item["signal"] == "SIGKILL"}
        self.assertTrue({child, grandchild} <= killed)
        self.assertIsNone(sentinel.poll())
        worker.close()

    def test_failed_process_cleanup_retains_lease_and_allows_later_close(self):
        worker = bare_worker()
        worker.process = Mock()
        worker.process_session = Mock()
        worker.process_session.close.side_effect = RuntimeError("survivor")
        lease, log = Mock(), Mock()
        worker.gpu_lock, worker.log = lease, log
        with self.assertRaisesRegex(WorkerCleanupError, "survivor"):
            worker.close()
        lease.close.assert_not_called()
        log.close.assert_called_once()
        self.assertIsNotNone(worker.process)
        worker.process_session.close.side_effect = None
        worker.process_session.close.return_value = {"exited": True}
        worker.close()
        lease.close.assert_called_once()

    def test_stream_close_error_does_not_skip_process_or_lease_cleanup(self):
        worker = bare_worker()
        worker.broken = True
        worker.stream = Mock()
        worker.stream.close.side_effect = OSError("close failed")
        worker.process = Mock()
        worker.process_session = Mock()
        lease = worker.gpu_lock = Mock()
        with self.assertRaisesRegex(WorkerCleanupError, "close failed"):
            worker.close()
        self.assertIsNone(worker.process)
        lease.close.assert_called_once()

    def constructor_failure(self, cancel):
        captured, released = [], []
        original_popen = subprocess.Popen
        original_monotonic = time.monotonic

        def spawn(args, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            process = original_popen([sys.executable, str(self.script), str(self.root), "wrapper"], **kwargs)
            captured.append(process)
            return process

        ticks = 0
        def clock():
            nonlocal ticks
            ticks += 1
            if cancel and ticks == 2:
                raise KeyboardInterrupt
            return original_monotonic()

        lease = Mock()
        lease.close.side_effect = lambda: released.append(
            [identity.pid for identity in OwnedProcessSession(captured[0]).live_members()] if captured else [])
        with patch("backends.worker_client.inspect_installation", return_value={"installation_ready": True, "isaac_path": str(self.root)}), \
             patch("backends.worker_client.GPULease", return_value=lease), \
             patch("backends.worker_client.subprocess.check_output", side_effect=["0", "fake-uuid"]), \
             patch("backends.worker_client.subprocess.Popen", side_effect=spawn), \
             patch("backends.worker_client.time.monotonic", side_effect=clock):
            with self.assertRaises(KeyboardInterrupt if cancel else TimeoutError):
                IsaacWorker(self.root / "output", 0, timeouts=FAST)
        self.assertEqual(released, [[]])
        self.assertIsNotNone(captured[0].poll())

    def test_startup_timeout_cleans_owned_process_tree_before_lease_release(self):
        self.constructor_failure(cancel=False)

    def test_constructor_cancellation_cleans_spawned_process(self):
        self.constructor_failure(cancel=True)

    def test_pid_reuse_and_disappearance_never_signal_an_unrelated_process(self):
        process, session = self.launch()
        member = session.live_members()[0]
        changed = type(member)(member.pid, member.parent, member.group,
                               member.session, member.started + 1, member.state)
        with patch("backends.worker_process.process_identity", return_value=changed), \
             patch("backends.worker_process.os.kill") as kill:
            session._signal([member], signal.SIGTERM)
        kill.assert_not_called()
        with patch("backends.worker_process.process_identity", return_value=None), \
             patch("backends.worker_process.os.kill") as kill:
            session._signal([member], signal.SIGTERM)
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
