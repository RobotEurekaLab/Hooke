"""A shared server's restricted /proc must not weaken owned-worker cleanup."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.worker_client import IsaacWorker, WorkerCleanupError, WorkerTimeouts
from backends.worker_process import OwnedProcessSession, ProcessIdentity, WorkerProcessError


class WorkerProcessPermissions(unittest.TestCase):
    def setUp(self):
        self.leader = ProcessIdentity(987650, 1, 987650, 987650, 42, "S")
        self.process = Mock(pid=self.leader.pid)
        with patch("backends.worker_process.process_identity", return_value=self.leader):
            self.session = OwnedProcessSession(self.process)

    def worker(self):
        worker = IsaacWorker.__new__(IsaacWorker)
        worker.stream = worker.connection = worker.directory = worker.log = None
        worker.process, worker.process_session = self.process, self.session
        worker.closed = worker.broken = False
        worker.cleanup_report = None
        worker.gpu_lock = Mock()
        worker.timeouts = WorkerTimeouts(shutdown=.01, terminate=.01, kill=.01)
        return worker

    def denied_scan(self, uid, *, known=False, vanished=False):
        pid = 987651
        if known:
            self.session.observed_members.add(pid)
        entry = Mock(name="proc_entry")
        entry.name = str(pid)
        if vanished:
            entry.stat.side_effect = FileNotFoundError
        elif isinstance(uid, Exception):
            entry.stat.side_effect = uid
        else:
            entry.stat.return_value.st_uid = uid
        def identity(candidate):
            if candidate == pid:
                raise PermissionError("restricted /proc stat")
            return None
        return patch("backends.worker_process.Path.iterdir", return_value=iter([entry])), \
               patch("backends.worker_process.process_identity", side_effect=identity)

    def test_unrelated_different_uid_denial_does_not_hold_lease(self):
        worker = self.worker()
        lease = worker.gpu_lock
        entries, identities = self.denied_scan(os.geteuid() + 1)
        with entries, identities, patch("backends.worker_process.os.kill") as kill:
            worker.close()
        self.assertIsNone(worker.process)
        self.assertIsNone(worker.gpu_lock)
        self.assertTrue(worker.cleanup_report["exited"])
        lease.close.assert_called_once()
        kill.assert_not_called()

    def test_same_uid_denial_preserves_lease_and_process(self):
        worker = self.worker()
        lease = worker.gpu_lock
        entries, identities = self.denied_scan(os.geteuid())
        with entries, identities, patch("backends.worker_process.os.kill") as kill:
            with self.assertRaisesRegex(WorkerCleanupError, "Cannot inspect possible owned process"):
                worker.close()
        self.assertIs(worker.process, self.process)
        self.assertIs(worker.gpu_lock, lease)
        lease.close.assert_not_called()
        kill.assert_not_called()

    def test_known_owned_member_denial_preserves_lease_even_if_uid_changed(self):
        worker = self.worker()
        lease = worker.gpu_lock
        entries, identities = self.denied_scan(os.geteuid() + 1, known=True)
        with entries, identities, patch("backends.worker_process.os.kill") as kill:
            with self.assertRaisesRegex(WorkerCleanupError, "Cannot inspect possible owned process"):
                worker.close()
        self.assertIs(worker.gpu_lock, lease)
        lease.close.assert_not_called()
        kill.assert_not_called()

    def test_inaccessible_owner_metadata_cannot_be_assumed_unrelated(self):
        worker = self.worker()
        lease = worker.gpu_lock
        entries, identities = self.denied_scan(PermissionError("ownership inaccessible"))
        with entries, identities:
            with self.assertRaisesRegex(WorkerCleanupError, "Cannot verify ownership"):
                worker.close()
        lease.close.assert_not_called()

    def test_entry_disappearing_before_owner_query_is_safe(self):
        entries, identities = self.denied_scan(None, vanished=True)
        with entries, identities:
            self.assertEqual(self.session.live_members(), [])

    def test_owned_member_is_remembered_before_later_denied_scan(self):
        child = ProcessIdentity(987651, 987650, 987650, 987650, 43, "S")
        with patch("backends.worker_process.Path.iterdir", return_value=iter([Path("/proc/987651")])), \
             patch("backends.worker_process.process_identity", side_effect=[self.leader, child]):
            self.assertEqual(self.session.live_members(), [child])
        entries, identities = self.denied_scan(os.geteuid() + 1)
        with entries, identities:
            with self.assertRaisesRegex(WorkerProcessError, "Cannot inspect possible owned process"):
                self.session.live_members()

    def test_denied_leader_identity_remains_terminal(self):
        with patch("backends.worker_process.process_identity", side_effect=PermissionError("owned leader denied")):
            with self.assertRaises(PermissionError):
                self.session.live_members()


if __name__ == "__main__":
    unittest.main(verbosity=2)
