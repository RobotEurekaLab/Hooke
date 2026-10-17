"""Bounded cleanup of the private Linux session created for one worker."""

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import time


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent: int
    group: int
    session: int
    started: int
    state: str


def process_identity(pid):
    """Read kernel identity safely even when a process name contains spaces."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = text.rsplit(")", 1)[1].split()
    return ProcessIdentity(int(pid), int(fields[1]), int(fields[2]),
                           int(fields[3]), int(fields[19]), fields[0])


class WorkerProcessError(RuntimeError):
    """Owned native processes remain live or their ownership cannot be verified."""


class OwnedProcessSession:
    """Own only a Popen child launched with start_new_session=True.

    Session membership survives the launcher's exit and covers children that
    create another process group. Zombies have no executing code or GPU context;
    the direct child is additionally reaped through its Popen handle.
    """

    def __init__(self, process):
        self.process = process
        self.session = process.pid
        self.uid = os.geteuid()
        self.observed_members = {self.session}
        self.leader = process_identity(process.pid)
        if self.session in (os.getsid(0), os.getpgrp()):
            raise WorkerProcessError("Refusing ownership of the caller's process session")
        if self.leader is not None and self.leader.session != self.session:
            raise WorkerProcessError("Worker was not launched in a private session")
        self.report = {"session": self.session, "signals": [], "remaining": []}

    def live_members(self):
        leader = process_identity(self.session)
        if leader is not None and self.leader is not None and leader.started != self.leader.started:
            raise WorkerProcessError("Worker session leader PID was reused; refusing signals")
        members = []
        for path in Path("/proc").iterdir():
            if path.name.isdecimal():
                pid = int(path.name)
                try:
                    identity = process_identity(pid)
                except PermissionError as exc:
                    try:
                        uid = path.stat().st_uid
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    except PermissionError as metadata_error:
                        raise WorkerProcessError(f"Cannot verify ownership of inaccessible process {pid}") from metadata_error
                    # Shared /proc may expose other users' entries without
                    # permitting stat reads. Never infer that our own or a
                    # previously observed member exited from a denied read.
                    if uid != self.uid and pid not in self.observed_members:
                        continue
                    raise WorkerProcessError(f"Cannot inspect possible owned process {pid}") from exc
                if identity is not None and identity.session == self.session and identity.state not in ("Z", "X"):
                    self.observed_members.add(pid)
                    members.append(identity)
        return members

    def _wait(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            self.process.poll()  # Reap the wrapper without mistaking its exit for child exit.
            members = self.live_members()
            if not members:
                return []
            if time.monotonic() >= deadline:
                return members
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def _signal(self, members, signum):
        # Recheck identities before signaling; a group ID can be reused.
        for member in members:
            current = process_identity(member.pid)
            if current is None or current.started != member.started or current.session != self.session:
                continue
            try:
                os.kill(member.pid, signum)
            except ProcessLookupError:
                continue
            self.report["signals"].append({"pid": member.pid, "started": member.started,
                                            "signal": signum.name})

    def close(self, *, grace, terminate, kill):
        members = self._wait(grace)
        for signum, timeout in ((signal.SIGTERM, terminate), (signal.SIGKILL, kill)):
            if not members:
                break
            self._signal(members, signum)
            members = self._wait(timeout)
        self.report["remaining"] = [member.pid for member in members]
        self.report["exited"] = not members
        if members:
            raise WorkerProcessError(f"Owned worker processes did not exit: {self.report['remaining']}")
        return self.report
