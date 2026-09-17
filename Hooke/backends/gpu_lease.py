"""One local GPU lease shared by native simulation and source rendering."""

import fcntl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class GPUBusy(RuntimeError):
    """A requested GPU belongs to another running job."""


class GPULease:
    def __init__(self, gpu):
        if gpu < 0:
            raise ValueError("GPU index must be nonnegative")
        folder = ROOT / "temp/backend_parity"
        folder.mkdir(parents=True, exist_ok=True)
        self.file = (folder / f"gpu-{gpu}.lock").open("a")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            raise GPUBusy(
                f"GPU {gpu} already has a Hooke job running; wait or stop that job"
            ) from None
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
