"""Single-owner simulation and EGL context controlled through local JSON IPC."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
import signal
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

from microscopy import OPERATIONS
from microscopy.control import apply_request, experiment_request
from microscopy.session import BACKENDS, ExperimentSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--operation", choices=OPERATIONS, default="push")
    parser.add_argument("--backend", choices=BACKENDS, default="mujoco")
    args = parser.parse_args()
    live_fps = float(os.environ.get("HOOKE_MICROSCOPY_LIVE_FPS", "4"))
    if not math.isfinite(live_fps) or not 1 <= live_fps <= 10:
        parser.error("Live capture FPS must be from 1 to 10")
    args.output.mkdir(parents=True, exist_ok=True)
    def stop(_signal, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    with ExitStack() as stack:
        session = ExperimentSession(args.output, args.backend, args.gpu, live_fps)
        stack.callback(session.close)
        print(json.dumps(dict(ready=True)), flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Command must be a mapping")
                print(json.dumps(session.command(request, args.operation)), flush=True)
            except Exception as error:
                fatal = not isinstance(error, (TypeError, ValueError))
                if fatal:
                    traceback.print_exc(file=sys.stderr)
                print(json.dumps(dict(ok=False, error=str(error), fatal=fatal)), flush=True)
                if fatal:
                    break


if __name__ == "__main__":
    main()
