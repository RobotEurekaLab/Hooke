"""Read-only environment diagnosis; never installs software or changes drivers."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
from backends.environment import isaac_environment
from backends.capabilities import registry
from backends.render_settings import RenderSettings


def diagnose(gpu=None):
    installation = isaac_environment(gpu)
    selected = installation["gpu"]
    checks = {
        "isaac_python_launcher": installation["checks"].get("launcher", False),
        "isaac_installation_access": installation["installation_ready"],
        "source_plugin": (
            Path(__file__).resolve().parents[1] / "libmjlab.so.3.3.0"
        ).is_file(),
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "meshplane_module_found": importlib.util.find_spec("meshplane")
        is not None,
    }
    packages = {}
    for name in ("mujoco", "numpy", "scipy", "jax", "toppra", "Pillow", "Flask"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    checks["source_packages"] = all(packages.values())
    report = dict(
        python=platform.python_version(),
        isaac_path=installation["isaac_path"],
        isaac_environment=installation,
        gpu=selected,
        checks=checks,
        render_settings=RenderSettings.from_environment().report(),
        packages=packages,
        driver_modified=False,
        capabilities=registry(),
    )
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(selected),
                "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
        index, uuid, name, driver, used, total = [s.strip() for s in line.split(",")]
        report["device"] = dict(
            index=int(index),
            uuid=uuid,
            name=name,
            driver=driver,
            memory_used_mib=int(used),
            memory_total_mib=int(total),
        )
        report["gpu_idle_for_startup"] = int(used) < 2048
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        checks["gpu_query"] = False
        report["gpu_error"] = str(error)
    report["environment_files_ready"] = all(checks.values())
    report["native_runtime_started"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.gpu)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    raise SystemExit(0 if result["environment_files_ready"] else 1)


if __name__ == "__main__":
    main()
