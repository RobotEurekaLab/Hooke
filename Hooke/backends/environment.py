"""Inspect the service account's Isaac installation without launching a GPU job."""

import os
from pathlib import Path
import pwd
import stat

from backends.config import account_config, installation_setting, isaac_gpu, gpu_index
from backends.discovery import IsaacDiscoveryError


class IsaacConfigurationError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["error"] + "\n" + "\n".join(report["remediation"]))


def inspect_installation(installation=None):
    try:
        user = pwd.getpwuid(os.geteuid()).pw_name
    except KeyError:
        user = str(os.geteuid())
    report = dict(
        service_user=user, service_uid=os.geteuid(), configuration_scope="server_process",
        account_configuration=str(account_config()), isaac_path=None, configuration_source=None,
        installation_ready=False, native_runtime_checked=False, checks={}, error=None,
        error_code=None, blocked_path=None, remediation=[],
    )

    def fail(code, path, message):
        report.update(error_code=code, blocked_path=str(path) if path else None, error=message)
        report["remediation"] = [
            "Web visitors share the server configuration; the Linux account running Hooke must configure the Isaac installation.",
            "Detailed installation, multi-user permissions, and service deployment notes: docs/isaac_setup.md.",
            "From the repo root, run ./scripts/start_hooke_backends.sh --configure-isaac --discover; "
            "a single available installation can be saved with --auto; non-standard directories can add --search-root /absolute/search/directory.",
            "From the repo's Hooke/ directory, run: ../.venv/bin/python -m backends.config "
            "--isaac-path /absolute/path/to/isaacsim --gpu <free GPU index>.",
            "Environment variables HOOKE_ISAAC_PATH / HOOKE_ISAAC_GPU take priority; restart the service after changing it.",
        ]
        if code in ("permission_denied", "not_executable"):
            report["remediation"].append(
                "Ask the installation owner or an administrator to grant the service account the necessary "
                "directory traversal, read, and execute permissions, or use an installation the service account "
                "can already access; the configured path cannot bypass permissions."
            )
        return report

    try:
        if installation is None:
            installation, source = installation_setting()
        else:
            installation, source = Path(installation).absolute(), "explicit"
        report.update(isaac_path=str(installation), configuration_source=source)
    except IsaacDiscoveryError as error:
        report["discovery"] = error.report
        return fail(error.code, None, str(error))
    except (ValueError, OSError) as error:
        return fail("configuration_invalid", None, str(error))

    # Check traversal separately: is_file() can hide an inaccessible parent as a missing file.
    def directory_error(path):
        try:
            mode = path.stat().st_mode
        except PermissionError:
            return fail("permission_denied", path, f"Service account {user} cannot access directory {path}.")
        except FileNotFoundError:
            return fail("installation_missing", path, f"Isaac installation directory does not exist: {path} (service account {user}).")
        except OSError as error:
            return fail("installation_unavailable", path, f"Cannot access Isaac directory {path}: {error}")
        if not stat.S_ISDIR(mode):
            return fail("installation_invalid", path, f"Isaac path is not a directory: {path}.")
        if not os.access(path, os.X_OK, effective_ids=True):
            return fail("permission_denied", path, f"Service account {user} cannot traverse directory {path}.")
        return None

    for path in (*reversed(installation.parents), installation):
        failure = directory_error(path)
        if failure:
            return failure
    report["checks"]["directory_traversal"] = True
    files = {
        "launcher": ("python.sh", True),
        "python_environment": ("setup_python_env.sh", False),
        "python_executable": ("kit/python/bin/python3", True),
        "carb_library": ("kit/libcarb.so", False),
    }
    for name, (relative, executable) in files.items():
        path = installation / relative
        for parent in reversed(path.relative_to(installation).parents[:-1]):
            failure = directory_error(installation / parent)
            if failure:
                return failure
        try:
            mode = path.stat().st_mode
        except PermissionError:
            return fail("permission_denied", path, f"Service account {user} cannot access {path}.")
        except FileNotFoundError:
            return fail("installation_incomplete", path, f"Isaac installation is missing a required file: {path}.")
        except OSError as error:
            return fail("installation_unavailable", path, f"Cannot read {path}: {error}")
        if not stat.S_ISREG(mode):
            return fail("installation_invalid", path, f"Isaac required file is not a regular file: {path}.")
        if not os.access(path, os.R_OK, effective_ids=True):
            return fail("permission_denied", path, f"Service account {user} cannot read {path}.")
        if executable and not os.access(path, os.X_OK, effective_ids=True):
            return fail("not_executable", path, f"Service account {user} cannot execute {path}.")
        report["checks"][name] = True
    report["installation_ready"] = True
    return report


def isaac_environment(gpu=None):
    report = inspect_installation()
    try:
        report["gpu"] = isaac_gpu() if gpu is None else gpu_index(gpu)
    except (ValueError, OSError) as error:
        report["gpu"] = None
        if report["installation_ready"]:
            report.update(installation_ready=False, error_code="configuration_invalid", error=str(error))
            report["remediation"] = ["Fix the gpu value in the service account configuration or the HOOKE_ISAAC_GPU environment variable; use a non-negative integer."]
    return report


def require_isaac_environment():
    report = isaac_environment()
    if not report["installation_ready"]:
        raise IsaacConfigurationError(report)
    return report
