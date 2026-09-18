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
            "网页访问者共用服务器配置；请由运行 Hooke 的 Linux 账号配置 Isaac 安装。",
            "详细安装、多用户权限及服务部署说明：docs/isaac_setup.md。",
            "从仓库根目录运行 ./scripts/start_hooke_backends.sh --configure-isaac --discover；"
            "唯一可用安装可用 --auto 保存；非标准目录可加 --search-root /absolute/search/directory。",
            "从仓库的 Hooke/ 目录运行：../.venv/bin/python -m backends.config "
            "--isaac-path /absolute/path/to/isaacsim --gpu <空闲GPU编号>。",
            "环境变量 HOOKE_ISAAC_PATH / HOOKE_ISAAC_GPU 优先；修改服务环境后重启服务。",
        ]
        if code in ("permission_denied", "not_executable"):
            report["remediation"].append(
                "请安装所有者或管理员为服务账号授予必要的目录遍历、读取和执行权限，"
                "或使用服务账号可访问的安装；配置路径不能绕过权限。"
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
            return fail("permission_denied", path, f"服务账号 {user} 无法访问目录 {path}。")
        except FileNotFoundError:
            return fail("installation_missing", path, f"Isaac 安装目录不存在：{path}（服务账号 {user}）。")
        except OSError as error:
            return fail("installation_unavailable", path, f"无法访问 Isaac 目录 {path}：{error}")
        if not stat.S_ISDIR(mode):
            return fail("installation_invalid", path, f"Isaac 路径不是目录：{path}。")
        if not os.access(path, os.X_OK, effective_ids=True):
            return fail("permission_denied", path, f"服务账号 {user} 无法遍历目录 {path}。")
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
            return fail("permission_denied", path, f"服务账号 {user} 无法访问 {path}。")
        except FileNotFoundError:
            return fail("installation_incomplete", path, f"Isaac 安装缺少必要文件：{path}。")
        except OSError as error:
            return fail("installation_unavailable", path, f"无法读取 {path}：{error}")
        if not stat.S_ISREG(mode):
            return fail("installation_invalid", path, f"Isaac 必要文件不是普通文件：{path}。")
        if not os.access(path, os.R_OK, effective_ids=True):
            return fail("permission_denied", path, f"服务账号 {user} 无法读取 {path}。")
        if executable and not os.access(path, os.X_OK, effective_ids=True):
            return fail("not_executable", path, f"服务账号 {user} 无法执行 {path}。")
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
            report["remediation"] = ["修正服务账号配置中的 gpu 或环境变量 HOOKE_ISAAC_GPU，使用非负整数。"]
    return report


def require_isaac_environment():
    report = isaac_environment()
    if not report["installation_ready"]:
        raise IsaacConfigurationError(report)
    return report
