"""Machine-specific installation settings stay outside version control."""

import json
import os
from pathlib import Path
import tempfile


LOCAL_CONFIG = Path(__file__).resolve().parents[2] / "temp/isaac_local.json"
DEFAULT_GPU = 0


def account_config():
    """Use the service account's configuration, independently of the checkout."""
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "hooke/isaac.json"


def read_settings(path):
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read Isaac configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Isaac configuration {path} must be a JSON object")
    return value


def local_settings():
    return read_settings(LOCAL_CONFIG)


def setting(name, environment, default):
    override = os.environ.get(environment)
    if override:
        return override, environment
    for path in (account_config(), LOCAL_CONFIG):
        value = read_settings(path).get(name)
        if value is not None:
            return value, str(path)
    return default, "default"


def installation_setting():
    value, source = setting("isaac_path", "HOOKE_ISAAC_PATH", None)
    if source == "default":
        from backends.discovery import discover_installations, select_installation

        return select_installation(discover_installations()), "auto-discovery"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Isaac installation path in {source} must be a nonempty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Isaac installation path in {source} must be absolute")
    return path, source


def isaac_installation():
    return installation_setting()[0]


def isaac_gpu():
    value, _source = setting("gpu", "HOOKE_ISAAC_GPU", DEFAULT_GPU)
    return gpu_index(value)


def gpu_index(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Isaac GPU index must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError("Isaac GPU index must be nonnegative")
    return value


def save_account_settings(installation, gpu=None):
    """Validate before atomically saving; never change installation permissions."""
    from backends.environment import inspect_installation, IsaacConfigurationError

    path = Path(installation).expanduser().absolute()
    report = inspect_installation(path)
    if not report["installation_ready"]:
        raise IsaacConfigurationError(report)
    try:
        value = read_settings(account_config())
    except ValueError:
        # An explicit setup command can repair malformed account configuration.
        value = {}
    value["isaac_path"] = str(path)
    if gpu is None:
        try:
            gpu = gpu_index(value.get("gpu", DEFAULT_GPU))
        except ValueError:
            gpu = DEFAULT_GPU
    value["gpu"] = gpu_index(gpu)
    destination = account_config()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    import argparse
    from backends.environment import isaac_environment, IsaacConfigurationError
    from backends.discovery import IsaacDiscoveryError, discover_installations, select_installation

    parser = argparse.ArgumentParser(description="Configure Isaac for the Linux account running Hooke.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--isaac-path", type=Path, help="Accessible Isaac installation containing python.sh")
    mode.add_argument("--discover", action="store_true", help="Read-only installation search; no GPU startup")
    mode.add_argument("--auto", action="store_true", help="Save the unique discovered installation for this account")
    parser.add_argument("--search-root", type=Path, action="append", default=[], help="Additional explicit search directory; repeatable")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.gpu is not None and not (args.isaac_path or args.auto):
        parser.error("--gpu requires --isaac-path or --auto when saving configuration")
    if args.search_root and not (args.discover or args.auto):
        parser.error("--search-root requires --discover or --auto")
    try:
        discovered = discover_installations(args.search_root) if args.discover or args.auto else None
        if args.discover:
            print(json.dumps(discovered, indent=2, ensure_ascii=False))
            raise SystemExit(0 if discovered["available_installations"] and not discovered["search_truncated"] else 1)
        installation = select_installation(discovered) if args.auto else args.isaac_path
        saved = save_account_settings(installation, args.gpu) if installation else None
        report = isaac_environment()
        if discovered is not None:
            report["discovery"] = discovered
        if saved:
            report["saved_configuration"] = str(saved)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["installation_ready"] else 1)
    except IsaacDiscoveryError as error:
        report = dict(error.report, error_code=error.code, error=str(error))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    except (IsaacConfigurationError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
