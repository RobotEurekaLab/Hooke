"""Find accessible standalone installations without running their programs."""

import os
from pathlib import Path


MAX_SEARCH_DIRECTORIES = 256
SEARCH_DEPTH = 3


class IsaacDiscoveryError(ValueError):
    def __init__(self, code, message, report):
        self.code = code
        self.report = report
        super().__init__(message)


def default_candidates():
    """Search this account and conventional shared locations, not other homes."""
    home = Path.home()
    candidates = [home / "isaacsim", home / "isaac-sim", home / "humanoid/isaacsim",
                  Path("/opt/isaacsim"), Path("/opt/isaac-sim"),
                  Path("/usr/local/isaacsim"), Path("/isaac-sim")]
    for root, pattern in ((home, "isaac-sim-*"), (home / "Downloads", "isaac-sim-*"),
                          (home / ".local/share/ov/pkg", "isaac-sim-*"),
                          (Path("/opt"), "isaac-sim-*"), (Path("/opt/nvidia"), "isaac*")):
        try:
            candidates.extend(sorted(root.glob(pattern)))
        except (OSError, RuntimeError):
            continue
    return candidates


def discover_installations(search_roots=()):
    """Inspect default candidates and an optional bounded, explicit directory scan."""
    from backends.environment import inspect_installation

    roots = [Path(root).expanduser().absolute() for root in search_roots]
    candidates = list(default_candidates())
    pending = [(root, 0) for root in reversed(roots)]
    visited = set()
    truncated = False
    while pending:
        root, depth = pending.pop()
        try:
            canonical = root.resolve()
            if canonical in visited:
                continue
            if len(visited) >= MAX_SEARCH_DIRECTORIES:
                truncated = True
                break
            visited.add(canonical)
            if depth == 0 or (root / "python.sh").is_file():
                candidates.append(root)
            if (root / "python.sh").is_file() or depth >= SEARCH_DEPTH:
                continue
            with os.scandir(root) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                        if len(pending) >= MAX_SEARCH_DIRECTORIES:
                            truncated = True
                            break
                        pending.append((Path(entry.path), depth + 1))
        except (OSError, RuntimeError):
            continue

    reports = []
    seen = set()
    for candidate in candidates:
        try:
            canonical = candidate.resolve()
        except (OSError, RuntimeError):
            canonical = candidate
        if canonical in seen:
            continue
        seen.add(canonical)
        reports.append(inspect_installation(candidate))
    available = [str(Path(item["isaac_path"]).resolve()) for item in reports if item["installation_ready"]]
    return dict(search_roots=[str(root) for root in roots], candidates=reports,
                available_installations=available, search_truncated=truncated,
                selection=available[0] if len(available) == 1 and not truncated else None,
                scope="Read-only account/shared search; explicit roots bounded to three levels and 256 directories")


def select_installation(report):
    if report["search_truncated"]:
        raise IsaacDiscoveryError("discovery_incomplete", "搜索达到目录上限，请缩小 --search-root 后重试。", report)
    available = report["available_installations"]
    if not available:
        raise IsaacDiscoveryError("installation_not_found", "没有发现服务账号可访问的完整 Isaac 安装，请指定路径或安装 Isaac。", report)
    if len(available) > 1:
        raise IsaacDiscoveryError("installation_ambiguous", "发现多个可用 Isaac 安装，请用 --isaac-path 明确选择：" + "、".join(available), report)
    return Path(available[0])
