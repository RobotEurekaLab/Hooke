"""Opt-in review presentation; source geometry and physics remain unchanged."""


REAL_LABS_CUTAWAY = "real_labs_cutaway_v1"
_CUTAWAY_PREFIXES = ("arch_roof_", "arch_cutaway_wall_", "arch_luminaire_")
_CUTAWAY_CAMERAS = frozenset(("overview", "workstation"))


def source_visibility_profile(metadata):
    """Resolve the existing laboratory overview/detail cutaway convention.

    All selected cameras must use the same review policy. Enclosed, mounted and
    scientific views are rejected rather than silently changing their scene.
    No profile means no visibility override for ordinary robot/microscopy runs.
    """
    profile = metadata.get("visual_profile")
    if profile is None:
        return {"name": None, "cameras": [], "hidden_geom_ids": []}
    if profile != REAL_LABS_CUTAWAY:
        raise ValueError(f"Unknown source visibility profile: {profile!r}")
    mapping = metadata.get("task_info", {}).get("camera_mapping")
    if not isinstance(mapping, dict) or not mapping or any(
        not isinstance(value, str) for value in mapping.values()
    ):
        raise ValueError("Real-labs cutaway requires an explicit camera mapping")
    cameras = list(dict.fromkeys(mapping.values()))
    names = metadata["names"]
    if any(name not in _CUTAWAY_CAMERAS or name not in names["camera"] for name in cameras):
        raise ValueError(
            "Real-labs cutaway supports only authored overview/workstation cameras; "
            "capture interior, mounted and scientific views without this profile"
        )
    hidden = [
        index for index, name in enumerate(names["geom"])
        if (name or "").startswith(_CUTAWAY_PREFIXES)
    ]
    if not hidden:
        raise ValueError("Real-labs cutaway profile found no authored cutaway geometry")
    return {
        "name": profile,
        "cameras": cameras,
        "hidden_geom_ids": hidden,
        "scope": "Review-camera visibility only; authored physical enclosure remains present",
    }


def apply_visibility_profile(stage, geom_paths, profile):
    """Hide only selected imageable prims; retain collision APIs and geometry.

    Resolve every target before changing any visibility so a broken conversion
    cannot leave a partially applied profile. USD is imported only when needed.
    """
    report = {**profile, "applied": False, "hidden_geom_paths": []}
    identifiers = profile["hidden_geom_ids"]
    if not identifiers:
        return report
    from pxr import UsdGeom

    targets = []
    for identifier in identifiers:
        path = geom_paths.get(identifier)
        if path is None:
            raise ValueError(f"Cutaway geometry {identifier} has no converted prim")
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        if not imageable:
            raise ValueError(f"Cutaway geometry {identifier} is not imageable: {path}")
        targets.append((path, imageable))
    for _, imageable in targets:
        imageable.MakeInvisible()
    report.update(applied=True, hidden_geom_paths=[path for path, _ in targets])
    return report
