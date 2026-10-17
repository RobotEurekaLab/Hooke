"""Select an asset family independently of its motor-control topology."""

import os


def requested_profile():
    profile = os.environ.get("HOOKE_MICROSCOPY_MANIPULATOR", "auto")
    choices = ("auto", "parallel-v4", "sutter-mp285-reference")
    if profile not in choices:
        raise ValueError("HOOKE_MICROSCOPY_MANIPULATOR must be " + ", ".join(choices))
    return profile


def installed_profile(model, fallback="linear"):
    from microscopy.cad_assets import asset_evidence

    profiles = {item["profile"] for item in asset_evidence(model).get("assemblies", [])
                if item.get("profile") in ("parallel-v4", "sutter-mp285-reference")}
    if len(profiles) > 1:
        raise ValueError("Mixed manipulator families require explicit per-tool controls")
    return next(iter(profiles), fallback)
