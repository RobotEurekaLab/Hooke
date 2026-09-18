"""Select an instrument stand without changing the task's SI coordinate frame."""

import os


def stand_profile(has_cad):
    value = os.environ.get("HOOKE_MICROSCOPY_STAND", "auto")
    if value not in ("auto", "openframe", "te2000-s-reference"):
        raise ValueError("HOOKE_MICROSCOPY_STAND must be auto, openframe or te2000-s-reference")
    if value == "auto":
        return "openframe" if has_cad else "photo-reference"
    if value == "openframe" and not has_cad:
        raise ValueError("The openFrame stand requires the CAD asset profile")
    if value == "te2000-s-reference":
        from microscopy.cad_optics import camera_profile
        if camera_profile() != "estimated":
            raise ValueError("The TE2000-S reference stand requires estimated optics; openFrame optical mounts do not fit this stand")
    return value
