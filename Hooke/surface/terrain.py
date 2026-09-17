"""Deterministic landscapes with a separately identified measured DEM patch."""

from pathlib import Path
import struct
import tempfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from worlds.pds_terrain import SOURCE_ROOT, load_crop


def landscape(mission, resolution=257):
    axis = np.linspace(
        -mission.terrain_half_size_m, mission.terrain_half_size_m, resolution
    )
    x, y = np.meshgrid(axis, axis)
    if mission.world == "lunar":
        height = 0.15 * np.sin(x / 4) * np.cos(y / 5)
        for cx, cy, radius, depth in (
            (6, 11, 6, 2.3),
            (-12, -12, 9, 3.0),
            (24, -8, 5, 1.7),
        ):
            r = np.hypot(x - cx, y - cy) / radius
            height += 1.1 * np.exp(-(((r - 1) / 0.13) ** 2)) - depth * np.exp(
                -((r / 0.7) ** 4)
            )
    else:
        height = 1.8 * np.sin(x / 11) * np.sin(y / 13) + 0.2 * np.cos(x / 2.5 + y / 4)
        height += 5 * np.exp(-(((x + 15) / 9) ** 2) - ((y - 17) / 16) ** 2)
        height += 3 * np.exp(-(((x - 22) / 8) ** 2) - ((y + 12) / 13) ** 2)
        height += 3 * np.exp(-(((x - 5) / 10) ** 2) - ((y - 18) / 9) ** 2)
    # A prepared access strip belongs to the generated mission layout.
    # It is not represented as measured terrain or as a soil preparation model.
    distance = np.maximum(np.abs(y) - 2.2, np.maximum(-x - 3, x - 10))
    height *= np.clip(distance / 3, 0, 1)
    height *= np.clip((np.hypot(x + 2.8, y - 3.5) - 2.3) / 2.5, 0, 1)
    crop, source = load_crop(SOURCE_ROOT / "assets/space/terrain" / mission.world)
    half_x, half_y = np.asarray(source["extent_m"]) / 2
    cx, cy = -23.0, 23.0
    patch_x, patch_y = x - cx, y - cy
    rows = np.linspace(-half_y, half_y, crop.shape[0])
    columns = np.linspace(-half_x, half_x, crop.shape[1])
    interpolator = RegularGridInterpolator(
        (rows, columns), crop, bounds_error=False, fill_value=None
    )
    elevations = interpolator(
        np.stack(
            (np.clip(patch_y, -half_y, half_y), np.clip(patch_x, -half_x, half_x)),
            axis=-1,
        )
    )
    outside = np.maximum(np.abs(patch_x) - half_x, np.abs(patch_y) - half_y)
    weight = 1 - np.clip(outside / 7, 0, 1)
    translation = -float(crop.mean())
    height = (1 - weight) * height + weight * (elevations + translation)
    return height, {
        "layout": f"{2 * mission.terrain_half_size_m:g} m generated landscape with prepared landing zone and rover access strip",
        "measured_patch": {
            "product_id": source["product"]["product_id"],
            "credit": source["product"]["credit"],
            "source_spacing_m": source["source_spacing_m"],
            "extent_m": source["extent_m"],
            "source_shape": source["shape"],
            "world_center_m": [cx, cy],
            "elevation_translation_m": translation,
            "height_sha256": source["height_sha256"],
        },
        "generated_features": "craters, ridges, rocks and the 7 m transition outside the measured patch",
        "interpolation": "linear resampling; source accuracy and resolution are unchanged",
        "soil_model": "rigid collision surface; no deformable soil or dust dynamics",
    }


def write_heightfield(path: Path, height):
    minimum, maximum = float(height.min()), float(height.max())
    normalized = ((height - minimum) / (maximum - minimum)).astype("<f4")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
        temporary.write(struct.pack("<ii", *height.shape) + normalized.tobytes())
    Path(temporary.name).replace(path)
    return minimum, maximum - minimum
