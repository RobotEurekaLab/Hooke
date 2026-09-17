"""Approximate fixed source point lights for enclosed native scenes.

OpenGL diffuse coefficients have no physical radiometric unit. The native
intensity scale is illustrative; attenuation, spot cones and ambient terms
are not equivalent. Existing global sun and sky illumination stay separate.
"""

import numpy as np


def point_light_specs(model):
    rows = []
    for index, directional in enumerate(model["light_directional"]):
        if directional or not model["light_active"][index]:
            continue
        if model["light_mode"][index] != 0:
            continue
        color = np.asarray(model["light_diffuse"][index], dtype=float)
        peak = float(color.max())
        if peak <= 0:
            continue
        rows.append(
            {
                "source_id": index,
                "body_id": int(model["light_bodyid"][index]),
                "position": model["light_pos"][index].tolist(),
                "color": (color / peak).tolist(),
                "intensity": 8000 * peak,
                "radius": max(0.06, float(model["light_bulbradius"][index])),
            }
        )
    return rows


def create_point_lights(stage, model, body_paths):
    from pxr import Gf, UsdGeom, UsdLux

    rows = point_light_specs(model)
    for row in rows:
        path = f"{body_paths[row['body_id']]}/source_light_{row['source_id']}"
        light = UsdLux.SphereLight.Define(stage, path)
        light.CreateRadiusAttr(row["radius"])
        light.CreateNormalizeAttr(True)
        light.CreateIntensityAttr(row["intensity"])
        light.CreateColorAttr(Gf.Vec3f(*row["color"]))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*row["position"]))
    return rows
