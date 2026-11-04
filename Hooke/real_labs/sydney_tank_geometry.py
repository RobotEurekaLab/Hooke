"""Photograph-informed tank with an authored, physically open dry traverse.

Tank scale, guides and inspection camera are estimates. The clear liquid and
blue illumination are visual context, without calibrated fluid or optical models.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.mounting import annular_mount

CAMERA = "side_heated_tank__inspection_camera"


def _material(b, key, rgba, **attributes):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=b.name + "__mat_" + key,
        rgba=numbers(rgba),
        **{k: str(v) for k, v in attributes.items()},
    )


def _box(b, parent, key, pos, size, material="metal"):
    return b.box(
        parent, pos, size, material, collision=True, name=b.name + "__sydney_" + key
    )


def build_tank(b, base, params):
    # The source shows a clear volume against blue illumination, not blue dye.
    # Keep this scene-local appearance estimate separate from fluid physics.
    _material(b, "blue_liquid", (0.93, 0.97, 1.0, 0.02), specular=0.65, shininess=0.5)
    _material(b, "blue_backlight", (0.025, 0.17, 0.42, 1), emission=0.15)
    for x in (-0.94, 0.94):
        for y in (-0.30, 0.30):
            b.box(base, (x, y, 0.40), (0.035, 0.035, 0.40), "dark", collision=True)
            b.box(base, (x, y, 0.018), (0.075, 0.065, 0.018), "metal")
        b.box(base, (x, 0, 0.79), (0.045, 0.37, 0.04), "dark", collision=True)
    for y in (-0.30, 0.30):
        b.box(base, (0, y, 0.78), (0.98, 0.03, 0.04), "dark", collision=True)
        b.box(base, (0, y, 0.30), (0.98, 0.025, 0.025), "metal")
    # Insulating supports connect both tank ends to the rigid stand.
    for x in (-0.67, 0.67):
        b.box(base, (x, 0, 0.845), (0.15, 0.23, 0.025), "cream", collision=True)
    b.box(base, (0, 0, 0.885), (0.81, 0.20, 0.015), "metal", collision=True)
    b.box(
        base,
        (0, 0, 1.26),
        (0.786, 0.168, 0.358),
        "blue_liquid",
        name=b.name + "__sydney_visual_liquid",
    )
    for y in (-0.19, 0.19):
        b.box(base, (0, y, 1.28), (0.80, 0.012, 0.38), "glass", collision=True)
        for z in (0.90, 1.66):
            b.box(base, (0, y, z), (0.84, 0.018, 0.025), "bright")
            for index in range(19):
                b.cylinder(
                    base,
                    (-0.77 + index * 0.085, y - 0.021, z),
                    0.007,
                    0.006,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
    b.box(base, (-0.80, 0, 1.28), (0.012, 0.19, 0.38), "glass", collision=True)
    b.box(base, (0.825, 0, 1.28), (0.027, 0.23, 0.43), "metal", collision=True)
    _box(b, base, "inner_end_plate", (0.786, 0, 1.28), (0.012, 0.17, 0.35), "copper")
    for z in (0.92, 1.06, 1.20, 1.34, 1.48, 1.62):
        for y in (-0.205, 0.205):
            b.cylinder(
                base, (0.858, y, z), 0.010, 0.008, "bright", euler=(0, math.pi / 2, 0)
            )
    for y in (-0.10, 0.10):
        b.rod(base, (0.89, y, 0.88), (0.89, y, 1.67), 0.012, "bright")
        b.rod(base, (0.83, y, 0.92), (0.94, y, 0.92), 0.012, "metal")
        b.rod(base, (0.89, y, 1.64), (0.82, y, 1.64), 0.012, "metal")
        for z in (1.02, 1.53):
            b.box(base, (0.89, y, z), (0.018, 0.035, 0.022), "dark")
    for x in (-0.59, 0.58):
        for z in (0.94, 1.61):
            b.box(base, (x, -0.215, z), (0.24, 0.018, 0.039), "dark")
    # Backlight and probe gantry are carried by the same stand, not floating decor.
    for x in (-0.94, 0.94):
        b.box(base, (x, 0.28, 1.35), (0.028, 0.028, 0.53), "dark", collision=True)
    _box(b, base, "x_rail", (0, 0.28, 1.88), (0.97, 0.035, 0.03))
    _box(b, base, "backlight", (0, 0.315, 1.32), (0.90, 0.012, 0.45), "blue_backlight")
    _supports(b, base)
    _traverse(b, base)
    _inspection_camera(b, base)
    _upper_apparatus(b, base)


def _upper_apparatus(b, base):
    """Source-visible raised clear enclosure on the right-hand end support.

    Its unobserved depth and attachment sizes are estimates. It is an empty
    fixed enclosure, without an inferred hydraulic connection to the main tank.
    The rear offset leaves the complete authored dry traverse unobstructed.
    """
    for x in (0.18, 0.88):
        _box(b, base, f"upper_support_{x}", (x, 0.56, 1.26), (0.025, 0.025, 0.46), "dark")
        _box(b, base, f"upper_stand_tie_{x}", (x, 0.42, 0.80), (0.025, 0.165, 0.025), "metal")
    _box(b, base, "upper_floor", (0.53, 0.56, 1.72), (0.40, 0.14, 0.018), "bright")
    for y in (0.426, 0.694):
        _box(b, base, f"upper_glass_{y}", (0.53, y, 2.00), (0.39, 0.006, 0.262), "glass")
        _box(b, base, f"upper_bottom_rail_{y}", (0.53, y, 1.754), (0.405, 0.018, 0.016), "bright")
        for i in range(10):
            b.cylinder(base, (0.17 + 0.08 * i, y - 0.020, 1.755), 0.004, 0.003,
                       "dark", euler=(math.pi / 2, 0, 0), density=0)
    for x in (0.134, 0.926):
        _box(b, base, f"upper_end_glass_{x}", (x, 0.56, 2.00), (0.006, 0.128, 0.262), "glass")
        _box(b, base, f"upper_end_rim_{x}", (x, 0.56, 2.267), (0.011, 0.14, 0.006), "metal")
    # Two ties carry the raised frame back to the original rear stand posts.
    _box(b, base, "upper_rear_base_tie", (0.53, 0.56, 0.80), (0.435, 0.025, 0.025), "dark")
    _box(b, base, "upper_right_stand_link", (0.94, 0.43, 0.80), (0.025, 0.155, 0.025), "dark")


def _supports(b, base):
    # The photographed end-pipe clamps and estimated backlight need real mounts.
    for y in (-0.10, 0.10):
        for z in (1.02, 1.53):
            _box(b, base, f"pipe_mount_{y}_{z}", (0.864, y, z), (0.013, 0.018, 0.012))
    for side in (-1, 1):
        for z in (0.95, 1.65):
            _box(
                b,
                base,
                f"backlight_bracket_{side}_{z}",
                (side * 0.914, 0.301, z),
                (0.026, 0.026, 0.018),
            )
            b.cylinder(
                base,
                (side * 0.914, 0.331, z),
                0.005,
                0.008,
                "dark",
                euler=(math.pi / 2, 0, 0),
            )


def _traverse(b, base):
    carrier = b.moving(
        base, "probe_x", (0, 0.28, 1.88), (1, 0, 0), (-0.55, 0.55), mass=0.30, kp=450
    )
    # Four connected faces around the existing full-length rectangular rail.
    for side in (-1, 1):
        _box(
            b,
            carrier,
            f"x_sleeve_z_{side}",
            (0, 0, side * 0.0456),
            (0.075, 0.0606, 0.015),
        )
        _box(
            b,
            carrier,
            f"x_sleeve_y_{side}",
            (0, side * 0.0481, 0),
            (0.075, 0.0125, 0.0306),
        )
    _box(b, carrier, "guide_reach", (0.030, -0.14, 0.0656), (0.014, 0.16, 0.012))
    annular_mount(
        b,
        carrier,
        "z_guide",
        (0, -0.28, 0.070),
        0.0036,
        0.022,
        0.070,
        material="metal",
        proxies=True,
        axis="z",
        namespace="sydney",
    )
    for z in (0.015, 0.125):
        _box(
            b,
            carrier,
            f"guide_band_{z}",
            (0.027, -0.28, z),
            (0.008, 0.015, 0.006),
            "dark",
        )
    target = b.moving(
        carrier,
        "probe_height",
        (0, -0.28, 0),
        (0, 0, 1),
        (-0.09, 0.09),
        mass=0.04,
        kp=180,
    )
    b.rod(
        target,
        (0, 0, 0.31),
        (0, 0, -0.64),
        0.003,
        "bright",
        collision=True,
        name=b.name + "__sydney_probe_shaft",
    )
    _box(b, target, "calibration_plate", (0, 0, -0.65), (0.015, 0.001, 0.01), "cream")
    _box(b, target, "plate_marker", (0, -0.0012, -0.65), (0.010, 0.0002, 0.001), "dark")
    b.site(target, "calibration_plate", (0, -0.0014, -0.65), 0.001)
    # Small fixed clamps at the stem/plate joint make the attachment inspectable.
    _box(b, target, "plate_mount", (0, 0.004, -0.635), (0.008, 0.004, 0.009))
    b.metadata["capabilities"] = [
        "probe_horizontal_traverse",
        "probe_height",
        "visible_inert_calibration_plate",
        "instrument_camera",
    ]
    b.metadata["inspection_interface"] = {
        "camera": CAMERA,
        "target_geom": b.name + "__sydney_plate_marker",
        "guide_clearance_m": 0.0006,
        "target_size_m": [0.030, 0.002, 0.020],
        "camera_is_calibrated": False,
    }
    b.metadata["limitations"].append(
        "Authored open guides and dry inert-plate positioning, with ideal joints and surrogate mass/gains. The clear volume and blue illumination have no fluid forces, heat transfer, convection, PIV/LIF output or calibrated imaging. The raised, tilted camera stand and hidden interfaces are estimated additions, not source-observed OEM hardware."
    )


def _inspection_camera(b, base):
    # Authored dry observation stand, separate from the source-supported tank.
    base = ET.SubElement(base, "body", name=b.name + "__camera_stand")
    for x in (-0.12, 0.12):
        _box(
            b, base, f"camera_foot_{x}", (x, -1.10, 0.020), (0.022, 0.22, 0.020), "dark"
        )
    _box(b, base, "camera_crossfoot", (0, -1.10, 0.057), (0.145, 0.025, 0.018))
    _box(b, base, "camera_post", (0, -1.10, 0.8075), (0.021, 0.021, 0.7325))
    _box(b, base, "camera_mount", (0, -1.10, 1.5525), (0.065, 0.055, 0.0125))
    # Raise the real lens and tilt the entire head, not only a render camera.
    # This authored viewpoint avoids the frontal softbox reflection at the plate.
    tilt = -math.atan2(0.370, 1.0115)
    pivot = (0, -1.0129 - 0.0871 * math.cos(tilt), 1.600 - 0.0871 * math.sin(tilt))
    for side in (-1, 1):
        _box(
            b,
            base,
            f"camera_cheek_{side}",
            (side * 0.070, pivot[1], (1.565 + pivot[2]) / 2),
            (0.007, 0.025, (pivot[2] - 1.565) / 2),
            "dark",
        )
        b.cylinder(
            base,
            (side * 0.067, pivot[1], pivot[2]),
            0.012,
            0.008,
            "metal",
            collision=True,
            euler=(0, math.pi / 2, 0),
            name=b.name + f"__sydney_camera_pivot_{side}",
        )
    base = ET.SubElement(
        base,
        "body",
        name=b.name + "__camera_head",
        pos=numbers(pivot),
        euler=numbers((tilt, 0, 0)),
    )
    _box(b, base, "camera_head", (0, 0, 0), (0.063, 0.048, 0.050), "dark")
    b.cylinder(
        base,
        (0, 0.062, 0),
        0.026,
        0.021,
        "metal",
        collision=True,
        euler=(math.pi / 2, 0, 0),
        name=b.name + "__sydney_camera_lens_barrel",
    )
    b.cylinder(
        base,
        (0, 0.085, 0),
        0.022,
        0.002,
        "lens",
        collision=True,
        euler=(math.pi / 2, 0, 0),
        name=b.name + "__sydney_camera_lens_face",
    )
    position = (0, 0.0871, 0)
    quaternion = (math.sqrt(0.5), math.sqrt(0.5), 0, 0)
    b.site(base, "inspection_camera", position, 0.001)
    ET.SubElement(
        base,
        "camera",
        name=CAMERA,
        pos=numbers(position),
        quat=numbers(quaternion),
        fovy="64",
        mode="fixed",
    )
    b.metadata["mounted_cameras"] = [
        dict(
            name=CAMERA,
            body=base.get("name"),
            site=b.metadata["sites"]["inspection_camera"],
            local_position_m=list(position),
            local_quaternion_wxyz=list(quaternion),
            fovy_deg=64,
            purpose="Authored dry inert-plate observation through the tank; not calibrated fluid imaging",
        )
    ]
