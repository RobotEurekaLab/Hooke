"""Original ground-test rover mast and cache interfaces, not NASA hardware CAD.

The public SLOPE preview informs the high-bay test context only. The routed mast,
open pan bearing, retained cache caps and stereo target are authored mechanical
surrogates. Optical baseline/FOV and bearing clearances are not calibrated.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount

CAMERAS = ("rover__stereo_left", "rover__stereo_right")
CAMERA_QUATERNION = (0.5, 0.5, -0.5, -0.5)


def _name(b, key):
    return f"{b.name}__mast_{key}"


def _box(b, parent, key, pos, half, material="metal", **kwargs):
    return b.box(
        parent, pos, half, material, collision=True, name=_name(b, key), **kwargs
    )


def _cylinder(b, parent, key, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=_name(b, key),
        **kwargs,
    )


def _registration_board(b, base, *, construction_base_z_m=None):
    """Build the supported target in its explicitly selected construction frame."""
    bx, by, bz = map(float, base.get("pos").split())
    if construction_base_z_m is not None:
        bz = float(construction_base_z_m)
    # NASA's anchored rover pose is yaw zero. Keep this restriction explicit;
    # a future different facility/placement must define its own supported target.
    if tuple(map(float, base.get("quat").split())) != (1.0, 0.0, 0.0, 0.0):
        raise ValueError("NASA stereo target requires the authored zero-yaw rover")
    angle = math.atan2(0.9, 2.22)
    board = ET.SubElement(
        b.root.find("worldbody"),
        "body",
        name=_name(b, "registration_fixture"),
        pos=f"{bx + 2.0} {by + 0.9} 0.3",
        euler=f"0 0 {angle}",
    )
    height = bz + 1.026 - 0.3
    _box(b, board, "target_foot", (0, 0, 0.018), (0.22, 0.20, 0.018), "dark")
    _cylinder(
        b,
        board,
        "target_column",
        (0.022, 0, height / 2 + 0.018),
        0.022,
        height / 2 - 0.018,
    )
    _box(b, board, "target_back", (0, 0, height), (0.010, 0.15, 0.12), "cream")
    for iy in range(-3, 4):
        for iz in range(-2, 3):
            if iy == iz == 0:
                continue
            _box(
                b,
                board,
                f"target_checker_{iy}_{iz}",
                (-0.0105, iy * 0.040, height + iz * 0.040),
                (0.0005, 0.020, 0.020),
                "dark" if (iy + iz) % 2 else "bright",
            )
    _cylinder(
        b,
        board,
        "registration_mark",
        (-0.011, 0, height),
        0.018,
        0.001,
        "orange",
        quat=(math.sqrt(0.5), 0, math.sqrt(0.5), 0),
    )
    site = b.site(board, "mast_registration_target", (-0.012, 0, height), 0.003)
    return {
        "site": site,
        "geom": _name(b, "registration_mark"),
        "diameter_m": 0.036,
        "purpose": "Authored inert geometric inspection target; not observed NASA facility hardware",
        "support": "Foot rests at z=0.30 m on the existing rigid soil-test lane; connected column and board",
    }


def build_mast(b, base, *, target_base_z_m=None):
    # Routed support clears all three original cache targets. The deck skin and
    # foot are connected; hidden electronics/cable delivery are not modeled.
    _box(b, base, "deck_skin", (0, 0, 0.403), (0.38, 0.30, 0.015))
    _box(b, base, "foot", (-0.22, 0.091, 0.425), (0.040, 0.018, 0.007), "dark")
    _cylinder(b, base, "column", (-0.22, 0.091, 0.673), 0.014, 0.241)
    _box(b, base, "upper_bridge", (-0.22, 0.0455, 0.914), (0.024, 0.0655, 0.016))
    _cylinder(b, base, "bearing_seat", (-0.22, 0, 0.923), 0.040, 0.007, "dark")
    _cylinder(b, base, "fixed_spindle", (-0.22, 0, 0.9535), 0.018, 0.0235, "bright")
    for sx in (-1, 1):
        for sy in (-1, 1):
            _cylinder(
                b,
                base,
                f"foot_fastener_{sx}_{sy}",
                (-0.22 + sx * 0.030, 0.091 + sy * 0.012, 0.434),
                0.004,
                0.002,
                "bright",
            )
    pan = b.moving(
        base,
        "mast_pan",
        (-0.22, 0, 1.026),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.5,
    )
    # Lower sleeve surface at 0.9306 m and bore radius 0.0186 m leave 0.6 mm
    # clearance from the seat/spindle. Top joins the head underside at 0.985 m.
    annular_mount(
        b,
        pan,
        "pan_sleeve",
        (0, 0, -0.0682),
        0.0186,
        0.031,
        0.0272,
        axis="z",
        proxies=True,
        material="metal",
        namespace="mast",
    )
    _box(b, pan, "head", (0, 0, 0), (0.044, 0.105, 0.041), "cream")
    _box(b, pan, "rear_panel", (-0.045, 0, 0), (0.001, 0.088, 0.030), "dark")
    cameras = []
    for label, y, camera in zip(("left", "right"), (-0.065, 0.065), CAMERAS):
        rotation = (math.sqrt(0.5), 0, math.sqrt(0.5), 0)
        _cylinder(
            b,
            pan,
            f"{label}_barrel",
            (0.052, y, 0),
            0.023,
            0.017,
            "dark",
            quat=rotation,
        )
        _cylinder(
            b, pan, f"{label}_lens", (0.071, y, 0), 0.018, 0.002, "lens", quat=rotation
        )
        position = (0.0731, y, 0)
        site = b.site(pan, f"stereo_{label}", position, 0.001)
        ET.SubElement(
            pan,
            "camera",
            name=camera,
            mode="fixed",
            pos=" ".join(map(str, position)),
            quat=" ".join(map(str, CAMERA_QUATERNION)),
            fovy="58",
        )
        cameras.append(
            dict(
                name=camera,
                body=pan.get("name"),
                site=site,
                local_position_m=list(position),
                local_quaternion_wxyz=list(CAMERA_QUATERNION),
                fovy_deg=58,
                purpose="Original body-mounted pinhole camera; authored stereo geometry, not calibrated imaging",
            )
        )
    b.site(pan, "stereo_camera", (0.075, 0, 0))
    b.metadata["mounted_cameras"] = cameras
    b.metadata["mast_inspection"] = {
        "control": "mast_pan",
        "camera_names": list(CAMERAS),
        "baseline_m": 0.130,
        "fovy_deg": 58,
        "target": _registration_board(
            b, base, construction_base_z_m=target_base_z_m
        ),
        "bearing_radial_clearance_m": 0.0006,
        "bearing_axial_clearance_m": 0.0006,
        "minimum_task_clearance_m": 0.0002,
        "scope": "Dry anchored-rover pan/target inspection and return; original representative mast, no NASA/OEM or stereo metrology claim",
        "limitations": [
            "Ideal pan constraint and original surrogate inertia/servo; no bearing stress/preload or optical calibration",
            "No physical moving service feed; internal electrical transfer remains unmodeled",
            "Separate target stand is authored test equipment, not source-observed NASA hardware",
        ],
    }


def build_caches(b, base):
    # All original target coordinates stay unchanged, 4 mm above visible caps.
    # Solid retained inert caps represent existing sample-cache targets; they
    # are not loadable sockets and do not establish sample pickup/transfer.
    _box(b, base, "cache_plinth", (-0.2, 0, 0.421), (0.10, 0.07, 0.003), "metal")
    _box(b, base, "cache_holder", (-0.2, 0, 0.449), (0.10, 0.07, 0.025), "dark")
    records = []
    for index in range(3):
        x = -0.26 + index * 0.06
        _cylinder(
            b, base, f"cache_cap_{index+1}", (x, 0, 0.4835), 0.022, 0.0095, "bright"
        )
        site = b.site(base, f"sample_cache_{index+1}", (x, 0, 0.497))
        records.append(
            {
                "site": site,
                "geom": _name(b, f"cache_cap_{index+1}"),
                "site_above_cap_m": 0.004,
                "approach_radius_m": 0.008,
                "approach_height_m": 0.15,
                "meaning": "Clear vertical 8 mm radius corridor above a retained inert cap; not a grasp/loading interface",
            }
        )
    b.metadata["cache_interfaces"] = records
