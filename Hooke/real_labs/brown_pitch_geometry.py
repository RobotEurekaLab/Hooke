"""Connected original dry positioner inside the documented Brown test section.

Only the facility's clear test-section dimensions and existence of pitch/yaw
positioning are source-established. These hidden bearings are authored.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount

PREFIX = "afam_test_section__brown_"


def _box(b, body, key, position, half, material="metal"):
    return b.box(body, position, half, material, collision=True, name=PREFIX + key)


def _bearing(b, body, key, center, inner, outer, half, axis):
    # Put visual shell and open contact sectors directly in the original body;
    # do not infer extra inertia by adding an unbudgeted fixed child body.
    holder = ET.Element("body")
    annular_mount(
        b,
        holder,
        key,
        (0, 0, 0),
        inner,
        outer,
        half,
        axis="z" if axis == "z" else "x",
        proxies=True,
        namespace="brown",
        material="dark",
    )
    for geom in holder:
        x, y, z = map(float, geom.get("pos", "0 0 0").split())
        if axis == "y":
            x, y = -y, x
            angle = float(geom.attrib.pop("euler", "0 0 0").split()[0]) / 2
            c, s = math.cos(angle) / math.sqrt(2), math.sin(angle) / math.sqrt(2)
            geom.set("quat", f"{c} {s} {s} {c}")
        geom.set("pos", " ".join(str(a + v) for a, v in zip(center, (x, y, z))))
        body.append(geom)


def add_positioner(b, base):
    _box(b, base, "pedestal_foot", (0, 0, 0.83), (0.15, 0.15, 0.03))
    b.cylinder(
        base,
        (0, 0, 1.065),
        0.033,
        0.205,
        "bright",
        collision=True,
        name=PREFIX + "pedestal_column",
    )
    _bearing(b, base, "yaw_bearing", (0, 0, 1.294), 0.0126, 0.080, 0.024, "z")
    yaw = b.moving(
        base,
        "model_yaw",
        (0, 0, 1.345),
        (0, 0, 1),
        (-0.45, 0.45),
        kind="hinge",
        mass=0.20,
        kp=160,
    )
    b.cylinder(
        yaw, (0, 0, 0), 0.078, 0.025, "metal", collision=True, name=PREFIX + "yaw_plate"
    )
    # Shaft stops 0.6 mm above the solid pedestal and overlaps its own plate.
    b.cylinder(
        yaw,
        (0, 0, -0.0467),
        0.012,
        0.0277,
        "bright",
        collision=True,
        name=PREFIX + "yaw_spindle",
    )
    for sign in (-1, 1):
        y = sign * 0.087
        _box(b, yaw, f"fork_post_{sign}", (0, y, 0.051), (0.031, 0.016, 0.026), "dark")
        _bearing(
            b, yaw, f"pitch_bearing_{sign}", (0, y, 0.100), 0.0126, 0.027, 0.012, "y"
        )
    pitch = b.moving(
        yaw,
        "model_pitch",
        (0, 0, 0.100),
        (0, 1, 0),
        (-0.32, 0.32),
        kind="hinge",
        mass=0.08,
        kp=100,
    )
    b.cylinder(
        pitch,
        (0, 0, 0),
        0.012,
        0.10,
        "bright",
        euler=(math.pi / 2, 0, 0),
        collision=True,
        name=PREFIX + "pitch_shaft",
    )
    for sign in (-1, 1):
        b.cylinder(
            pitch,
            (0, sign * 0.105, 0),
            0.018,
            0.005,
            "metal",
            euler=(math.pi / 2, 0, 0),
            collision=True,
            name=PREFIX + f"shaft_retainer_{sign}",
        )
    b.geom(
        pitch,
        "ellipsoid",
        (0.28, 0.055, 0.014),
        (0, 0, 0.040),
        "orange",
        collision=True,
        name=PREFIX + "inert_coupon",
    )
    _box(b, pitch, "coupon_mount", (0, 0, 0.020), (0.025, 0.025, 0.020))
    b.site(pitch, "aerodynamic_coupon", (0, 0, 0.055))
    # Actual body axes, rather than commands, define dry orientation telemetry.
    b.site(pitch, "coupon_center", (0, 0, 0.040), 0.002)
    b.metadata["positioner_interface"] = dict(
        revision=2,
        coordinate_migration="None: original joint frames, angle zero, ranges, masses, inertias and drives retained.",
        yaw_range_rad=[-0.45, 0.45],
        pitch_range_rad=[-0.32, 0.32],
        radial_bearing_clearance_m=0.0006,
        yaw_spindle_to_pedestal_clearance_m=0.0006,
        pitch_retainer_axial_clearance_m=0.001,
        sample="Original clamped 560 x 110 x 28 mm inert ellipsoid",
        reference_axis="Authored longitudinal tunnel X axis; no measured wind direction or aerodynamic angle calibration",
        source_boundary="Facility documents pitch/yaw positioning. Forks, open bearings, shaft retainers and ideal joint load support are original estimates, not recovered positioner CAD.",
        scope="Dry inert coupon orientation and return; no loading, airflow, animal experiment or aerodynamic response.",
    )
