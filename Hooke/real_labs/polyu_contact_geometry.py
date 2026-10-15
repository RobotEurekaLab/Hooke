"""Photo-informed UMT exterior and an explicitly authored ball-on-flat fixture.

The installed photograph supports the dark rounded case, clear front, cream rear
and compact load head. Bearings, dimensions and contact tooling are estimates;
this is not a manufacturer model or a calibrated tribology instrument.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.mounting import annular_mount

PREFIX = "umt_station__polyu_"


def _glass_front(b, base):
    """One continuous curved visible surface with convex contact sectors."""
    count = 48
    vertices = []
    for z in (0.18, 0.84):
        for offset in (0, 0.003):
            for i in range(count + 1):
                a = math.pi + math.pi * i / count
                vertices.append(
                    (
                        (0.251 + offset) * math.cos(a),
                        0.02 + (0.270 + offset) * math.sin(a),
                        z,
                    )
                )
    stride = count + 1
    faces = []
    for i in range(count):
        for a, c, reverse in (
            (0, stride, False),
            (stride * 2, stride * 3, True),
            (0, stride * 2, True),
            (stride, stride * 3, False),
        ):
            triangles = ((a + i, a + i + 1, c + i + 1), (a + i, c + i + 1, c + i))
            faces.extend(tuple(reversed(f)) if reverse else f for f in triangles)
    for i in (0, count):
        triangles = (
            (i, stride + i, 3 * stride + i),
            (i, 3 * stride + i, 2 * stride + i),
        )
        faces.extend(tuple(reversed(f)) if i else f for f in triangles)
    mesh = PREFIX + "clear_front_mesh"
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for p in vertices for v in p]),
        face=" ".join(str(i) for f in faces for i in f),
        maxhullvert="4",
    )
    b.geom(base, "mesh", (), mesh=mesh, material="glass", name=PREFIX + "clear_front")
    for i in range(count):
        ids = [
            i,
            i + 1,
            stride + i,
            stride + i + 1,
            2 * stride + i,
            2 * stride + i + 1,
            3 * stride + i,
            3 * stride + i + 1,
        ]
        name = PREFIX + f"guard_contact_{i}"
        ET.SubElement(
            b.root.find("asset"),
            "mesh",
            name=name + "_mesh",
            vertex=numbers([v for j in ids for v in vertices[j]]),
        )
        b.geom(
            base,
            "mesh",
            (),
            mesh=name + "_mesh",
            material="glass",
            collision=True,
            name=name,
            group=3,
            rgba=(0.5, 0.5, 0.5, 0),
        )


def build(b, base, params):
    if params.get("interface_revision", 2) != 2:
        raise ValueError(
            "PolyU contact fixture requires interface_revision=2; replay revision1 from its archived package"
        )

    def box(parent, key, pos, half, material="metal", **kw):
        return b.box(
            parent, pos, half, material, collision=True, name=PREFIX + key, **kw
        )

    def cylinder(parent, key, pos, radius, half, material="metal", **kw):
        return b.cylinder(
            parent, pos, radius, half, material, collision=True, name=PREFIX + key, **kw
        )

    b.feet(base, 0.51, 0.57)
    b.housing(
        base,
        [(0.024, 0.25, 0.28, 0), (0.14, 0.27, 0.30, 0), (0.18, 0.25, 0.28, 0)],
        material="dark",
        radius=0.075,
    )
    base.findall("geom")[-1].set("contype", "1")
    base.findall("geom")[-1].set("conaffinity", "1")
    box(base, "base_core", (0, 0, 0.099), (0.23, 0.25, 0.075), "dark")
    box(base, "bed", (0, -0.015, 0.177), (0.21, 0.215, 0.003), "bright")
    box(base, "rear_case", (0, 0.245, 0.51), (0.24, 0.035, 0.33), "shell")
    for sign in (-1, 1):
        box(
            base,
            f"side_case_{sign}",
            (sign * 0.233, 0.118, 0.51),
            (0.018, 0.092, 0.33),
            "shell",
        )
    b.housing(
        base,
        [(0.84, 0.25, 0.28, 0), (0.94, 0.25, 0.28, 0), (0.96, 0.23, 0.26, 0)],
        material="dark",
        radius=0.05,
    )
    base.findall("geom")[-1].set("contype", "1")
    base.findall("geom")[-1].set("conaffinity", "1")
    box(base, "top_core", (0, 0, 0.892), (0.23, 0.25, 0.052), "dark")
    _glass_front(b, base)

    # Rail direction matches the original X joint; both ends bear on the bed.
    for sign in (-1, 1):
        y = -0.025 + sign * 0.08
        cylinder(
            base,
            f"x_rail_{sign}",
            (0, y, 0.193),
            0.006,
            0.162,
            "bright",
            euler=(0, math.pi / 2, 0),
        )
        for end in (-1, 1):
            box(
                base,
                f"x_foot_{sign}_{end}",
                (end * 0.174, y, 0.193),
                (0.012, 0.018, 0.013),
                "dark",
            )
    # Two source-visible low round covers: external geometry only, not motor internals.
    for sign in (-1, 1):
        cylinder(
            base,
            f"drive_cover_{sign}",
            (sign * 0.075, -0.180, 0.205),
            0.025,
            0.032,
            "metal",
            euler=(math.pi / 2, 0, 0),
        )
        cylinder(
            base,
            f"drive_face_{sign}",
            (sign * 0.075, -0.213, 0.205),
            0.021,
            0.001,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    slider = b.moving(
        base,
        "coupon_x",
        (0, -0.025, 0.229),
        (1, 0, 0),
        (-0.065, 0.065),
        mass=0.11,
        kp=240,
    )
    box(slider, "platen", (0, 0, 0), (0.09, 0.11, 0.018))
    for sign in (-1, 1):
        annular_mount(
            b,
            slider,
            f"x_bearing_{sign}",
            (0, sign * 0.08, -0.036),
            0.0066,
            0.0126,
            0.045,
            axis="x",
            proxies=True,
            namespace="polyu",
            material="dark",
        )
        box(
            slider,
            f"x_bearing_web_{sign}",
            (0, sign * 0.08, -0.021),
            (0.045, 0.008, 0.003),
        )
    cylinder(slider, "specimen_puck", (0, 0, 0.022), 0.039, 0.004, "dark")
    box(slider, "coupon", (0, 0, 0.028), (0.020, 0.020, 0.002), "sample")
    b.site(slider, "tribology_coupon", (0, 0, 0.030), 0.0005)
    # Peripheral pads retain the full-size coupon without occupying its scan path.
    for sign in (-1, 1):
        box(
            slider,
            f"retaining_post_{sign}",
            (0, sign * 0.026, 0.025),
            (0.010, 0.003, 0.007),
            "dark",
        )
        box(
            slider,
            f"retaining_pad_{sign}",
            (0, sign * 0.021, 0.031),
            (0.010, 0.008, 0.001),
            "metal",
        )

    box(base, "column", (0, 0.165, 0.51), (0.081, 0.016, 0.33), "dark")
    for sign in (-1, 1):
        x = sign * 0.044
        box(base, f"z_foot_{sign}", (x, 0.105, 0.200), (0.015, 0.015, 0.02))
        cylinder(base, f"z_guide_{sign}", (x, 0.105, 0.51), 0.006, 0.29, "bright")
        box(base, f"z_cap_{sign}", (x, 0.125, 0.812), (0.015, 0.04, 0.012))
        box(base, f"z_foot_brace_{sign}", (x, 0.135, 0.2), (0.015, 0.015, 0.02))
    head = b.moving(
        base,
        "probe_height",
        (0, -0.025, 0.62),
        (0, 0, 1),
        (-0.10, 0.08),
        mass=0.16,
        kp=280,
    )
    for sign in (-1, 1):
        annular_mount(
            b,
            head,
            f"z_bearing_{sign}",
            (sign * 0.044, 0.13, 0),
            0.0066,
            0.0126,
            0.035,
            axis="z",
            proxies=True,
            namespace="polyu",
            material="metal",
        )
    box(head, "rear_bridge", (0, 0.13, 0), (0.0314, 0.0126, 0.035))
    box(head, "cantilever", (0, 0.055, -0.012), (0.030, 0.0624, 0.022))
    box(head, "load_head", (0, 0, -0.04), (0.066, 0.037, 0.045), "metal")
    box(head, "load_cell", (0, 0, -0.125), (0.031, 0.029, 0.04), "dark")
    cylinder(head, "load_column", (0, 0, -0.2125), 0.019, 0.0475, "bright")
    cylinder(head, "collet", (0, 0, -0.273), 0.011, 0.013, "dark")
    cylinder(head, "tip_stem", (0, 0, -0.3), 0.004, 0.014, "bright")
    tip = b.geom(
        head,
        "sphere",
        (0.003,),
        (0, 0, -0.318),
        "bright",
        collision=True,
        name=PREFIX + "counterface",
    )
    # A 1 mm overlap inside the collet joins the stem and spherical insert.
    cylinder(head, "tip_socket", (0, 0, -0.314), 0.0035, 0.002, "metal")
    b.site(head, "counterface_bottom", (0, 0, -0.321), 0.0005)
    for x in (-0.048, 0, 0.048):
        cylinder(
            head,
            f"head_screw_{x}",
            (x, -0.039, -0.025),
            0.005,
            0.002,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    b.screen(base, (-0.08, -0.279, 0.892), 0.12, 0.042)
    for x, mat in ((-0.12, "led"), (-0.03, "warning"), (0.08, "guard_red")):
        cylinder(
            base,
            f"button_bezel_{x}",
            (x, -0.295, 0.125),
            0.012,
            0.006,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
        cylinder(
            base,
            f"button_{x}",
            (x, -0.300, 0.125),
            0.009,
            0.003,
            mat,
            euler=(math.pi / 2, 0, 0),
        )
    b.metadata["capabilities"] = [
        "inert_coupon_translation",
        "authored_ball_on_flat_contact",
        "measured_loaded_reciprocation",
    ]
    b.metadata["contact_interface"] = dict(
        revision=2,
        counterface=tip.get("name"),
        coupon=PREFIX + "coupon",
        counterface_site=f"{b.name}__site_counterface_bottom",
        contact_height_m=-0.04,
        contact_radius_m=0.003,
        coupon_size_m=[0.04, 0.04, 0.004],
        coordinate_migration="Joint frames, directions, zero positions, full ranges, gains and lumped inertias unchanged. Revision2 replaces the unreachable revision1 needle with a connected generic load tool; former noncontact semantics are obsolete. Legacy packages remain separately replayable.",
        source_boundary="Exterior appearance from installed PolyU photograph; internal open bearings, load tool, retention and all dimensions are authored estimates.",
        working_envelope="Full joint limits are retained, not a collision-free Cartesian box. Approach near centered coupon; loaded X travel is bounded to +/-10 mm. Lower probe commands encounter the physical coupon.",
        contact_model=dict(
            solref=[0.008, 1.0],
            friction=[0.8, 0.01, 0.001],
            interpretation="Authored numerical compliance, not a measured material law. Original 8 ms contact lost engagement at the rejected faster path. Slower smooth reciprocation preserves default contact settings, global solver, 2 ms timestep and original drives.",
        ),
        control_model="Original ideal gravity-compensated position servos and rigid retained coupon; no force sensor or force controller. Constraint forces are observed, not prescribed.",
    )
    b.metadata["limitations"].append(
        "Uncalibrated generic rigid ball-on-flat contact only. No material-specific friction, wear, deformation, calibrated normal load or manufacturer performance; clear enclosure is fixed and does not model a loading door."
    )
