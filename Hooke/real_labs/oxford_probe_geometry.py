"""Estimated split guides and asymmetric shafts for Oxford's probe vignette.

Only the broad foreground shaft, slimmer rear holder and small clamp details
are photo-supported. The slide bearings, scale and motorization are authored
mechanical interfaces. These tips intentionally never touch the specimen.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers


def _brass(b):
    name = f"{b.name}__mat_probe_brass"
    if b.root.find(f'./asset/material[@name="{name}"]') is None:
        ET.SubElement(
            b.root.find("asset"),
            "material",
            name=name,
            rgba=".49 .41 .21 1",
            specular=".5",
            shininess=".5",
        )


def _split_guide(b, arm, side):
    """A C-shaped collar has a real 8.6 mm bore around the 8 mm guide post."""
    inner, outer = 0.0086, 0.015 if side > 0 else 0.013
    half, z = (0.014, -0.008) if side > 0 else (0.012, 0.006)
    # The 18 degree slit faces rearward, away from the front shaft bracket.
    angles = [math.pi / 2 + 0.15708 + (math.tau - 0.31416) * i / 40 for i in range(41)]
    vertices, faces = [], []
    for height in (z - half, z + half):
        for radius in (outer, inner):
            vertices.extend(
                (radius * math.cos(a), radius * math.sin(a), height) for a in angles
            )
    n = len(angles)
    for i in range(n - 1):
        for a, c, d, e in (
            (i, i + 1, 2 * n + i + 1, 2 * n + i),
            (n + i + 1, n + i, 3 * n + i, 3 * n + i + 1),
            (i + 1, i, n + i, n + i + 1),
            (2 * n + i, 2 * n + i + 1, 3 * n + i + 1, 3 * n + i),
        ):
            faces.extend(((a, c, d), (a, d, e)))
    for i, reverse in ((0, False), (n - 1, True)):
        quad = (i, 2 * n + i, 3 * n + i, n + i)
        for face in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
            faces.append(tuple(reversed(face)) if reverse else face)
    mesh = b.unique("probe_open_guide_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for p in vertices for v in p]),
        face=" ".join(str(v) for f in faces for v in f),
    )
    b.geom(
        arm, "mesh", (), material="metal", mesh=mesh, name=b.unique("probe_open_guide")
    )
    # Narrow convex wall cells preserve the visible bore and rear slit. The
    # inner flat face stays outside the guide post even at intermediate travel.
    for a0, a1 in zip(angles, angles[1:]):
        angle = (a0 + a1) / 2
        radius = (inner + outer) / 2
        b.box(
            arm,
            (radius * math.cos(angle), radius * math.sin(angle), z),
            ((outer - inner) / 2, outer * math.tan((a1 - a0) / 2), half),
            "metal",
            euler=(0, 0, angle),
            collision=True,
            group=3,
            rgba=(0, 0, 0, 0),
            name=b.unique("probe_guide_contact"),
        )
    b.box(
        arm,
        (0, -0.017, -0.003),
        (0.007, 0.007, 0.010),
        "metal",
        collision=True,
        name=b.unique("probe_shaft_mount"),
    )
    # Visible split-clamp ears and fastener are outside the through-bore.
    for x in (-0.006, 0.006):
        b.box(
            arm,
            (x, 0.014, z),
            (0.003, 0.004, 0.005),
            "metal",
            collision=True,
            name=b.unique("probe_clamp_ear"),
        )
    b.cylinder(
        arm,
        (0.010, 0.014, z),
        0.003,
        0.002,
        "dark",
        euler=(0, math.pi / 2, 0),
        collision=True,
        name=b.unique("probe_clamp_screw"),
    )


def add_probe_holder(b, arm, side):
    """Retain original fine-tip endpoints while distinguishing the two holders."""
    _brass(b)
    _split_guide(b, arm, side)
    if side > 0:
        start = (0, -0.020, -0.006)
        shoulder = (-0.073, -0.061, -0.014)
        b.rod(
            arm,
            start,
            shoulder,
            0.008,
            "probe_brass",
            collision=True,
            name=b.unique("probe_broad_foreground_shaft"),
        )
        # A shorter sleeve joins the larger shaft to the fine bent holder.
        sleeve = (-0.067, -0.0576, -0.0133)
        b.rod(
            arm,
            sleeve,
            shoulder,
            0.010,
            "metal",
            collision=True,
            name=b.unique("probe_foreground_collar"),
        )
        b.rod(
            arm,
            shoulder,
            (-0.142, -0.079, -0.026),
            0.0015,
            "dark",
            collision=True,
            name=b.unique("probe_fine_holder"),
        )
    else:
        start = (0, -0.017, 0.002)
        shoulder = (0.079, -0.034, -0.009)
        elbow = (0.102, -0.055, -0.020)
        b.rod(
            arm,
            start,
            shoulder,
            0.0045,
            "probe_brass",
            collision=True,
            name=b.unique("probe_rear_brass_shaft"),
        )
        b.rod(
            arm,
            (0.071, -0.0323, -0.0079),
            shoulder,
            0.006,
            "metal",
            collision=True,
            name=b.unique("probe_rear_collar"),
        )
        for a, c in ((shoulder, elbow), (elbow, (0.142, -0.079, -0.026))):
            b.rod(
                arm,
                a,
                c,
                0.0015,
                "dark",
                collision=True,
                name=b.unique("probe_fine_holder"),
            )
    # Keep both endpoints and the 120 micrometre tip radius exactly unchanged.
    b.rod(
        arm,
        (-side * 0.142, -0.079, -0.026),
        (-side * 0.151, -0.082, -0.030),
        0.00012,
        "bright",
        collision=True,
        name=b.unique("probe_fine_tip"),
    )
