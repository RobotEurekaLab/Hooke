"""Original inspection-head estimates informed by the Yale cleanroom photograph.

The partially occluded source establishes exterior cues, not an instrument
identity, dimensions, hidden optical paths or the authored stage's motorization.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers


REFERENCE_PHOTO = (
    "https://news.yale.edu/sites/default/files/ynews-12-03-52_r5a_0133.jpg"
)


def _name(b, key):
    return f"{b.name}__yale_{key}"


def _mesh(b, key, vertices, faces=None):
    attributes = dict(
        name=_name(b, key), vertex=numbers([v for p in vertices for v in p])
    )
    if faces is not None:
        attributes["face"] = " ".join(str(v) for f in faces for v in f)
    ET.SubElement(b.root.find("asset"), "mesh", **attributes)
    return attributes["name"]


def _casting(b, parent, key, levels):
    """A closed rounded casting; its convex exterior also participates in contact."""
    sides = 64
    vertices = []
    for z, width, depth, center_y in levels:
        for i in range(sides):
            angle = i * math.tau / sides
            cx, sy = math.cos(angle), math.sin(angle)
            vertices.append(
                (
                    width * math.copysign(abs(cx) ** 0.8, cx),
                    center_y + depth * math.copysign(abs(sy) ** 0.8, sy),
                    z,
                )
            )
    faces = []
    for level in range(len(levels) - 1):
        first, second = level * sides, (level + 1) * sides
        for i in range(sides):
            j = (i + 1) % sides
            faces.extend(
                (
                    (first + i, first + j, second + j),
                    (first + i, second + j, second + i),
                )
            )
    for ring, orientation in ((0, -1), ((len(levels) - 1) * sides, 1)):
        center = len(vertices)
        z, _, _, y = levels[0 if orientation == -1 else -1]
        vertices.append((0, y, z))
        for i in range(sides):
            a, c = ring + i, ring + (i + 1) % sides
            faces.append((center, c, a) if orientation == -1 else (center, a, c))
    mesh = _mesh(b, key + "_mesh", vertices, faces)
    b.geom(
        parent,
        "mesh",
        (),
        material="cream",
        mesh=mesh,
        collision=True,
        name=_name(b, key),
    )


def _annular_geometry(b, key, profile, inner_radius):
    """One watertight visual annulus and convex wall sectors with the same bore."""
    if min(r for _, r in profile) <= inner_radius:
        raise ValueError("Optical sleeve must have positive wall thickness")
    sides = 32
    section = tuple(profile) + (
        (profile[-1][0], inner_radius),
        (profile[0][0], inner_radius),
    )
    vertices = [
        (r * math.cos(i * math.tau / sides), r * math.sin(i * math.tau / sides), z)
        for z, r in section
        for i in range(sides)
    ]
    faces = []
    for level in range(len(section)):
        first, second = level * sides, (level + 1) % len(section) * sides
        for i in range(sides):
            j = (i + 1) % sides
            faces.extend(
                (
                    (first + i, first + j, second + j),
                    (first + i, second + j, second + i),
                )
            )
    visual = _mesh(b, key + "_mesh", vertices, faces)
    contacts = []
    for i in range(sides):
        points = [
            (r * math.cos(angle), r * math.sin(angle), z)
            for angle in (i * math.tau / sides, (i + 1) * math.tau / sides)
            for z, r in section
        ]
        contacts.append(_mesh(b, f"{key}_contact_mesh_{i}", points))
    return visual, contacts


def _annulus(b, parent, key, geometry, position, axis, material):
    visual, contacts = geometry
    b.geom(
        parent,
        "mesh",
        (),
        position,
        material,
        mesh=visual,
        zaxis=axis,
        name=_name(b, key),
        density=0,
    )
    for i, mesh in enumerate(contacts):
        b.geom(
            parent,
            "mesh",
            (),
            position,
            material,
            mesh=mesh,
            zaxis=axis,
            collision=True,
            group=3,
            rgba=(0, 0, 0, 0),
            density=0,
            name=_name(b, f"{key}_contact_{i}"),
        )


def retaining_pads(b, stage):
    """Original rigid retaining fingers meet the unchanged coupon's upper face."""
    for side in (-1, 1):
        x = side * 0.060
        b.box(
            stage,
            (x, 0, 0.016),
            (0.014, 0.012, 0.008),
            "orange",
            collision=True,
            name=_name(b, f"clamp_block_{side}"),
        )
        b.rod(
            stage,
            (x, 0, 0.024),
            (side * 0.040, 0, 0.016),
            0.003,
            "bright",
            collision=True,
            name=_name(b, f"clamp_finger_{side}"),
        )
        b.box(
            stage,
            (side * 0.040, 0, 0.0135),
            (0.006, 0.006, 0.0015),
            "rubber",
            collision=True,
            name=_name(b, f"retaining_pad_{side}"),
        )
        b.cylinder(stage, (x, 0, 0.025), 0.004, 0.002, "metal")


def stand(b, base):
    """Rounded estimated stand, with the retained focus datum and clear stage bay."""
    b.box(base, (0, 0, .018), (.25, .21, .018), "cream", collision=True)
    # The stage's complete Y range reaches y=.110 m. The rounded casting's
    # forward surface starts at .118 m, retaining a real 8 mm travel clearance.
    _casting(b, base, "stand_casting", (
        (.036, .057, .042, .160),
        (.060, .057, .042, .160),
        (.280, .052, .040, .160),
        (.420, .047, .037, .160),
        (.442, .042, .033, .160),
    ))
    # The narrow rear guide stays separate from the cream housing. Its partly
    # concealed shape is an estimate, not an OEM mechanical reconstruction.
    b.box(base, (0, .151, .29), (.018, .050, .19), "metal")


def inspection_head(b, parent):
    """Separate rounded head, lower barrel and true open-ended eyepiece sleeves."""
    # The visible source objective and eyepieces are dark, with subtle neutral
    # coatings. Do not render those recessed faces as bright cyan glass.
    lens = b.root.find(f"asset/material[@name='{b.name}__mat_lens']")
    lens.set("rgba", ".055 .065 .060 1")
    _casting(
        b,
        parent,
        "main_casting",
        (
            (0.044, 0.043, 0.046, 0.015),
            (0.050, 0.058, 0.059, 0.008),
            (0.062, 0.074, 0.068, 0.004),
            (0.090, 0.077, 0.067, 0.006),
            (0.116, 0.073, 0.061, 0.011),
            (0.135, 0.062, 0.051, 0.018),
            (0.146, 0.048, 0.039, 0.022),
        ),
    )
    _casting(
        b,
        parent,
        "binocular_housing",
        (
            (0.132, 0.050, 0.030, -0.033),
            (0.140, 0.070, 0.039, -0.038),
            (0.150, 0.077, 0.042, -0.043),
            (0.166, 0.072, 0.042, -0.048),
            (0.181, 0.060, 0.035, -0.050),
            (0.186, 0.050, 0.029, -0.050),
        ),
    )
    b.cylinder(
        parent,
        (0, 0, 0.030),
        0.050,
        0.032,
        "cream",
        collision=True,
        name=_name(b, "lower_barrel"),
    )
    b.cylinder(
        parent, (0, 0, 0.035), 0.052, 0.004, "metal", name=_name(b, "barrel_seam")
    )
    objective = _annular_geometry(
        b,
        "objective_collar",
        ((-0.037, 0.057), (-0.031, 0.060), (-0.006, 0.059), (-0.001, 0.054)),
        inner_radius=0.044,
    )
    _annulus(b, parent, "objective_collar", objective, (0, 0, 0), (0, 0, 1), "dark")
    b.cylinder(
        parent,
        (0, 0, -0.031),
        0.045,
        0.004,
        "lens",
        collision=True,
        name=_name(b, "objective_lens"),
    )

    axis = (0, -math.sin(0.9), math.cos(0.9))
    cuff = _annular_geometry(
        b,
        "eyepiece_cuff",
        ((-0.034, 0.020), (-0.028, 0.024), (-0.004, 0.025), (0, 0.0245)),
        inner_radius=0.0178,
    )
    for side in (-1, 1):
        mouth = (side * 0.042, -0.137, 0.231)
        center = tuple(v - 0.076 * a for v, a in zip(mouth, axis))
        b.cylinder(
            parent,
            center,
            0.021,
            0.048,
            "cream",
            zaxis=axis,
            collision=True,
            name=_name(b, f"eyepiece_stem_{side}"),
        )
        _annulus(b, parent, f"eyepiece_cuff_{side}", cuff, mouth, axis, "rubber")
        lens = tuple(v - 0.006 * a for v, a in zip(mouth, axis))
        b.cylinder(
            parent,
            lens,
            0.0184,
            0.0015,
            "lens",
            zaxis=axis,
            collision=True,
            name=_name(b, f"eyepiece_lens_{side}"),
        )
        b.cylinder(
            parent,
            (side * 0.082, 0.006, 0.090),
            0.019,
            0.010,
            "metal",
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=_name(b, f"knob_spindle_{side}"),
        )
        b.cylinder(
            parent,
            (side * 0.096, 0.006, 0.090),
            0.029,
            0.012,
            "dark",
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=_name(b, f"zoom_knob_{side}"),
        )
        b.cylinder(
            parent,
            (side * 0.109, 0.006, 0.090),
            0.020,
            0.0015,
            "rubber",
            euler=(0, math.pi / 2, 0),
        )
    b.metadata["visual_reference_details"] = {
        "source": REFERENCE_PHOTO,
        "basis": "Original exterior estimates from a partly blurred and occluded photograph; no identifiable OEM model or CAD.",
        "observed_features": [
            "Rounded central casting and distinct binocular housing",
            "Cream lower barrel with a broad dark objective collar",
            "Inclined dark eyepiece tubes and large lateral knobs",
            "Green work plate and separate orange-ended guide fixture",
        ],
        "estimated_geometry": "Dimensions, rear stand, internal lens geometry and hidden mounts remain estimates.",
        "optics": "Open eyepiece sleeves, recessed lens surfaces and matching wall contacts; no calibrated magnification or optical image formation.",
        "fixture": "The rigid retaining pads and motorized XY/focus motions are authored surrogates. The coupon remains kinematically clamped; no clamp-force or wafer metrology model.",
    }


def guide_details(b, base):
    """Visible ferrules/fasteners join the existing separate handling guide."""
    for side in (-1, 1):
        b.cylinder(
            base,
            (side * 0.204, -0.165, 0.108),
            0.013,
            0.009,
            "metal",
            euler=(0, math.pi / 2, 0),
        )
        for z in (0.062, 0.088):
            b.cylinder(
                base,
                (side * 0.21, -0.188, z),
                0.003,
                0.0015,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
