"""Original exterior estimates from the Kyoto installed TIRF photograph.

The visible housing arrangement informs these shapes, not manufacturer CAD.
Hidden supports, dimensions and optical paths remain authored approximations.
"""

import math
import xml.etree.ElementTree as ET


KYOTO_TIRF_PHOTO = (
    "https://www.mbsys.me.kyoto-u.ac.jp/wp-content/uploads/2023/09/"
    "Nano3-1-scaled.jpg"
)


def _shell(builder, parent, key, position, levels):
    body = ET.SubElement(
        parent,
        "body",
        name=f"{builder.name}__reference_{key}",
        pos=" ".join(str(value) for value in position),
    )
    builder.housing(body, levels, radius=0.013, material="cream")
    # Each shell is convex and lies behind the sample travel envelope. Its
    # actual exterior participates in contact instead of relying on no-contact
    # cosmetic geometry to conceal interference with a moving stage.
    geom = body.find("geom")
    geom.set("name", f"{builder.name}__reference_{key}_shell")
    geom.set("contype", "1")
    geom.set("conaffinity", "1")


def _beam(builder, parent, key, first, second, half_width, half_depth):
    direction = tuple(b - a for a, b in zip(first, second))
    builder.box(
        parent,
        tuple((a + b) / 2 for a, b in zip(first, second)),
        (half_width, half_depth, math.sqrt(sum(v * v for v in direction)) / 2),
        "cream",
        zaxis=direction,
        collision=True,
        name=f"{builder.name}__reference_{key}",
    )


def kyoto_microscope_shell(builder, parent):
    """Add fixed, supported housings around the unchanged sample-stage frame."""
    b = builder

    def name(key):
        return f"{b.name}__reference_{key}"

    b.box(
        parent,
        (0, 0, 0.035),
        (0.20, 0.26, 0.035),
        "cream",
        collision=True,
        name=name("foot"),
    )
    # Retain the previous rear structural contact column and its footprint.
    b.box(
        parent,
        (0, 0.18, 0.26),
        (0.075, 0.07, 0.20),
        "cream",
        collision=True,
        name=name("rear_support"),
    )
    _shell(
        b,
        parent,
        "lower_body",
        (0, 0, 0),
        [
            (0.065, 0.18, 0.115, 0.11),
            (0.18, 0.16, 0.080, 0.15),
            (0.29, 0.145, 0.078, 0.158),
            (0.35, 0.11, 0.063, 0.175),
            (0.43, 0.085, 0.065, 0.178),
        ],
    )

    # A broad hinge and solid slanted casting carry the upper illumination
    # assembly. It is separate from the right-hand binocular/photo head.
    b.cylinder(
        parent,
        (-0.125, 0.17, 0.27),
        0.047,
        0.032,
        "cream",
        euler=(0, math.pi / 2, 0),
        collision=True,
        name=name("arm_hinge"),
    )
    b.cylinder(
        parent,
        (-0.160, 0.17, 0.27),
        0.028,
        0.006,
        "dark",
        euler=(0, math.pi / 2, 0),
        collision=True,
        name=name("hinge_cap"),
    )
    _beam(
        b,
        parent,
        "slanted_arm",
        (-0.125, 0.17, 0.27),
        (-0.125, 0.32, 0.65),
        0.031,
        0.035,
    )
    b.box(
        parent,
        (-0.125, 0.30, 0.698),
        (0.078, 0.095, 0.052),
        "cream",
        euler=(-0.35, 0, 0),
        collision=True,
        name=name("illumination_housing"),
    )
    b.cylinder(
        parent,
        (-0.125, 0.192, 0.660),
        0.052,
        0.033,
        "dark",
        euler=(math.pi / 2 - 0.35, 0, 0),
        collision=True,
        name=name("illumination_adapter"),
    )
    for z in (0.685, 0.701, 0.717):
        b.box(parent, (-0.204, 0.30, z), (0.0008, 0.060, 0.002), "dark")

    # The arm carries the fixed condenser on an attached lateral bracket.
    # Its optical performance and hidden internal light path are not modeled.
    _beam(
        b,
        parent,
        "condenser_crosshead",
        (-0.125, 0.265, 0.52),
        (0, 0.18, 0.52),
        0.020,
        0.025,
    )
    _beam(b, parent, "condenser_reach", (0, 0.18, 0.52), (0, -0.09, 0.52), 0.022, 0.025)
    b.cylinder(
        parent,
        (0, -0.09, 0.47),
        0.028,
        0.047,
        "dark",
        collision=True,
        name=name("condenser"),
    )

    _shell(
        b,
        parent,
        "viewing_head",
        (0.065, 0.055, 0.415),
        [
            (0, 0.062, 0.065, 0.050),
            (0.055, 0.100, 0.080, 0.025),
            (0.078, 0.090, 0.065, 0.035),
        ],
    )
    for side in (-1, 1):
        start = (0.065 + side * 0.029, 0.044, 0.475)
        end = (0.065 + side * 0.044, -0.076, 0.525)
        b.rod(
            parent,
            start,
            end,
            0.022,
            "dark",
            collision=True,
            name=name(f"eyepiece_tube_{side}"),
        )
        b.rod(
            parent,
            end,
            (end[0], -0.105, 0.538),
            0.026,
            "rubber",
            collision=True,
            name=name(f"eyecup_{side}"),
        )
    for key, z, radius, half, material in (
        ("photo_tube", 0.550, 0.033, 0.065, "dark"),
        ("camera_collar", 0.621, 0.040, 0.014, "cream"),
        ("camera_body", 0.666, 0.038, 0.034, "dark"),
    ):
        b.cylinder(
            parent,
            (0.065, 0.105, z),
            radius,
            half,
            material,
            collision=True,
            name=name(key),
        )

    # A large side port aligns with the pre-existing external optical tube.
    # Its rearward offset leaves the entire XY specimen envelope unobstructed.
    for key, x, radius, half, material in (
        ("port_flange", -0.151, 0.066, 0.014, "dark"),
        ("port_barrel", -0.207, 0.054, 0.047, "dark"),
        ("port_adapter", -0.274, 0.049, 0.021, "metal"),
    ):
        b.cylinder(
            parent,
            (x, 0.080, 0.254),
            radius,
            half,
            material,
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=name(key),
        )
    b.box(
        parent,
        (-0.329, 0.080, 0.254),
        (0.035, 0.061, 0.046),
        "dark",
        collision=True,
        name=name("side_camera"),
    )
    # Focus controls sit on the opposite side rather than inside the camera port.
    for key, x, radius, half in (
        ("focus_coarse", 0.171, 0.043, 0.018),
        ("focus_fine", 0.194, 0.024, 0.007),
    ):
        b.cylinder(
            parent,
            (x, 0.120, 0.160),
            radius,
            half,
            "dark",
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=name(key),
        )

    # Connect the new camera to the existing station-level service-lead start.
    b.rod(parent, (0.065, 0.105, 0.697), (0, 0.18, 0.754), 0.004, "rubber")
    b.metadata["visual_reference_details"] = {
        "source_url": KYOTO_TIRF_PHOTO,
        "modeled_cues": [
            "Separate slanted illumination arm and binocular/photo head",
            "Broad lower casting, asymmetric side camera port and focus controls",
            "Continuous fixed hinge, bracket and housing supports",
        ],
        "dimension_basis": "Estimated exterior proportions; no OEM drawings or surveyed dimensions",
        "hidden_geometry": "Support attachments, internal optical path and unseen surfaces are authored estimates",
        "collision_boundary": "Primary housings, arm, camera barrels and viewing head have contact geometry; vents and cables are appearance details",
        "optical_simulation": False,
    }
