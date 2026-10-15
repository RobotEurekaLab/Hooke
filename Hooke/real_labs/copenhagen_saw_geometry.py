"""Versioned, source-informed unpowered saw access with an inert retained core.

Photographs establish the arrangement, not hidden hinge/bearing dimensions.
The feed fixture and contact compliance are authored mechanical surrogates.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.copenhagen_ice_geometry import _bearing_y
from real_labs.mounting import annular_mount

PREFIX = "ice_station__pice_v2_"


def slotted_plate(b, parent, key, xs, ys, z, half, *, open_row):
    """One connected U-shaped surface and matching enabled contact cells."""
    cells = {(i, j) for i in range(3) for j in range(2)} - {(1, open_row)}
    vertices, lookup, faces = [], {}, []

    def vertex(i, j, k):
        point = (xs[i], ys[j], z + (-half, half)[k])
        if point not in lookup:
            lookup[point] = len(vertices)
            vertices.append(point)
        return lookup[point]

    def quad(points):
        ids = [vertex(*point) for point in points]
        faces.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))

    for i, j in sorted(cells):
        quad(((i, j, 1), (i + 1, j, 1), (i + 1, j + 1, 1), (i, j + 1, 1)))
        quad(((i, j, 0), (i, j + 1, 0), (i + 1, j + 1, 0), (i + 1, j, 0)))
        for neighbor, points in (
            ((i - 1, j), ((i, j, 0), (i, j, 1), (i, j + 1, 1), (i, j + 1, 0))),
            (
                (i + 1, j),
                ((i + 1, j, 0), (i + 1, j + 1, 0), (i + 1, j + 1, 1), (i + 1, j, 1)),
            ),
            ((i, j - 1), ((i, j, 0), (i + 1, j, 0), (i + 1, j, 1), (i, j, 1))),
            (
                (i, j + 1),
                ((i, j + 1, 0), (i, j + 1, 1), (i + 1, j + 1, 1), (i + 1, j + 1, 0)),
            ),
        ):
            if neighbor not in cells:
                quad(points)
        b.box(
            parent,
            ((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, z),
            ((xs[i + 1] - xs[i]) / 2, (ys[j + 1] - ys[j]) / 2, half),
            "metal",
            collision=True,
            group=3,
            name=PREFIX + key + f"_contact_{i}_{j}",
        )
    name = PREFIX + key + "_mesh"
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=numbers([v for p in vertices for v in p]),
        face=" ".join(str(v) for p in faces for v in p),
    )
    b.geom(
        parent, "mesh", (), material="metal", mesh=name, name=PREFIX + key + "_surface"
    )


def supported_guard(b, base):
    """Keep the original hinge frame/range, adding a real open pivot assembly."""
    cover = b.moving(
        base,
        "wheel_guard",
        (-0.285, -0.023, 1.48),
        (0, 0, -1),
        (0, 1.1),
        kind="hinge",
        mass=0.28,
    )
    cover.set("euler", numbers((0, 0, -0.75)))
    cover.find("joint").set("ref", ".75")
    b.box(
        cover,
        (0.278, 0.012, 0),
        (0.268, 0.015, 0.295),
        "shell",
        collision=True,
        name=PREFIX + "guard_panel",
    )
    for i, (a, c) in enumerate(
        (
            ((0.04, -0.011, -0.26), (0.51, -0.011, 0.26)),
            ((0.04, -0.011, 0.26), (0.51, -0.011, -0.26)),
        )
    ):
        b.rod(
            cover,
            a,
            c,
            0.008,
            "metal",
            collision=True,
            name=PREFIX + f"guard_brace_{i}",
        )
    b.cylinder(
        cover,
        (0.48, -0.047, 0),
        0.025,
        0.018,
        "dark",
        euler=(math.pi / 2, 0, 0),
        collision=True,
        name=PREFIX + "guard_knob",
    )
    b.rod(
        cover,
        (0.48, -0.029, 0),
        (0.48, -0.003, 0),
        0.008,
        "metal",
        collision=True,
        name=PREFIX + "guard_knob_stem",
    )
    b.cylinder(
        base,
        (-0.285, -0.023, 1.48),
        0.004,
        0.25,
        "bright",
        collision=True,
        name=PREFIX + "hinge_pin",
    )
    for sign in (-1, 1):
        z = 1.48 + sign * 0.23
        b.box(
            base,
            (-0.285, -0.004, z),
            (0.004, 0.019, 0.016),
            "metal",
            collision=True,
            name=PREFIX + f"hinge_leaf_a_{sign}",
        )
        b.box(
            base,
            (-0.273, 0.011, z),
            (0.012, 0.006, 0.016),
            "metal",
            collision=True,
            name=PREFIX + f"hinge_leaf_b_{sign}",
        )
        annular_mount(
            b,
            cover,
            f"guard_knuckle_{sign}",
            (0, 0, sign * 0.16),
            0.0046,
            0.009,
            0.022,
            axis="z",
            proxies=True,
            namespace="pice_v2",
        )
        b.box(
            cover,
            (0.010, 0.005, sign * 0.16),
            (0.005, 0.005, 0.016),
            "metal",
            collision=True,
            name=PREFIX + f"guard_moving_leaf_{sign}",
        )
    # Connect the passive upper-wheel hub to the rear housing.
    b.cylinder(
        base,
        (0, 0.12, 1.48),
        0.014,
        0.035,
        "bright",
        euler=(math.pi / 2, 0, 0),
        collision=True,
        name=PREFIX + "passive_wheel_spindle",
    )
    return cover


def feed_interface(b, base):
    """Move the unchanged-size retained dummy into a real stationary blade face."""
    # Thin vertical ribbon, not the legacy 26 mm thick bar. Dimensions inferred
    # from the photograph and selected for the explicit inert contact contract.
    b.box(
        base,
        (0.205, -0.034, 0.9595),
        (0.0004, 0.006, 0.2145),
        "bright",
        collision=True,
        name=PREFIX + "stationary_blade",
    )
    b.box(
        base,
        (0.17, 0.02, 1.11),
        (0.045, 0.07, 0.064),
        "guard_red",
        collision=True,
        name=PREFIX + "upper_blade_guard",
    )
    b.box(
        base,
        (0.205, 0.016, 1.04),
        (0.025, 0.006, 0.02),
        "guard_red",
        collision=True,
        name=PREFIX + "upper_guide_bridge",
    )
    # The blade connects to the guide at its back edge; its front contact face
    # stays exposed below the upper red guard and above the slotted worktable.
    b.box(
        base,
        (0.10, 0.10, 0.710),
        (0.15, 0.07, 0.045),
        "cream",
        collision=True,
        name=PREFIX + "lower_guide_arm",
    )
    b.box(
        base,
        (0.205, 0.015, 0.755),
        (0.025, 0.085, 0.012),
        "guard_red",
        collision=True,
        name=PREFIX + "lower_blade_guide",
    )
    slotted_plate(
        b,
        base,
        "table",
        [-0.35, 0.201, 0.209, 0.35],
        [-0.375, 0.015, 0.205],
        0.86,
        0.016,
        open_row=0,
    )
    carrier = b.moving(
        base, "core_feed_v2", (0.205, -0.225, 0.92), (0, 1, 0), (0, 0.152), mass=0.18
    )
    slotted_plate(
        b,
        carrier,
        "carrier",
        [-0.102, -0.004, 0.004, 0.102],
        [-0.064, -0.035, 0.064],
        0,
        0.014,
        open_row=1,
    )
    b.geom(
        carrier,
        "cylinder",
        (0.034, 0.066),
        (0, 0, 0.048),
        "glass",
        euler=(0, math.pi / 2, 0),
        collision=True,
        name=PREFIX + "inert_core",
    )
    b.geom(
        carrier,
        "cylinder",
        (0.026, 0.052),
        (0, 0, 0.048),
        "shell",
        euler=(0, math.pi / 2, 0),
        name=PREFIX + "core_inner_appearance",
    )
    for sign in (-1, 1):
        b.box(
            carrier,
            (sign * 0.072, 0, 0.038),
            (0.006, 0.049, 0.019),
            "blue",
            collision=True,
            name=PREFIX + f"retaining_clamp_{sign}",
        )
        b.box(
            carrier,
            (sign * 0.072, 0, 0.0165),
            (0.006, 0.049, 0.0025),
            "metal",
            collision=True,
            name=PREFIX + f"retaining_pad_{sign}",
        )
    b.site(carrier, "inert_ice_look_core", (0, 0, 0.048), 0.02)
    b.site(base, "unpowered_blade_contact", (0.205, -0.040, 0.968), 0.002)
    for i, x in enumerate((0.100, 0.300)):
        b.rod(
            base,
            (x, -0.295, 0.891),
            (x, -0.030, 0.891),
            0.007,
            "bright",
            collision=True,
            name=PREFIX + f"feed_rail_{i}",
        )
        for j, y in enumerate((-0.295, -0.030)):
            b.box(
                base,
                (x, y, 0.8855),
                (0.019, 0.012, 0.0095),
                "metal",
                collision=True,
                name=PREFIX + f"feed_rail_{i}_support_{j}",
            )
        _bearing_y(b, carrier, f"v2_bearing_{i}", x - 0.205)
    b.metadata["mechanical_interface"] = {
        "version": 2,
        "supersedes_control": "inert_core_carriage",
        "active_feed_control": "core_feed_v2",
        "old_frame_m": [-0.115, -0.22, 0.92],
        "old_range_m": [-0.075, 0.075],
        "new_frame_m": [0.205, -0.225, 0.92],
        "new_range_m": [0, 0.152],
        "migration": "No coordinate remapping; the old separated carriage is available only with interface_revision=1. Version 2 supplies a physically aligned new feed control.",
        "rationale": "Y-only travel could not overcome the original 320 mm X centerline mismatch. The unchanged-size inert core is now aligned with an unpowered blade contact face.",
        "core_dimensions_m": [0.132, 0.068, 0.068],
        "blade_dimensions_m": [0.0008, 0.012, 0.429],
        "nominal_contact_feed_m": 0.151,
        "maximum_commanded_compliance_overtravel_m": 0.001,
        "table_and_carrier_kerf_width_m": 0.008,
        "guard_hinge_radial_clearance_m": 0.0006,
        "physical_scope": "Retained rigid dummy approaches and loads a stationary blade surrogate; no cutting, blade drive, fracture, ice properties or refrigeration.",
        "source_boundary": "Source photographs establish the saw/guide/table arrangement; all hidden mount dimensions and feed mechanism are authored estimates.",
    }
    return carrier
