"""Source-informed glovebox appearance and explicitly inferred mounting hardware.

The ETH photograph establishes the shaped pale face, dark tapered sleeves and
visible frame fasteners. Bearings, guide pedestals and the pass-through hinge
are original engineering details, not recovered OEM geometry. The gloves remain
fixed presentation meshes; no atmosphere, containment or glove mechanics exist.
"""

import math
from functools import partial
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.mounting import annular_mount


RUNNING_CLEARANCE_M = 0.0005


def name(b, key):
    return f"{b.name}__glovebox_{key}"


def _mesh(b, parent, key, vertices, faces, material):
    mesh = name(b, key + "_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for point in vertices for v in point]),
        face=" ".join(str(i) for face in faces for i in face),
    )
    return b.geom(parent, "mesh", (), material=material, mesh=mesh, name=name(b, key))


_annulus = partial(annular_mount, namespace="glovebox")


def add_carriage_support(b, base, carriage):
    """Bridge the 21 mm deck gap and the 3 mm tray gap without filling the bore."""
    for row, y in enumerate((-0.15, 0.15)):
        for end, x in enumerate((-0.51, 0.51)):
            b.box(
                base,
                (x, y, 0.9755),
                (0.021, 0.020, 0.0105),
                "metal",
                collision=True,
                name=name(b, f"rail_{row}_pedestal_{end}"),
            )
            for side in (-1, 1):
                b.box(
                    base,
                    (x, y + side * 0.016, 0.984),
                    (0.021, 0.006, 0.019),
                    "metal",
                    collision=True,
                    name=name(b, f"rail_{row}_clamp_{end}_{side}"),
                )
        for index, x in enumerate((-0.135, 0.135)):
            _annulus(
                b,
                carriage,
                f"bearing_{row}_{index}",
                (x, y, -0.038),
                0.011 + RUNNING_CLEARANCE_M,
                0.020,
                0.038,
                half=True,
                proxies=True,
                material="metal",
            )
            b.cylinder(
                carriage,
                (x, y, 0.0255),
                0.006,
                0.0015,
                "bright",
                name=name(b, f"bearing_{row}_{index}_fastener"),
            )
    b.box(
        base,
        (-0.82, 0.2, 2.1225),
        (0.215, 0.265, 0.0025),
        "rubber",
        collision=True,
        name=name(b, "upper_accessory_pad"),
    )


def add_passage_details(b, base, lid):
    """Smooth appearance over the existing open collision rods; inferred hinge."""
    _annulus(
        b, base, "barrel", (1.6775, 0, 1.36), 0.216, 0.260, 0.3775, material="metal"
    )
    for index, x in enumerate((1.34, 2.055)):
        _annulus(b, base, f"flange_{index}", (x, 0, 1.36), 0.222, 0.276, 0.027)
        for bolt in range(12):
            angle = math.tau * bolt / 12
            b.cylinder(
                base,
                (x + 0.0285, 0.258 * math.cos(angle), 1.36 + 0.258 * math.sin(angle)),
                0.005,
                0.0015,
                "dark",
                euler=(0, math.pi / 2, 0),
                name=name(b, f"flange_{index}_fastener_{bolt}"),
            )
    # The bracket reaches the outer barrel wall. Upper/lower lugs lie outside
    # the lid's swept edge; an annular moving hub leaves 0.5 mm around the pin.
    b.box(
        base,
        (2.08, -0.277, 1.36),
        (0.040, 0.023, 0.12),
        "metal",
        collision=True,
        name=name(b, "hinge_mount"),
    )
    for side in (-1, 1):
        z = 1.36 + side * 0.10
        b.rod(
            base,
            (2.105, -0.277, z),
            (2.105, -0.23, z),
            0.009,
            "bright",
            collision=True,
            name=name(b, f"hinge_lug_{side}"),
        )
    b.cylinder(
        base,
        (2.105, -0.23, 1.36),
        0.004,
        0.109,
        "bright",
        collision=True,
        name=name(b, "hinge_pin"),
    )
    _annulus(
        b,
        lid,
        "hinge_hub",
        (0, 0, 0),
        0.0045,
        0.011,
        0.018,
        axis="z",
        proxies=True,
        material="metal",
    )


def _glove_digit(b, base, key, x, sections):
    """Closed, tapered lobe with a relaxed curve and blunt fingertip."""
    count = 24
    vertices = [
        (x + dx + rx * math.cos(a), y, z + rz * math.sin(a))
        for dx, y, z, rx, rz in sections
        for a in (math.tau * i / count for i in range(count))
    ]
    faces = []
    for row in range(len(sections) - 1):
        for column in range(count):
            a, c = row * count + column, row * count + (column + 1) % count
            faces.extend(((a, c, c + count), (a, c + count, a + count)))
    for row in (0, len(sections) - 1):
        dx, y, z, _, _ = sections[row]
        center = len(vertices)
        vertices.append((x + dx, y, z))
        for column in range(count):
            a, c = row * count + column, row * count + (column + 1) % count
            faces.append((center, c, a) if row == 0 else (center, a, c))
    _mesh(b, base, key, vertices, faces, "rubber")


def add_gloves(b, base):
    """Tapered, relaxed presentation sleeves; no new joints or collision shapes."""
    for index, x in enumerate((-0.60, 0.60)):
        b.cylinder(
            base,
            (x, -0.57, 1.44),
            0.145,
            0.069,
            "metal",
            euler=(math.pi / 2, 0, 0),
            name=name(b, f"port_{index}"),
        )
        b.ring(base, (x, -0.645, 1.44), 0.14, 0.012, "bright", plane="xz")
        for bolt in range(8):
            angle = math.tau * (bolt + 0.5) / 8
            b.cylinder(
                base,
                (x + 0.136 * math.cos(angle), -0.660, 1.44 + 0.136 * math.sin(angle)),
                0.004,
                0.002,
                "dark",
                euler=(math.pi / 2, 0, 0),
                name=name(b, f"port_{index}_fastener_{bolt}"),
            )
        # y, z, lateral radius, vertical radius: irregular taper, wrist and
        # flattened terminal shape all stay inside the previous glove envelope.
        sections = (
            (-0.632, 1.44, 0.118, 0.116),
            (-0.67, 1.434, 0.112, 0.107),
            (-0.72, 1.425, 0.105, 0.101),
            (-0.79, 1.410, 0.096, 0.089),
            (-0.87, 1.395, 0.087, 0.077),
            (-0.96, 1.372, 0.073, 0.060),
            (-1.04, 1.345, 0.051, 0.038),
            (-1.09, 1.332, 0.060, 0.030),
            (-1.13, 1.327, 0.057, 0.021),
            (-1.15, 1.325, 0.047, 0.015),
        )
        count = 48
        vertices = []
        for row, (y, z, rx, rz) in enumerate(sections):
            for column in range(count):
                angle = math.tau * column / count
                fold = 1 + 0.025 * math.sin(5 * angle + row * 1.7)
                vertices.append(
                    (
                        x + rx * math.cos(angle) * fold,
                        y + (0.003 * math.cos(4 * angle) if row > 7 else 0),
                        z + rz * math.sin(angle) * fold,
                    )
                )
        faces = []
        for row in range(len(sections) - 1):
            for column in range(count):
                a, c = row * count + column, row * count + (column + 1) % count
                faces.extend(((a, c, c + count), (a, c + count, a + count)))
        for row in (0, len(sections) - 1):
            y, z, _, _ = sections[row]
            center = len(vertices)
            vertices.append((x, y, z))
            for column in range(count):
                a, c = row * count + column, row * count + (column + 1) % count
                faces.append((center, c, a) if row == 0 else (center, a, c))
        _mesh(b, base, f"glove_{index}", vertices, faces, "rubber")
        # The photograph shows a flattened hand with folded distal fingers,
        # rather than a taper ending in a point. These intersecting fixed skin
        # lobes are presentation geometry, not a deformable hand or sealed glove.
        for finger, (dx, length) in enumerate(
            ((-0.045, 0.080), (-0.015, 0.112), (0.015, 0.105), (0.045, 0.078))
        ):
            sections = [
                (
                    dx * (1 - 0.07 * t),
                    -1.12 - length * t,
                    1.330 - 0.018 * math.sin(math.pi * t * 0.80),
                    radius,
                    radius * 0.78,
                )
                for t, radius in (
                    (0, 0.014),
                    (0.35, 0.0135),
                    (0.70, 0.011),
                    (0.93, 0.008),
                    (1, 0.003),
                )
            ]
            _glove_digit(b, base, f"glove_{index}_finger_{finger}", x, sections)
        _glove_digit(
            b,
            base,
            f"glove_{index}_thumb",
            x,
            (
                (0.046, -1.050, 1.341, 0.022, 0.022),
                (0.065, -1.075, 1.333, 0.020, 0.019),
                (0.074, -1.105, 1.320, 0.016, 0.014),
                (0.068, -1.135, 1.316, 0.012, 0.010),
                (0.056, -1.153, 1.320, 0.004, 0.004),
            ),
        )


def add_front_details(b, base):
    """Photographic brow, inset frame and repeated fasteners; dimensions inferred."""
    columns = (
        (-1.34, 2.000),
        (-1.31, 1.965),
        (-1.26, 1.948),
        (-1.15, 1.948),
        (-1.07, 1.970),
        (-1.015, 2.005),
        (-0.94, 2.015),
        (0.94, 2.015),
        (1.015, 2.005),
        (1.07, 1.970),
        (1.15, 1.948),
        (1.26, 1.948),
        (1.31, 1.965),
        (1.34, 2.000),
    )
    vertices = [
        (x, y, z)
        for x, lower in columns
        for y, z in ((-0.613, lower), (-0.613, 2.12), (-0.547, lower), (-0.547, 2.12))
    ]
    faces = []
    for i in range(len(columns) - 1):
        a, n = i * 4, (i + 1) * 4
        for quad in (
            (a, n, n + 1, a + 1),
            (a + 2, a + 3, n + 3, n + 2),
            (a, a + 2, n + 2, n),
            (a + 1, n + 1, n + 3, a + 3),
        ):
            faces.extend(((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])))
    faces.extend(((0, 1, 3), (0, 3, 2)))
    end = (len(columns) - 1) * 4
    faces.extend(((end, end + 2, end + 3), (end, end + 3, end + 1)))
    _mesh(b, base, "shaped_brow", vertices, faces, "cream")
    for side in (-1, 1):
        for index, height in enumerate((-0.42, -0.28, -0.14, 0, 0.14, 0.28, 0.42)):
            y = -0.50 - 0.040 * math.cos(0.15) + height * math.sin(0.15)
            z = 1.50 + 0.040 * math.sin(0.15) + height * math.cos(0.15)
            b.cylinder(
                base,
                (side * 1.27, y, z),
                0.006,
                0.003,
                "bright",
                euler=(math.pi / 2 - 0.15, 0, 0),
                name=name(b, f"frame_fastener_{side}_{index}"),
            )
            b.rod(
                base,
                (side * 1.27 - 0.0025, y - 0.0035, z),
                (side * 1.27 + 0.0025, y - 0.0035, z),
                0.0006,
                "dark",
            )
    for x in (-1.232, 1.232):
        b.rod(
            base,
            (x, -0.576, 1.016),
            (x, -0.430, 1.984),
            0.005,
            "rubber",
            name=name(b, f"window_gasket_{x}"),
        )
    b.rod(
        base,
        (-1.232, -0.576, 1.016),
        (1.232, -0.576, 1.016),
        0.005,
        "rubber",
        name=name(b, "window_gasket_lower"),
    )
    # Screen attachment is an inferred short upper-frame bracket.
    b.box(
        base,
        (-0.90, -0.574, 1.957),
        (0.032, 0.022, 0.018),
        "metal",
        name=name(b, "screen_bracket"),
    )
