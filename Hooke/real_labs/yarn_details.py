"""Original bearing and pressure-arm details for the Leeds yarn workstation.

The installed photograph establishes an exposed, layered roller face, rounded
pale guards, small adjusters and blue side shells. Bearing dimensions and hidden
mounts below are authored mechanical adaptations, not OEM drawings. Existing
joint constraints carry the surrogate; no bearing stress or yarn dynamics are
modeled.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers


RUNNING_CLEARANCE_M = 0.0005


def _name(b, key):
    return f"{b.name}__yarn_{key}"


def _bearing(b, parent, key, center, inner, outer, half_length, *, split=False):
    """Hollow X-axis bearing with a visible annulus and clearance-aware proxies."""
    count = 16 if split else 32
    extent = math.pi if split else math.tau
    angles = [extent * i / count for i in range(count + 1)]
    vertices = []
    for x in (-half_length, half_length):
        for radius in (inner, outer):
            vertices.extend(
                (
                    center[0] + x,
                    center[1] + radius * math.cos(a),
                    center[2] + radius * math.sin(a),
                )
                for a in angles
            )
    stride = count + 1
    faces = []
    for i in range(count):
        for first, second, reverse in (
            (0, stride, False),
            (2 * stride, 3 * stride, True),
            (0, 2 * stride, True),
            (stride, 3 * stride, False),
        ):
            a, c, d, e = first + i, first + i + 1, second + i + 1, second + i
            triangles = ((a, c, d), (a, d, e))
            faces.extend(
                tuple(reversed(face)) if reverse else face for face in triangles
            )
    if split:
        for i in (0, count):
            triangles = (
                (i, stride + i, 3 * stride + i),
                (i, 3 * stride + i, 2 * stride + i),
            )
            faces.extend(tuple(reversed(face)) if i else face for face in triangles)
    mesh = _name(b, key + "_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for point in vertices for v in point]),
        face=" ".join(str(i) for face in faces for i in face),
    )
    b.geom(
        parent, "mesh", (), material="metal", mesh=mesh, name=_name(b, key + "_shell")
    )
    radius = (inner + outer) / 2
    half_radial = (outer - inner) / 2
    half_tangent = outer * math.tan(extent / (2 * count))
    for i in range(count):
        angle = extent * (i + 0.5) / count
        b.box(
            parent,
            (
                center[0],
                center[1] + radius * math.cos(angle),
                center[2] + radius * math.sin(angle),
            ),
            (half_length, half_radial, half_tangent),
            "metal",
            euler=(angle, 0, 0),
            collision=True,
            rgba=(0.5, 0.5, 0.5, 0),
            name=_name(b, f"{key}_proxy_{i}"),
        )


def add_feed_bearings(b, base, shaft):
    """Join the original 1.28 m shaft to two frame-mounted hollow bearings."""
    for side, sign in (("left", -1), ("right", 1)):
        # The journal overlaps the main shaft axially, passes through the bore,
        # and terminates in a small outboard retainer; no solid bearing blocks.
        b.cylinder(
            shaft,
            (sign * 0.6725, 0, 0),
            0.020,
            0.0475,
            "bright",
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=_name(b, f"{side}_journal"),
        )
        b.cylinder(
            shaft,
            (sign * 0.723, 0, 0),
            0.026,
            0.003,
            "dark",
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=_name(b, f"{side}_journal_retainer"),
        )
        _bearing(
            b,
            base,
            f"{side}_feed_bearing",
            (sign * 0.69, -0.165, 1.53),
            0.020 + RUNNING_CLEARANCE_M,
            0.052,
            0.025,
        )
        # A rearward bracket reaches the existing sloped side-frame member.
        b.box(
            base,
            (sign * 0.70, -0.014, 1.51),
            (0.042, 0.101, 0.022),
            "shell",
            collision=True,
            name=_name(b, f"{side}_bearing_bracket"),
        )
        for y in (-0.071, 0.047):
            b.cylinder(
                base,
                (sign * 0.747, y, 1.51),
                0.009,
                0.005,
                "dark",
                euler=(0, math.pi / 2, 0),
                name=_name(b, f"{side}_bracket_bolt_{y}"),
            )


def add_carriage_bearings(b, base, carriage):
    """Upper split bearing shoes connect the plate while clearing the low deck."""
    for index, y in enumerate((-0.38, -0.19)):
        for end, x in enumerate((-0.44, 0.44)):
            b.box(
                base,
                (x, y, 0.585),
                (0.025, 0.019, 0.005),
                "metal",
                collision=True,
                name=_name(b, f"rail_{index}_pedestal_{end}"),
            )
            b.cylinder(
                base,
                (x, y, 0.592),
                0.008,
                0.006,
                "dark",
                name=_name(b, f"rail_{index}_fastener_{end}"),
            )
        for index_x, x in enumerate((-0.11, 0.11)):
            _bearing(
                b,
                carriage,
                f"linear_bearing_{index}_{index_x}",
                (x, y + 0.285, -0.041),
                0.012 + RUNNING_CLEARANCE_M,
                0.024,
                0.045,
                split=True,
            )
            b.cylinder(
                carriage,
                (x, y + 0.285, 0.024),
                0.009,
                0.004,
                "dark",
                name=_name(b, f"linear_bearing_bolt_{index}_{index_x}"),
            )


def add_pressure_modules(b, base):
    """Layered fixed guards, pressure arms and roller pairs from visible cues.

    These small roller/arm details are fixed exterior context. Only the original
    upper feed shaft and retained bobbin carriage have controlled motion.
    """
    rows = ((-0.40, 0.88), (-0.25, 1.13), (-0.10, 1.38), (0.055, 1.63))
    for row, (y, z) in enumerate(rows):
        b.rod(
            base,
            (-0.69, y + 0.07, z),
            (0.69, y + 0.07, z),
            0.035,
            "metal",
            collision=True,
            name=_name(b, f"row_{row}_crossbar"),
        )
        for column, x in enumerate((-0.48, -0.16, 0.16, 0.48)):
            key = f"pressure_{row}_{column}"
            module = ET.SubElement(
                base, "body", name=_name(b, key), pos=numbers((x, y, z))
            )
            b.box(
                module,
                (0, 0.021, 0),
                (0.082, 0.049, 0.040),
                "metal",
                collision=True,
                name=_name(b, key + "_rear_mount"),
            )
            # Tapered guard instead of a repeated square block. The central
            # backing remains collision-capable; thin cosmetic edges do not
            # imply small contact features beyond the existing surrogate.
            b.housing(
                module,
                [
                    (-0.063, 0.071, 0.036, -0.017),
                    (-0.040, 0.103, 0.039, -0.017),
                    (0.045, 0.103, 0.039, -0.017),
                    (0.069, 0.071, 0.036, -0.017),
                ],
                radius=0.014,
                material="cream",
            )
            b.box(
                module,
                (0, -0.011, 0),
                (0.083, 0.030, 0.051),
                "cream",
                collision=True,
                name=_name(b, key + "_guard_proxy"),
            )
            for label, px, pz, radius in (
                ("lower", -0.061, -0.021, 0.037),
                ("upper", 0.058, 0.030, 0.030),
            ):
                b.cylinder(
                    module,
                    (px, -0.075, pz),
                    radius,
                    0.027,
                    "bright",
                    euler=(math.pi / 2, 0, 0),
                    collision=True,
                    name=_name(b, f"{key}_{label}_roller"),
                )
                b.cylinder(
                    module,
                    (px, -0.106, pz),
                    radius * 0.76,
                    0.005,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                    name=_name(b, f"{key}_{label}_bearing"),
                )
                b.cylinder(
                    module,
                    (px, -0.113, pz),
                    radius * 0.47,
                    0.004,
                    "blue",
                    euler=(math.pi / 2, 0, 0),
                    name=_name(b, f"{key}_{label}_cap"),
                )
            b.rod(
                module,
                (-0.061, -0.121, -0.021),
                (0.058, -0.121, 0.030),
                0.012,
                "metal",
                name=_name(b, key + "_pressure_arm"),
            )
            # Pivot/adjuster housings have visible links back to the guard.
            b.rod(
                module,
                (0.075, -0.040, 0.055),
                (0.075, -0.115, 0.055),
                0.006,
                "bright",
                name=_name(b, key + "_adjuster_stem"),
            )
            b.cylinder(
                module,
                (0.075, -0.124, 0.055),
                0.012,
                0.010,
                "dark",
                euler=(math.pi / 2, 0, 0),
                name=_name(b, key + "_adjuster_knob"),
            )
            b.box(
                module,
                (0, -0.058, 0.052),
                (0.020, 0.002, 0.008),
                "dark",
                name=_name(b, key + "_indicator"),
            )
