"""Original exterior details informed by Nottingham's historical brochure.

The mobile pan's rounded jacket, tapered frame and open tray are visible in
the photograph. Their dimensions, the dry inspection cover hinge and hidden
attachments are inferred; none implements a thermal or food process model.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers


def _mesh(b, parent, key, vertices, faces, material, *, collision=False):
    mesh = b.unique(key + "_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([value for vertex in vertices for value in vertex]),
        face=" ".join(str(value) for face in faces for value in face),
    )
    return b.geom(
        parent,
        "mesh",
        (),
        material=material,
        mesh=mesh,
        name=b.unique(key),
        collision=collision,
    )


def add_side_plates(b, base):
    """Tapered load-bearing side webs join the feet and trunnion housings."""
    outline = ((-0.32, 0.155), (0.30, 0.155), (0.09, 1.00), (-0.085, 1.00))
    faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7))
    sides = tuple(
        face
        for i in range(4)
        for face in ((i, (i + 1) % 4, (i + 1) % 4 + 4), (i, (i + 1) % 4 + 4, i + 4))
    )
    for x in (-0.355, 0.355):
        vertices = [(x + dx, y, z) for dx in (-0.027, 0.027) for y, z in outline]
        _mesh(
            b,
            base,
            "pan_tapered_support",
            vertices,
            (*faces, *sides),
            "cream",
            collision=True,
        )
        for y, z in ((-0.20, 0.25), (0.06, 0.47), (0.015, 0.73)):
            b.cylinder(
                base,
                (x + math.copysign(0.030, x), y, z),
                0.010,
                0.004,
                "metal",
                euler=(0, math.pi / 2, 0),
            )


def add_open_tray(b, base):
    """Supported wire lattice has genuine openings, without a hidden solid deck."""
    for x in (-0.30, 0.30):
        b.box(
            base,
            (x, -0.30, 0.539),
            (0.013, 0.23, 0.016),
            "cream",
            collision=True,
            name=b.unique("pan_tray_frame"),
        )
    for y in (-0.52, -0.08):
        b.box(
            base,
            (0, y, 0.539),
            (0.30, 0.010, 0.016),
            "cream",
            collision=True,
            name=b.unique("pan_tray_frame"),
        )
    # Two cantilever rails meet both side webs, so the tray does not hang from
    # the wire lattice. Their exterior ends remain within the existing frame.
    for x in (-0.30, 0.30):
        b.box(
            base,
            (x, -0.14, 0.516),
            (0.016, 0.365, 0.014),
            "metal",
            collision=True,
            name=b.unique("pan_tray_support"),
        )
        b.box(
            base,
            (math.copysign(0.328, x), 0.07, 0.516),
            (0.030, 0.033, 0.020),
            "metal",
            collision=True,
            name=b.unique("pan_tray_attachment"),
        )
    for i in range(20):
        x = -0.285 + i * 0.030
        b.rod(
            base,
            (x, -0.51, 0.545),
            (x, -0.09, 0.545),
            0.0028,
            "bright",
            collision=True,
            name=b.unique("pan_tray_wire"),
        )
    for i in range(15):
        y = -0.51 + i * 0.030
        b.rod(
            base,
            (-0.29, y, 0.549),
            (0.29, y, 0.549),
            0.0028,
            "bright",
            collision=True,
            name=b.unique("pan_tray_crosswire"),
        )


def _hinge_knuckle(b, parent, key, center, half):
    """Shared-vertex hollow barrel and separate convex walls leave a real bore."""
    outer, inner, count = 0.017, 0.0068, 32
    vertices, faces = [], []
    for dx in (-half, half):
        for radius in (outer, inner):
            for i in range(count):
                angle = math.tau * i / count
                vertices.append(
                    (
                        center[0] + dx,
                        center[1] + radius * math.cos(angle),
                        center[2] + radius * math.sin(angle),
                    )
                )
    for i in range(count):
        j = (i + 1) % count
        for a, c, d, e in (
            (i, j, 2 * count + j, 2 * count + i),
            (count + j, count + i, 3 * count + i, 3 * count + j),
            (j, i, count + i, count + j),
            (2 * count + i, 2 * count + j, 3 * count + j, 3 * count + i),
        ):
            faces.extend(((a, c, d), (a, d, e)))
    _mesh(b, parent, key, vertices, faces, "metal")
    # Tangential boxes conservatively approximate the annular wall while their
    # inner flat face stays outside the 6.8 mm bore. Axle radius is only 6 mm.
    radial = (outer + inner) / 2
    for i in range(count):
        angle = math.tau * i / count
        b.box(
            parent,
            (
                center[0],
                center[1] + radial * math.cos(angle),
                center[2] + radial * math.sin(angle),
            ),
            (half, (outer - inner) / 2, outer * math.tan(math.pi / count)),
            "metal",
            euler=(angle, 0, 0),
            collision=True,
            name=b.unique(key + "_contact"),
            rgba=(0, 0, 0, 0),
            group=3,
        )


def add_cover_hinge(b, pan, lid):
    """Inferred split hinge follows the existing authored cover pivot exactly."""
    for x in (-0.060, 0.060):
        # A shallow leaf runs beneath the closed cover into the barrel's lower
        # wall. It must clear both the lid and central moving barrel even though
        # the engine normally filters contact between adjacent joint bodies.
        b.box(
            pan,
            (x, 0.274, 0.1995),
            (0.024, 0.023, 0.0105),
            "metal",
            collision=True,
            name=b.unique("pan_cover_fixed_leaf"),
        )
        _hinge_knuckle(b, pan, "pan_cover_fixed_knuckle", (x, 0.294, 0.225), 0.022)
    _hinge_knuckle(b, lid, "pan_cover_moving_knuckle", (0, 0, 0), 0.034)
    b.box(
        lid,
        (0, -0.028, 0),
        (0.030, 0.013, 0.008),
        "metal",
        collision=True,
        name=b.unique("pan_cover_moving_leaf"),
    )
    b.cylinder(
        pan,
        (0, 0.294, 0.225),
        0.006,
        0.100,
        "bright",
        euler=(0, math.pi / 2, 0),
        collision=True,
        name=b.unique("pan_cover_axle"),
    )
    for x in (-0.105, 0.105):
        b.cylinder(
            pan,
            (x, 0.294, 0.225),
            0.011,
            0.005,
            "bright",
            euler=(0, math.pi / 2, 0),
            name=b.unique("pan_cover_axle_head"),
        )


def add_external_trunnion_details(b, base, pan):
    """Photographed side fittings, with dry estimated routes and no pipe model."""
    for x in (-0.355, 0.355):
        b.cylinder(
            base,
            (x + math.copysign(0.051, x), 0, 0.93),
            0.055,
            0.012,
            "copper",
            euler=(0, math.pi / 2, 0),
            name=b.unique("pan_bearing_cap"),
        )
        for i in range(4):
            angle = i * math.pi / 2
            b.cylinder(
                base,
                (
                    x + math.copysign(0.066, x),
                    0.043 * math.cos(angle),
                    0.93 + 0.043 * math.sin(angle),
                ),
                0.006,
                0.004,
                "metal",
                euler=(0, math.pi / 2, 0),
            )
    # Small visible side fitting and curved lower return, attached to the fixed
    # right side frame. These are exterior cues, not connected steam plumbing.
    b.rod(base, (0.409, 0.08, 0.91), (0.45, 0.15, 0.88), 0.012, "copper")
    b.rod(base, (0.45, 0.15, 0.88), (0.45, 0.19, 0.61), 0.012, "metal")
    b.rod(base, (0.45, 0.19, 0.61), (0.355, 0.12, 0.56), 0.012, "metal")
    for x in (-0.43, 0.43):
        b.rod(pan, (x, 0.06, 0.30), (x, 0.08, 0.41), 0.020, "rubber")


def _guard_wire(p, key, start, end, radius, color):
    geom = p.rod(key, start, end, radius, color)
    geom.set("contype", "1")
    geom.set("conaffinity", "1")


def add_hall_guard(p, y, steel):
    """Open welded-wire panels sit on bank posts, not on the reserved aisle."""
    for z in (0.25, 0.89):
        p.box("lower_guard_rail", (0, y + 0.24, z), (2.30, 0.012, 0.012), steel, True)
    for i in range(93):
        x = -2.30 + i * 0.05
        _guard_wire(
            p,
            "lower_guard_wire",
            (x, y + 0.24, 0.25),
            (x, y + 0.24, 0.89),
            0.002,
            steel,
        )
    for i in range(13):
        z = 0.27 + i * 0.05
        _guard_wire(
            p,
            "lower_guard_crosswire",
            (-2.30, y + 0.24, z),
            (2.30, y + 0.24, z),
            0.002,
            steel,
        )
    for i in range(98):
        x = -2.425 + i * 0.05
        _guard_wire(
            p,
            "upper_guard_wire",
            (x, y + 0.30, 0.985),
            (x, y + 0.30, 1.17),
            0.002,
            steel,
        )


def add_rear_hall_apparatus(p, white, steel, dark):
    """Distinct source-visible rear silhouettes; equipment identity is unknown."""
    blue = (0.10, 0.32, 0.47, 1)
    # Tall open rack with blue tray edges visible at the back-left of the photo.
    for x in (-1.94, -1.06):
        for y in (2.76, 3.32):
            p.box("rear_rack_post", (x, y, 0.94), (0.023, 0.023, 0.94), steel, True)
    for z in (0.13, 0.45, 0.77, 1.09, 1.41, 1.73):
        p.box("rear_rack_shelf", (-1.50, 3.04, z), (0.46, 0.30, 0.013), steel, True)
        p.box("rear_rack_tray", (-1.50, 3.02, z + 0.046), (0.40, 0.255, 0.033), blue)
    # White square-grid front, distinct from the open rack and taller right unit.
    p.box("rear_grid_cabinet", (0, 3.04, 0.72), (0.52, 0.34, 0.62), white, True)
    for x in (-0.42, 0.42):
        p.box("rear_grid_foot", (x, 3.04, 0.05), (0.035, 0.27, 0.05), dark, True)
    p.box("rear_grid_inset", (-0.07, 2.694, 0.87), (0.315, 0.009, 0.335), dark)
    for x in (-0.28, -0.07, 0.14):
        p.box("rear_grid_vertical", (x, 2.681, 0.87), (0.011, 0.009, 0.335), white)
    for z in (0.65, 0.87, 1.09):
        p.box("rear_grid_horizontal", (-0.07, 2.679, z), (0.315, 0.010, 0.011), white)
    p.box("rear_grid_control_face", (0.385, 2.688, 0.89), (0.08, 0.012, 0.34), blue)
    for z in (0.70, 0.88, 1.06):
        p.box("rear_grid_control", (0.385, 2.672, z), (0.033, 0.006, 0.025), dark)
    p.box("rear_tall_unit", (1.5, 3.04, 0.99), (0.48, 0.34, 0.89), white, True)
    for x in (1.10, 1.90):
        p.box("rear_tall_foot", (x, 3.04, 0.05), (0.035, 0.27, 0.05), dark, True)
    p.box("rear_tall_panel", (1.5, 2.69, 1.43), (0.34, 0.012, 0.27), steel)
    p.box("rear_tall_display", (1.5, 2.674, 1.50), (0.16, 0.006, 0.065), dark)
    for z in (0.34, 0.40, 0.46, 0.52):
        p.box("rear_tall_vent", (1.5, 2.69, z), (0.29, 0.013, 0.007), dark)
    # Wall mounts and articulated white extraction/service arms are visible in
    # the hall photo. Routing and joints are fixed estimates, not extra controls.
    for x, side in ((-0.45, -1), (1.45, 1)):
        p.box("rear_service_mount", (x, 3.40, 2.45), (0.095, 0.105, 0.18), white, True)
        points = (
            (x, 3.34, 2.50),
            (x + side * 0.22, 3.0, 2.81),
            (x + side * 0.42, 2.72, 2.32),
            (x + side * 0.32, 2.65, 2.08),
        )
        for start, end in zip(points, points[1:]):
            p.rod("rear_service_arm", start, end, 0.038, white)
        for point in points[1:-1]:
            p.box("rear_service_joint", point, (0.047, 0.047, 0.047), dark)
