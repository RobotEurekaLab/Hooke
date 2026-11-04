"""Photo-constrained exterior details for the UCSD inspection surrogate.

The small official photograph establishes a blue counterweight crossing the
arm and a curved white carrier exterior, not their internal construction. The
split clamp, shell thickness and open inspection rim below are inferred.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount


_PRISM_FACES = (
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
)


def add_pedestal_surface(b, base):
    """Continuous cast exterior over the retained convex contact sectors.

    The open front arch is retained. Smooth radial normals remove artificial
    seams between collision pieces; the nominal taper stays unchanged.
    """
    start, stop = 27 * math.tau / 32, (32 + 21) * math.tau / 32
    count = 193
    angles = [start + (stop - start) * i / (count - 1) for i in range(count)]
    vertices, normals, faces = [], [], []
    for inner in (False, True):
        offset = len(vertices)
        radii = (0.51, 0.25) if inner else (0.74, 0.41)
        slope = (radii[0] - radii[1]) / 0.94
        length = math.sqrt(1 + slope * slope)
        sign = -1 if inner else 1
        for z, radius in zip((0.10, 1.04), radii):
            for angle in angles:
                vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
                normals.append(tuple(sign * v / length for v in (math.cos(angle), math.sin(angle), slope)))
        for i in range(count - 1):
            a, c = offset + i, offset + count + i
            triangles = ((a, a + 1, c + 1), (a, c + 1, c))
            faces.extend(tuple(reversed(t)) if inner else t for t in triangles)
    # The existing sectors retain horizontal surfaces and arch end faces.
    mesh = b.unique('ucsd_cast_pedestal_surface_mesh')
    ET.SubElement(b.root.find('asset'), 'mesh', name=mesh,
                  vertex=' '.join(str(v) for p in vertices for v in p),
                  normal=' '.join(str(v) for p in normals for v in p),
                  face=' '.join(str(v) for f in faces for v in f))
    b.geom(base, 'mesh', (), mesh=mesh, material='blue', density=0,
           name='{}__ucsd_cast_pedestal_surface'.format(b.name))


def _mesh(b, parent, vertices, faces, name, *, visible=True):
    asset = b.unique(f"{name}_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=asset,
        vertex=" ".join(str(value) for point in vertices for value in point),
        face=" ".join(str(value) for face in faces for value in face),
    )
    attributes = {
        "name": b.unique(name),
        "type": "mesh",
        "mesh": asset,
        "density": "0",
        "contype": "0" if visible else "1",
        "conaffinity": "0" if visible else "1",
    }
    if visible:
        attributes["material"] = f"{b.name}__mat_cream"
    else:
        attributes.update(rgba="0 0 0 0", group="3")
    return ET.SubElement(parent, "geom", **attributes)


def _named_box(b, parent, name, pos, half_size, material):
    geom = b.box(parent, pos, half_size, material, collision=True)
    geom.set("name", b.unique(name))
    return geom


def add_counterweight(b, arm):
    """Split blue block seats on both beam flanges; no floating plate stack."""
    # Two cheeks tie the upper and lower caps together around the beam. Small
    # overlaps represent the inferred bolted split clamp, not movable payload.
    for side in (-1, 1):
        _named_box(
            b,
            arm,
            "ucsd_counterweight_cheek",
            (-1.08, side * 0.262, 0.112),
            (0.18, 0.042, 0.333),
            "blue",
        )
    for z, half_height in ((0.268, 0.173), (-0.155, 0.062)):
        _named_box(
            b,
            arm,
            "ucsd_counterweight_cap",
            (-1.08, 0, z),
            (0.18, 0.223, half_height),
            "blue",
        )
    for side in (-1, 1):
        for x in (-1.20, -0.96):
            for z in (-0.15, 0.37):
                bolt = b.cylinder(
                    arm,
                    (x, side * 0.310, z),
                    0.012,
                    0.007,
                    "metal",
                    euler=(math.pi / 2, 0, 0),
                )
                bolt.set("name", b.unique("ucsd_counterweight_fastener"))
    # The source arm has a continuous painted side and rounded end bosses.
    # These attached skins/caps leave the load-bearing beam and pivot gap intact.
    for side in (-1, 1):
        b.box(arm, (.13, side * .184, 0), (1.55, .004, .078), "blue",
              name=b.unique("ucsd_arm_side_skin"))
        b.cylinder(arm, (1.45, side * .543, -.05), .080, .008, "blue",
                   euler=(math.pi / 2, 0, 0), collision=True,
                   name=b.unique("ucsd_outer_pivot_cap"))
        b.cylinder(arm, (1.45, side * .553, -.05), .022, .004, "metal",
                   euler=(math.pi / 2, 0, 0),
                   name=b.unique("ucsd_outer_pivot_fastener"))


def add_basket_bearings(b, arm):
    """Open pivot sleeves clear both rotating shafts and suspension straps.

    A 0.6 mm radial gap and 1 mm strap-to-sleeve gap are authored inspection
    clearances, carried by the existing ideal hinge, not bearing calibration.
    """
    for side in (-1, 1):
        sleeve = ET.SubElement(arm, "body", pos=f"1.45 {side * .50} -.05",
                               euler=f"{math.pi / 2} 0 0")
        annular_mount(b, sleeve, f"basket_bearing_{side}",
                      (0, 0, 0), .0506, .085, .035,
                      axis="z", proxies=True, material="blue", namespace="ucsd")


def add_basket_exterior(b, basket):
    """Curved hollow exterior retains the existing supported, visible target.

    Convex shell strips preserve a real top opening and collision surfaces.
    The authored rigid coupon and its support datum are left to the caller.
    This open inspection state does not claim the photographed shroud's hidden
    construction, wall thickness or its real high-g suspension design.
    """
    floor = _named_box(
        b,
        basket,
        "ucsd_basket_floor",
        (0, 0, -0.63),
        (0.405, 0.37, 0.035),
        "cream",
    )
    # Flat suspension straps meet the floor and pivot bosses continuously.
    for side in (-1, 1):
        _named_box(
            b,
            basket,
            "ucsd_basket_hanger",
            (0, side * 0.446, -0.300),
            (0.035, 0.018, 0.303),
            "metal",
        )
        # The floor is narrower than the bulged sidewall, so a cross-member
        # explicitly carries each hanger into the floor rather than ending in air.
        _named_box(
            b,
            basket,
            "ucsd_basket_saddle",
            (0, side * 0.404, -0.610),
            (0.055, 0.060, 0.025),
            "metal",
        )
        pivot = b.cylinder(
            basket,
            (0, side * 0.46, 0),
            0.05,
            0.026,
            "metal",
            euler=(math.pi / 2, 0, 0),
        )
        pivot.set("name", b.unique("ucsd_basket_pivot"))

    # A rounded lower section with a flat supported floor fits the documented
    # 0.6 x 0.7 m rectangular container without cutting through its corners.
    angles = [-math.pi / 2 + i * (5 * math.pi / 6) / 32 for i in range(33)]
    visual_vertices, visual_faces = [], []
    for side in (-1, 1):
        # The visible shell is one shared-vertex bent plate. Closing every
        # collision strip separately would create false bevel/corrugation seams.
        offset = len(visual_vertices)
        count = len(angles)
        for x in (-0.41, 0.41):
            for thickness in (0, 0.018):
                visual_vertices.extend(
                    (
                        x,
                        side * (0.32 + (0.145 - thickness) * math.cos(angle)),
                        -0.455 + (0.20 - thickness) * math.sin(angle),
                    )
                    for angle in angles
                )

        def vertex(end, inner, angle):
            return offset + (end * 2 + inner) * count + angle

        def quad(a, c, d, e, *, reverse=False):
            faces = ((a, c, d), (a, d, e))
            visual_faces.extend(
                tuple(reversed(face)) if reverse else face for face in faces
            )

        for index in range(count - 1):
            for inner in (0, 1):
                quad(
                    vertex(0, inner, index),
                    vertex(1, inner, index),
                    vertex(1, inner, index + 1),
                    vertex(0, inner, index + 1),
                    reverse=(side == 1) != bool(inner),
                )
            for end in (0, 1):
                quad(
                    vertex(end, 0, index),
                    vertex(end, 0, index + 1),
                    vertex(end, 1, index + 1),
                    vertex(end, 1, index),
                    reverse=(end == 0) == (side == 1),
                )
        for index in (0, count - 1):
            quad(
                vertex(0, 0, index),
                vertex(0, 1, index),
                vertex(1, 1, index),
                vertex(1, 0, index),
                reverse=(index == 0) == (side == 1),
            )
        for first, second in zip(angles, angles[1:]):
            section = []
            for angle, thickness in (
                (first, 0),
                (second, 0),
                (second, 0.018),
                (first, 0.018),
            ):
                section.append(
                    (
                        side * (0.32 + (0.145 - thickness) * math.cos(angle)),
                        -0.455 + (0.20 - thickness) * math.sin(angle),
                    )
                )
            vertices = [(x, y, z) for x in (-0.41, 0.41) for y, z in section]
            _mesh(
                b,
                basket,
                vertices,
                _PRISM_FACES,
                "ucsd_basket_shell_contact",
                visible=False,
            )

    # Curved end plates join the side shell and floor. Their upper edges do not
    # bridge the opening; the clamped sample remains inspectable from above.
    outline = [
        (-0.32 - 0.145 * math.cos(a), -0.455 + 0.20 * math.sin(a))
        for a in reversed(angles)
    ] + [(0.32 + 0.145 * math.cos(a), -0.455 + 0.20 * math.sin(a)) for a in angles]
    count = len(outline)
    for side in (-1, 1):
        xs = sorted((side * 0.400, side * 0.420))
        vertices = [(x, y, z) for x in xs for y, z in outline]
        faces = []
        for index in range(1, count - 1):
            faces.extend(
                ((0, index + 1, index), (count, count + index, count + index + 1))
            )
        for index in range(count):
            following = (index + 1) % count
            faces.extend(
                (
                    (index, following, count + following),
                    (index, count + following, count + index),
                )
            )
        _mesh(b, basket, vertices, faces, "ucsd_basket_end_contact", visible=False)
        offset = len(visual_vertices)
        visual_vertices.extend(vertices)
        visual_faces.extend(tuple(offset + index for index in face) for face in faces)
    _mesh(b, basket, visual_vertices, visual_faces, "ucsd_basket_curved_exterior")
    return floor
