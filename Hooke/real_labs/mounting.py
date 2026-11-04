"""Reusable open annular mounts with explicit radial collision clearance.

Presentation meshes retain the bore; optional sector boxes model contact
without a convex mesh filling that opening. They do not model bearing stresses.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers


def annular_mount(
    b,
    parent,
    key,
    center,
    inner,
    outer,
    half_length,
    *,
    half=False,
    axis="x",
    proxies=False,
    material="bright",
    namespace="mount",
):
    """Open bore presentation mesh, optionally with separated collision sectors."""
    if axis not in {"x", "z"} or not 0 < inner < outer or half_length <= 0:
        raise ValueError(
            "Annular mounts require a valid axis, ordered radii and positive length"
        )

    def named(key):
        return f"{b.name}__{namespace}_{key}"

    count = 24 if half else 48
    extent = math.pi if half else math.tau
    angles = [extent * i / count for i in range(count + 1)]

    def point(t, radial, angle):
        values = (t, radial * math.cos(angle), radial * math.sin(angle))
        if axis == "z":
            values = (values[1], values[2], values[0])
        return tuple(a + v for a, v in zip(center, values))

    vertices = [
        point(t, r, a)
        for t in (-half_length, half_length)
        for r in (inner, outer)
        for a in angles
    ]
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
            faces.extend(tuple(reversed(f)) if reverse else f for f in triangles)
    if half:
        for i in (0, count):
            triangles = (
                (i, stride + i, 3 * stride + i),
                (i, 3 * stride + i, 2 * stride + i),
            )
            faces.extend(tuple(reversed(f)) if i else f for f in triangles)
    mesh = named(key + "_shell_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for point in vertices for v in point]),
        face=" ".join(str(i) for face in faces for i in face),
    )
    b.geom(parent, "mesh", (), material=material, mesh=mesh, name=named(key + "_shell"))
    if not proxies:
        return
    radius, thickness = (inner + outer) / 2, (outer - inner) / 2
    tangent = outer * math.tan(extent / (2 * count))
    for i in range(count):
        angle = extent * (i + 0.5) / count
        size = (half_length, thickness, tangent)
        rotation = (angle, 0, 0)
        if axis == "z":
            size, rotation = (thickness, tangent, half_length), (0, 0, angle)
        b.box(
            parent,
            point(0, radius, angle),
            size,
            material,
            euler=rotation,
            collision=True,
            rgba=(0.5, 0.5, 0.5, 0),
            name=named(f"{key}_proxy_{i}"),
        )
