"""Original open stage plates; dimensions are authored, not OEM specifications."""

import xml.etree.ElementTree as ET


def _plate_mesh(builder, key, half_size, opening):
    """One manifold annulus avoids artificial bevel seams between contact boxes."""
    width, depth, thickness = half_size
    clear_x, clear_y = opening
    rings = [
        (-width, -depth),
        (width, -depth),
        (width, depth),
        (-width, depth),
        (-clear_x, -clear_y),
        (clear_x, -clear_y),
        (clear_x, clear_y),
        (-clear_x, clear_y),
    ]
    vertices = [(x, y, z) for z in (-thickness, thickness) for x, y in rings]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend(
            [
                (i + 8, j + 8, j + 12),
                (i + 8, j + 12, i + 12),
                (i, j + 4, j),
                (i, i + 4, j + 4),
                (i, j, j + 8),
                (i, j + 8, i + 8),
                (i + 4, i + 12, j + 12),
                (i + 4, j + 12, j + 4),
            ]
        )
    name = f"{builder.name}__{key}_mesh"
    ET.SubElement(
        builder.root.find("asset"),
        "mesh",
        name=name,
        vertex=" ".join(str(v) for point in vertices for v in point),
        face=" ".join(str(v) for face in faces for v in face),
    )
    return name


def aperture_plate(
    builder, parent, key, position, half_size, opening, material, *, collision=True
):
    """Build a supported rectangular plate with a real through opening.

    Four non-overlapping contact boxes match one continuous visual mesh. Both
    representations leave the opening clear; no convex hull closes it in physics.
    """
    x, y, z = position
    width, depth, thickness = half_size
    clear_x, clear_y = opening
    if not (0 < clear_x < width and 0 < clear_y < depth and thickness > 0):
        raise ValueError("Stage opening must lie inside a positive plate envelope")
    mesh = _plate_mesh(builder, key, half_size, opening)
    builder.geom(
        parent,
        "mesh",
        (),
        position,
        material,
        mesh=mesh,
        name=f"{builder.name}__{key}_surface",
    )
    for side in (-1, 1):
        builder.box(
            parent,
            (x, y + side * (depth + clear_y) / 2, z),
            (width, (depth - clear_y) / 2, thickness),
            material,
            collision=collision,
            group="3",
            name=f"{builder.name}__{key}_crossbar_{side}",
        )
        builder.box(
            parent,
            (x + side * (width + clear_x) / 2, y, z),
            ((width - clear_x) / 2, clear_y, thickness),
            material,
            collision=collision,
            group="3",
            name=f"{builder.name}__{key}_side_{side}",
        )


def slide_holder_details(builder, parent, key, surface_z, outer_size):
    """Estimated insert frame and fastening details for a 75 × 25 mm slide."""
    width, depth = outer_size
    # The insert supports the ends and long edges without closing the aperture.
    aperture_plate(
        builder,
        parent,
        key + "_insert",
        (0, 0, surface_z - 0.0004),
        (0.042, 0.020, 0.0004),
        (0.029, 0.009),
        "dark",
        collision=True,
    )
    for x in (-width + 0.009, width - 0.009):
        for y in (-depth + 0.009, depth - 0.009):
            builder.cylinder(
                parent,
                (x, y, surface_z + 0.0007),
                0.003,
                0.0007,
                "bright",
                name=f"{builder.name}__{key}_fastener_{x}_{y}",
            )
            builder.cylinder(
                parent, (x, y, surface_z + 0.00142), 0.0011, 0.00004, "dark"
            )
    # Separate rail edges and small end stops read at workstation distance.
    for side in (-1, 1):
        builder.box(
            parent,
            (side * (width - 0.003), 0, surface_z + 0.0006),
            (0.001, depth - 0.016, 0.0006),
            "bright",
        )
