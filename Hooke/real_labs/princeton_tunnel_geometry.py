"""BAM tunnel exterior casings and an explicitly authored probe roof slot.

Turning radii, wall thicknesses and support details are photo-informed estimates.
These collision shells are hardware exteriors, not a fluid discretization.
"""

from collections import defaultdict
import math
import xml.etree.ElementTree as ET


_PRISM = (
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


def _outward_faces(vertices, faces):
    """Keep shared-vertex shells closed and consistently outward oriented."""
    edges = defaultdict(list)
    for index, face in enumerate(faces):
        for first, second in zip(face, (*face[1:], face[0])):
            edges[tuple(sorted((first, second)))].append((index, first < second))
    adjacency = defaultdict(list)
    for neighbors in edges.values():
        if len(neighbors) != 2:
            raise ValueError("Expected a closed casing mesh")
        (first, direction), (second, other) = neighbors
        adjacency[first].append((second, direction == other))
        adjacency[second].append((first, direction == other))
    flip = {}
    for seed in range(len(faces)):
        if seed in flip:
            continue
        flip[seed] = False
        pending, component = [seed], []
        while pending:
            current = pending.pop()
            component.append(current)
            for other, reverse in adjacency[current]:
                expected = flip[current] ^ reverse
                if other not in flip:
                    flip[other] = expected
                    pending.append(other)
                elif flip[other] != expected:
                    raise ValueError("Inconsistent casing face orientation")
        volume = 0.0
        for index in component:
            face = faces[index][::-1] if flip[index] else faces[index]
            a, c, d = (vertices[point] for point in face)
            volume += (
                a[0] * (c[1] * d[2] - c[2] * d[1])
                + a[1] * (c[2] * d[0] - c[0] * d[2])
                + a[2] * (c[0] * d[1] - c[1] * d[0])
            )
        if volume < 0:
            for index in component:
                flip[index] = not flip[index]
    return [face[::-1] if flip[index] else face for index, face in enumerate(faces)]


def _mesh(b, base, name, vertices, faces, *, material="cyan", collision=False):
    mesh_name = b.unique(f"{name}_mesh")
    # Keep small wall thicknesses near the local origin: MuJoCo stores compiled
    # mesh vertices as float32, while the enclosing geom transform is float64.
    center = tuple(
        sum(point[axis] for point in vertices) / len(vertices) for axis in range(3)
    )
    vertices = [
        tuple(point[axis] - center[axis] for axis in range(3)) for point in vertices
    ]
    faces = _outward_faces(vertices, faces)
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh_name,
        vertex=" ".join(str(v) for point in vertices for v in point),
        face=" ".join(str(v) for face in faces for v in face),
    )
    attributes = dict(
        name=b.unique(name),
        type="mesh",
        mesh=mesh_name,
        pos=" ".join(str(value) for value in center),
        density="0",
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
    )
    if collision:
        attributes.update(group="3", rgba="0 0 0 0")
    else:
        attributes["material"] = f"{b.name}__mat_{material}"
    return ET.SubElement(base, "geom", **attributes)


def _sweep(b, base, name, rings):
    """Hollow rectangular wall sweep with independent convex contact cells."""
    vertices = [point for ring in rings for point in ring]
    faces = []

    def quad(a, c, d, e):
        faces.extend(((a, c, d), (a, d, e)))

    for section in range(len(rings) - 1):
        first, second = section * 8, (section + 1) * 8
        for edge in range(4):
            following = (edge + 1) % 4
            for inner in (0, 4):
                quad(
                    first + edge + inner,
                    second + edge + inner,
                    second + following + inner,
                    first + following + inner,
                )
            cell = [
                rings[index][point]
                for index in (section, section + 1)
                for point in (edge, following, following + 4, edge + 4)
            ]
            _mesh(b, base, f"{name}_contact", cell, _PRISM, collision=True)
    for section in (0, len(rings) - 1):
        offset = section * 8
        for edge in range(4):
            following = (edge + 1) % 4
            quad(
                offset + edge,
                offset + following,
                offset + following + 4,
                offset + edge + 4,
            )
    return _mesh(b, base, name, vertices, faces)


def _straight_ring(x, y, half_y, half_z):
    return [
        (x, y + dy * (half_y - thickness), 2.15 + dz * (half_z - thickness))
        for thickness in (0, 0.035)
        for dy, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def add_return_casings(b, base):
    """Rounded corner shells stay inside the existing apparatus envelope."""
    for side in (-1, 1):
        _sweep(
            b,
            base,
            "bam_contraction_casing",
            [
                _straight_ring(side * 1.53, -1.8, 0.65, 0.65),
                _straight_ring(side * 2.78, -1.8, 1.05, 1.05),
            ],
        )
        _sweep(
            b,
            base,
            "bam_return_transition",
            [
                _straight_ring(side * 2.35, 2.6, 0.82, 0.85),
                _straight_ring(side * 2.78, 2.6, 1.05, 1.05),
            ],
        )
        for back in (False, True):
            center_y = 1.5 if back else -0.7
            angles = [i * math.pi / 32 for i in range(17)]
            rings = []
            for angle in angles:
                # Front: (-pi/2, 0); rear: (0, pi/2), mirrored in X.
                angle = angle if back else angle - math.pi / 2
                rings.append(
                    [
                        (
                            side * (2.78 + radius * math.cos(angle)),
                            center_y + radius * math.sin(angle),
                            z,
                        )
                        for thickness in (0, 0.035)
                        for radius, z in (
                            (0.05 + thickness, 1.10 + thickness),
                            (2.15 - thickness, 1.10 + thickness),
                            (2.15 - thickness, 3.20 - thickness),
                            (0.05 + thickness, 3.20 - thickness),
                        )
                    ]
                )
            _sweep(b, base, "bam_rounded_turn", rings)

        # Longitudinal side leg between the two corner modules.
        rings = [
            [
                (
                    side * (3.88 + dx * (1.05 - thickness)),
                    y,
                    2.15 + dz * (1.05 - thickness),
                )
                for thickness in (0, 0.035)
                for dx, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            for y in (-0.7, 1.5)
        ]
        _sweep(b, base, "bam_side_return", rings)

        # Source-informed open underframes, with bearing members reaching the
        # casing's lower wall; capacities and exact sections are not measured.
        for y in (-0.45, 1.25):
            for dx in (-0.72, 0.72):
                x = side * (3.88 + dx)
                b.box(base, (x, y, 0.025), (0.15, 0.15, 0.025), "cyan", collision=True)
                b.box(base, (x, y, 0.55), (0.06, 0.06, 0.50), "cyan", collision=True)
            support = b.box(
                base, (side * 3.88, y, 1.08), (0.83, 0.08, 0.04), "cyan", collision=True
            )
            support.set("name", b.unique("bam_return_bearing_crossbar"))
        for dx in (-0.72, 0.72):
            b.rod(
                base,
                (side * (3.88 + dx), -0.45, 0.12),
                (side * (3.88 + dx), 1.25, 1.04),
                0.025,
                "cyan",
            )

        # Thin casing-colored flanges replace the old oversized black bands.
        for x, width in ((side * 1.535, 0.65), (side * 2.78, 1.05)):
            for sign in (-1, 1):
                b.box(
                    base,
                    (x, -1.8 + sign * (width + 0.012), 2.15),
                    (0.009, 0.016, width + 0.025),
                    "cyan",
                )
                b.box(
                    base,
                    (x, -1.8, 2.15 + sign * (width + 0.012)),
                    (0.009, width + 0.025, 0.016),
                    "cyan",
                )
            for dy in (-width, width):
                for dz in (-width * 0.7, 0, width * 0.7):
                    b.cylinder(
                        base,
                        (x + side * 0.012, -1.8 + dy, 2.15 + dz),
                        0.010,
                        0.004,
                        "metal",
                        euler=(0, math.pi / 2, 0),
                    )
    _sweep(
        b,
        base,
        "bam_back_return",
        [
            _straight_ring(-2.35, 2.6, 0.82, 0.85),
            _straight_ring(2.35, 2.6, 0.82, 0.85),
        ],
    )


def add_probe_roof(b, base):
    """True 44 x 770 mm inferred slot around the unchanged probe path.

    A single visible annular pane retains the original panel envelope; four
    contact boxes leave the slot open while keeping the adjacent roof solid.
    This is an authored traverse interface, not a photographed OEM mechanism.
    """
    outer = ((0.54, -2.4096), (1.46, -2.4096), (1.46, -1.1904), (0.54, -1.1904))
    inner = ((0.878, -2.185), (0.922, -2.185), (0.922, -1.415), (0.878, -1.415))
    vertices = [
        (x, y, z) for z in (2.7596, 2.7836) for loop in (outer, inner) for x, y in loop
    ]
    faces = []
    for edge in range(4):
        following = (edge + 1) % 4
        for a, c, d, e in (
            (edge, following, following + 8, edge + 8),
            (edge + 4, following + 4, following + 12, edge + 12),
            (edge, following, following + 4, edge + 4),
            (edge + 8, following + 8, following + 12, edge + 12),
        ):
            faces.extend(((a, c, d), (a, d, e)))
    for xmin, xmax, ymin, ymax in (
        (0.54, 0.878, -2.4096, -1.1904),
        (0.922, 1.46, -2.4096, -1.1904),
        (0.878, 0.922, -2.4096, -2.185),
        (0.878, 0.922, -1.415, -1.1904),
    ):
        geom = b.box(
            base,
            ((xmin + xmax) / 2, (ymin + ymax) / 2, 2.7716),
            ((xmax - xmin) / 2, (ymax - ymin) / 2, 0.012),
            "glass",
            collision=True,
        )
        geom.set("name", b.unique("bam_roof_slot_contact"))
        geom.set("rgba", "0 0 0 0")
        geom.set("group", "3")
    return _mesh(b, base, "test_horizontal_panel", vertices, faces, material="glass")
