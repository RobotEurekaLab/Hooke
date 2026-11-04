"""Original open Cornell loading mechanism, in metres.

The photograph supports the dense exterior manifold, not these hidden passages
or the access mechanism. Bores, bearing clearances and hinge are authored aids
for geometric inspection; they are not vacuum seals or OEM hardware dimensions.
"""

import math
import xml.etree.ElementTree as ET


def _mesh(builder, name, vertices, faces):
    name = f"{builder.name}__loading_{name}_mesh"
    ET.SubElement(
        builder.root.find("asset"),
        "mesh",
        name=name,
        vertex=" ".join(f"{v:.10g}" for point in vertices for v in point),
        face=" ".join(str(i) for face in faces for i in face),
    )
    return name


def tube(
    builder,
    parent,
    key,
    center,
    inner,
    outer,
    sections,
    *,
    axis="y",
    segments=48,
    omitted=None,
    material="bright",
):
    """One watertight visible tube plus separate convex sector contact pieces.

    Optional omitted (axial section, angular sector) cells form real side ports.
    Internal cell faces cancel from the visual mesh, avoiding rendered seams.
    Convex collision pieces never span the central bore or a missing cell.
    """
    center = tuple(center)
    vertices, faces, lookup, boundary = [], [], {}, {}

    def vertex(radius, theta, length):
        a, b = radius * math.cos(theta), radius * math.sin(theta)
        xyz = {"y": (a, length, b), "x": (length, a, b), "z": (a, b, length)}[axis]
        xyz = tuple(round(center[i] + xyz[i], 11) for i in range(3))
        if xyz not in lookup:
            lookup[xyz] = len(vertices)
            vertices.append(xyz)
        return lookup[xyz]

    quads = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    # Mapping the cross section to X/Z with axial Y reverses handedness.
    # Correct both visible boundary normals and each convex sector's winding.
    if axis == "y":
        quads = tuple(tuple(reversed(quad)) for quad in quads)
    for row, (lo, hi) in enumerate(zip(sections, sections[1:])):
        for col in range(segments):
            if omitted and omitted(row, col):
                continue
            theta, phi = math.tau * col / segments, math.tau * (col + 1) / segments
            cell = [
                vertex(r, a, y)
                for y in (lo, hi)
                for r, a in ((inner, theta), (outer, theta), (outer, phi), (inner, phi))
            ]
            local_faces = []
            for quad in quads:
                polygon = tuple(cell[i] for i in quad)
                signature = tuple(sorted(polygon))
                if signature in boundary:
                    del boundary[signature]
                else:
                    boundary[signature] = polygon
                local_faces.extend(
                    ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3]))
                )
            mesh = _mesh(
                builder,
                f"{key}_contact_{row}_{col}",
                [vertices[i] for i in cell],
                local_faces,
            )
            builder.geom(
                parent,
                "mesh",
                (),
                material=material,
                mesh=mesh,
                collision=True,
                group="3",
                name=f"{builder.name}__loading_{key}_contact_{row}_{col}",
            )
    for quad in boundary.values():
        faces.extend(((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])))
    # Drop vertices belonging only to omitted/internal cells for compact meshes.
    used = sorted({i for face in faces for i in face})
    remap = {old: new for new, old in enumerate(used)}
    mesh = _mesh(
        builder,
        key + "_surface",
        [vertices[i] for i in used],
        [tuple(remap[i] for i in face) for face in faces],
    )
    builder.geom(
        parent,
        "mesh",
        (),
        material=material,
        mesh=mesh,
        name=f"{builder.name}__loading_{key}_surface",
    )


def _box(builder, parent, key, pos, size, material="metal", **kwargs):
    return builder.box(
        parent,
        pos,
        size,
        material,
        collision=True,
        name=f"{builder.name}__loading_{key}",
        **kwargs,
    )


def _rectangular_passage(builder, parent, key, start, end, y, z):
    """Four walls leave a 90 by 110 mm open holder passage along X."""
    length, x = (end - start) / 2, (start + end) / 2
    for side in (-1, 1):
        _box(
            builder,
            parent,
            key + f"_side_{side}",
            (x, y + side * 0.050, z),
            (length, 0.005, 0.065),
            "bright",
        )
        _box(
            builder,
            parent,
            key + f"_roof_{side}",
            (x, y, z + side * 0.060),
            (length, 0.045, 0.005),
            "bright",
        )


def add_loading(builder, base):
    """Preserve both original body frames, axis ranges and target dimensions."""
    # The middle axial interval has matched left/right side openings. The
    # adjacent ring material, not an invisible weld, supports the branch walls.
    tube(
        builder,
        base,
        "chamber",
        (1.31, 0, 1.22),
        0.143,
        0.155,
        (-0.30, 0.005, 0.135, 0.22),
        omitted=lambda row, col: row == 1 and (col < 3 or col >= 45 or 21 <= col < 27),
    )
    tube(
        builder,
        base,
        "front_rim",
        (1.31, 0, 1.22),
        0.1275,
        0.187,
        (-0.317, -0.299),
        segments=48,
    )
    builder.cylinder(
        base,
        (1.31, 0.227, 1.22),
        0.155,
        0.007,
        "metal",
        euler=(math.pi / 2, 0, 0),
        collision=True,
        name=f"{builder.name}__loading_rear_cap",
    )
    _rectangular_passage(builder, base, "left_branch", 1.015, 1.205, 0.07, 1.22)
    _rectangular_passage(builder, base, "right_branch", 1.415, 1.565, 0.07, 1.22)
    # Shorter sleeve keeps the handle clear at the full retained 200 mm stroke.
    tube(
        builder,
        base,
        "rod_sleeve",
        (0, 0.07, 1.22),
        0.016,
        0.033,
        (1.49, 1.68),
        axis="x",
        segments=32,
    )
    for x in (1.50, 1.65):
        tube(
            builder,
            base,
            f"guide_collar_{x}",
            (0, 0.07, 1.22),
            0.016,
            0.045,
            (x - 0.008, x + 0.008),
            axis="x",
            segments=32,
        )
        _box(builder, base, f"guide_stand_{x}", (x, 0.07, 1.070), (0.023, 0.030, 0.105))
    # Both legs contact the existing worktop at z=.965 and meet collar bottoms.
    # Side branch end plate supports the sleeve through a genuine rod opening.
    tube(
        builder,
        base,
        "sleeve_reducer",
        (0, 0.07, 1.22),
        0.016,
        0.057,
        (1.548, 1.562),
        axis="x",
        segments=32,
    )
    # Front knuckles meet a fixed rear bracket; moving leaf meets the door disc.
    pivot = (1.125, -0.335, 1.22)
    builder.cylinder(
        base,
        pivot,
        0.006,
        0.061,
        "metal",
        collision=True,
        name=f"{builder.name}__loading_hinge_pin",
    )
    for side in (-1, 1):
        tube(
            builder,
            base,
            f"fixed_knuckle_{side}",
            pivot,
            0.006,
            0.016,
            tuple(sorted((side * 0.034, side * 0.056))),
            axis="z",
            segments=24,
        )
        _box(
            builder,
            base,
            f"hinge_bracket_{side}",
            (1.125, -0.294, 1.22 + side * 0.045),
            (0.018, 0.041, 0.011),
        )
    door = builder.moving(
        base,
        "load_lock_door",
        pivot,
        (0, 0, -1),
        (0, 1.25),
        kind="hinge",
        mass=0.7,
        kp=450,
    )
    tube(
        builder,
        door,
        "moving_knuckle",
        (0, 0, 0),
        0.007,
        0.015,
        (-0.026, 0.026),
        axis="z",
        segments=24,
    )
    _box(builder, door, "hinge_leaf", (0.042, 0, 0), (0.029, 0.006, 0.022))
    builder.cylinder(
        door,
        (0.185, 0, 0),
        0.158,
        0.010,
        "bright",
        euler=(math.pi / 2, 0, 0),
        collision=True,
        name=f"{builder.name}__loading_door_disc",
    )
    # A shallow outer-face insert is opaque; it is not a fictitious viewing bore.
    builder.cylinder(
        door, (0.185, -0.012, 0), 0.11, 0.003, "metal", euler=(math.pi / 2, 0, 0)
    )
    builder.rod(
        door,
        (0.29, -0.043, -0.052),
        (0.29, -0.043, 0.052),
        0.011,
        "dark",
        collision=True,
        name=f"{builder.name}__loading_door_handle",
    )
    for z in (-0.052, 0.052):
        builder.rod(
            door,
            (0.29, -0.01, z),
            (0.29, -0.043, z),
            0.008,
            "dark",
            collision=True,
            name=f"{builder.name}__loading_handle_stem_{z}",
        )
    builder.site(door, "door_handle", (0.29, -0.043, 0))
    transfer = builder.moving(
        base,
        "transfer_rod",
        (1.82, 0.07, 1.22),
        (-1, 0, 0),
        (0, 0.20),
        mass=0.25,
        kp=600,
    )
    builder.rod(
        transfer,
        (-0.57, 0, 0),
        (0.18, 0, 0),
        0.012,
        "bright",
        collision=True,
        name=f"{builder.name}__loading_transfer_shaft",
    )
    builder.cylinder(
        transfer,
        (0.18, 0, 0),
        0.04,
        0.055,
        "dark",
        euler=(0, math.pi / 2, 0),
        collision=True,
        name=f"{builder.name}__loading_transfer_handle",
    )
    _box(
        builder, transfer, "carrier", (-0.54, 0, 0.016), (0.034, 0.028, 0.006), "copper"
    )
    _box(builder, transfer, "coupon", (-0.54, 0, 0.024), (0.021, 0.019, 0.002), "lens")
    # Small rim clamps explicitly support the rigid attachment; they do not
    # imply gripping physics or scientific thin-film growth.
    for side in (-1, 1):
        _box(
            builder,
            transfer,
            f"coupon_clamp_{side}",
            (-0.54, side * 0.022, 0.024),
            (0.024, 0.003, 0.004),
            "metal",
        )
    builder.site(transfer, "substrate", (-0.54, 0, 0.026))
    builder.metadata["loading_inspection"] = {
        "hatch_open_rad": 1.10,
        "inspection_transfer_m": 0.04,
        "camera": "loading_inspection",
        "source_geometry": "authored interior",
        "scope": "Rigid coupon access and measured return; no vacuum process",
    }


def add_manifold_details(builder, base, flange, hose):
    """Source-visible density and support cues; hidden plumbing is unverified."""
    # Two extrusion rails carry the observed lower right-hand service manifold.
    for y in (-0.39, 0.33):
        _box(
            builder, base, f"manifold_rail_{y}", (1.31, y, 0.986), (0.38, 0.021, 0.021)
        )
    for x in (1.11, 1.36, 1.55):
        # Rear branches remain behind the clear front viewing/access cone.
        builder.rod(base, (x, 0.31, 1.01), (x, 0.31, 1.11), 0.038, "bright")
        flange(builder, base, (x, 0.31, 1.10), 0.052, bolts=8)
        _box(
            builder, base, f"manifold_mount_{x}", (x, 0.31, 1.002), (0.05, 0.046, 0.016)
        )
    builder.rod(base, (1.06, 0.31, 1.10), (1.60, 0.31, 1.10), 0.032, "bright")
    for i, x in enumerate((1.11, 1.36, 1.55)):
        hose(
            builder,
            base,
            [
                (x, 0.31, 1.10),
                (x + 0.08, 0.53, 1.08),
                (0.55 + i * 0.08, 0.58, 0.89),
                (0.49 + i * 0.08, 0.48, 0.76),
            ],
            0.004,
            "dark" if i % 2 else "blue",
        )
    # The source's short valve forest is asymmetric, with different actuator
    # heights and exposed flanges. Keep it behind the measured loading sweep.
    for index, (x, height) in enumerate(((1.02, .13), (1.20, .21), (1.43, .16), (1.63, .27))):
        builder.rod(base, (x, .43, 1.01), (x, .43, 1.10 + height), .022, "bright")
        flange(builder, base, (x, .43, 1.14), .036, bolts=6)
        builder.box(base, (x, .43, 1.13 + height), (.032, .039, .030), "dark",
                    name=builder.unique(f"manifold_valve_head_{index}"))
        builder.box(base, (x, .43, 1.005), (.037, .055, .019), "metal",
                    name=builder.unique(f"manifold_branch_mount_{index}"))
        hose(builder, base, [(x, .47, 1.13 + height), (x + .035, .58, 1.04),
             (1.48, .58, .79), (.57, .48, .76)], .003,
             ("blue", "dark", "warning", "dark")[index])
    # Two unequal black controllers are visible at the front-left vessel;
    # their brackets connect to its existing frame without inventing internals.
    for index, (x, y, width, depth, z) in enumerate(((-1.14, -.50, .26, .20, 1.03),
                                                    (-.68, -.43, .32, .24, 1.13))):
        builder.box(base, (x, y, z), (width / 2, depth / 2, .085), "dark",
                    name=builder.unique(f"asymmetric_controller_{index}"))
        builder.rod(base, (x, y + depth / 2, z - .06), (x, -.23, .925), .012, "metal")
        for row in range(6):
            builder.box(base, (x, y - depth / 2 - .001, z - .055 + row * .019),
                        (width * .32, .001, .003), "rubber",
                        name=builder.unique(f"controller_vent_{index}_{row}"))
