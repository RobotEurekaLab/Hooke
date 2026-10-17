"""Parametric laboratory furniture with explicit mechanical simplifications.

Optical-table construction follows published manufacturer component concepts.
The dimensions of each installed bench remain authored scene estimates. Hole
marks and isolator bellows are visual geometry, not threads or air springs.
"""

from functools import lru_cache
import math
import xml.etree.ElementTree as ET


OPTICAL_REFERENCES = [
    "https://www.newport.com/p/M-INT1-46-8-A",
    "https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=7597&pn=TF1220A7",
]
STEEL = (0.60, 0.64, 0.67, 1)
FRAME = (0.20, 0.23, 0.25, 1)
RUBBER = (0.035, 0.04, 0.045, 1)


def _numbers(values):
    return " ".join(f"{value:.9g}" for value in values)


def _box(parent, name, size, position, color, *, collision=True, **attributes):
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type="box",
        size=_numbers(size),
        pos=_numbers(position),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
        **attributes,
    )


def _cylinder(parent, name, radius, bottom, top, xy, color, *, collision=True):
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type="cylinder",
        size=_numbers((radius, (top - bottom) / 2)),
        pos=_numbers((*xy, (bottom + top) / 2)),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
    )


def _rod(parent, name, first, second, radius, color, *, collision=True):
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type="cylinder",
        fromto=_numbers((*first, *second)),
        size=str(radius),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
    )


@lru_cache(maxsize=32)
def optical_grid_mesh(width, depth, pitch=0.025, border=0.0125):
    """Batch a reusable tile of shallow marks into a closed, non-contact mesh.

    The top octagons represent nominal M6 openings. Their shallow closed backs
    keep the mesh portable through MuJoCo; the load-bearing skin remains solid.
    """
    columns = max(0, math.floor((width - 2 * border + 1e-9) / pitch) + 1)
    rows = max(0, math.floor((depth - 2 * border + 1e-9) / pitch) + 1)
    vertices, faces = [], []
    for i in range(columns):
        x = -width / 2 + border + i * pitch
        for j in range(rows):
            y = -depth / 2 + border + j * pitch
            start = len(vertices)
            vertices.extend(
                (
                    x + 0.003 * math.cos(k * math.tau / 8),
                    y + 0.003 * math.sin(k * math.tau / 8),
                    0.0,
                )
                for k in range(8)
            )
            vertices.append((x, y, -0.00010))
            faces.extend((start, start + k, start + k + 1) for k in range(1, 7))
            faces.extend((start + 8, start + (k + 1) % 8, start + k) for k in range(8))
    return (
        _numbers(value for point in vertices for value in point),
        " ".join(str(value) for face in faces for value in face),
        columns,
        rows,
    )


def _optical(parent, item, asset):
    if asset is None:
        raise ValueError("Optical bench construction requires the MJCF asset section")
    name = item["id"]
    width, depth, height = item["size"]
    thickness = float(item.get("tabletop_thickness", 0.203))
    if not 0.075 <= thickness < height - 0.22:
        raise ValueError("Optical tabletop thickness leaves no usable support frame")
    underside, skin = height - thickness, 0.0034
    metal = name + "__mat_metal"
    ET.SubElement(
        asset,
        "material",
        name=metal,
        rgba=_numbers(STEEL),
        specular="0.45",
        shininess="0.35",
    )
    _box(
        parent,
        name + "_top",
        (width / 2, depth / 2, skin / 2),
        (0, 0, height - skin / 2),
        STEEL,
        material=metal,
    )
    _box(
        parent,
        name + "_sandwich_core",
        (width / 2 - 0.002, depth / 2 - 0.002, (thickness - 2 * skin) / 2),
        (0, 0, height - thickness / 2),
        (0.15, 0.17, 0.18, 1),
    )
    _box(
        parent,
        name + "_steel_bottom_skin",
        (width / 2, depth / 2, skin / 2),
        (0, 0, underside + skin / 2),
        STEEL,
        material=metal,
    )
    for side in (-1, 1):
        _box(
            parent,
            f"{name}_edge_trim_x_{side}",
            (0.002, depth / 2, (thickness - 2 * skin) / 2),
            (side * (width / 2 - 0.002), 0, height - thickness / 2),
            FRAME,
        )
        _box(
            parent,
            f"{name}_edge_trim_y_{side}",
            (width / 2 - 0.004, 0.002, (thickness - 2 * skin) / 2),
            (0, side * (depth / 2 - 0.002), height - thickness / 2),
            FRAME,
        )
    pitch, border, tile_size = 0.025, 0.0125, 20
    columns = math.floor((width - 2 * border + 1e-9) / pitch) + 1
    rows = math.floor((depth - 2 * border + 1e-9) / pitch) + 1
    mesh_names, instances = set(), 0
    for first_col in range(0, columns, tile_size):
        for first_row in range(0, rows, tile_size):
            tile_cols = min(tile_size, columns - first_col)
            tile_rows = min(tile_size, rows - first_row)
            mesh = f"bench_grid_m6_{tile_cols}_{tile_rows}"
            if asset.find(f"mesh[@name='{mesh}']") is None:
                vertex, face, _, _ = optical_grid_mesh(
                    tile_cols * pitch, tile_rows * pitch
                )
                ET.SubElement(
                    asset,
                    "mesh",
                    name=mesh,
                    vertex=vertex,
                    face=face,
                    inertia="shell",
                    maxhullvert="4",
                )
            mesh_names.add(mesh)
            x = -width / 2 + border + (first_col + (tile_cols - 1) / 2) * pitch
            y = -depth / 2 + border + (first_row + (tile_rows - 1) / 2) * pitch
            ET.SubElement(
                parent,
                "geom",
                name=f"{name}_m6_grid_visual_{instances}",
                type="mesh",
                mesh=mesh,
                pos=_numbers((x, y, height + 0.00012)),
                rgba=_numbers(RUBBER),
                contype="0",
                conaffinity="0",
                mass="0",
            )
            instances += 1
    leg_x, leg_y = width / 2 - 0.15, depth / 2 - 0.15
    for i, x in enumerate((-leg_x, leg_x)):
        for j, y in enumerate((-leg_y, leg_y)):
            prefix = f"{name}_isolator_{i}_{j}"
            _cylinder(parent, prefix + "_rubber_foot", 0.072, 0, 0.024, (x, y), RUBBER)
            _cylinder(
                parent, prefix + "_steel_level_stem", 0.018, 0.024, 0.066, (x, y), STEEL
            )
            _box(
                parent,
                prefix + "_base_plate",
                (0.11, 0.11, 0.012),
                (x, y, 0.078),
                FRAME,
            )
            _cylinder(
                parent, prefix + "_can", 0.09, 0.09, underside - 0.065, (x, y), FRAME
            )
            _cylinder(
                parent,
                prefix + "_steel_collar",
                0.094,
                underside - 0.078,
                underside - 0.064,
                (x, y),
                STEEL,
            )
            _cylinder(
                parent,
                prefix + "_rubber_bellows",
                0.064,
                underside - 0.065,
                underside - 0.016,
                (x, y),
                RUBBER,
            )
            for ring in range(4):
                z = underside - 0.061 + ring * 0.011
                _cylinder(
                    parent,
                    prefix + f"_rubber_rib_{ring}",
                    0.07,
                    z,
                    z + 0.005,
                    (x, y),
                    RUBBER,
                    collision=False,
                )
            _cylinder(
                parent,
                prefix + "_steel_piston",
                0.048,
                underside - 0.022,
                underside - 0.009,
                (x, y),
                STEEL,
            )
            _cylinder(
                parent,
                prefix + "_load_cap",
                0.080,
                underside - 0.009,
                underside,
                (x, y),
                FRAME,
            )
    _box(parent, name + "_frame_rear", (leg_x, 0.025, 0.030), (0, leg_y, 0.19), FRAME)
    for i, x in enumerate((-leg_x, leg_x)):
        _box(
            parent,
            name + f"_frame_side_{i}",
            (0.025, leg_y, 0.030),
            (x, 0, 0.19),
            FRAME,
        )
    return dict(
        tabletop_thickness_m=thickness,
        steel_skin_thickness_m=skin,
        optical_grid=dict(
            pitch_m=0.025,
            nominal_thread="M6",
            columns=columns,
            rows=rows,
            visual_marks=columns * rows,
            visual_meshes=len(mesh_names),
            visual_mesh_instances=instances,
            physical_thread_contact=False,
        ),
        support="Four fixed isolator-shaped columns, support frame and levelling feet",
        pneumatic_dynamics=False,
        source_urls=OPTICAL_REFERENCES,
        geometry_basis="Representative manufacturer construction adapted to authored bench dimensions; installed model identity unverified",
        collision_boundary="Solid top skin, core, frame and supports; mounting-hole marks and bellows ribs are appearance geometry",
    )


def _open_frame(parent, item):
    name = item["id"]
    width, depth, height = item["size"]
    underside = height - 0.035
    leg_x, leg_y = width / 2 - 0.15, depth / 2 - 0.15
    for i, x in enumerate((-leg_x, leg_x)):
        for j, y in enumerate((-leg_y, leg_y)):
            prefix = f"{name}_leg_{i}_{j}"
            _cylinder(parent, prefix + "_rubber_foot", 0.055, 0, 0.025, (x, y), RUBBER)
            _cylinder(
                parent, prefix + "_steel_level_stem", 0.011, 0.025, 0.065, (x, y), STEEL
            )
            _box(
                parent,
                prefix,
                (0.035, 0.035, (underside - 0.065) / 2),
                (x, y, (underside + 0.065) / 2),
                FRAME,
            )
            for axis in (0, 1):
                offset = 0.040 if axis == 0 else -0.040
                pos = (
                    (x + offset, y, underside - 0.075)
                    if axis == 0
                    else (x, y + offset, underside - 0.075)
                )
                size = (0.005, 0.027, 0.075) if axis == 0 else (0.027, 0.005, 0.075)
                _box(
                    parent,
                    prefix + f"_steel_bracket_{axis}",
                    size,
                    pos,
                    STEEL,
                    collision=False,
                )
                first = (
                    (pos[0] - 0.006, pos[1], pos[2])
                    if axis == 0
                    else (pos[0], pos[1] - 0.006, pos[2])
                )
                second = (
                    (pos[0] + 0.006, pos[1], pos[2])
                    if axis == 0
                    else (pos[0], pos[1] + 0.006, pos[2])
                )
                _rod(
                    parent,
                    prefix + f"_steel_fastener_{axis}",
                    first,
                    second,
                    0.007,
                    STEEL,
                    collision=False,
                )
    for j, y in enumerate((-leg_y, leg_y)):
        _box(
            parent,
            name + f"_apron_x_{j}",
            (leg_x, 0.025, 0.025),
            (0, y, underside - 0.025),
            FRAME,
        )
    for i, x in enumerate((-leg_x, leg_x)):
        _box(
            parent,
            name + f"_apron_y_{i}",
            (0.025, leg_y, 0.025),
            (x, 0, underside - 0.025),
            FRAME,
        )
        _box(
            parent,
            name + f"_lower_side_{i}",
            (0.025, leg_y, 0.020),
            (x, 0, 0.22),
            FRAME,
        )
    _box(parent, name + "_brace", (leg_x, 0.025, 0.025), (0, leg_y, 0.22), FRAME)
    for side in (-1, 1):
        _rod(
            parent,
            name + f"_rear_diagonal_{side}",
            (-leg_x, leg_y, 0.24 if side == 1 else underside - 0.055),
            (leg_x, leg_y, underside - 0.055 if side == 1 else 0.24),
            0.010,
            FRAME,
        )
    return dict(
        support="Fixed rectangular frame, rear cross-bracing, corner brackets and levelling feet",
        geometry_basis="Authored representative construction; installed frame identity unverified",
        collision_boundary="Top, structural frame and feet have contact geometry; bracket fasteners are appearance details",
        source_urls=[],
    )


def _cabinet(parent, item, color):
    name = item["id"]
    width, depth, height = item["size"]
    underside = height - 0.035
    _box(
        parent,
        name + "_toe",
        (width / 2 - 0.055, depth / 2 - 0.06, 0.055),
        (0, 0, 0.055),
        FRAME,
    )
    _box(
        parent,
        name + "_cabinet",
        (width / 2 - 0.025, depth / 2 - 0.043, (underside - 0.11) / 2),
        (0, 0, (underside + 0.11) / 2),
        color,
    )
    count = max(2, round(width / 0.55))
    for index in range(count):
        x = -width / 2 + (index + 0.5) * width / count
        _box(
            parent,
            name + f"_door_{index}",
            (width / count / 2 - 0.011, 0.009, (underside - 0.15) / 2),
            (x, -depth / 2 + 0.034, (underside + 0.15) / 2),
            color,
        )
        z, y = height - 0.13, -depth / 2 + 0.010
        _rod(
            parent,
            name + f"_steel_pull_{index}",
            (x - 0.065, y, z),
            (x + 0.065, y, z),
            0.006,
            STEEL,
            collision=False,
        )
        for side in (-1, 1):
            _rod(
                parent,
                name + f"_steel_pull_mount_{index}_{side}",
                (x + side * 0.06, y, z),
                (x + side * 0.06, -depth / 2 + 0.028, z),
                0.004,
                STEEL,
                collision=False,
            )
    return dict(
        support="Fixed cabinet carcase and recessed plinth; attached pull bars remain within countertop footprint",
        geometry_basis="Authored representative cabinetry; doors remain fixed",
        collision_boundary="Solid cabinet and counter proxies; pull bars are visual hardware, not graspable articulated drawers",
        source_urls=[],
    )


def build_furniture(parent, item, accent, asset=None):
    """Build within the existing bench envelope and return explicit assumptions."""
    width, depth, height = map(float, item["size"])
    if min(width, depth) <= 0.32 or height <= 0.35:
        raise ValueError("Bench dimensions do not accommodate the support design")
    style, name = item.get("style", "cabinet"), item["id"]
    if style == "optical":
        record = _optical(parent, item, asset)
    else:
        _box(
            parent,
            name + "_top",
            (width / 2, depth / 2, 0.0175),
            (0, 0, height - 0.0175),
            item.get("top_color", (0.13, 0.15, 0.16, 1)),
        )
        if style == "open_frame":
            record = _open_frame(parent, item)
        elif style == "cabinet":
            record = _cabinet(parent, item, item.get("color", accent))
        else:
            raise ValueError(f"Unknown bench style: {style}")
    return dict(
        bench_id=name,
        style=style,
        working_height_m=height,
        footprint_m=[width, depth],
        simulated_isolation=False,
        **record,
    )
