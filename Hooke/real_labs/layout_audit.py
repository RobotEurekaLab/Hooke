"""Offline plan checks and an inspectable SVG; no simulator or renderer imports."""

from itertools import combinations
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


SCOPE = (
    "Authored geometry checks, not measured-source accuracy or building-code approval. "
    "The 1 m doorway approach is a screening convention, not an egress standard. "
    "Only catalogued benches are tested as obstacles; equipment envelopes, furnishings, "
    "door swings, human reach and dynamic collision clearance need separate review."
)
_FLOOR_KINDS = {"sequencer", "phenotyping_booth", "mobile_robot", "rover"}
_TOLERANCE = 1e-8


def model_obstacles(model, data):
    """Project collision bounds intersecting z=0.12..1.8 m into world XY.

    ``data`` must already have current forward-kinematics transforms. Local
    ``geom_aabb`` centres and half extents are transformed without importing a
    simulator. These conservative boxes include fixtures and equipment; they
    are not exact occupied silhouettes or a pedestrian-clearance calculation.
    """
    obstacles = []
    for index in range(model.ngeom):
        if not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        name = model.geom(index).name or f"geom_{index}"
        is_floor = (
            name == "floor"
            or name.startswith("arch_floor")
            or name.endswith("_floor")
            or name.startswith("floor_")
        )
        is_frame = name.startswith("arch_opening_") and (
            "_jamb_" in name or "_lintel_" in name or name.endswith("_roller")
        )
        if is_floor or is_frame or name.startswith("arch_roof"):
            continue
        bounds = np.asarray(model.geom_aabb[index], dtype=float)
        rotation = np.asarray(data.geom_xmat[index], dtype=float).reshape(3, 3)
        position = np.asarray(data.geom_xpos[index], dtype=float)
        if (
            bounds.shape != (6,)
            or position.shape != (3,)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[3:] < 0)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(position))
        ):
            raise ValueError(f"Invalid collision bounds or transform: {name}")
        center = position + rotation @ bounds[:3]
        half_extent = np.abs(rotation) @ bounds[3:]
        minimum, maximum = center - half_extent, center + half_extent
        if not np.all(np.isfinite(np.r_[minimum, maximum])):
            raise ValueError(f"Nonfinite projected collision bounds: {name}")
        if maximum[2] < 0.12 or minimum[2] > 1.8:
            continue
        if np.any(maximum[:2] - minimum[:2] <= _TOLERANCE):
            continue
        obstacles.append(
            {
                "name": name,
                "minimum_xy": minimum[:2].tolist(),
                "maximum_xy": maximum[:2].tolist(),
            }
        )
    return obstacles


def _obstacle_polygon(obstacle):
    minimum = np.asarray(obstacle["minimum_xy"], dtype=float)
    maximum = np.asarray(obstacle["maximum_xy"], dtype=float)
    if (
        minimum.shape != (2,)
        or maximum.shape != (2,)
        or not np.all(np.isfinite(np.r_[minimum, maximum]))
        or np.any(maximum < minimum)
    ):
        raise ValueError(f"Invalid projected obstacle: {obstacle['name']}")
    return np.array(
        (minimum, (maximum[0], minimum[1]), maximum, (minimum[0], maximum[1]))
    )


def doorway_obstacle_warnings(definition, obstacles):
    """Flag conservative bound intersections for review, never certify egress."""
    width, depth = definition["room"]["size"][:2]
    projected = [(row["name"], _obstacle_polygon(row)) for row in obstacles]
    warnings = []
    for opening in definition["room"].get("architecture", {}).get("openings", []):
        if opening["kind"] not in ("door", "shutter"):
            continue
        approach = _door_approach(opening, width, depth)
        intersections = [
            name for name, polygon in projected if _overlap(approach, polygon)
        ]
        if intersections:
            warnings.append(
                {
                    "opening": opening["id"],
                    "approach_depth_m": 1.0,
                    "intersecting_bounds": intersections,
                    "scope": "Conservative projected collision bounds; inspect actual clearance. Does not alter bench-audit pass status.",
                }
            )
    return warnings


def _footprint(bench):
    width, depth = bench["size"][:2]
    angle = float(bench.get("yaw", 0))
    rotation = np.array(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    )
    corners = np.array(
        (
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2),
        )
    )
    return corners @ rotation.T + np.asarray(bench["pos"][:2])


def _overlap(first, second):
    """Separating-axis test; edge/corner contact is not an area overlap."""
    for polygon in (first, second):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.array((-edge[1], edge[0]))
            length = np.linalg.norm(axis)
            if length <= _TOLERANCE:
                continue
            axis /= length
            left, right = first @ axis, second @ axis
            if (
                min(left.max(), right.max()) - max(left.min(), right.min())
                <= _TOLERANCE
            ):
                return False
    return True


def _door_approach(opening, width, depth):
    center, half = float(opening["offset"]), float(opening["width"]) / 2
    if opening["wall"] == "back":
        return np.array(
            (
                (center - half, depth / 2 - 1),
                (center + half, depth / 2 - 1),
                (center + half, depth / 2),
                (center - half, depth / 2),
            )
        )
    if opening["wall"] == "left":
        return np.array(
            (
                (-width / 2, center - half),
                (-width / 2 + 1, center - half),
                (-width / 2 + 1, center + half),
                (-width / 2, center + half),
            )
        )
    raise ValueError(f"Unsupported doorway wall: {opening['wall']}")


def _contains_origin(bench, position):
    angle = float(bench.get("yaw", 0))
    offset = np.asarray(position[:2]) - np.asarray(bench["pos"][:2])
    # Inverse rotation maps the point to the bench's local rectangle.
    local = (
        np.array(
            ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle)))
        )
        @ offset
    )
    return bool(np.all(np.abs(local) <= np.asarray(bench["size"][:2]) / 2 + _TOLERANCE))


def audit_layout(definition):
    """Check authored bench/door geometry and expose unverified support origins.

    ``passed`` covers the listed geometric checks only. ``manual_inspection`` is
    deliberately separate: an instrument origin over a bench is not proof of
    full-footprint support, and elevated floor platforms are outside this audit.
    """
    report = {
        "scene_id": definition["id"],
        "scope": SCOPE,
        "passed": True,
        "checks": [],
        "equipment_support": [],
        "manual_inspection": [],
        "manual_inspection_count": 0,
        "all_equipment_origins_classified": False,
    }

    def check(name, passed, **details):
        report["checks"].append({"check": name, "passed": bool(passed), **details})
        if not passed:
            report["passed"] = False

    dimensions = np.asarray(definition["room"]["size"], dtype=float)
    valid_room = (
        dimensions.shape == (3,)
        and np.all(np.isfinite(dimensions))
        and np.all(dimensions > 0)
    )
    check("positive_finite_room_dimensions", valid_room)
    if not valid_room:
        return report
    width, depth, height = dimensions
    benches, polygons = definition.get("benches", []), {}
    names = [bench["id"] for bench in benches]
    check("unique_bench_identifiers", len(names) == len(set(names)))
    for bench in benches:
        size = np.asarray(bench["size"], dtype=float)
        position = np.asarray(bench["pos"], dtype=float)
        valid = (
            size.shape == (3,)
            and position.shape == (3,)
            and np.all(np.isfinite(size))
            and np.all(size > 0)
            and np.all(np.isfinite(position))
            and math.isfinite(bench.get("yaw", 0))
        )
        check("valid_bench_dimensions", valid, bench=bench["id"])
        if not valid:
            continue
        polygon = _footprint(bench)
        polygons[bench["id"]] = polygon
        margins = np.asarray((width / 2, depth / 2)) - np.abs(polygon)
        check(
            "bench_inside_room",
            np.all(margins >= -_TOLERANCE),
            bench=bench["id"],
            minimum_boundary_clearance_m=float(margins.min()),
        )
    for (first, a), (second, b) in combinations(polygons.items(), 2):
        check(
            "bench_footprints_do_not_overlap",
            not _overlap(a, b),
            benches=[first, second],
        )

    openings = definition["room"].get("architecture", {}).get("openings", [])
    for opening in openings:
        if opening["kind"] not in ("door", "shutter"):
            continue
        approach = _door_approach(opening, width, depth)
        blocked_by = [
            name for name, polygon in polygons.items() if _overlap(approach, polygon)
        ]
        check(
            "doorway_approach_clear_of_benches",
            not blocked_by,
            opening=opening["id"],
            approach_depth_m=1.0,
            blocked_by=blocked_by,
        )

    interior = definition.get("camera", {}).get("interior")
    if interior is None:
        check("interior_camera_inside_room", False, detail="Interior camera missing")
    else:
        position = np.asarray(interior["position"], dtype=float)
        inside = (
            position.shape == (3,)
            and np.all(np.isfinite(position))
            and abs(position[0]) < width / 2
            and abs(position[1]) < depth / 2
            and 0 < position[2] < height
        )
        check("interior_camera_inside_room", inside, position_m=position.tolist())

    for equipment in definition.get("equipment", []):
        position = equipment["pos"]
        check(
            "equipment_origin_inside_room",
            abs(position[0]) <= width / 2
            and abs(position[1]) <= depth / 2
            and 0 <= position[2] < height,
            equipment=equipment["id"],
            origin_m=position,
        )
        supports = [
            b
            for b in benches
            if b["id"] in polygons
            and _contains_origin(b, position)
            and -0.005 <= position[2] - (b["pos"][2] + b["size"][2]) <= 0.08
        ]
        row = {"equipment": equipment["id"], "origin_m": position}
        if supports:
            row.update(
                status="origin_over_bench",
                benches=[b["id"] for b in supports],
                detail="Origin only; 0.08 m allowance for local asset-origin offsets. Full footprint and contact are unverified.",
            )
        elif (
            equipment["kind"] in _FLOOR_KINDS
            or equipment.get("support", {}).get("type") == "floor"
        ) and abs(position[2]) <= 0.005:
            row.update(
                status="expected_floor_equipment",
                detail="Authored floor-mounted category; contact and full footprint are unverified.",
            )
        else:
            row.update(
                status="manual_inspection_required",
                detail="No matching bench top at this origin; inspect a floor mount, platform, offset or missing support.",
            )
            if equipment.get("params", {}).get("authored_pose"):
                row["authored_pose"] = equipment["params"]["authored_pose"]
            report["manual_inspection"].append(row.copy())
        report["equipment_support"].append(row)
    report["manual_inspection_count"] = len(report["manual_inspection"])
    report["all_equipment_origins_classified"] = not report["manual_inspection"]
    return report


def write_floorplan(definition, output_path, obstacles=()):
    """Write a labelled, metre-scaled plan SVG with estimates visibly identified."""
    width, depth, _ = map(float, definition["room"]["size"])
    if not all(math.isfinite(v) and v > 0 for v in (width, depth)):
        raise ValueError("Floorplan requires positive finite room dimensions")
    report = audit_layout(definition)
    obstacles = list(obstacles)
    scale = min(980 / width, 570 / depth)
    left, top = 100 + (980 - width * scale) / 2, 125
    plan_bottom = top + depth * scale
    benches, equipment = definition.get("benches", []), definition.get("equipment", [])
    legend_top = plan_bottom + 98
    canvas_height = int(legend_top + max(len(benches), len(equipment)) * 24 + 108)
    context = (
        "Grey: conservative collision bounds at z=0.12–1.8 m, including fixtures. Door swings remain unverified."
        if obstacles
        else "Bench footprint checks only. Furnishings, equipment envelopes and door swings are not represented."
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{canvas_height}" viewBox="0 0 1200 {canvas_height}">',
        '<rect width="100%" height="100%" fill="#f7f8f5"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#24323a} .minor{fill:#56636b;font-size:15px}</style>",
        f'<text x="60" y="45" font-size="27" font-weight="bold">{escape(definition.get("title", definition["id"]))}</text>',
        '<text x="60" y="74" class="minor">Authored layout; room dimensions and placements are estimates. Institutional accuracy is unverified.</text>',
        f'<text x="60" y="99" class="minor">{escape(context)}</text>',
    ]

    def point(x, y):
        return left + (x + width / 2) * scale, top + (depth / 2 - y) * scale

    def polygon(points, attributes):
        coordinates = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (point(*p) for p in points)
        )
        lines.append(f'<polygon points="{coordinates}" {attributes}/>')

    def text(x, y, label, attributes='font-size="14"'):
        lines.append(
            f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{escape(str(label))}</text>'
        )

    polygon(
        np.array(
            (
                (-width / 2, -depth / 2),
                (width / 2, -depth / 2),
                (width / 2, depth / 2),
                (-width / 2, depth / 2),
            )
        ),
        'fill="#e9ece8" stroke="#37484d" stroke-width="4"',
    )
    for obstacle in obstacles:
        polygon(
            _obstacle_polygon(obstacle),
            'fill="#b7bfbd" fill-opacity="0.30" stroke="#929f9d" stroke-width="0.65"',
        )
    for opening in definition["room"].get("architecture", {}).get("openings", []):
        if opening["kind"] not in ("door", "shutter"):
            continue
        approach = _door_approach(opening, width, depth)
        polygon(
            approach,
            'fill="#dcebdc" fill-opacity="0.72" stroke="#598164" stroke-dasharray="6 4" stroke-width="1.5"',
        )
        center, half = opening["offset"], opening["width"] / 2
        endpoints = (
            ((center - half, depth / 2), (center + half, depth / 2))
            if opening["wall"] == "back"
            else ((-width / 2, center - half), (-width / 2, center + half))
        )
        (x1, y1), (x2, y2) = [point(*p) for p in endpoints]
        lines.append(
            f'<path d="M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}" stroke="#f7f8f5" stroke-width="7"/>'
        )
        label_x, label_y = point(*np.mean(approach, axis=0))
        text(
            label_x,
            label_y,
            f'{opening["width"]:g} m entry',
            'text-anchor="middle" font-size="12" fill="#416149"',
        )

    for index, bench in enumerate(benches, 1):
        color = bench.get("color", definition["room"]["accent"])
        fill = "#" + "".join(
            f"{round(max(0, min(1, float(c))) * 255):02x}" for c in color[:3]
        )
        polygon(
            _footprint(bench),
            f'fill="{fill}" fill-opacity="0.52" stroke="#465157" stroke-width="1.5"',
        )
        x, y = point(*bench["pos"][:2])
        text(
            x,
            y + 5,
            f"B{index}",
            'text-anchor="middle" font-size="16" font-weight="bold"',
        )
        text(
            60,
            legend_top + (index - 1) * 24,
            f'B{index}: {bench["id"].replace("_", " ")} — {bench["size"][0]:g} × {bench["size"][1]:g} m',
        )
    for index, item in enumerate(equipment, 1):
        x, y = point(*item["pos"][:2])
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#bc5938" stroke="white" stroke-width="1.5"/>'
        )
        text(x + 7, y - 7, f"E{index}", 'font-size="13" font-weight="bold"')
        text(
            610,
            legend_top + (index - 1) * 24,
            f'E{index}: {item["id"].replace("_", " ")} (origin z={item["pos"][2]:g} m)',
        )
    interior = definition.get("camera", {}).get("interior")
    if interior:
        x, y = point(*interior["position"][:2])
        tx, ty = point(*interior["target"][:2])
        vector = np.array((tx - x, ty - y), dtype=float)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector *= 34 / norm
            lines.append(
                f'<path d="M{x:.2f},{y:.2f} l{vector[0]:.2f},{vector[1]:.2f}" stroke="#315b95" stroke-width="3"/>'
            )
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#315b95"/>')
        text(x + 9, y + 17, "Interior camera", 'font-size="12"')
    text(
        left + width * scale / 2,
        plan_bottom + 30,
        f"{width:g} m",
        'text-anchor="middle" font-size="18"',
    )
    text(
        left - 20,
        top + depth * scale / 2,
        f"{depth:g} m",
        'text-anchor="end" font-size="18"',
    )
    text(60, legend_top - 29, "BENCHES", 'font-size="15" font-weight="bold"')
    text(610, legend_top - 29, "EQUIPMENT ORIGINS", 'font-size="15" font-weight="bold"')
    footer = canvas_height - 58
    text(
        60,
        footer,
        f'Geometric checks: {"PASS" if report["passed"] else "FAIL"}; support origins needing manual inspection: {report["manual_inspection_count"]}.',
        'font-size="14"',
    )
    text(
        60,
        footer + 22,
        "Green dashed areas: 1 m entry screening zones. These are not a building-code approval.",
        'class="minor"',
    )
    if obstacles:
        warnings = doorway_obstacle_warnings(definition, obstacles)
        text(
            60,
            footer + 44,
            f"Grey projected bounds: {len(obstacles)}; entry zones requiring collision-bound review: {len(warnings)}. Bench PASS is independent.",
            'class="minor"',
        )
    lines.append("</svg>")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
