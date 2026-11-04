"""Historical room-photo context with supported, contact-capable exteriors.

The profiler envelope follows a published GeoMx dimension table. Its closed
exterior and the other parked items have no scientific or electrical behavior.
Shelf occupancy, small-device shapes and all placements are authored estimates.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.equipment import _Builder


PROFILER_SIZE_M = (0.76, 0.73, 0.61)


def _body(world, name, position, yaw=0):
    return ET.SubElement(
        world,
        "body",
        name=f"rochester_context__{name}",
        pos=" ".join(map(str, position)),
        euler=f"0 0 {yaw}",
    )


def _profiler(b, world):
    """Photo-visible historical apparatus, with an explicit supported envelope."""
    body = _body(world, "historical_profiler", (-3.76, -0.55, 0.92), math.pi / 2)
    for x in (-0.30, 0.30):
        for y in (-0.29, 0.29):
            b.cylinder(
                body,
                (x, y, 0.013),
                0.025,
                0.013,
                "dark",
                collision=True,
                name=f"rochester_context__profiler_foot_{x}_{y}",
            )
    b.box(
        body,
        (0, 0, 0.102),
        (0.355, 0.338, 0.076),
        "dark",
        collision=True,
        name="rochester_context__profiler_base_contact",
    )
    b.housing(
        body,
        [(0.026, 0.355, 0.338, 0), (0.065, 0.38, 0.365, 0), (0.164, 0.38, 0.365, 0)],
        radius=0.023,
        material="dark",
    )
    b.box(
        body,
        (0, 0, 0.381),
        (0.35, 0.325, 0.217),
        "shell",
        collision=True,
        name="rochester_context__profiler_shell_contact",
    )
    b.housing(
        body,
        [(0.164, 0.375, 0.359, 0), (0.569, 0.375, 0.350, 0), (0.61, 0.352, 0.328, 0)],
        radius=0.025,
    )
    # Dark lower frontage and a shallow opening handle are visible in the photo.
    b.box(body, (0, -0.352, 0.292), (0.354, 0.008, 0.119), "dark")
    b.box(body, (0, -0.361, 0.181), (0.351, 0.003, 0.003), "metal")
    b.box(body, (0, -0.355, 0.225), (0.095, 0.010, 0.012), "metal")
    b.box(body, (0.27, -0.360, 0.396), (0.011, 0.002, 0.005), "plant")
    for side in (-1, 1):
        for z in (0.092, 0.128):
            for i in range(9):
                b.box(
                    body,
                    (side * 0.379, -0.22 + i * 0.041, z),
                    (0.0007, 0.011, 0.002),
                    "dark",
                )
    b.site(body, "profiler_exterior_front", (0, -0.365, 0.29))


def _carton(b, world, name, position, size, color):
    width, depth, height = size
    body = _body(world, name, position)
    b.box(
        body,
        (0, 0, height / 2),
        (width / 2, depth / 2, height / 2),
        "cream",
        collision=True,
        rgba=color,
        name=f"rochester_context__{name}_contact",
    )
    b.box(
        body,
        (0, -depth / 2 - 0.0007, height * 0.47),
        (width * 0.30, 0.0007, min(0.022, height * 0.16)),
        "shell",
    )
    b.box(
        body,
        (0, 0, height + 0.0005),
        (0.016, depth / 2, 0.0005),
        "cream",
        rgba=(0.67, 0.53, 0.30, 1),
    )
    for x in (-width / 2, width / 2):
        b.box(body, (x, 0, height * 0.93), (0.0004, depth / 2, 0.001), "dark")


def _shelf_occupancy(b, world):
    # Deterministic observed categories, not repeated identical decorative bins.
    colors = (
        (0.64, 0.42, 0.19, 1),
        (0.82, 0.83, 0.78, 1),
        (0.20, 0.30, 0.42, 1),
        (0.62, 0.70, 0.75, 1),
    )
    for geom in list(world.findall("geom")):
        name = geom.get("name", "")
        if name.startswith("wetlab_genomics_stock_"):
            world.remove(geom)
        elif name in ("wetlab_genomics_shelf_0", "wetlab_genomics_shelf_1"):
            geom.set("rgba", ".64 .42 .19 1")
    for level, z in enumerate((2.06, 2.55)):
        for column, x in enumerate((-3.61, -0.95, 1.70)):
            b.box(
                world,
                (x, 1.587, z - 0.038),
                (0.020, 0.145, 0.013),
                "metal",
                collision=True,
                name=f"rochester_context__shelf_bracket_{level}_{column}",
            )
        x = -3.39
        for index in range(13):
            width = (0.31, 0.25, 0.37, 0.28, 0.34)[(index + level) % 5]
            height = (
                (0.20, 0.14, 0.16, 0.12)[index % 4]
                if level == 0
                else (0.18, 0.14, 0.17)[index % 3]
            )
            depth = (0.28, 0.23, 0.30)[index % 3]
            _carton(
                b,
                world,
                f"stock_{level}_{index}",
                (x, 1.58, z + 0.025),
                (width, depth, height),
                colors[(index + 2 * level) % 4],
            )
            if level == 0 and index in (1, 6, 10):
                _carton(
                    b,
                    world,
                    f"stack_{index}",
                    (x, 1.58, z + 0.025 + height),
                    (width * 0.84, depth * 0.91, 0.09),
                    colors[1],
                )
            x += (
                width / 2
                + (0.31, 0.25, 0.37, 0.28, 0.34)[(index + level + 1) % 5] / 2
                + 0.070
            )


def _wipe_pack(b, world, position):
    body = _body(world, "bench_wipes", position)
    b.box(
        body,
        (0, 0, 0.030),
        (0.12, 0.07, 0.030),
        "cream",
        collision=True,
        rgba=(0.31, 0.53, 0.42, 1),
        name="rochester_context__wipe_body",
    )
    for sign in (-1, 1):
        b.box(
            body,
            (0, sign * 0.051, 0.064),
            (0.12, 0.019, 0.004),
            "shell",
            collision=True,
        )
        b.box(
            body,
            (sign * 0.086, 0, 0.064),
            (0.034, 0.032, 0.004),
            "shell",
            collision=True,
        )
    b.box(body, (0, -0.071, 0.031), (0.093, 0.001, 0.020), "shell")
    # A folded tissue is a fixed thin visual sheet emerging through the real slot.
    vertices = [
        (-0.046, -0.004, 0.052),
        (0.046, -0.004, 0.052),
        (-0.045, 0.012, 0.114),
        (0.045, 0.012, 0.114),
        (-0.037, -0.011, 0.151),
        (0.039, -0.016, 0.140),
    ]
    name = "rochester_context__wipe_sheet"
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=" ".join(str(v) for point in vertices for v in point),
        face="0 1 3 0 3 2 2 3 5 2 5 4",
    )
    b.geom(body, "mesh", (), material="shell", mesh=name)


def _bench_units(b, world):
    for index, x in enumerate((0.64, 1.10)):
        body = _body(world, f"parked_unit_{index}", (x, -0.77, 0.92), math.pi)
        for px in (-0.063, 0.063):
            for py in (-0.073, 0.073):
                b.cylinder(
                    body,
                    (px, py, 0.005),
                    0.012,
                    0.005,
                    "dark",
                    collision=True,
                    name=f"rochester_context__unit_foot_{index}_{px}_{py}",
                )
        b.box(
            body,
            (0, 0, 0.062),
            (0.075, 0.081, 0.052),
            "cream",
            collision=True,
            name=f"rochester_context__unit_core_{index}",
        )
        b.housing(
            body,
            [(0.01, 0.073, 0.080, 0), (0.034, 0.09, 0.10, 0), (0.115, 0.083, 0.095, 0)],
            radius=0.018,
            material="cream",
        )
        if index == 0:
            b.housing(
                body,
                [
                    (0.115, 0.080, 0.089, 0),
                    (0.17, 0.069, 0.075, 0),
                    (0.24, 0.025, 0.040, 0),
                ],
                radius=0.020,
                material="glass",
            )
            b.box(
                body,
                (0, 0, 0.162),
                (0.045, 0.055, 0.048),
                "blue",
                collision=True,
                name="rochester_context__closed_blue_cover_contact",
            )
        else:
            b.cylinder(body, (0, 0, 0.126), 0.053, 0.011, "blue", collision=True)
            b.cylinder(body, (0, 0, 0.145), 0.027, 0.008, "dark", collision=True)
        b.box(body, (0.033, -0.092, 0.064), (0.021, 0.003, 0.012), "dark")
        b.cylinder(
            body,
            (-0.035, -0.096, 0.055),
            0.012,
            0.004,
            "blue",
            euler=(math.pi / 2, 0, 0),
        )
        b.box(
            body,
            (0.054, 0.093, 0.041),
            (0.009, 0.007, 0.009),
            "dark",
            name=f"rochester_context__unit_cable_port_{index}",
        )
        # Both fixed cable ends enter modeled ports; the loop stays on the bench.
        points = [
            (x - 0.054, -0.87, 0.961),
            (x - 0.075, -0.915, 0.958),
            (x - 0.045, -0.96, 0.936),
            (0.89, -0.99, 0.934),
            (0.87 + index * 0.07, -1.01, 0.961),
        ]
        for j, (start, end) in enumerate(zip(points, points[1:])):
            b.rod(
                world,
                start,
                end,
                0.0025,
                "dark",
                name=f"rochester_context__unit_cable_{index}_{j}",
            )
    b.box(
        world,
        (0.90, -1.045, 0.949),
        (0.17, 0.035, 0.029),
        "shell",
        collision=True,
        name="rochester_context__power_strip",
    )
    for index in range(2):
        b.box(
            world, (0.87 + index * 0.07, -1.012, 0.961), (0.010, 0.006, 0.012), "dark"
        )
    _wipe_pack(b, world, (0.75, -0.48, 0.92))


def add_rochester_context(root, world):
    """Add only the photographed room's distinct parked apparatus and storage."""
    b = _Builder(root, "rochester_context")
    _profiler(b, world)
    _shelf_occupancy(b, world)
    _bench_units(b, world)
    # These closed shells are convex. Use their visible envelopes for contact;
    # interior block approximations alone would leave the rounded covers hollow.
    for geom in world.iter("geom"):
        if geom.get("mesh", "").startswith("rochester_context__housing_mesh_"):
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
