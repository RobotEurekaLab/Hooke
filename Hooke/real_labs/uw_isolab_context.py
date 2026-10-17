"""Original UW analyzer context informed by the published Picarro assembly manual.

The 432 x 432 x 191 mm analyzer envelope and 12 x 32 mm vial have published
bases. Stand, rack, pump, vaporizer and hidden mechanics remain authored
estimates. No vendor CAD, source photograph or scientific process is embedded.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount


def add_static_assembly(root, parent, b):
    palette = {
        "case_paint": (0.29, 0.30, 0.29, 1),
        "stand_paint": (0.82, 0.80, 0.73, 1),
        "vaporizer_paint": (0.38, 0.39, 0.38, 1),
    }
    for key, color in palette.items():
        ET.SubElement(
            root.find("asset"),
            "material",
            name=b.name + "__mat_" + key,
            rgba=" ".join(map(str, color)),
            specular=".22",
            shininess=".2",
        )

    def box(key, pos, half, material="stand_paint", body=parent, **kw):
        return b.box(
            body,
            pos,
            half,
            material,
            collision=True,
            name=b.name + "__oem_" + key,
            **kw,
        )

    def cyl(key, pos, radius, half, material="bright", body=parent, **kw):
        return b.cylinder(
            body,
            pos,
            radius,
            half,
            material,
            collision=True,
            name=b.name + "__oem_" + key,
            **kw,
        )

    # 432 W x 432 D x 191 H mm, including the authored 12 mm feet.
    cabinet = ET.SubElement(parent, "body", name="oem_analyzer_enclosure")
    for i, x in enumerate((-0.181, 0.181)):
        for j, y in enumerate((-0.181, 0.181)):
            cyl(
                f"case_foot_{i}_{j}",
                (x, y, 0.006),
                0.014,
                0.006,
                "rubber",
                body=cabinet,
            )
    box("case_bottom", (0, 0, 0.014), (0.216, 0.216, 0.002), "case_paint", body=cabinet)
    box("case_roof", (0, 0, 0.189), (0.216, 0.216, 0.002), "case_paint", body=cabinet)
    for side in (-1, 1):
        box(
            f"case_side_{side}",
            (side * 0.214, 0, 0.1015),
            (0.002, 0.216, 0.0855),
            "case_paint",
            body=cabinet,
        )
        box(
            f"case_face_{side}",
            (0, side * 0.214, 0.1015),
            (0.212, 0.002, 0.0855),
            "case_paint",
            body=cabinet,
        )
    # Narrow metal edge strip and a flush front status button; no fake readings.
    box(
        "front_edge",
        (0, -0.2155, 0.184),
        (0.208, 0.0005, 0.0015),
        "bright",
        body=cabinet,
    )
    cyl(
        "power_ring",
        (-0.187, -0.2155, 0.045),
        0.010,
        0.0005,
        "bright",
        body=cabinet,
        euler=(math.pi / 2, 0, 0),
    )
    cyl(
        "power_button",
        (-0.187, -0.216, 0.045),
        0.0072,
        0.0001,
        "led",
        body=cabinet,
        euler=(math.pi / 2, 0, 0),
    )
    box(
        "front_label",
        (0.120, -0.216, 0.160),
        (0.069, 0.0001, 0.009),
        "cream",
        body=cabinet,
    )
    for x in (-0.196, 0.196):
        for z in (0.03, 0.17):
            cyl(
                f"case_screw_{x}_{z}",
                (x, -0.216, z),
                0.0021,
                0.0001,
                "bright",
                body=cabinet,
                euler=(math.pi / 2, 0, 0),
            )
    # Rear fan grille and connector panels follow the manual's external layout.
    box(
        "fan_backing",
        (0.10, 0.216, 0.121),
        (0.036, 0.0001, 0.034),
        "dark",
        body=cabinet,
    )
    for i in range(8):
        box(
            f"fan_grille_{i}",
            (0.10, 0.2162, 0.092 + i * 0.008),
            (0.033, 0.0002, 0.0009),
            "metal",
            body=cabinet,
        )
    box(
        "io_panel",
        (0.025, 0.2162, 0.047),
        (0.167, 0.0002, 0.020),
        "metal",
        body=cabinet,
    )
    for i in range(6):
        box(
            f"rear_connector_{i}",
            (-0.102 + i * 0.051, 0.217, 0.046),
            (0.017, 0.001, 0.007),
            "dark",
            body=cabinet,
        )

    # A separate two-foot stand straddles, rather than growing out of, the case.
    for i, x in enumerate((-0.239, 0.239)):
        box(f"stand_foot_{i}", (x, 0.007, 0.008), (0.018, 0.233, 0.008), "bright")
        box(f"stand_column_{i}", (x, 0.170, 0.208), (0.014, 0.014, 0.192), "bright")
        box(f"stand_clamp_{i}", (x, 0.166, 0.393), (0.024, 0.026, 0.017), "stand_paint")
        cyl(
            f"clamp_screw_{i}",
            (x, 0.139, 0.393),
            0.005,
            0.002,
            "bright",
            euler=(math.pi / 2, 0, 0),
        )
    box("rail_housing", (0, 0.166, 0.432), (0.290, 0.041, 0.032))
    box("rail_front_stripe", (0, 0.124, 0.432), (0.273, 0.001, 0.006), "cyan")
    box("rail_top_track", (0, 0.164, 0.465), (0.274, 0.025, 0.002), "dark")
    for x in (-0.279, 0.279):
        box(f"rail_end_{x}", (x, 0.166, 0.432), (0.011, 0.042, 0.033), "cream")

    # The trays hang from formed brackets attached to the rail, as on p9.
    for key, x, width in (("wash", -0.172, 0.035), ("tray", -0.069, 0.051)):
        box(f"{key}_clamp", (x, 0.166, 0.394), (width, 0.023, 0.006), "bright")
        box(f"{key}_hanger", (x, 0.145, 0.329), (0.009, 0.003, 0.059))
        box(f"{key}_back", (x, 0.131, 0.274), (width, 0.014, 0.005))
        box(f"{key}_shelf", (x, 0.012, 0.266), (width, 0.129, 0.003))
    # One dense rectangular rack. Pitch/count are estimated from the photo.
    cx, cy, pitch = -0.069, 0.012, 0.014
    for i in range(7):
        box(
            f"rack_grid_x_{i}",
            (cx + (i - 3) * pitch, cy, 0.275),
            (0.0007, 0.112, 0.006),
            "bright",
        )
    for j in range(17):
        box(
            f"rack_grid_y_{j}",
            (cx, cy + (j - 8) * pitch, 0.275),
            (0.042, 0.0007, 0.006),
            "bright",
        )
    for i in range(6):
        for j in range(16):
            x, y = cx + (i - 2.5) * pitch, cy + (j - 7.5) * pitch
            annular_mount(
                b,
                parent,
                f"rack_bore_{i}_{j}",
                (x, y, 0.275),
                0.00615,
                0.007,
                0.006,
                axis="z",
                material="bright",
                namespace="oem",
                proxies=False,
            )
    # Two wash containers plus a smaller waste container; inert static context.
    for j, y in enumerate((0.078, 0.002)):
        cyl(f"wash_base_{j}", (-0.172, y, 0.273), 0.026, 0.004, "bright")
        cyl(f"wash_glass_{j}", (-0.172, y, 0.294), 0.025, 0.017, "glass")
        cyl(f"wash_cap_{j}", (-0.172, y, 0.316), 0.027, 0.005, "blue")
        cyl(f"wash_center_{j}", (-0.172, y, 0.3215), 0.006, 0.0005, "cream")
    cyl("waste_glass", (-0.172, -0.071, 0.289), 0.012, 0.02, "glass")
    cyl("waste_neck", (-0.172, -0.071, 0.313), 0.006, 0.004, "glass")
    # Separate vaporizer on the right, suspended from an adjustable bracket.
    box("vaporizer_hanger", (0.120, 0.166, 0.365), (0.031, 0.028, 0.035), "bright")
    box("vaporizer_mount", (0.120, 0.105, 0.333), (0.050, 0.073, 0.005), "bright")
    box(
        "vaporizer_case",
        (0.120, 0.057, 0.295),
        (0.070, 0.114, 0.033),
        "vaporizer_paint",
    )
    box(
        "vaporizer_face",
        (0.120, -0.058, 0.295),
        (0.069, 0.001, 0.032),
        "vaporizer_paint",
    )
    box(
        "vaporizer_display_frame",
        (0.099, -0.060, 0.308),
        (0.025, 0.001, 0.011),
        "rubber",
    )
    box("vaporizer_display", (0.099, -0.061, 0.308), (0.020, 0.0003, 0.007), "screen")
    for j in range(5):
        box(
            f"vaporizer_vent_{j}",
            (0.1905, 0.017 + 0.018 * j, 0.295),
            (0.0005, 0.005, 0.012),
            "dark",
        )
    cyl("vaporizer_port_base", (0.120, 0.060, 0.334), 0.016, 0.006, "bright")
    cyl("vaporizer_port", (0.120, 0.060, 0.343), 0.008, 0.003, "dark")
    # Estimated external tool cover; hidden axes will be authored separately.
    box("y_axis_cover", (-0.070, 0.073, 0.491), (0.029, 0.136, 0.025), "cream")
    box("z_tower", (-0.070, -0.037, 0.595), (0.022, 0.024, 0.129), "cream")
    box("z_guide_recess", (-0.070, -0.0615, 0.572), (0.011, 0.0005, 0.092), "dark")
    box("z_face_rail_left", (-0.087, -0.063, 0.572), (0.002, 0.002, 0.095), "bright")
    box("z_face_rail_right", (-0.053, -0.063, 0.572), (0.002, 0.002, 0.095), "bright")
    box("tool_carriage", (-0.070, -0.065, 0.499), (0.014, 0.003, 0.015), "metal")
    cyl("tool_holder", (-0.070, -0.070, 0.470), 0.005, 0.014, "bright")
    cyl("dry_stylus", (-0.070, -0.070, 0.427), 0.0006, 0.029, "bright")
    # Two separate external pump boxes are source-supported; placement estimated.
    for i, x in enumerate((-0.155, 0.155)):
        for dx in (-0.052, 0.052):
            for y in (0.283, 0.387):
                cyl(
                    f"pump_foot_{i}_{dx}_{y}",
                    (x + dx, y, 0.007),
                    0.007,
                    0.007,
                    "rubber",
                )
        box(
            f"pump_case_{i}",
            (x, 0.335, 0.047),
            (0.070, 0.065, 0.033),
            "vaporizer_paint",
        )
        for dx in (-0.04, 0.04):
            box(
                f"pump_handle_post_{i}_{dx}",
                (x + dx, 0.335, 0.089),
                (0.003, 0.003, 0.011),
                "bright",
            )
        box(f"pump_handle_{i}", (x, 0.335, 0.102), (0.043, 0.003, 0.003), "bright")
        cyl(
            f"pump_port_{i}",
            (x, 0.266, 0.052),
            0.006,
            0.004,
            "bright",
            euler=(math.pi / 2, 0, 0),
        )
    # Continuous exterior tubing to those ports; no flow or pump model.
    for i, x in enumerate((-0.16, -0.12)):
        cyl(
            f"rear_service_port_{i}",
            (x, 0.218, 0.128),
            0.006,
            0.004,
            "bright",
            euler=(math.pi / 2, 0, 0),
        )
        target_x = -0.155 if i == 0 else 0.155
        points = [
            (x, 0.222, 0.128),
            (x, 0.242, 0.12),
            (target_x, 0.247, 0.090),
            (target_x, 0.262, 0.052),
        ]
        for j, (a, z) in enumerate(zip(points, points[1:])):
            b.rod(
                parent,
                a,
                z,
                0.004,
                "rubber",
                collision=True,
                name=f"isotope_analyzer__oem_cable_{i}_{j}",
            )
    return b.metadata
