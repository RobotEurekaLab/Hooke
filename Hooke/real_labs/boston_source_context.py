"""Photographed PML console furniture and passive SEM exterior services.

The BU photograph supports these visible objects, not their measured dimensions
or hidden wiring. Display decorations and the periodic-table layout are original
schematics. They do not reproduce measured spectra or third-party poster artwork.
All additions are fixed; the seven-axis dry specimen mechanism is unchanged.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.equipment import _Builder


def add_source_context(root, base, console, namespace):
    b = _Builder(root, namespace)
    asset = root.find("asset")
    for key, color in {
        "navy": "0.025 0.10 0.18 1",
        "poster": "0.015 0.065 0.25 1",
        "paper": "0.72 0.77 0.73 1",
        "pink": "0.72 0.39 0.46 1",
        "yellow": "0.80 0.67 0.12 1",
        "green": "0.27 0.49 0.32 1",
    }.items():
        ET.SubElement(asset, "material", name=f"{namespace}__mat_{key}", rgba=color)

    def box(key, pos, half, material="cream", parent=base, **attrs):
        return b.box(
            parent,
            pos,
            half,
            material,
            collision=True,
            mass=0,
            name=f"{namespace}__{key}",
            **attrs,
        )

    def rod(key, points, radius=0.004, material="rubber", parent=base):
        for index, (start, end) in enumerate(zip(points, points[1:])):
            b.rod(
                parent,
                start,
                end,
                radius,
                material,
                collision=True,
                mass=0,
                name=f"{namespace}__{key}_{index}",
            )

    def cylinder(key, pos, radius, half, material="metal", parent=base, **attrs):
        b.cylinder(
            parent,
            pos,
            radius,
            half,
            material,
            collision=True,
            mass=0,
            name=f"{namespace}__{key}",
            **attrs,
        )

    # Two parked seats with a continuous floor/caster/spoke/column/cushion path.
    # Hidden lower frames are estimates; no rolling chair mechanism is claimed.
    for index, x in enumerate((0.69, 1.66)):
        chair = ET.Element("body")
        for spoke in range(5):
            angle = 2 * math.pi * spoke / 5
            px, py = 0.25 * math.cos(angle), 0.25 * math.sin(angle)
            cylinder(
                f"chair_{index}_wheel_{spoke}",
                (px, py, 0.030),
                0.030,
                0.022,
                "dark",
                chair,
                quat=(math.sqrt(0.5), math.sqrt(0.5), 0, 0),
            )
            rod(
                f"chair_{index}_spoke_{spoke}",
                [(0, 0, 0.080), (px, py, 0.055)],
                0.018,
                "cream",
                chair,
            )
        cylinder(f"chair_{index}_column", (0, 0, 0.263), 0.027, 0.200, "metal", chair)
        box(f"chair_{index}_pan", (0, 0, 0.454), (0.205, 0.205, 0.014), "dark", chair)
        b.geom(
            chair,
            "ellipsoid",
            (0.235, 0.228, 0.040),
            (0, 0, 0.478),
            "navy",
            collision=True,
            mass=0,
            name=f"{namespace}__chair_{index}_cushion",
        )
        rod(
            f"chair_{index}_back_support",
            [(0, -0.18, 0.451), (0, -0.255, 0.55), (0, -0.27, 0.88)],
            0.020,
            "cream",
            chair,
        )
        b.geom(
            chair,
            "ellipsoid",
            (0.223, 0.052, 0.192),
            (0, -0.247, 0.82),
            "navy",
            collision=True,
            mass=0,
            name=f"{namespace}__chair_{index}_back",
        )

        # Bake the parked placement into these fixed geoms. Adding an inertial
        # body would unnecessarily change MuJoCo's automatic model statistics.
        for geom in chair:
            field = "fromto" if "fromto" in geom.attrib else "pos"
            values = [float(value) for value in geom.get(field).split()]
            geom.set(
                field,
                " ".join(
                    f"{value + (x, -1.17, 0)[i % 3]:.8g}"
                    for i, value in enumerate(values)
                ),
            )
            base.append(geom)

    # Physical key caps touch the existing supported keyboard panel. Extra
    # control groups follow the photograph without inventing a functional GUI.
    box("keyboard_support", (0, -0.018, 0.056), (0.25, 0.073, 0.006), "dark", console)
    for row in range(4):
        for col in range(13):
            box(
                f"key_{row}_{col}",
                (-0.222 + col * 0.036, -0.073 + row * 0.032, 0.065),
                (0.014, 0.011, 0.003),
                "paper",
                console,
            )
    for side in (-1, 1):
        for row in range(3):
            cylinder(
                f"side_knob_{side}_{row}",
                (side * (0.39 - row * 0.024), 0.045 + row * 0.026, 0.064),
                0.009,
                0.016,
                "dark",
                console,
            )

    # Supported small joystick controllers, mouse bodies and connected cables.
    for key, x, y in (("aux", 1.57, -0.165), ("left", 0.57, -0.38)):
        box(key + "_controller", (x, y, 0.816), (0.095, 0.065, 0.026))
        box(key + "_display", (x, y + 0.015, 0.843), (0.070, 0.035, 0.001), "dark")
        cylinder(
            key + "_joystick_foot", (x + 0.042, y - 0.037, 0.846), 0.017, 0.006, "dark"
        )
        rod(
            key + "_joystick",
            [(x + 0.042, y - 0.037, 0.847), (x + 0.042, y - 0.037, 0.880)],
            0.005,
            "rubber",
        )
    for index, (x, material) in enumerate(((0.81, "dark"), (1.83, "cream"))):
        b.geom(
            base,
            "ellipsoid",
            (0.036, 0.057, 0.019),
            (x, -0.60, 0.809),
            material,
            collision=True,
            mass=0,
            name=f"{namespace}__mouse_{index}",
        )
        box(f"mouse_{index}_wheel", (x, -0.580, 0.827), (0.003, 0.009, 0.002), "dark")
        rod(
            f"mouse_{index}_cable",
            [
                (x, -0.545, 0.805),
                (x + 0.065, -0.48, 0.795),
                (x + 0.07, -0.24, 0.795),
                (x, 0.015, 0.827),
            ],
        )
    for index, x in enumerate((0.76, 1.31, 1.86)):
        rod(
            f"monitor_{index}_cable",
            [
                (x, 0.075, 1.035),
                (x + 0.045, 0.14, 0.98),
                (x + 0.075, 0.15, 0.82),
                (x + 0.075, 0.14, 0.775),
                (x, 0.12, 0.68),
                (0.57, 0.12, 0.65),
                (0.31, 0.12, 0.65),
            ],
        )
        # Neutral blank application panes: visual context, not SEM measurements.
        box(
            f"display_{index}_backing", (x, 0.015, 1.095), (0.232, 0.005, 0.128), "navy"
        )
        box(
            f"display_{index}_pane",
            (x - 0.036, 0.009, 1.075),
            (0.18, 0.001, 0.095),
            "paper",
        )
        box(f"display_{index}_header", (x, 0.009, 1.206), (0.222, 0.001, 0.013), "navy")
        for row in range(5):
            box(
                f"display_{index}_button_{row}",
                (x + 0.186, 0.009, 1.14 - row * 0.028),
                (0.029, 0.001, 0.009),
                "paper",
            )

    # Original colour-block periodic layout on the source's prominent blue
    # wall poster. The unmeasured room depth is explicitly estimated.
    box("poster_board", (1.19, 1.542, 1.82), (0.62, 0.008, 0.40), "poster")
    for row in range(7):
        columns = (
            (0, 17)
            if row == 0
            else (*range(2), *range(12, 18)) if row < 3 else range(18)
        )
        for col in columns:
            material = (
                "yellow"
                if col < 2
                else "pink" if col < 12 else "green" if col < 16 else "paper"
            )
            box(
                f"periodic_{row}_{col}",
                (0.642 + 0.0645 * col, 1.532, 2.10 - row * 0.067),
                (0.030, 0.002, 0.031),
                material,
            )
    for row in range(2):
        for col in range(14):
            box(
                f"periodic_f_{row}_{col}",
                (0.90 + 0.060 * col, 1.532, 1.58 - row * 0.054),
                (0.028, 0.002, 0.024),
                "paper",
            )

    # Additional source-supported upper chamber ports. Keep them above the
    # moving access assembly; the obscured door face is deliberately untouched.
    for index, x in enumerate((-0.125, -0.052)):
        cylinder(f"roof_port_{index}_flange", (x, -0.020, 1.283), 0.027, 0.008)
        cylinder(f"roof_port_{index}_neck", (x, -0.020, 1.313), 0.017, 0.025, "dark")
        cylinder(f"roof_port_{index}_cap", (x, -0.020, 1.341), 0.022, 0.006)
        for bolt in range(4):
            angle = math.pi * bolt / 2
            cylinder(
                f"roof_port_{index}_bolt_{bolt}",
                (x + 0.021 * math.cos(angle), -0.020 + 0.021 * math.sin(angle), 1.293),
                0.003,
                0.003,
                "bright",
            )
    rod(
        "upper_service",
        [
            (-0.266, 0.048, 1.441),
            (-0.28, 0.11, 1.46),
            (-0.24, 0.14, 1.40),
            (-0.15, 0.12, 1.34),
            (-0.125, -0.020, 1.347),
        ],
        0.006,
        "rubber",
    )
    return {
        "version": 1,
        "basis": "BU PML installed microscope and operator-console photograph",
        "scope": "Parked seating, populated console, wall chart, fixed cables and upper chamber fittings",
        "dimensions": "Estimated from the photograph; room and obscured supports are not surveyed",
        "limits": "Passive context only; schematic displays, no OEM CAD, cable forces or scientific image formation",
    }
