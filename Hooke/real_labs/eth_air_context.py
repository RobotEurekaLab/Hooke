"""Photo-informed closed benchtop exterior, without instrument behavior.

The published CFX96 Touch envelope supplies overall dimensions. Cover curvature,
feet, display and vent details are original estimates from the ETH tour image.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.equipment import _Builder


EXTERIOR_SIZE_M = (0.33, 0.46, 0.36)
POSITION_M = (0.17, 1.30, 0.90)


def _closed_shell(b, body, name, levels, radius, material):
    b.housing(body, levels, radius=radius, material=material)
    geom = body.findall("geom")[-1]
    geom.set("name", f"eth_air_context__{name}")
    geom.set("contype", "1")
    geom.set("conaffinity", "1")


def add_eth_air_context(root, world):
    """Place one supported, unpowered closed apparatus on the existing counter."""
    b = _Builder(root, "eth_air_context")
    body = ET.SubElement(
        world,
        "body",
        name="eth_air_context__closed_instrument",
        pos=" ".join(map(str, POSITION_M)),
    )
    for x in (-0.115, 0.115):
        for y in (-0.175, 0.175):
            b.cylinder(
                body,
                (x, y, 0.006),
                0.012,
                0.006,
                "rubber",
                collision=True,
                name=f"eth_air_context__foot_{x}_{y}",
            )
    # The wedge front carries the display; the rear supports the closed hood.
    _closed_shell(
        b,
        body,
        "base",
        [
            (0.012, 0.149, 0.213, 0),
            (0.058, 0.165, 0.230, 0),
            (0.157, 0.162, 0.155, 0.075),
        ],
        radius=0.012,
        material="cream",
    )
    _closed_shell(
        b,
        body,
        "seam",
        [(0.157, 0.159, 0.153, 0.075), (0.165, 0.159, 0.153, 0.075)],
        radius=0.016,
        material="rubber",
    )
    _closed_shell(
        b,
        body,
        "hood",
        [
            (0.165, 0.157, 0.151, 0.075),
            (0.307, 0.150, 0.145, 0.079),
            (0.349, 0.141, 0.135, 0.083),
            (0.360, 0.124, 0.118, 0.085),
        ],
        radius=0.021,
        material="cream",
    )
    display = ET.SubElement(
        body,
        "body",
        name="eth_air_context__display",
        pos="0 -0.1575 0.10820",
        euler=f"{math.atan(0.66)} 0 0",
    )
    b.box(
        display,
        (0, 0, 0),
        (0.128, 0.076, 0.002),
        "shell",
        collision=True,
        name="eth_air_context__display_bezel",
    )
    b.box(display, (0, 0, 0.0029), (0.115, 0.061, 0.0009), "screen")
    b.box(
        body,
        (0, -0.078, 0.191),
        (0.018, 0.004, 0.003),
        "metal",
        name="eth_air_context__front_latch",
    )
    for side in (-1, 1):
        # Thin trim follows the taper of the side cover; it is not a real vent.
        panel = ET.SubElement(
            body,
            "body",
            name=f"eth_air_context__side_panel_{side}",
            pos=f"{side * 0.1539} 0.080 0.235",
            euler=f"0 {-side * math.atan(0.007 / 0.142)} 0",
        )
        b.box(
            panel,
            (0, 0, 0),
            (0.0005, 0.102, 0.049),
            "shell",
            name=f"eth_air_context__side_trim_{side}",
        )
        for index in range(14):
            b.box(
                panel,
                (side * 0.0007, -0.086 + index * 0.0132, 0),
                (0.00025, 0.0011, 0.032),
                "metal",
            )
    b.site(body, "exterior_front", (0, -0.23, 0.10))
