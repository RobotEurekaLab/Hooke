"""Reference-informed vacuum and probe workstations, in metres.

These are authored exterior reconstructions. Controls are mechanical surrogates;
they do not implement vacuum, beam optics, deposition or electrical metrology.
Source photographs remain external references and are not distributed as textures.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box


CORNELL = "https://schlom.mse.cornell.edu/labtour"
ANU = "https://science.anu.edu.au/study/360-virtual-tours/sensitive-high-resolution-ion-microprobe-shrimp-360deg-facility-tour"
OXFORD = "https://interface.web.ox.ac.uk/infrastructure"


def _flange(b, parent, center, radius, plane="xy", bolts=12):
    """Open bolted flange; the center remains a physical access opening."""
    b.ring(parent, center, radius, radius * 0.10, "bright", plane, 32)
    b.ring(parent, center, radius * 0.84, radius * 0.025, "dark", plane, 32)
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    for index in range(bolts):
        point = list(center)
        angle = math.tau * index / bolts
        point[axes[0]] += radius * math.cos(angle)
        point[axes[1]] += radius * math.sin(angle)
        b.geom(parent, "sphere", (radius * 0.058,), point, "dark")


def _cable(b, parent, points, radius=0.004, material="dark"):
    for start, end in zip(points, points[1:]):
        b.rod(parent, start, end, radius, material)


def _frame(b, parent, x, y, width, depth, top):
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = x + sx * width / 2, y + sy * depth / 2
            b.cylinder(parent, (px, py, 0.022), 0.045, 0.022, "rubber", collision=True)
            b.box(
                parent,
                (px, py, (top + 0.044) / 2),
                (0.025, 0.025, (top - 0.044) / 2),
                "metal",
                collision=True,
            )
    for z in (0.20, top):
        for sy in (-1, 1):
            b.box(
                parent,
                (x, y + sy * depth / 2, z),
                (width / 2, 0.025, 0.025),
                "metal",
                collision=True,
            )
        for sx in (-1, 1):
            b.box(
                parent,
                (x + sx * width / 2, y, z),
                (0.025, depth / 2, 0.025),
                "metal",
                collision=True,
            )


def _mbe(b, base, params):
    # The two tall cylindrical chambers and their steel frames follow the
    # photographed Schlom apparatus silhouette; dimensions are estimated.
    for cx, cy, radius in ((-0.72, 0.20, 0.34), (0.40, 0.20, 0.29)):
        _frame(b, base, cx, cy, radius * 2.4, 0.86, 1.95)
        # A load-bearing subframe connects all four uprights at vessel level.
        for sy in (-1, 1):
            b.box(
                base,
                (cx, cy + sy * 0.43, 0.925),
                (radius * 1.2, 0.025, 0.025),
                "metal",
                collision=True,
            )
        for sx in (-1, 1):
            b.box(
                base,
                (cx + sx * radius * 0.57, cy, 0.925),
                (0.033, 0.43, 0.025),
                "metal",
                collision=True,
            ).set("name", b.unique("vessel_cradle"))
        b.cylinder(base, (cx, cy, 1.24), radius, 0.29, "metal", collision=True)
        for z in (0.94, 1.54):
            _flange(b, base, (cx, cy, z), radius * 1.08)
        b.cylinder(base, (cx, cy, 1.67), radius * 0.67, 0.12, "bright")
        _flange(b, base, (cx, cy, 1.80), radius * 0.77)
        # Front view port has an annular opening and translucent optical face.
        b.rod(
            base,
            (cx, cy - radius, 1.26),
            (cx, cy - radius - 0.16, 1.26),
            0.082,
            "metal",
        )
        _flange(b, base, (cx, cy - radius - 0.16, 1.26), 0.094, "xz", 8)
        b.cylinder(
            base,
            (cx, cy - radius - 0.17, 1.26),
            0.068,
            0.003,
            "lens",
            euler=(math.pi / 2, 0, 0),
        )
        for index in range(6):
            angle = math.tau * index / 6
            dx, dy = math.cos(angle), math.sin(angle)
            lower = (cx + dx * 0.18, cy + dy * 0.18, 0.72)
            upper = (cx + dx * 0.13, cy + dy * 0.13, 0.99)
            b.rod(base, lower, upper, 0.041, "bright")
            b.cylinder(base, (lower[0], lower[1], 0.69), 0.046, 0.025, "dark")
            b.cylinder(base, (lower[0], lower[1], 0.65), 0.047, 0.011, "warning")
            _cable(
                b,
                base,
                [lower, (lower[0], lower[1], 0.40), (cx - 0.30, cy - 0.37, 0.22)],
                0.005,
                "blue" if index % 2 else "dark",
            )
        for dx, dy, length in (
            (-0.12, 0.03, 0.48),
            (0.08, -0.02, 0.65),
            (0.05, 0.16, 0.37),
        ):
            b.cylinder(
                base, (cx + dx, cy + dy, 1.79 + length / 2), 0.031, length / 2, "bright"
            )
            _flange(b, base, (cx + dx, cy + dy, 1.89), 0.05, bolts=6)
            b.cylinder(base, (cx + dx, cy + dy, 1.79 + length), 0.042, 0.025, "dark")
        # RHEED/viewing side arms use short branch pipes and bolted interfaces.
        for sy in (-1, 1):
            b.rod(
                base,
                (cx + 0.18, cy + sy * 0.18, 1.35),
                (cx + 0.44, cy + sy * 0.43, 1.35),
                0.035,
                "metal",
            )
            b.box(
                base, (cx + 0.44, cy + sy * 0.43, 1.35), (0.082, 0.062, 0.068), "dark"
            )
    b.rod(base, (-0.38, 0.20, 1.22), (0.11, 0.20, 1.22), 0.081, "bright")
    for x in (-0.34, 0.07):
        _flange(b, base, (x, 0.20, 1.22), 0.105, "yz", 8)
    # A lower load-lock frame and open-faced chamber stand to the right.
    _frame(b, base, 1.31, -0.07, 0.72, 0.95, 0.91)
    b.box(base, (1.31, -0.07, 0.94), (0.40, 0.52, 0.025), "bright", collision=True)
    b.rod(base, (0.69, 0.20, 1.22), (1.31, 0.20, 1.22), 0.062, "metal")
    b.cylinder(
        base, (1.31, -0.04, 1.22), 0.155, 0.26, "metal", euler=(math.pi / 2, 0, 0)
    )
    # Two saddles bridge the 100 mm gap between worktop and horizontal vessel.
    for y in (-0.19, 0.11):
        b.box(
            base, (1.31, y, 1.015), (0.067, 0.045, 0.05), "metal", collision=True
        ).set("name", b.unique("loadlock_saddle"))
    _flange(b, base, (1.31, -0.31, 1.22), 0.17, "xz")
    door = b.moving(
        base,
        "load_lock_door",
        (1.125, -0.335, 1.22),
        (0, 0, -1),
        (0, 1.25),
        kind="hinge",
        mass=0.7,
        kp=450,
    )
    b.cylinder(
        door,
        (0.185, 0, 0),
        0.158,
        0.010,
        "bright",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    b.cylinder(
        door, (0.185, -0.012, 0), 0.11, 0.003, "glass", euler=(math.pi / 2, 0, 0)
    )
    b.rod(door, (0.29, -0.043, -0.052), (0.29, -0.043, 0.052), 0.011, "dark")
    b.site(door, "door_handle", (0.29, -0.043, 0))
    # Transfer rod has its own side sleeve; no moving part crosses the door.
    b.rod(base, (1.47, 0.07, 1.22), (1.88, 0.07, 1.22), 0.033, "metal")
    _flange(b, base, (1.50, 0.07, 1.22), 0.057, "yz", 8)
    transfer = b.moving(
        base,
        "transfer_rod",
        (1.82, 0.07, 1.22),
        (-1, 0, 0),
        (0, 0.20),
        mass=0.25,
        kp=600,
    )
    b.rod(transfer, (-0.57, 0, 0), (0.18, 0, 0), 0.012, "bright")
    b.cylinder(transfer, (0.18, 0, 0), 0.04, 0.055, "dark", euler=(0, math.pi / 2, 0))
    b.box(transfer, (-0.54, 0, 0.016), (0.034, 0.028, 0.006), "copper", collision=True)
    b.box(transfer, (-0.54, 0, 0.024), (0.021, 0.019, 0.002), "lens")
    b.site(transfer, "substrate", (-0.54, 0, 0.026))
    # A separate shielded electronics rack and restrained cable routes.
    b.box(base, (-1.37, 0.63, 0.65), (0.23, 0.25, 0.61), "dark", collision=True)
    for x in (-1.55, -1.19):
        for y in (0.43, 0.83):
            b.cylinder(base, (x, y, 0.02), 0.028, 0.02, "rubber", collision=True).set(
                "name", b.unique("rack_foot")
            )
    for z in (0.29, 0.47, 0.65, 0.83, 1.01):
        b.box(base, (-1.37, 0.369, z), (0.20, 0.01, 0.070), "metal")
        b.screen(base, (-1.46, 0.351, z), 0.12, 0.065)
        b.cylinder(
            base, (-1.24, 0.345, z), 0.018, 0.012, "dark", euler=(math.pi / 2, 0, 0)
        )
    for i in range(9):
        x = -1.05 + i * 0.17
        _cable(
            b,
            base,
            [
                (x, 0.59, 1.82),
                (x + 0.06, 0.70, 1.35),
                (x + 0.04, 0.74, 0.22),
                (-1.2, 0.72, 0.18 + i * 0.009),
            ],
            0.004,
            ("blue", "dark", "warning")[i % 3],
        )
    b.metadata["capabilities"] = [
        "load-lock access articulation",
        "200 mm representative substrate-transfer stroke",
        "clamped substrate and door-handle target sites",
    ]
    b.metadata["limitations"].append(
        "Transfer stroke and access mechanism are authored surrogates; no vacuum pump-down, interlocks, RHEED, film growth or source-temperature process is simulated."
    )


def _cornell_features(world, definition):
    box(
        world,
        "cornell_service_raceway",
        (2.4, 0.045, 0.065),
        (0, 1.9, 1.72),
        (0.55, 0.58, 0.55, 1),
    )
    for i in range(6):
        box(
            world,
            f"cornell_service_socket_{i}",
            (0.06, 0.018, 0.08),
            (-1.8 + i * 0.7, 1.83, 1.72),
            (0.83, 0.83, 0.77, 1),
            collision=False,
        )
    # The reference constrains apparatus, not the room perimeter or services.
    for i in range(3):
        box(
            world,
            f"cornell_pump_{i}",
            (0.19, 0.24, 0.17),
            (-0.85 + i * 0.65, 1.48, 0.21),
            (0.22, 0.27, 0.29, 1),
        )
        box(
            world,
            f"cornell_pump_foot_{i}",
            (0.22, 0.27, 0.02),
            (-0.85 + i * 0.65, 1.48, 0.02),
            (0.1, 0.11, 0.12, 1),
        )


def _shrimp(b, base, params):
    # Grounded white cabinet chassis follows the inspected ANU preview. The
    # curved separator, large magnet and source tower remain room-scale assets.
    for x in (-1.90, -0.95, 0.0, 0.95, 1.90):
        for y in (-0.76, 0.76):
            b.box(base, (x, y, 0.035), (0.09, 0.10, 0.035), "cream", collision=True)
            b.box(base, (x, y, 0.105), (0.034, 0.036, 0.05), "metal", collision=True)
    b.box(base, (0, 0, 0.49), (2.23, 0.84, 0.36), "shell", collision=True)
    b.box(base, (0, 0, 0.875), (2.30, 0.92, 0.025), "cream", collision=True)
    for index, x in enumerate((-1.86, -1.12, -0.38, 0.36, 1.10, 1.84)):
        b.box(base, (x, -0.85, 0.49), (0.35, 0.014, 0.31), "metal")
        b.box(base, (x, -0.868, 0.49), (0.332, 0.006, 0.29), "shell")
        b.box(base, (x + 0.25, -0.879, 0.63), (0.007, 0.012, 0.048), "dark")
        if index % 2:
            for z in (0.30, 0.43, 0.56, 0.69):
                b.box(base, (x, -0.882, z), (0.255, 0.005, 0.045), "metal")
                b.screen(base, (x - 0.13, -0.893, z), 0.10, 0.049)
                b.geom(base, "sphere", (0.006,), (x + 0.19, -0.9, z), "led")
    # White sector-magnet covers above and below the exposed curved vacuum tube.
    cx, cy, radius = -0.70, 0.02, 0.85
    angles = [-0.65 + index * 2.75 / 22 for index in range(23)]
    for index, (a, c) in enumerate(zip(angles, angles[1:])):
        mid = (a + c) / 2
        x, y = cx + radius * math.cos(mid), cy + radius * math.sin(mid)
        length = radius * (c - a) / 2 + 0.006
        for z in (1.145, 1.535):
            b.box(
                base,
                (x, y, z),
                (length, 0.23, 0.125),
                "cream",
                euler=(0, 0, mid + math.pi / 2),
                collision=True,
            )
        start = (cx + radius * math.cos(a), cy + radius * math.sin(a), 1.34)
        end = (cx + radius * math.cos(c), cy + radius * math.sin(c), 1.34)
        b.rod(base, start, end, 0.062, "bright")
        # Recessed coil faces between the magnet and its return yoke.
        b.box(
            base,
            (
                cx + (radius + 0.15) * math.cos(mid),
                cy + (radius + 0.15) * math.sin(mid),
                1.34,
            ),
            (length, 0.041, 0.06),
            "copper",
            euler=(0, 0, mid + math.pi / 2),
        )
        if index in (0, 5, 11, 16, 21):
            # Inferred exterior return-yoke supports join both cover banks.
            # They sit outside the coil and preserve the central tube opening.
            b.box(
                base,
                (
                    cx + (radius + 0.20) * math.cos(mid),
                    cy + (radius + 0.20) * math.sin(mid),
                    1.34,
                ),
                (0.031, 0.028, 0.085),
                "metal",
                euler=(0, 0, mid + math.pi / 2),
                collision=True,
            ).set("name", b.unique("return_yoke_support"))
        if index % 5 == 0:
            local = ET.SubElement(
                base,
                "body",
                name=b.unique("analyzer_flange"),
                pos=f"{x} {y} 1.34",
                euler=f"0 0 {mid + math.pi / 2}",
            )
            _flange(b, local, (0, 0, 0), 0.091, "yz", 8)
    for x, y in ((-1.40, 0.57), (-0.62, 0.76), (0.0, 0.18)):
        b.box(base, (x, y, 0.968), (0.19, 0.16, 0.068), "metal", collision=True)
        b.box(base, (x, y, 1.05), (0.23, 0.19, 0.016), "bright")
    # White analyzer housing on the left includes the observed broad panels.
    # Four inferred standoffs bridge cabinet top to housing without relying on
    # a distant sector pedestal or an unseen suspension structure.
    for x in (-2.05, -1.53):
        for y in (-0.14, 0.46):
            b.box(
                base,
                (x, y, 0.98),
                (0.035, 0.045, 0.08),
                "metal",
                collision=True,
            ).set("name", b.unique("analyzer_standoff"))
    b.box(base, (-1.79, 0.16, 1.34), (0.35, 0.39, 0.28), "cream", collision=True)
    b.box(base, (-1.79, -0.241, 1.34), (0.30, 0.008, 0.19), "shell")
    b.box(base, (-1.88, -0.252, 1.34), (0.16, 0.004, 0.032), "blue")
    b.box(base, (-1.56, -0.255, 1.36), (0.028, 0.004, 0.036), "warning")
    b.rod(base, (-2.13, 0.16, 1.34), (-2.29, 0.16, 1.34), 0.096, "bright")
    _flange(b, base, (-2.29, 0.16, 1.34), 0.125, "yz")
    # Vacuum source body, grounded pedestal, high ion column and lateral optics.
    b.box(base, (1.26, 0.10, 0.925), (0.41, 0.36, 0.025), "metal", collision=True)
    for x in (0.98, 1.54):
        for y in (-0.17, 0.37):
            b.cylinder(base, (x, y, 1.00), 0.025, 0.075, "metal", collision=True)
    b.cylinder(base, (1.26, 0.10, 1.30), 0.255, 0.235, "bright", collision=True)
    for z in (1.06, 1.545):
        _flange(b, base, (1.26, 0.10, z), 0.275, bolts=16)
    b.cylinder(base, (1.26, 0.10, 1.69), 0.11, 0.14, "metal")
    _flange(b, base, (1.26, 0.10, 1.835), 0.15)
    b.cylinder(base, (1.26, 0.10, 1.98), 0.074, 0.14, "bright")
    b.cylinder(base, (1.26, 0.10, 2.155), 0.09, 0.035, "dark")
    for z in (1.86, 1.94, 2.02, 2.10):
        _flange(b, base, (1.26, 0.10, z), 0.084, bolts=6)
    b.rod(base, (1.03, 0.10, 1.34), (0.05, -0.10, 1.34), 0.069, "metal")
    for x in (0.28, 0.65, 0.96):
        _flange(b, base, (x, -0.08 + x * 0.18, 1.34), 0.095, "yz", 8)
    for y in (-0.22, 0.41):
        b.rod(base, (1.26, y, 1.31), (1.82, y, 1.31), 0.043, "bright")
        _flange(b, base, (1.80, y, 1.31), 0.069, "yz", 8)
        b.cylinder(
            base, (1.89, y, 1.31), 0.059, 0.083, "dark", euler=(0, math.pi / 2, 0)
        )
    # A visible load-lock is below the source. Its access door, side-transfer
    # carriage and puck rotation are independent geometry-review controls.
    b.box(base, (1.27, -0.49, 1.105), (0.22, 0.22, 0.026), "metal", collision=True)
    for x in (1.12, 1.42):
        for y in (-0.63, -0.35):
            b.cylinder(base, (x, y, 0.99), 0.019, 0.09, "metal", collision=True).set(
                "name", b.unique("loadlock_standoff")
            )
    # Fixed guide rails support the translating sample carrier over its full
    # travel. Their top meets the carriage's lower face at z = 1.193 m.
    for x in (1.215, 1.325):
        b.box(
            base, (x, -0.40, 1.162), (0.014, 0.13, 0.031), "metal", collision=True
        ).set("name", b.unique("carrier_guide"))
    _flange(b, base, (1.27, -0.715, 1.265), 0.145, "xz")
    b.ring(base, (1.27, -0.38, 1.265), 0.145, 0.024, "bright", "xz")
    for x in (1.12, 1.42):
        b.rod(base, (x, -0.70, 1.265), (x, -0.37, 1.265), 0.018, "metal")
    hatch = b.moving(
        base,
        "load_lock_door",
        (1.108, -0.746, 1.265),
        (0, 0, -1),
        (0, 1.20),
        kind="hinge",
        mass=0.4,
        kp=450,
    )
    b.cylinder(
        hatch,
        (0.162, 0, 0),
        0.137,
        0.008,
        "glass",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    _flange(b, hatch, (0.162, 0, 0), 0.137, "xz", 10)
    b.rod(hatch, (0.25, -0.035, -0.04), (0.25, -0.035, 0.04), 0.009, "dark")
    b.site(hatch, "door_handle", (0.25, -0.035, 0))
    transfer = b.moving(
        base,
        "sample_transfer",
        (1.27, -0.46, 1.205),
        (0, 1, 0),
        (0, 0.12),
        mass=0.15,
        kp=500,
    )
    b.box(transfer, (0, 0, 0), (0.076, 0.06, 0.012), "dark", collision=True)
    b.rod(transfer, (0.06, 0, 0), (0.62, 0, 0), 0.010, "bright")
    b.cylinder(transfer, (0.62, 0, 0), 0.027, 0.037, "dark", euler=(0, math.pi / 2, 0))
    rotation = b.moving(
        transfer,
        "sample_rotation",
        (0, 0, 0.022),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.06,
        kp=120,
    )
    b.cylinder(rotation, (0, 0, 0), 0.043, 0.009, "copper", collision=True)
    b.cylinder(rotation, (0, 0, 0.010), 0.030, 0.001, "cream")
    for index in range(9):
        angle = math.tau * index / 9
        b.geom(
            rotation,
            "ellipsoid",
            (0.002, 0.001, 0.0005),
            (0.018 * math.cos(angle), 0.018 * math.sin(angle), 0.0115),
            "dark",
        )
    b.site(rotation, "mineral_mount", (0, 0, 0.012))
    # Grounded cable stalks and curved hoses visible in the source image.
    for x in (0.87, 1.79):
        b.box(base, (x, 0.62, 0.919), (0.07, 0.10, 0.019), "metal", collision=True)
        b.rod(base, (x, 0.62, 0.93), (x, 0.62, 2.25), 0.018, "metal")
    b.rod(base, (0.87, 0.62, 2.25), (1.79, 0.62, 2.25), 0.018, "metal")
    for index in range(12):
        dx = index * 0.055
        _cable(
            b,
            base,
            [
                (1.07 + dx, 0.07, 1.60),
                (0.95 + dx, 0.58, 2.15),
                (0.93 + dx, 0.68, 1.13),
                (0.72 + dx, 0.61, 0.94),
            ],
            0.0035,
            "blue" if index % 4 == 0 else "dark",
        )
    for i in range(3):
        points = [
            (1.72 + i * 0.1, -0.05, 1.30),
            (1.82 + i * 0.1, -0.25, 1.05),
            (1.85 + i * 0.1, -0.50, 0.96),
            (1.70 + i * 0.1, -0.73, 0.94),
        ]
        _cable(b, base, points, 0.018, "metal")
    b.metadata["capabilities"] = [
        "load-lock access articulation",
        "120 mm representative mineral-mount transfer",
        "mineral-mount rotation and target site",
    ]
    b.metadata["limitations"].append(
        "Source/extraction optics, ion transport, magnet fields, detector response and isotopic ages are not simulated. Access and stage strokes are authored mechanical surrogates, not the SHRIMP operating procedure."
    )


def _anu_features(world, definition):
    # Rectangular overhead air-handling boxes and a pale green wall band are
    # visible in the official preview. Exact dimensions and routes are inferred.
    for i, x in enumerate((-2.25, -0.65, 0.95, 2.55)):
        box(
            world,
            f"anu_air_box_{i}",
            (0.66, 0.30, 0.25),
            (x, 2.20, 2.50),
            (0.76, 0.80, 0.77, 1),
        )
        box(
            world,
            f"anu_air_vent_{i}",
            (0.22, 0.009, 0.12),
            (x + 0.22, 1.887, 2.49),
            (0.15, 0.20, 0.21, 1),
            collision=False,
        )
        for row in range(9):
            box(
                world,
                f"anu_vent_grille_{i}_{row}",
                (0.22, 0.01, 0.003),
                (x + 0.22, 1.876, 2.385 + row * 0.026),
                (0.61, 0.67, 0.65, 1),
                collision=False,
            )
    box(
        world,
        "anu_wall_band",
        (3.4, 0.016, 0.055),
        (0, 2.48, 1.22),
        (0.22, 0.48, 0.42, 1),
        collision=False,
    )
    box(
        world,
        "anu_control_screen_stand",
        (0.024, 0.025, 0.13),
        (2.85, 0.95, 0.98),
        (0.38, 0.41, 0.42, 1),
    )
    box(
        world,
        "anu_control_screen",
        (0.28, 0.025, 0.17),
        (2.85, 0.95, 1.23),
        (0.04, 0.07, 0.10, 1),
    )


BUILDERS = {"oxide_mbe_cluster": _mbe, "shrimp_ion_microprobe": _shrimp}
SOURCES = {
    "oxide_mbe_cluster": {
        "reference": "Cornell Schlom dual oxide MBE photograph; authored exterior and load-lock mechanism",
        "url": CORNELL,
        "dimensions_m": [3.7, 1.9, 2.47],
        "dimension_basis": "Estimated from photograph; no surveyed or OEM dimensions verified",
    }
}
SAMPLE_INTERFACES = {
    "oxide_mbe_cluster": (
        "substrate",
        (0.042, 0.038, 0.004),
        "clamped",
        "rigid substrate coupon clamped to transfer holder",
    )
}
FEATURES = {"cornell_schlom_mbe": _cornell_features}
SOURCES["shrimp_ion_microprobe"] = {
    "reference": "ANU SHRIMP RG facility preview; authored large cabinet chassis, curved analyzer and vacuum-source exterior",
    "url": ANU,
    "dimensions_m": [4.6, 2.1, 2.28],
    "dimension_basis": "Estimated from official room preview; source gives no measured instrument dimensions",
}
SAMPLE_INTERFACES["shrimp_ion_microprobe"] = (
    "mineral_mount",
    (0.060, 0.060, 0.003),
    "clamped",
    "rigid polished mineral mount with illustrative grain marks",
)
FEATURES["anu_shrimp_geochronology"] = _anu_features
