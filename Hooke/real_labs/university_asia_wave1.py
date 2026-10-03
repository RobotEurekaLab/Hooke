"""Photo-informed Asian laboratory assemblies, in metres.

Declared axes are mechanical surrogates. Geometry does not implement calibrated
optical, magnetic, aerodynamic or hydrodynamic measurements.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers


KYOTO = "https://www.mbsys.me.kyoto-u.ac.jp/album/lab/"
SJTU = "https://oe.sjtu.edu.cn/list.php?id=26&t=3"
KAIST = "https://spintronics.kaist.ac.kr/facilities.html"
HKUST = "https://aaf.ust.hk/fac/1/"

SOURCES = {
    "optical_manipulation_workspace": {
        "reference": "Yokokawa cB1S03 actual TIRF workstation and entrance photographs",
        "url": KYOTO,
        "dimensions_m": [2.2, 1.15, 0.87],
        "dimension_basis": "Estimated external optical assembly; no surveyed geometry or OEM CAD. Slide is a conventional 75 x 25 mm surrogate.",
    },
    "towing_carriage_basin": {
        "reference": "SJTU actual deepwater-basin hall and offshore-platform model photographs",
        "url": SJTU,
        "dimensions_m": [18.0, 26.4, 5.55],
        "dimension_basis": "Estimated display-scale basin and spanning carriage, not the measured SJTU basin. Two-axis support is an authored mechanical proxy; water is visual only.",
    },
    "magnetic_microscopy_probe": {
        "reference": "KAIST USDL installed MOKE probe-station photograph with Nikon LV-IM label",
        "url": KAIST,
        "dimensions_m": [0.88, 0.76, 1.0],
        "dimension_basis": "Authored metric estimates of plate, column, microscope and probes. No OEM internal design, magnet field map or precision-control specification reproduced.",
    },
    "windtunnel_model_stage": {
        "reference": "HKUST AAF actual suspended-hydrofoil photograph and official test-section dimensions",
        "url": HKUST,
        "dimensions_m": [3.10, 14.40, 3.32],
        "dimension_basis": "Published clear test-section L/W/H is 14/2.5/2 m. External frame, raised floor, specimen and mounts are authored estimates; full return circuit is excluded.",
    },
}


def _body(parent, name, position):
    return ET.SubElement(parent, "body", name=name, pos=numbers(position))


def _bolts(b, parent, xs, ys, z):
    for x in xs:
        for y in ys:
            b.cylinder(parent, (x, y, z), 0.003, 0.001, "dark")


def _optical_station(b, base, params):
    b.box(base, (0, 0, 0.018), (1.08, 0.55, 0.018), "metal", collision=True)
    _bolts(b, base, [i * 0.12 for i in range(-8, 9)], [-0.48, -0.36, 0.36, 0.48], 0.037)
    scope = _body(base, b.name + "__microscope_frame", (0.51, 0.1, 0.036))
    b.box(scope, (0, 0, 0.035), (0.20, 0.26, 0.035), "cream", collision=True)
    b.housing(
        scope,
        [
            (0.065, 0.16, 0.08, 0.16),
            (0.46, 0.10, 0.09, 0.17),
            (0.53, 0.075, 0.07, 0.16),
        ],
        material="cream",
    )
    b.box(scope, (0, 0.18, 0.26), (0.075, 0.07, 0.20), "cream", collision=True)
    # The upper camera mast is carried by the rear microscope frame.
    b.rod(scope, (0, 0.18, 0.44), (0, 0.18, 0.70), 0.026, "bright")
    b.box(scope, (0, 0.18, 0.74), (0.075, 0.065, 0.055), "dark")
    b.rod(scope, (0, 0.18, 0.52), (0, -0.09, 0.52), 0.026, "cream")
    b.cylinder(scope, (0, -0.09, 0.47), 0.024, 0.047, "dark")
    for side in (-1, 1):
        b.rod(
            scope,
            (side * 0.028, 0.12, 0.40),
            (side * 0.045, -0.08, 0.49),
            0.023,
            "cream",
        )
        b.rod(
            scope,
            (side * 0.045, -0.08, 0.49),
            (side * 0.048, -0.125, 0.51),
            0.025,
            "rubber",
        )
    b.cylinder(
        scope, (-0.177, 0.09, 0.18), 0.045, 0.025, "dark", euler=(0, math.pi / 2, 0)
    )
    # A bridge supports the two nested XY carriers, with clear optical access.
    for x in (-0.095, 0.095):
        b.box(scope, (x, -0.09, 0.115), (0.018, 0.105, 0.05), "dark")
        b.box(scope, (x, -0.09, 0.18), (0.014, 0.13, 0.018), "metal")
    xstage = b.moving(
        scope,
        "slide_x",
        (0, -0.09, 0.205),
        (1, 0, 0),
        (-0.018, 0.018),
        mass=0.10,
        kp=200,
    )
    b.box(xstage, (0, 0, 0), (0.13, 0.10, 0.012), "dark", collision=True)
    ystage = b.moving(
        xstage, "slide_y", (0, 0, 0.019), (0, 1, 0), (-0.014, 0.014), mass=0.055, kp=180
    )
    b.box(ystage, (0, 0, 0), (0.065, 0.055, 0.007), "metal", collision=True)
    b.box(ystage, (0, 0, 0.0075), (0.0375, 0.0125, 0.0005), "glass")
    b.box(ystage, (0, 0, 0.00808), (0.001, 0.001, 0.00008), "sample")
    for x in (-0.035, 0.035):
        b.box(ystage, (x, 0.015, 0.010), (0.008, 0.006, 0.002), "dark")
    b.site(ystage, "inert_target", (0, 0, 0.00816), 0.0003)
    focus = b.moving(
        scope,
        "objective_focus",
        (0, -0.09, 0.132),
        (0, 0, 1),
        (-0.003, 0.003),
        mass=0.04,
        kp=180,
    )
    b.cylinder(focus, (0, 0, 0), 0.018, 0.027, "metal")
    b.cylinder(focus, (0, 0, 0.030), 0.008, 0.009, "lens")
    b.rod(scope, (0, -0.09, 0.08), (0, -0.09, 0.12), 0.023, "dark")
    # External camera and optics are mechanically carried by table-mounted posts.
    for index, (x, y, height) in enumerate(
        (
            (-0.76, 0.18, 0.29),
            (-0.44, 0.18, 0.29),
            (-0.12, 0.18, 0.29),
            (-0.72, -0.20, 0.22),
            (-0.39, -0.20, 0.22),
        )
    ):
        b.box(base, (x, y, 0.05), (0.06, 0.045, 0.014), "dark")
        b.rod(base, (x, y, 0.062), (x, y, height), 0.012, "bright")
        b.ring(base, (x, y, height), 0.037, 0.008, "dark", plane="yz")
        b.cylinder(
            base, (x, y, height), 0.026, 0.004, "lens", euler=(0, math.pi / 2, 0)
        )
        b.box(base, (x, y + 0.04, height), (0.022, 0.018, 0.022), "dark")
    b.box(base, (-0.96, 0.18, 0.21), (0.075, 0.10, 0.173), "cream")
    b.rod(base, (-0.90, 0.18, 0.29), (-0.79, 0.18, 0.29), 0.032, "dark")
    b.rod(base, (-0.13, 0.18, 0.29), (0.34, 0.18, 0.29), 0.026, "dark")
    b.box(base, (-0.79, -0.40, 0.105), (0.16, 0.075, 0.069), "metal")
    b.screen(base, (-0.79, -0.477, 0.105), 0.13, 0.06)
    for path in (
        (
            (0.51, 0.28, 0.79),
            (0.77, 0.36, 0.72),
            (0.88, 0.38, 0.045),
            (0.20, 0.43, 0.045),
            (-0.86, 0.42, 0.045),
        ),
        (
            (-0.96, 0.27, 0.22),
            (-1.01, 0.37, 0.13),
            (-0.94, 0.47, 0.045),
            (-0.68, 0.44, 0.045),
        ),
    ):
        for start, end in zip(path, path[1:]):
            b.rod(base, start, end, 0.004, "rubber")
    b.metadata["capabilities"] = [
        "xy_slide_position",
        "objective_focus_surrogate",
        "visible_2mm_inert_target",
    ]
    b.metadata["limitations"].append(
        "The inert target is not a cell. No TIRF illumination, image formation, fluorescence, molecular dynamics or calibrated optical measurement."
    )


def _kyoto_features(world, definition):
    light = (0.78, 0.79, 0.74, 1)
    dark = (0.045, 0.05, 0.055, 1)
    # A genuine opening connects the front optical room to the bright rear bay.
    for name, center, half in (("left", -1.775, 1.225), ("right", 1.825, 1.175)):
        box(
            world,
            "kyoto_partition_" + name,
            (half, 0.06, 1.45),
            (center, 1.45, 1.45),
            light,
        )
    box(world, "kyoto_door_lintel", (0.60, 0.06, 0.35), (0.05, 1.45, 2.55), light)
    # Curtain rails are attached to posts; curtain folds remain outside aisles.
    for x in (-1.1, 2.05):
        box(
            world,
            f"kyoto_curtain_post_{x}",
            (0.022, 0.022, 1.39),
            (x, 0.88, 1.39),
            (0.44, 0.46, 0.47, 1),
        )
    box(
        world,
        "kyoto_curtain_rail",
        (1.60, 0.025, 0.025),
        (0.475, 0.88, 2.77),
        (0.45, 0.47, 0.49, 1),
    )
    for index in range(22):
        x = -1.07 + index * 0.145
        if -0.63 < x < 0.74:
            continue  # Preserve a visible walk-through opening to the rear bay.
        box(
            world,
            f"kyoto_curtain_fold_{index}",
            (0.074, 0.025, 1.25),
            (x, 0.87 + 0.022 * (index % 2), 1.48),
            dark,
            collision=False,
        )
    # Rear shelves belong to the adjoining preparation bay, not a second lab.
    for x in (-2.5, 1.8):
        for z in (0.95, 1.45, 1.95):
            box(world, f"kyoto_shelf_{x}_{z}", (0.34, 0.32, 0.022), (x, 2.70, z), light)
        for dx in (-0.31, 0.31):
            box(
                world,
                f"kyoto_shelf_post_{x}_{dx}",
                (0.015, 0.025, 1.0),
                (x + dx, 2.70, 1.0),
                (0.5, 0.52, 0.54, 1),
            )
    box(
        world,
        "kyoto_floor_cable_cover",
        (0.055, 1.40, 0.012),
        (-1.43, -0.50, 0.012),
        dark,
    )
    for name, half, position in (
        ("monitor_foot", (0.16, 0.13, 0.017), (-2.25, -0.10, 0.767)),
        ("monitor_post", (0.025, 0.025, 0.14), (-2.25, -0.07, 0.91)),
        ("monitor", (0.29, 0.035, 0.18), (-2.25, -0.07, 1.12)),
        ("keyboard", (0.25, 0.09, 0.013), (-2.25, -0.56, 0.763)),
        ("computer", (0.12, 0.24, 0.28), (-2.25, 0.48, 0.28)),
    ):
        box(world, "kyoto_acquisition_" + name, half, position, dark)
    box(
        world,
        "kyoto_acquisition_screen",
        (0.27, 0.002, 0.16),
        (-2.25, -0.107, 1.12),
        (0.04, 0.13, 0.17, 1),
        collision=False,
    )


def _towing_basin(b, base, params):
    # The basin is physically supported above the room floor; no hidden void.
    b.box(base, (0, 0, 0.09), (8.4, 13.0, 0.09), "metal", collision=True)
    for side in (-1, 1):
        b.box(base, (side * 8.3, 0, 0.68), (0.20, 13.0, 0.50), "metal", collision=True)
        b.box(base, (0, side * 12.8, 0.68), (8.1, 0.20, 0.50), "metal", collision=True)
        b.box(base, (side * 8.3, 0, 1.22), (0.28, 13.1, 0.045), "bright")
        # Full-length rails carry both ends of the moving bridge.
        b.box(base, (side * 8.65, 0, 2.30), (0.15, 12.8, 0.12), "blue", collision=True)
        b.box(base, (side * 8.65, 0, 2.45), (0.065, 12.8, 0.035), "bright")
        for y in (-11.5, -6, 0, 6, 11.5):
            b.box(
                base, (side * 8.65, y, 1.10), (0.14, 0.15, 1.10), "blue", collision=True
            )
            b.box(base, (side * 8.65, y, 0.05), (0.28, 0.30, 0.05), "dark")
    # Opaque tinted surface is a visual water proxy, deliberately non-colliding.
    for key, color in (
        ("basin_water", "0.015 0.07 0.09 1"),
        ("water_ripple", "0.045 0.15 0.17 1"),
    ):
        ET.SubElement(
            b.root.find("asset"),
            "material",
            name=b.name + "__mat_" + key,
            rgba=color,
            specular="0.8",
            shininess="0.8",
        )
    b.box(base, (0, 0, 1.05), (8.09, 12.59, 0.012), "basin_water")
    for row in range(17):
        y = -11.4 + row * 1.34
        for index in range(20):
            x1, x2 = -7.8 + index * 0.78, -7.8 + (index + 1) * 0.78
            b.rod(
                base,
                (x1, y + 0.07 * math.sin(x1 * 1.8 + row), 1.063),
                (x2, y + 0.07 * math.sin(x2 * 1.8 + row), 1.063),
                0.004,
                "water_ripple",
            )
    # Slatted absorber edge is visible in the reference foreground.
    for index in range(40):
        x = -7.8 + index * 0.40
        b.box(base, (x, -11.80, 0.97), (0.075, 0.83, 0.07), "metal", euler=(0.13, 0, 0))
    bridge = b.moving(
        base,
        "carriage_longitudinal",
        (0, 0, 2.58),
        (0, 1, 0),
        (-8.0, 8.0),
        mass=75,
        kp=3000,
        force=10000,
    )
    for side in (-1, 1):
        b.box(bridge, (side * 8.65, 0, -0.045), (0.21, 0.55, 0.052), "blue")
        b.box(
            bridge, (side * 8.65, 0, 0.60), (0.13, 0.23, 0.60), "blue", collision=True
        )
        for y in (-0.36, 0.36):
            b.cylinder(
                bridge,
                (side * 8.65, y, -0.085),
                0.11,
                0.065,
                "dark",
                euler=(0, math.pi / 2, 0),
            )
    for y in (-0.31, 0.31):
        for z in (1.22, 2.78):
            b.box(bridge, (0, y, z), (8.76, 0.08, 0.09), "blue", collision=True)
        for index in range(11):
            x = -8.5 + index * 1.7
            b.rod(bridge, (x, y, 1.22), (x, y, 2.78), 0.055, "blue")
            if index < 10:
                b.rod(bridge, (x, y, 1.22), (x + 1.7, y, 2.78), 0.048, "blue")
    b.box(bridge, (0, 0, 1.20), (8.60, 0.40, 0.065), "metal")
    for x in (-8.65, 8.65):
        b.rod(bridge, (x, -0.31, 1.22), (x, 0.31, 2.78), 0.06, "blue")
    trolley = b.moving(
        bridge,
        "model_transverse",
        (0, 0, 1.02),
        (1, 0, 0),
        (-5.5, 5.5),
        mass=8,
        kp=900,
        force=1000,
    )
    b.box(trolley, (0, 0, 0), (0.35, 0.35, 0.11), "warning", collision=True)
    b.rod(trolley, (0, 0, -0.10), (0, 0, -1.82), 0.028, "bright", collision=True)
    # Supported offshore test body: the rig, rather than fake buoyancy, carries it.
    deck_z = -1.95
    b.box(trolley, (0, 0, deck_z), (0.80, 0.44, 0.10), "warning", collision=True)
    for x in (-0.60, 0.60):
        for y in (-0.30, 0.30):
            b.cylinder(trolley, (x, y, deck_z - 0.32), 0.075, 0.22, "warning")
    for y in (-0.30, 0.30):
        b.box(trolley, (0, y, deck_z - 0.54), (0.78, 0.13, 0.08), "warning")
    # A small lattice mast and crane preserve the model's observed silhouette.
    for dx in (-0.10, 0.10):
        for dy in (-0.10, 0.10):
            b.rod(
                trolley,
                (-0.42 + dx, dy, deck_z + 0.10),
                (-0.42 + dx * 0.45, dy * 0.45, deck_z + 1.22),
                0.012,
                "shell",
            )
    for z in (0.3, 0.6, 0.9):
        b.rod(
            trolley,
            (-0.51, -0.09, deck_z + z),
            (-0.33, 0.09, deck_z + z + 0.22),
            0.009,
            "shell",
        )
    b.rod(
        trolley, (0.36, 0.1, deck_z + 0.1), (0.36, 0.1, deck_z + 0.50), 0.035, "orange"
    )
    b.rod(
        trolley, (0.36, 0.1, deck_z + 0.50), (1.00, 0.1, deck_z + 0.88), 0.022, "orange"
    )
    b.rod(
        trolley, (0.36, 0.1, deck_z + 0.60), (1.00, 0.1, deck_z + 0.88), 0.012, "bright"
    )
    b.site(trolley, "offshore_model", (0, 0, deck_z), 0.02)
    b.metadata["capabilities"] = [
        "longitudinal_towing_carriage",
        "transverse_supported_model_position",
        "visible_offshore_test_body",
    ]
    b.metadata["limitations"].append(
        "Water and wave highlights are static visual geometry. No buoyancy, fluid forces, wave generation, mooring dynamics or offshore structural response. Model is rigidly attached to the visible carriage support."
    )


def _basin_features(world, definition):
    steel = (0.24, 0.33, 0.37, 1)
    # Hall columns support the high roof; two generous edge aisles stay open.
    for y in (-13, -6.5, 0, 6.5, 13):
        for x in (-11.6, 11.6):
            box(
                world,
                f"basin_hall_column_{x}_{y}",
                (0.19, 0.20, 4.75),
                (x, y, 4.75),
                steel,
            )
        box(world, f"basin_hall_crossbeam_{y}", (11.8, 0.16, 0.24), (0, y, 9.26), steel)
    for index in range(11):
        x = -10 + index * 2.0
        box(
            world,
            f"basin_back_window_{index}",
            (0.75, 0.014, 0.60),
            (x, 16.94, 4.1),
            (0.48, 0.63, 0.67, 1),
            collision=False,
        )
    for side in (-1, 1):
        box(
            world,
            f"basin_clear_aisle_mark_{side}",
            (0.025, 12.7, 0.001),
            (side * 10.1, 0, 0.002),
            (0.9, 0.71, 0.13, 1),
            collision=False,
        )


def _magnetic_probe(b, base, params):
    b.box(base, (0, 0, 0.018), (0.43, 0.37, 0.018), "bright", collision=True)
    _bolts(b, base, [-0.38, -0.2, 0, 0.2, 0.38], [-0.32, 0.32], 0.037)
    # Four posts and a rear microscope column join the breadboard to the plate.
    for x in (-0.34, 0.34):
        for y in (-0.28, 0.28):
            b.cylinder(base, (x, y, 0.195), 0.023, 0.159, "bright", collision=True)
    for x in (-0.26, 0.26):
        b.box(base, (x, 0, 0.36), (0.16, 0.35, 0.014), "bright", collision=True)
    for y in (-0.24, 0.24):
        b.box(base, (0, y, 0.36), (0.10, 0.11, 0.014), "bright", collision=True)
    b.box(base, (0, 0, 0.076), (0.12, 0.11, 0.04), "dark", collision=True)
    b.cylinder(base, (0, 0, 0.185), 0.087, 0.068, "metal")
    for index in range(12):
        b.ring(
            base, (0, 0, 0.126 + index * 0.010), 0.092, 0.0045, "copper", segments=20
        )
    b.cylinder(base, (0, 0, 0.26), 0.10, 0.009, "metal")
    b.box(base, (0, 0, 0.306), (0.062, 0.047, 0.037), "dark")
    xstage = b.moving(
        base, "specimen_x", (0, 0, 0.355), (1, 0, 0), (-0.012, 0.012), mass=0.08, kp=220
    )
    b.box(xstage, (0, 0, 0), (0.074, 0.060, 0.013), "metal", collision=True)
    ystage = b.moving(
        xstage,
        "specimen_y",
        (0, 0, 0.023),
        (0, 1, 0),
        (-0.010, 0.010),
        mass=0.04,
        kp=200,
    )
    b.box(ystage, (0, 0, 0), (0.035, 0.026, 0.010), "dark", collision=True)
    b.box(ystage, (0, 0, 0.0105), (0.006, 0.006, 0.0005), "sample")
    for x in (-0.003, 0.003):
        b.box(ystage, (x, 0, 0.0111), (0.001, 0.002, 0.0001), "copper")
    b.site(ystage, "magnetic_coupon", (0, 0, 0.0112), 0.0003)
    # A pole-mounted white industrial microscope, rather than a generic box.
    b.cylinder(base, (-0.30, 0.15, 0.46), 0.032, 0.424, "bright", collision=True)
    b.cylinder(base, (-0.30, 0.15, 0.89), 0.05, 0.012, "metal")
    b.box(base, (-0.30, 0.15, 0.64), (0.055, 0.06, 0.16), "dark")
    b.box(base, (-0.15, 0.15, 0.66), (0.15, 0.065, 0.045), "cream")
    focus = b.moving(
        base,
        "microscope_focus",
        (0, 0, 0.64),
        (0, 0, 1),
        (-0.01, 0.01),
        mass=0.65,
        kp=350,
    )
    b.box(focus, (-0.11, 0.12, 0), (0.08, 0.08, 0.10), "cream")
    b.housing(
        focus,
        [
            (-0.08, 0.125, 0.105, 0.01),
            (0.075, 0.125, 0.105, 0.01),
            (0.105, 0.105, 0.09, 0.01),
        ],
        radius=0.012,
        material="cream",
    )
    b.cylinder(focus, (0, 0, -0.103), 0.082, 0.025, "dark")
    b.cylinder(focus, (0, 0, -0.157), 0.025, 0.031, "bright", collision=True)
    b.cylinder(focus, (0, 0, -0.19), 0.017, 0.003, "lens")
    b.cylinder(focus, (0, 0.02, 0.185), 0.045, 0.08, "cream")
    b.box(focus, (0, 0.02, 0.29), (0.050, 0.045, 0.030), "metal")
    b.cylinder(focus, (-0.17, 0.02, 0), 0.038, 0.018, "dark", euler=(math.pi / 2, 0, 0))
    # Three supported micrometer probe blocks surround, rather than hide, the die.
    for index, (x, y, color) in enumerate(
        ((0.25, 0, "sample"), (0, -0.23, "blue"), (0, 0.23, "blue"))
    ):
        b.box(base, (x, y, 0.398), (0.060, 0.060, 0.024), "dark")
        b.box(base, (x, y, 0.435), (0.047, 0.042, 0.015), color)
        b.cylinder(
            base,
            (x + 0.065, y, 0.435),
            0.012,
            0.024,
            "copper",
            euler=(0, math.pi / 2, 0),
        )
        b.cylinder(base, (x, y, 0.466), 0.013, 0.018, "copper")
        length = math.hypot(x, y)
        sx, sy = x / length, y / length
        b.rod(base, (x, y, 0.427), (sx * 0.075, sy * 0.075, 0.408), 0.006, "metal")
        b.rod(
            base,
            (sx * 0.075, sy * 0.075, 0.408),
            (sx * 0.021, sy * 0.021, 0.399),
            0.0007,
            "bright",
        )
        b.rod(base, (x, y, 0.440), (x + 0.06, y + 0.05, 0.40), 0.003, "rubber")
    b.metadata["capabilities"] = [
        "specimen_xy_position",
        "microscope_focus_clearance",
        "visible_inert_12mm_coupon",
    ]
    b.metadata["limitations"].append(
        "Probes remain above the inert coupon; no electrical contact, magnetization, field generation, Kerr image formation or force sensing. Manual-looking micrometers are visual; only the three declared axes are actuated surrogates."
    )


def _kaist_features(world, definition):
    metal = (0.26, 0.29, 0.31, 1)
    # Dedicated instrument rack at the right edge of the optical bench.
    for x in (1.62, 2.23):
        for y in (0.50, 1.13):
            box(
                world,
                f"kaist_rack_post_{x}_{y}",
                (0.025, 0.025, 0.91),
                (x, y, 0.91),
                metal,
            )
    for index, z in enumerate((0.19, 0.54, 0.89, 1.24, 1.60)):
        box(
            world,
            f"kaist_rack_shelf_{index}",
            (0.33, 0.34, 0.018),
            (1.925, 0.815, z),
            metal,
        )
        if index < 4:
            box(
                world,
                f"kaist_readout_{index}",
                (0.25, 0.27, 0.09),
                (1.925, 0.80, z + 0.11),
                (0.70, 0.72, 0.71, 1),
            )
            box(
                world,
                f"kaist_readout_screen_{index}",
                (0.11, 0.003, 0.045),
                (1.85, 0.525, z + 0.11),
                (0.035, 0.16, 0.20, 1),
                collision=False,
            )
    box(
        world,
        "kaist_service_raceway",
        (2.05, 0.035, 0.06),
        (-0.20, 1.85, 1.16),
        (0.73, 0.74, 0.72, 1),
    )
    for index in range(7):
        box(
            world,
            f"kaist_socket_{index}",
            (0.045, 0.008, 0.042),
            (-1.9 + index * 0.55, 1.806, 1.16),
            (0.92, 0.91, 0.84, 1),
            collision=False,
        )


def _windtunnel(b, base, params):
    # Clear section is 2.5 m wide, 14 m long, 2 m high; external frame estimated.
    for y in (-6.8, -4.5, -2.2, 0, 2.2, 4.5, 6.8):
        for x in (-1.42, 1.42):
            b.box(base, (x, y, 0.30), (0.11, 0.16, 0.30), "metal", collision=True)
            b.box(base, (x, y, 0.05), (0.17, 0.23, 0.05), "dark")
    b.box(base, (0, 0, 0.61), (1.52, 7.15, 0.07), "dark", collision=True)
    b.box(base, (0, 0, 0.685), (0.77, 0.72, 0.005), "shell")
    for side in (-1, 1):
        # Steel frame at the pane edges supports the dark overhead panels.
        for y in (-7, -4.5, -2.2, 0, 2.2, 4.5, 7):
            b.box(
                base,
                (side * 1.33, y, 1.71),
                (0.075, 0.065, 1.03),
                "metal",
                collision=True,
            )
        for z in (0.75, 2.68):
            b.box(
                base, (side * 1.33, 0, z), (0.075, 7.15, 0.055), "dark", collision=True
            )
        # Individual glazed panels leave visible external access and framing.
        for y, half in (
            (-5.78, 1.16),
            (-3.35, 1.08),
            (-1.10, 1.02),
            (1.10, 1.02),
            (3.35, 1.08),
            (5.78, 1.16),
        ):
            b.box(
                base,
                (side * 1.26, y, 1.70),
                (0.010, half, 0.94),
                "glass",
                collision=True,
            )
    # Partial roof panels preserve the overhead mounting channel and inspection view.
    for x in (-0.91, 0.91):
        b.box(base, (x, 0, 2.77), (0.40, 7.15, 0.09), "dark", collision=True)
    for y in (-6.9, -4.5, -2.2, 2.2, 4.5, 6.9):
        b.box(base, (0, y, 2.80), (1.40, 0.09, 0.10), "metal", collision=True)
    # A support bridge carries the yaw bearing and model's long sting.
    for y in (-0.58, 0.58):
        b.box(base, (0, y, 2.87), (1.40, 0.065, 0.065), "metal", collision=True)
    for x in (-0.36, 0.36):
        b.box(base, (x, 0, 2.92), (0.055, 0.64, 0.09), "metal", collision=True)
    b.box(base, (0, 0, 3.03), (0.43, 0.38, 0.04), "dark")
    b.cylinder(base, (0, 0, 3.095), 0.20, 0.025, "metal")
    yaw = b.moving(
        base,
        "model_yaw",
        (0, 0, 3.13),
        (0, 0, 1),
        (-0.65, 0.65),
        kind="hinge",
        mass=1.0,
        kp=240,
        force=100,
    )
    b.cylinder(yaw, (0, 0, 0), 0.17, 0.015, "dark")
    b.rod(yaw, (0, 0, -0.015), (0, 0, -1.30), 0.022, "bright", collision=True)
    pitch = b.moving(
        yaw,
        "model_pitch",
        (0, 0, -1.30),
        (1, 0, 0),
        (-0.26, 0.26),
        kind="hinge",
        mass=0.28,
        kp=180,
        force=35,
    )
    b.cylinder(pitch, (0, 0, 0), 0.035, 0.037, "metal", euler=(0, math.pi / 2, 0))
    b.rod(pitch, (0, -0.50, 0), (0, 0.50, 0), 0.019, "bright", collision=True)
    for y, width in ((-0.38, 0.58), (0.38, 0.44)):
        b.geom(
            pitch,
            "ellipsoid",
            (width, 0.075, 0.015),
            (0, y, -0.018),
            "bright",
            collision=True,
        )
        for side in (-1, 1):
            b.rod(
                pitch,
                (side * width * 0.70, y, -0.018),
                (side * width, y, 0.035),
                0.012,
                "bright",
            )
    b.site(pitch, "hydrofoil_model", (0, 0, 0), 0.012)
    for y in (-3.5, -1.0, 1.0, 3.5):
        for x in (-0.84, 0.84):
            b.box(base, (x, y, 2.668), (0.20, 0.24, 0.008), "shell")
    b.metadata["capabilities"] = [
        "supported_model_yaw",
        "supported_model_pitch",
        "visible_rigid_hydrofoil_model",
    ]
    b.metadata["limitations"].append(
        "Only the local straight test section and mechanically attached model are represented. No airflow, lift, drag, acoustic prediction or validated tunnel control; the full return circuit is absent."
    )


def _wind_features(world, definition):
    dark = (0.08, 0.10, 0.12, 1)
    # Observation console lives alongside the actual glazed test section.
    box(
        world,
        "wind_console_base",
        (0.48, 0.35, 0.37),
        (2.8, -0.2, 0.37),
        (0.57, 0.60, 0.59, 1),
    )
    box(world, "wind_console_top", (0.57, 0.43, 0.035), (2.8, -0.2, 0.775), dark)
    box(world, "wind_console_monitor_foot", (0.16, 0.10, 0.02), (2.8, 0.0, 0.83), dark)
    box(
        world,
        "wind_console_monitor_post",
        (0.025, 0.025, 0.18),
        (2.8, 0.04, 1.01),
        dark,
    )
    box(world, "wind_console_display", (0.27, 0.035, 0.18), (2.8, 0.04, 1.20), dark)
    box(
        world,
        "wind_console_screen",
        (0.25, 0.002, 0.16),
        (2.8, 0.002, 1.20),
        (0.04, 0.19, 0.22, 1),
        collision=False,
    )
    for side in (-1, 1):
        box(
            world,
            f"wind_aisle_mark_{side}",
            (0.025, 6.9, 0.001),
            (side * 2.0, 0, 0.002),
            (0.89, 0.72, 0.16, 1),
            collision=False,
        )


BUILDERS = {
    "optical_manipulation_workspace": _optical_station,
    "towing_carriage_basin": _towing_basin,
    "magnetic_microscopy_probe": _magnetic_probe,
    "windtunnel_model_stage": _windtunnel,
}
SAMPLE_INTERFACES = {
    "optical_manipulation_workspace": (
        "inert_target",
        (0.002, 0.002, 0.00016),
        "clamped",
        "Visible inert target fixed to a standard-sized glass slide",
    ),
    "towing_carriage_basin": (
        "offshore_model",
        (1.6, 0.88, 1.9),
        "clamped",
        "Visible rigid offshore-platform test body mechanically suspended from the carriage",
    ),
    "magnetic_microscopy_probe": (
        "magnetic_coupon",
        (0.012, 0.012, 0.001),
        "clamped",
        "Visible inert coupon with contact-pad geometry on a supported XY carrier",
    ),
    "windtunnel_model_stage": (
        "hydrofoil_model",
        (1.18, 1.05, 0.09),
        "clamped",
        "Visible rigid hydrofoil test model suspended from an overhead yaw/pitch mount",
    ),
}
FEATURES = {
    "kyoto_yokokawa_nanometrics": _kyoto_features,
    "sjtu_deepwater_offshore_basin": _basin_features,
    "kaist_usdl_magnetic_probe": _kaist_features,
    "hkust_aaf_wind_tunnel": _wind_features,
}
