"""Non-biological university apparatus with inert mechanical inspection targets."""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box


NJU = "https://rise.nju.edu.cn/f9/17/c59634a719127/page.htm"
HKU = "https://www.civil.hku.hk/h0_facilities_lab.html"
SNU = "https://acustica.snu.ac.kr/research/"
FUDAN = "https://saflab.fudan.edu.cn/sysgk/main.psp"


def _rack(b, parent, x, y, height=1.65):
    b.box(parent, (x, y, height / 2), (0.23, 0.21, height / 2), "dark", collision=True)
    for index in range(int(height / 0.17) - 1):
        z = 0.15 + index * 0.17
        b.box(parent, (x, y - 0.215, z), (0.208, 0.014, 0.066), "metal")
        b.screen(parent, (x - 0.065, y - 0.232, z), 0.12, 0.06)
        for dx in (0.07, 0.12, 0.17):
            b.cylinder(
                parent,
                (x + dx, y - 0.235, z),
                0.007,
                0.003,
                "dark",
                euler=(math.pi / 2, 0, 0),
            )


def _cryostat(b, base, params):
    # The photographed vessel stays closed and statically supported by its upper flange.
    for x in (-0.75, 0.75):
        for y in (-0.45, 0.53):
            b.box(base, (x, y, 0.032), (0.08, 0.08, 0.032), "dark", collision=True)
            b.box(base, (x, y, 1.24), (0.04, 0.04, 1.20), "metal", collision=True)
            b.box(base, (x - 0.025, y - 0.041, 1.24), (0.006, 0.002, 1.18), "dark")
        for z in (0.12, 1.85, 2.44):
            b.box(base, (x, 0.04, z), (0.04, 0.53, 0.04), "metal")
    for y in (-0.45, 0.53):
        for z in (0.12, 1.85, 2.44):
            b.box(base, (0, y, z), (0.79, 0.04, 0.04), "metal")
    for x in (-0.75, 0.75):
        b.rod(base, (x, -0.4, 0.18), (x, 0.48, 1.81), 0.026, "metal")
        b.rod(base, (x, 0.48, 0.18), (x, -0.4, 1.81), 0.026, "metal")
    for side in (-1, 1):
        b.rod(
            base, (side * 0.71, -0.44, 1.34), (side * 0.30, -0.44, 1.84), 0.034, "metal"
        )
    b.box(base, (0, 0.04, 1.85), (0.69, 0.47, 0.04), "bright", collision=True)
    b.cylinder(base, (0, 0.04, 1.25), 0.345, 0.55, "shell", collision=True)
    b.cylinder(base, (0, 0.04, 1.795), 0.40, 0.022, "metal")
    b.cylinder(base, (0, 0.04, 2.12), 0.155, 0.23, "shell")
    b.cylinder(
        base, (0.43, 0.04, 2.23), 0.068, 0.23, "metal", euler=(0, math.pi / 2, 0)
    )
    for angle in range(20):
        a = angle * math.tau / 20
        b.cylinder(
            base,
            (0.372 * math.cos(a), 0.04 + 0.372 * math.sin(a), 1.82),
            0.012,
            0.018,
            "bright",
        )
    b.cylinder(base, (-0.38, 0.11, 2.075), 0.11, 0.16, "metal")
    for index in range(3):
        b.rod(
            base,
            (-0.25 + index * 0.10, 0.40, 2.25),
            (-0.60 + index * 0.10, 0.50, 1.86),
            0.018,
            "rubber",
        )
    _rack(b, base, -1.13, 0.14, 1.82)
    # Inspection trolley follows the photo; its compact XY coupon fixture is an added proxy.
    for x in (-0.30, 0.43):
        for y in (-1.25, -0.72):
            b.cylinder(
                base, (x, y, 0.04), 0.04, 0.012, "rubber", euler=(math.pi / 2, 0, 0)
            )
            b.rod(base, (x, y, 0.05), (x, y, 0.84), 0.015, "metal")
    for z in (0.19, 0.84):
        b.box(base, (0.065, -0.985, z), (0.39, 0.29, 0.022), "metal", collision=True)
    b.box(base, (-0.12, -0.93, 1.03), (0.20, 0.19, 0.167), "shell", collision=True)
    b.screen(base, (-0.17, -1.13, 1.04), 0.25, 0.20)
    for index in range(4):
        b.cylinder(
            base,
            (0.045, -1.13, 0.97 + index * 0.038),
            0.011,
            0.005,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    b.box(base, (0.29, -0.995, 0.89), (0.105, 0.16, 0.028), "dark", collision=True)
    sx = b.moving(
        base,
        "inspection_x",
        (0.29, -1.07, 0.932),
        (1, 0, 0),
        (-0.032, 0.032),
        mass=0.05,
        kp=200,
    )
    b.box(sx, (0, 0, 0), (0.062, 0.09, 0.014), "metal", collision=True)
    sy = b.moving(
        sx,
        "inspection_y",
        (0, 0, 0.026),
        (0, 1, 0),
        (-0.035, 0.035),
        mass=0.035,
        kp=170,
    )
    b.box(sy, (0, 0, 0), (0.040, 0.041, 0.011), "dark", collision=True)
    b.box(sy, (0, 0, 0.013), (0.010, 0.010, 0.002), "sample")
    b.site(sy, "inspection_coupon", (0, 0, 0.015), 0.0005)
    b.metadata["capabilities"] = [
        "external_inspection_xy",
        "visible_inert_coupon",
        "statically_supported_closed_vessel",
    ]
    b.metadata["limitations"].append(
        "Cryostat is a closed static exterior. The trolley coupon fixture is an authored auxiliary mechanism; no cold-stage internals, refrigeration, vacuum or quantum computation."
    )


def _nju_features(world, definition):
    for index in range(6):
        box(
            world,
            f"nju_rear_panel_{index}",
            (0.36, 0.018, 1.1),
            (-2.0 + index * 0.78, 2.0, 1.15),
            (0.55, 0.60, 0.63, 1),
            collision=False,
        )


BUILDERS = {"quantum_cryostat_inspection_bay": _cryostat}
SOURCES = {
    "quantum_cryostat_inspection_bay": dict(
        reference="NJU RISE Electronics Building Room 109 installed cryostat photo",
        url=NJU,
        dimensions_m=[2.65, 2.10, 2.50],
        dimension_basis="Undimensioned external photograph; braced frame, white vessel and racks observed. Inspection trolley XY fixture is explicitly authored, not part of documented cryostat internals.",
    )
}
SAMPLE_INTERFACES = {
    "quantum_cryostat_inspection_bay": (
        "inspection_coupon",
        (0.020, 0.020, 0.004),
        "clamped",
        "Visible inert target on the auxiliary trolley fixture",
    )
}
FEATURES = {"nju_rise_quantum_109": _nju_features}


def _rock_frame(b, base, params):
    b.box(base, (0, 0, 0.15), (0.57, 0.46, 0.15), "dark", collision=True)
    b.box(base, (0, 0.27, 1.24), (0.48, 0.17, 0.94), "metal", collision=True)
    for x in (-0.43, 0.43):
        b.cylinder(base, (x, -0.07, 1.25), 0.055, 0.96, "bright", collision=True)
        for z in (0.34, 1.98):
            b.cylinder(base, (x, -0.07, z), 0.095, 0.055, "dark")
    b.box(base, (0, 0, 2.14), (0.54, 0.41, 0.23), "dark", collision=True)
    b.box(base, (0, -0.418, 2.16), (0.39, 0.012, 0.16), "metal")
    b.box(base, (0, 0, 0.53), (0.27, 0.30, 0.23), "metal", collision=True)
    b.cylinder(base, (0, -0.07, 0.845), 0.16, 0.09, "dark", collision=True)
    b.cylinder(base, (0, -0.07, 0.949), 0.12, 0.014, "bright", collision=True)
    b.cylinder(base, (0, -0.07, 1.029), 0.035, 0.066, "metal")
    for index in range(5):
        b.ring(
            base, (0, -0.07, 0.98 + index * 0.024), 0.0355, 0.0008, "dark", segments=16
        )
    b.site(base, "rock_core", (0, -0.07, 1.04), 0.0005)
    head = b.moving(
        base,
        "crosshead_height",
        (0, -0.07, 1.45),
        (0, 0, 1),
        (-0.10, 0.10),
        mass=0.7,
        kp=500,
    )
    b.box(head, (0, 0.015, 0.05), (0.29, 0.145, 0.12), "metal", collision=True)
    b.cylinder(head, (0, 0, -0.095), 0.09, 0.055, "dark")
    b.cylinder(head, (0, 0, -0.165), 0.11, 0.015, "bright", collision=True)
    b.cylinder(base, (0, 0.0, 1.88), 0.13, 0.18, "dark")
    door = b.moving(
        base,
        "guard_open",
        (-0.51, -0.46, 1.20),
        (0, 0, -1),
        (0, 1.03),
        kind="hinge",
        mass=0.35,
        kp=260,
    )
    b.box(door, (0.50, 0, 0), (0.46, 0.008, 0.65), "glass", collision=True)
    for x in (0.04, 0.96):
        b.box(door, (x, 0, 0), (0.018, 0.018, 0.68), "dark")
    for z in (-0.665, 0.665):
        b.box(door, (0.5, 0, z), (0.48, 0.018, 0.018), "dark")
    b.rod(door, (0.88, -0.047, -0.11), (0.88, -0.047, 0.11), 0.013, "metal")
    for z in (-0.10, 0.10):
        b.rod(door, (0.88, -0.015, z), (0.88, -0.047, z), 0.01, "metal")
    _rack(b, base, -0.99, 0.12, 1.35)
    # Side display is carried by its rack top, not suspended in open space.
    b.box(base, (-0.99, 0.12, 1.372), (0.16, 0.12, 0.022), "dark")
    b.box(base, (-0.99, 0.16, 1.52), (0.018, 0.023, 0.13), "metal")
    b.screen(base, (-0.99, 0.16, 1.68), 0.35, 0.25)
    b.metadata["capabilities"] = [
        "non_contact_crosshead_position",
        "guard_access",
        "supported_inert_rock_core",
    ]
    b.metadata["limitations"].append(
        "Crosshead stops above an inert rigid core. Guard and travel are authored mechanical proxies; no load-pressure rating, triaxial cell, rock fracture or calibrated force is simulated."
    )


def _rock_features(world, definition):
    for x in (1.55, 2.65):
        box(
            world,
            f"hku_board_foot_{x}",
            (0.30, 0.12, 0.024),
            (x, 0.8, 0.024),
            (0.15, 0.18, 0.20, 1),
        )
        box(
            world,
            f"hku_board_post_{x}",
            (0.018, 0.018, 0.82),
            (x, 0.8, 0.84),
            (0.5, 0.55, 0.58, 1),
        )
    box(
        world,
        "hku_unbranded_display_board",
        (0.59, 0.025, 0.65),
        (2.1, 0.8, 1.2),
        (0.83, 0.85, 0.85, 1),
    )
    for i in range(3):
        box(
            world,
            f"hku_board_abstract_panel_{i}",
            (0.14, 0.001, 0.32),
            (1.7 + i * 0.39, 0.773, 1.18),
            (0.43, 0.51, 0.55, 1),
            collision=False,
        )
    for z in (2.40, 2.58):
        box(
            world,
            f"hku_rear_conduit_{z}",
            (2.55, 0.027, 0.015),
            (0, 2.17, z),
            (0.79, 0.81, 0.80, 1),
            collision=False,
        )


BUILDERS["guarded_rock_inspection_frame"] = _rock_frame
SOURCES["guarded_rock_inspection_frame"] = dict(
    reference="HKU Civil Engineering installed MTS 815 rock laboratory photo",
    url=HKU,
    dimensions_m=[1.78, 1.86, 2.37],
    dimension_basis="Estimated loading-frame exterior, guard and control rack from photo. Inert core, travel and guard hinge are authored proxies; hidden triaxial cell and loading capability are not reconstructed.",
)
SAMPLE_INTERFACES["guarded_rock_inspection_frame"] = (
    "rock_core",
    (0.070, 0.070, 0.132),
    "clamped",
    "Inert rigid cylindrical rock surrogate on lower platen",
)
FEATURES["hku_rock_mechanics"] = _rock_features


def _impedance_tube(b, base, params):
    # Only the actual installed upper-left photograph is used, not the vendor diagram.
    for x, height in ((-0.72, 0.205), (-0.20, 0.205), (0.58, 0.205)):
        b.box(base, (x, 0, 0.023), (0.075, 0.105, 0.023), "dark", collision=True)
        b.box(base, (x, 0, height / 2), (0.027, 0.028, height / 2), "dark")
    for x, length, radius in (
        (-0.70, 0.30, 0.102),
        (-0.37, 0.34, 0.071),
        (0.34, 0.49, 0.037),
        (0.67, 0.13, 0.032),
    ):
        b.cylinder(
            base,
            (x, 0, 0.205),
            radius,
            length / 2,
            "dark",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
    for x, radius in (
        (-0.86, 0.12),
        (-0.54, 0.078),
        (-0.18, 0.095),
        (0.08, 0.055),
        (0.59, 0.045),
    ):
        b.cylinder(
            base, (x, 0, 0.205), radius, 0.012, "dark", euler=(0, math.pi / 2, 0)
        )
        for angle in range(8):
            a = angle * math.tau / 8
            b.cylinder(
                base,
                (
                    x - 0.014,
                    radius * 0.85 * math.cos(a),
                    0.205 + radius * 0.85 * math.sin(a),
                ),
                0.004,
                0.003,
                "metal",
                euler=(0, math.pi / 2, 0),
            )
    for index, x in enumerate((-0.49, -0.39, -0.29, 0.18, 0.28)):
        if x > 0:
            b.cylinder(base, (x, 0, 0.256), 0.012, 0.020, "metal")
        b.cylinder(base, (x, 0, 0.318), 0.009, 0.047, "metal")
        b.cylinder(base, (x, 0, 0.365), 0.006, 0.01, "dark")
        b.rod(base, (x, 0, 0.38), (x + 0.05, 0.18, 0.28), 0.0025, "rubber")
    # Open sample collar is a mechanical inspection vignette, not a sealed waveguide.
    # Ideal linear guide: contact carries normal load, while the slide joint
    # constrains tangential motion instead of imposing dry sliding friction.
    b.box(
        base,
        (-0.05, 0, 0.047),
        (0.105, 0.105, 0.026),
        "metal",
        collision=True,
        condim=1,
    )
    sample = b.moving(
        base,
        "coupon_axial",
        (-0.05, 0, 0.085),
        (1, 0, 0),
        (-0.026, 0.026),
        mass=0.045,
        kp=180,
    )
    b.box(sample, (0, 0, 0), (0.045, 0.072, 0.012), "dark", collision=True, condim=1)
    b.box(sample, (0, 0, 0.060), (0.012, 0.018, 0.048), "metal")
    b.cylinder(
        sample,
        (0, 0, 0.120),
        0.052,
        0.004,
        "sample",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    b.site(sample, "acoustic_coupon", (0, 0, 0.12), 0.0005)
    b.box(base, (0.36, 0.18, 0.023), (0.09, 0.10, 0.023), "metal")
    b.rod(base, (0.36, 0.18, 0.04), (0.36, 0.18, 0.48), 0.014, "bright")
    b.box(base, (0.36, 0.10, 0.46), (0.032, 0.10, 0.022), "dark")
    probe = b.moving(
        base,
        "probe_clearance",
        (0.36, 0, 0.425),
        (0, 0, 1),
        (-0.035, 0.035),
        mass=0.055,
        kp=190,
    )
    b.cylinder(probe, (0, 0, -0.015), 0.018, 0.04, "metal", collision=True)
    b.rod(probe, (0, 0, -0.055), (0, 0, -0.105), 0.003, "bright")
    b.box(probe, (0, 0.09, 0), (0.03, 0.09, 0.018), "dark")
    b.metadata["capabilities"] = [
        "inert_coupon_axial_position",
        "microphone_clearance_proxy",
        "visible_inert_coupon",
    ]
    b.metadata["limitations"].append(
        "Sample collar is shown open for mechanical inspection; no sealed tube, sound propagation, absorption coefficient, microphone calibration or operating procedure."
    )


def _snu_features(world, definition):
    # Plain wall and cable trunk are estimated local bay context, not a surveyed room.
    box(
        world,
        "snu_wall_trunk",
        (1.25, 0.018, 0.04),
        (0, 1.68, 1.28),
        (0.82, 0.81, 0.77, 1),
        collision=False,
    )
    for index in range(4):
        box(
            world,
            f"snu_outlet_{index}",
            (0.035, 0.005, 0.025),
            (-0.8 + index * 0.4, 1.655, 1.28),
            (0.45, 0.48, 0.48, 1),
            collision=False,
        )


BUILDERS["impedance_tube_inspection_workspace"] = _impedance_tube
SOURCES["impedance_tube_inspection_workspace"] = dict(
    reference="SNU Acoustics and Vibration installed impedance-tube photograph, upper-left panel only",
    url=SNU,
    dimensions_m=[1.75, 0.43, 0.50],
    dimension_basis="Estimated black tube, microphone bosses and support feet; collar access and motion ranges are original mechanical inspection proxies.",
)
SAMPLE_INTERFACES["impedance_tube_inspection_workspace"] = (
    "acoustic_coupon",
    (0.008, 0.104, 0.104),
    "clamped",
    "Inert circular coupon visible in an open inspection collar",
)
FEATURES["snu_acoustic_impedance"] = _snu_features


def _materials_glovebox(b, base, params):
    b.box(base, (0, 0, 0.045), (0.58, 0.32, 0.045), "dark", collision=True)
    b.box(base, (0, 0, 0.43), (0.60, 0.34, 0.34), "shell", collision=True)
    for x in (-0.30, 0.30):
        b.box(base, (x, -0.348, 0.43), (0.285, 0.010, 0.30), "shell")
        b.rod(base, (x, -0.37, 0.37), (x, -0.37, 0.51), 0.008, "metal")
    b.box(base, (0, 0, 0.81), (0.62, 0.36, 0.04), "metal", collision=True)
    b.box(base, (0, 0.33, 1.29), (0.60, 0.026, 0.44), "shell", collision=True)
    for x in (-0.585, 0.585):
        b.box(base, (x, 0, 1.29), (0.026, 0.34, 0.44), "shell", collision=True)
    b.box(base, (0, 0, 1.77), (0.64, 0.38, 0.06), "shell", collision=True)
    b.box(base, (0, -0.354, 1.31), (0.554, 0.007, 0.39), "glass")
    for z in (0.91, 1.70):
        b.box(base, (0, -0.355, z), (0.595, 0.022, 0.022), "metal")
    for x in (-0.26, 0.26):
        b.cylinder(
            base, (x, -0.368, 1.29), 0.105, 0.012, "bright", euler=(math.pi / 2, 0, 0)
        )
        b.cylinder(
            base, (x, -0.382, 1.29), 0.090, 0.015, "rubber", euler=(math.pi / 2, 0, 0)
        )
        b.rod(base, (x, -0.40, 1.29), (x - 0.01, -0.44, 1.07), 0.047, "rubber")
        b.rod(base, (x - 0.01, -0.44, 1.07), (x + 0.02, -0.43, 0.98), 0.036, "rubber")
        b.geom(
            base, "ellipsoid", (0.039, 0.018, 0.055), (x + 0.02, -0.43, 0.945), "rubber"
        )
        for index in range(4):
            dx = (index - 1.5) * 0.018
            b.rod(
                base,
                (x + 0.02 + dx, -0.43, 0.92),
                (x + 0.024 + dx * 1.10, -0.43, 0.867 + abs(index - 1.5) * 0.008),
                0.008,
                "rubber",
            )
    b.cylinder(
        base,
        (0.85, 0.03, 1.30),
        0.205,
        0.265,
        "shell",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    for x in (0.598, 1.10):
        b.cylinder(
            base, (x, 0.03, 1.30), 0.224, 0.017, "metal", euler=(0, math.pi / 2, 0)
        )
    for angle in range(8):
        a = angle * math.tau / 8
        b.cylinder(
            base,
            (1.124, 0.03 + 0.19 * math.cos(a), 1.30 + 0.19 * math.sin(a)),
            0.012,
            0.020,
            "bright",
            euler=(0, math.pi / 2, 0),
        )
    b.cylinder(
        base, (1.145, 0.03, 1.30), 0.05, 0.016, "dark", euler=(0, math.pi / 2, 0)
    )
    for angle in range(3):
        a = angle * math.tau / 3
        b.rod(
            base,
            (1.16, 0.03, 1.30),
            (1.16, 0.03 + 0.14 * math.cos(a), 1.30 + 0.14 * math.sin(a)),
            0.01,
            "metal",
        )
    b.box(base, (0.67, -0.10, 0.60), (0.07, 0.12, 0.16), "shell")
    b.cylinder(
        base, (0.67, -0.223, 0.62), 0.03, 0.009, "guard_red", euler=(math.pi / 2, 0, 0)
    )
    for y in (-0.12, 0.16):
        b.box(base, (0, y, 0.882), (0.31, 0.025, 0.029), "metal")
    tray = b.moving(
        base,
        "inspection_tray_x",
        (0, 0.02, 0.925),
        (1, 0, 0),
        (-0.12, 0.12),
        mass=0.14,
        kp=270,
    )
    b.box(tray, (0, 0, 0), (0.17, 0.13, 0.014), "metal", collision=True)
    turn = b.moving(
        tray,
        "coupon_theta",
        (0, 0, 0.033),
        (0, 0, 1),
        (-0.90, 0.90),
        kind="hinge",
        mass=0.06,
        kp=180,
    )
    b.cylinder(turn, (0, 0, 0), 0.080, 0.018, "dark", collision=True)
    b.box(turn, (0, 0, 0.022), (0.027, 0.018, 0.004), "sample")
    b.box(turn, (0.04, 0, 0.0185), (0.019, 0.003, 0.0005), "warning")
    b.site(turn, "materials_coupon", (0, 0, 0.026), 0.0005)
    # Adjacent cabinet is only the externally visible context, not a functional second instrument.
    b.box(base, (-0.96, 0.04, 0.91), (0.22, 0.36, 0.91), "dark", collision=True)
    b.box(base, (-0.96, -0.328, 0.98), (0.20, 0.007, 0.78), "blue")
    b.metadata["capabilities"] = [
        "inert_inspection_tray_translation",
        "inert_coupon_rotation",
        "visible_closed_compartment",
    ]
    b.metadata["limitations"].append(
        "Sealed-atmosphere physics, glove deformation and chemical processes are absent. Tray and rotary holder are authored mechanical demonstrations, not verified manufacturer automation."
    )


def _fudan_features(world, definition):
    for x in (1.50, 1.75):
        box(
            world,
            f"fudan_wall_pipe_{x}",
            (0.013, 0.014, 0.58),
            (x, 1.94, 1.39),
            (0.48, 0.52, 0.53, 1),
            collision=False,
        )
    box(
        world,
        "fudan_wall_meter_panel",
        (0.24, 0.028, 0.12),
        (1.63, 1.91, 1.55),
        (0.76, 0.77, 0.74, 1),
        collision=False,
    )


BUILDERS["materials_glovebox_inspection_workspace"] = _materials_glovebox
SOURCES["materials_glovebox_inspection_workspace"] = dict(
    reference="Fudan Superassembly Framework Laboratory installed two-glove cabinet photograph",
    url=FUDAN,
    dimensions_m=[2.40, 0.90, 1.86],
    dimension_basis="Estimated glovebox cabinet, front gloves, transfer cylinder and adjacent cabinet exterior. Internal mechanical inspection tray is an authored proxy. Separate bench photograph does not determine room adjacency.",
)
SAMPLE_INTERFACES["materials_glovebox_inspection_workspace"] = (
    "materials_coupon",
    (0.054, 0.036, 0.008),
    "clamped",
    "Inert rectangular materials coupon on the internal inspection holder",
)
FEATURES["fudan_superassembly_materials"] = _fudan_features
