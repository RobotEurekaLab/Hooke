"""Reference-informed optical and materials workstations; mechanical proxies only."""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box


KYOTO = "https://dpe.energy.kyoto-u.ac.jp/en/research/facilities/"
CITYU = "https://personal.cityu.edu.hk/arogatch/lab.htm"
POLYU = "https://www.polyu.edu.hk/me/research/laboratories/materials-and-mechanics-technology-laboratory/"
CUHK = "https://www.mpil.mae.cuhk.edu.hk/facilities"


def _board(b, parent, width, depth):
    b.box(parent, (0, 0, 0.02), (width / 2, depth / 2, 0.02), "metal", collision=True)
    for x in range(-int(width / 0.12), int(width / 0.12) + 1):
        for y in range(-int(depth / 0.12), int(depth / 0.12) + 1):
            b.cylinder(parent, (x * 0.06, y * 0.06, 0.0405), 0.002, 0.0005, "dark")


def _optic(b, parent, x, y, height, radius=0.035):
    """A plate-mounted post supports a transverse lens, including its lower foot."""
    b.box(parent, (x, y, 0.05), (0.045, 0.035, 0.01), "dark")
    b.rod(parent, (x, y, 0.06), (x, y, height), 0.009, "bright")
    b.cylinder(parent, (x, y, height), radius, 0.012, "dark", euler=(math.pi / 2, 0, 0))
    b.cylinder(
        parent,
        (x, y - 0.013, height),
        radius * 0.76,
        0.001,
        "lens",
        euler=(math.pi / 2, 0, 0),
    )
    for dx in (-0.025, 0.025):
        b.cylinder(
            parent,
            (x + dx, y - 0.017, height + 0.025),
            0.005,
            0.007,
            "metal",
            euler=(math.pi / 2, 0, 0),
        )


def _display(b, parent, x, y, z, width=0.42):
    b.box(parent, (x, y, z + 0.012), (0.12, 0.10, 0.012), "dark")
    b.box(parent, (x, y + 0.035, z + 0.16), (0.018, 0.023, 0.145), "metal")
    b.screen(parent, (x, y + 0.035, z + 0.32), width, width * 0.63)


def _kyoto_terahertz(b, base, params):
    _board(b, base, 1.48, 0.92)
    for x in (-0.56, 0.56):
        for y in (-0.22, 0.28):
            b.box(base, (x, y, 0.06), (0.075, 0.07, 0.02), "dark", collision=True)
            b.box(base, (x, y, 0.35), (0.032, 0.032, 0.29), "dark", collision=True)
    b.box(base, (0, 0.035, 0.66), (0.64, 0.32, 0.03), "dark", collision=True)
    for x in (-0.32, 0.31):
        b.cylinder(base, (x, 0.22, 0.90), 0.045, 0.21, "dark")
        b.cylinder(base, (x, 0.22, 1.10), 0.031, 0.035, "metal")
        b.box(base, (x, 0.22, 1.155), (0.042, 0.035, 0.030), "dark")
        b.box(
            base,
            (x, 0.184, 1.155),
            (0.034, 0.001, 0.021),
            "guard_red" if x > 0 else "shell",
        )
        for z in (0.82, 0.93, 1.02):
            b.ring(base, (x, 0.22, z), 0.046, 0.004, "metal", segments=16)
        b.rod(base, (x, 0.255, 1.18), (x + 0.11, 0.41, 0.70), 0.003, "rubber")
    # Front specimen carriage is authored geometry; the source does not expose its mechanics.
    for x in (-0.23, 0.23):
        b.box(base, (x, -0.13, 0.714), (0.026, 0.11, 0.024), "metal")
    xstage = b.moving(
        base,
        "target_x",
        (0, -0.13, 0.751),
        (1, 0, 0),
        (-0.075, 0.075),
        mass=0.1,
        kp=240,
    )
    b.box(xstage, (0, 0, 0), (0.14, 0.10, 0.013), "metal", collision=True)
    ystage = b.moving(
        xstage, "target_y", (0, 0, 0.027), (0, 1, 0), (-0.045, 0.045), mass=0.07, kp=220
    )
    b.box(ystage, (0, 0, 0), (0.075, 0.07, 0.012), "dark", collision=True)
    b.box(ystage, (0, 0, 0.014), (0.01, 0.01, 0.002), "sample")
    b.site(ystage, "thz_target", (0, 0, 0.016), 0.0005)
    for x in (-0.55, 0.56):
        b.box(base, (x, 0.12, 0.82), (0.035, 0.08, 0.13), "dark")
        b.cylinder(
            base, (x, 0.02, 0.87), 0.105, 0.015, "dark", euler=(math.pi / 2, 0, 0)
        )
        b.cylinder(
            base, (x, 0.003, 0.87), 0.072, 0.004, "metal", euler=(math.pi / 2, 0, 0)
        )
        for angle in range(12):
            a = angle * math.tau / 12
            b.cylinder(
                base,
                (x + 0.092 * math.cos(a), -0.001, 0.87 + 0.092 * math.sin(a)),
                0.003,
                0.002,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
    for x, y, h in (
        (-0.57, -0.37, 0.25),
        (-0.3, -0.37, 0.25),
        (0.03, -0.35, 0.30),
        (0.30, -0.30, 0.25),
        (0.63, -0.30, 0.29),
    ):
        _optic(b, base, x, y, h)
    for x in (-0.68, 0.69):
        b.cylinder(base, (x, 0.29, 0.36), 0.018, 0.32, "bright")
        b.box(base, (x, 0.29, 0.052), (0.045, 0.07, 0.012), "dark")
    b.metadata["capabilities"] = [
        "target_xy",
        "visible_inert_coupon",
        "two_level_optical_support",
    ]
    b.metadata["limitations"].append(
        "No terahertz radiation, optical ray tracing, spectroscopy or OEM stage precision. Coarse XY carrier is an original mechanical proxy."
    )


def _kyoto_features(world, definition):
    # Visible window blinds provide local context; room dimensions are estimates.
    for index in range(23):
        box(
            world,
            f"kawayama_blind_{index}",
            (1.30, 0.013, 0.013),
            (-0.30, 1.61, 1.22 + index * 0.045),
            (0.84, 0.84, 0.80, 1),
            collision=False,
        )


BUILDERS = {"terahertz_alignment_workspace": _kyoto_terahertz}
SOURCES = {
    "terahertz_alignment_workspace": {
        "reference": "Kyoto Kawayama installed laser terahertz microscope photograph",
        "url": KYOTO,
        "dimensions_m": [1.48, 0.92, 1.19],
        "dimension_basis": "Estimated exterior from undimensioned official photo. Two-level bridge, camera columns and round mounts are observed; XY travel and hidden structure are original proxies.",
    },
}
SAMPLE_INTERFACES = {
    "terahertz_alignment_workspace": (
        "thz_target",
        (0.02, 0.02, 0.004),
        "clamped",
        "Visible inert alignment coupon on the XY carrier",
    )
}
FEATURES = {"kyoto_kawayama_terahertz": _kyoto_features}


def _cityu_spectrometer(b, base, params):
    # An authored open service bay exposes the inert holder. Source shows closed covers.
    for x in (-0.68, 0.68):
        module = ET.SubElement(base, "body", pos=f"{x} 0 0")
        b.feet(module, 0.58, 0.62)
        b.housing(
            module,
            [(0.025, 0.29, 0.31, 0), (0.43, 0.29, 0.31, 0), (0.46, 0.275, 0.295, 0)],
            material="shell",
            radius=0.055,
        )
        b.box(module, (0, 0, 0.23), (0.275, 0.29, 0.20), "shell", collision=True)
        b.box(module, (0, -0.311, 0.25), (0.25, 0.001, 0.001), "metal")
        for dx in (-0.25, 0.25):
            b.cylinder(
                module,
                (dx, -0.312, 0.40),
                0.004,
                0.001,
                "dark",
                euler=(math.pi / 2, 0, 0),
            )
    b.box(base, (0, 0, 0.05), (0.37, 0.31, 0.025), "shell", collision=True)
    for x in (-0.34, 0.34):
        b.box(base, (x, 0, 0.25), (0.03, 0.31, 0.20), "shell", collision=True)
    b.box(base, (0, 0.285, 0.25), (0.32, 0.025, 0.20), "shell", collision=True)
    b.box(base, (0, -0.29, 0.10), (0.32, 0.02, 0.05), "shell")
    for index in range(28):
        a = index * math.tau / 28
        b.box(
            base,
            (0.255 * math.cos(a), 0.255 * math.sin(a), 0.445),
            (0.03, 0.062, 0.018),
            "shell",
            euler=(0, 0, a + math.pi / 2),
        )
    b.cylinder(base, (0, -0.04, 0.18), 0.105, 0.10, "metal", collision=True)
    theta = b.moving(
        base,
        "coupon_theta",
        (0, -0.04, 0.292),
        (0, 0, 1),
        (-0.85, 0.85),
        kind="hinge",
        mass=0.07,
        kp=180,
    )
    b.cylinder(theta, (0, 0, 0), 0.08, 0.012, "dark", collision=True)
    b.box(theta, (0, 0, 0.016), (0.017, 0.010, 0.004), "sample")
    b.box(theta, (0.035, 0, 0.0125), (0.018, 0.003, 0.0005), "warning")
    b.site(theta, "pl_coupon", (0, 0, 0.02), 0.0005)
    # Both fixed guide and sliding stem are visible, continuously supported by rear wall.
    b.box(base, (0, 0.33, 0.43), (0.045, 0.035, 0.38), "metal")
    lid = b.moving(
        base, "service_lid_lift", (0, 0, 0.67), (0, 0, 1), (0, 0.12), mass=0.20, kp=300
    )
    b.box(lid, (0, 0.33, -0.04), (0.027, 0.025, 0.20), "bright")
    b.box(lid, (0, 0.165, 0), (0.027, 0.165, 0.015), "metal")
    b.cylinder(lid, (0, 0, 0), 0.235, 0.018, "dark", collision=True)
    b.cylinder(lid, (0, 0, 0.021), 0.223, 0.008, "shell")
    for x in (-0.05, 0.05):
        b.box(lid, (x, 0, 0.055), (0.01, 0.013, 0.026), "dark")
    b.box(lid, (0, 0, 0.083), (0.06, 0.013, 0.01), "dark")
    detector = ET.SubElement(base, "body", pos="-0.74 -0.35 0.025")
    b.housing(
        detector,
        [(0, 0.17, 0.20, 0), (0.32, 0.17, 0.20, 0), (0.40, 0.12, 0.16, 0)],
        material="shell",
        radius=0.015,
    )
    for x in (-0.075, 0.075):
        b.cylinder(
            detector, (x, -0.20, 0.16), 0.055, 0.012, "metal", euler=(math.pi / 2, 0, 0)
        )
        b.rod(
            base, (-0.74 + x, -0.56, 0.185), (-0.94 + x, -0.62, 0.03), 0.018, "rubber"
        )
    _display(b, base, 0.94, 0.36, 0.024, 0.43)
    b.box(base, (0.93, -0.38, 0.035), (0.16, 0.05, 0.011), "dark")
    b.box(base, (-0.1, 0.37, 0.48), (0.33, 0.085, 0.065), "shell")
    b.screen(base, (-0.1, 0.283, 0.48), 0.36, 0.085)
    b.metadata["capabilities"] = [
        "inert_coupon_rotation",
        "supported_service_lid",
        "visible_inert_coupon",
    ]
    b.metadata["limitations"].append(
        "Open service aperture, lifting lid and internal coupon stage are authored demonstration geometry, not documented FLS1000 mechanisms. No spectra, cooling or optical propagation."
    )


def _cityu_features(world, definition):
    for index, x in enumerate((-0.9, 0.45)):
        box(
            world,
            f"cityu_unbranded_wall_panel_{index}",
            (0.30, 0.012, 0.40),
            (x, 1.90, 1.95),
            (0.22, 0.26, 0.30, 1),
            collision=False,
        )
        for line in range(5):
            box(
                world,
                f"cityu_panel_line_{index}_{line}",
                (0.22, 0.002, 0.006),
                (x, 1.884, 2.15 - line * 0.075),
                (0.61, 0.68, 0.68, 1),
                collision=False,
            )
    box(
        world,
        "cityu_service_trunk",
        (0.04, 0.025, 0.68),
        (-1.65, 1.88, 1.80),
        (0.83, 0.84, 0.80, 1),
        collision=False,
    )


def _polyu_tribometer(b, base, params):
    b.feet(base, 0.51, 0.57)
    b.housing(
        base,
        [(0.023, 0.25, 0.28, 0), (0.14, 0.27, 0.30, 0), (0.18, 0.25, 0.28, 0)],
        material="blue",
        radius=0.075,
    )
    b.box(base, (0, 0, 0.09), (0.23, 0.25, 0.06), "blue", collision=True)
    b.box(base, (0, 0.225, 0.515), (0.24, 0.055, 0.335), "shell", collision=True)
    for x in (-0.21, 0.21):
        b.box(base, (x, 0.02, 0.51), (0.022, 0.22, 0.33), "metal", collision=True)
    b.housing(
        base,
        [(0.83, 0.25, 0.28, 0), (0.94, 0.25, 0.28, 0), (0.96, 0.23, 0.26, 0)],
        material="dark",
        radius=0.05,
    )
    b.box(base, (0, 0, 0.88), (0.23, 0.25, 0.04), "dark", collision=True)
    # Segmented clear guard follows the visible rounded front, leaving instruments readable.
    for index in range(11):
        a = math.pi + (index + 0.5) * math.pi / 11
        b.box(
            base,
            (0.251 * math.cos(a), 0.02 + 0.27 * math.sin(a), 0.50),
            (0.004, 0.037, 0.32),
            "glass",
            euler=(0, 0, a),
        )
    for x in (-0.09, 0.09):
        b.box(base, (x, -0.035, 0.193), (0.014, 0.13, 0.018), "bright")
    slider = b.moving(
        base,
        "coupon_x",
        (0, -0.025, 0.229),
        (1, 0, 0),
        (-0.065, 0.065),
        mass=0.11,
        kp=240,
    )
    b.box(slider, (0, 0, 0), (0.09, 0.11, 0.018), "metal", collision=True)
    b.cylinder(slider, (0, 0, 0.022), 0.039, 0.004, "dark")
    b.box(slider, (0, 0, 0.028), (0.020, 0.020, 0.002), "sample")
    b.site(slider, "tribology_coupon", (0, 0, 0.030), 0.0005)
    b.box(base, (0, 0.11, 0.53), (0.055, 0.05, 0.27), "dark")
    head = b.moving(
        base,
        "probe_height",
        (0, -0.025, 0.62),
        (0, 0, 1),
        (-0.10, 0.08),
        mass=0.16,
        kp=280,
    )
    b.box(head, (0, 0.07, 0.03), (0.075, 0.05, 0.08), "metal", collision=True)
    b.box(head, (0, 0.018, -0.02), (0.055, 0.07, 0.03), "dark")
    b.cylinder(head, (0, 0, -0.09), 0.022, 0.055, "metal")
    b.rod(head, (0, 0, -0.145), (0, 0, -0.18), 0.004, "bright")
    b.screen(base, (-0.08, -0.279, 0.892), 0.12, 0.042)
    for x, mat in ((-0.12, "led"), (-0.03, "warning"), (0.08, "guard_red")):
        b.cylinder(
            base, (x, -0.286, 0.125), 0.009, 0.003, mat, euler=(math.pi / 2, 0, 0)
        )
    b.metadata["capabilities"] = [
        "inert_coupon_translation",
        "probe_clearance",
        "visible_inert_coupon",
    ]
    b.metadata["limitations"].append(
        "Probe stops above the coupon. No contact-force calibration, friction, wear, material deformation or branded instrument accuracy."
    )


def _polyu_features(world, definition):
    for index in range(4):
        box(
            world,
            f"polyu_wall_service_{index}",
            (0.035, 0.014, 0.04),
            (-1.0 + index * 0.23, 1.68, 1.13),
            (0.70, 0.71, 0.68, 1),
            collision=False,
        )


BUILDERS.update(
    {
        "photoluminescence_service_workspace": _cityu_spectrometer,
        "umt_mechanical_workspace": _polyu_tribometer,
    }
)
SOURCES.update(
    {
        "photoluminescence_service_workspace": dict(
            reference="CityU Rogach Optics Lab 2 installed FLS1000 photo",
            url=CITYU,
            dimensions_m=[2.28, 1.04, 1.02],
            dimension_basis="Exterior and support estimated from photo; open service bay and mechanical controls explicitly authored proxies, not OEM internal reproduction.",
        ),
        "umt_mechanical_workspace": dict(
            reference="PolyU Materials and Mechanics Technology installed UMT photograph",
            url=POLYU,
            dimensions_m=[0.54, 0.60, 0.96],
            dimension_basis="Estimated rounded enclosure, rear frame and stage; no published apparatus dimensions or calibrated travel verified.",
        ),
    }
)
SAMPLE_INTERFACES.update(
    {
        "photoluminescence_service_workspace": (
            "pl_coupon",
            (0.034, 0.020, 0.008),
            "clamped",
            "Visible inert optical coupon in authored open service bay",
        ),
        "umt_mechanical_workspace": (
            "tribology_coupon",
            (0.040, 0.040, 0.004),
            "clamped",
            "Visible inert mechanical coupon on carriage",
        ),
    }
)
FEATURES.update(
    {
        "cityu_rogach_photoluminescence": _cityu_features,
        "polyu_materials_tribology": _polyu_features,
    }
)


def _cuhk_ultrafast(b, base, params):
    _board(b, base, 2.16, 1.70)
    # Elongated laser covers occupy the right, with a dense optical row at left.
    for x, y, length in ((0.62, -0.33, 0.70), (0.64, 0.43, 0.63), (-0.78, 0.57, 0.42)):
        b.box(base, (x, y, 0.17), (0.30, length / 2, 0.12), "cream", collision=True)
        b.housing(
            ET.SubElement(base, "body", pos=f"{x} {y} 0"),
            [
                (0.05, 0.31, length / 2, 0),
                (0.27, 0.31, length / 2, 0),
                (0.30, 0.28, length / 2 - 0.02, 0),
            ],
            material="cream",
            radius=0.025,
        )
        for index in range(5):
            b.box(
                base,
                (x + 0.302, y - length * 0.33 + index * 0.05, 0.18),
                (0.002, 0.018, 0.028),
                "dark",
            )
        b.cylinder(
            base,
            (x - 0.25, y - length / 2 - 0.006, 0.16),
            0.024,
            0.008,
            "metal",
            euler=(math.pi / 2, 0, 0),
        )
    for x, y, h in (
        (-0.81, -0.60, 0.23),
        (-0.53, -0.60, 0.23),
        (-0.28, -0.57, 0.23),
        (-0.92, -0.22, 0.28),
        (-0.43, 0.20, 0.25),
        (-0.16, 0.18, 0.25),
        (0.11, 0.17, 0.25),
        (-0.34, 0.49, 0.28),
        (-0.08, 0.60, 0.28),
        (0.14, 0.62, 0.28),
    ):
        _optic(b, base, x, y, h, 0.03)
    for x, y, width in ((-0.55, -0.37, 0.66), (-0.18, 0.37, 0.65), (0.29, 0.1, 0.03)):
        b.box(base, (x, y, 0.31), (width / 2, 0.012, 0.26), "dark")
        for side in (-1, 1):
            b.box(
                base,
                (x + side * max(width / 2 - 0.04, 0), y, 0.054),
                (0.04, 0.08, 0.014),
                "dark",
            )
    b.box(base, (-0.05, -0.60, 0.082), (0.22, 0.13, 0.04), "metal", collision=True)
    translation = b.moving(
        base,
        "coupon_x",
        (-0.05, -0.60, 0.139),
        (1, 0, 0),
        (-0.08, 0.08),
        mass=0.11,
        kp=250,
    )
    b.box(translation, (0, 0, 0), (0.105, 0.095, 0.017), "dark", collision=True)
    b.box(translation, (0, 0, 0.020), (0.014, 0.014, 0.003), "sample")
    b.site(translation, "alignment_coupon", (0, 0, 0.023), 0.0005)
    # An independently actuated optic has its bearing directly on a table-mounted post.
    b.box(base, (-0.67, 0.11, 0.052), (0.075, 0.07, 0.012), "dark")
    b.cylinder(base, (-0.67, 0.11, 0.20), 0.021, 0.136, "bright")
    turn = b.moving(
        base,
        "mirror_theta",
        (-0.67, 0.11, 0.35),
        (0, 0, 1),
        (-0.60, 0.60),
        kind="hinge",
        mass=0.045,
        kp=170,
    )
    b.cylinder(turn, (0, 0, 0), 0.055, 0.014, "metal", collision=True)
    b.box(turn, (0, 0, 0.052), (0.057, 0.014, 0.040), "dark")
    b.box(turn, (0, -0.015, 0.052), (0.047, 0.001, 0.033), "lens")
    for x in (-0.96, -0.45):
        b.rod(base, (x, 0.71, 0.06), (x, 0.71, 0.63), 0.018, "bright")
        b.box(base, (x, 0.71, 0.052), (0.06, 0.06, 0.012), "dark")
        b.screen(base, (x, 0.71, 0.75), 0.40, 0.27)
    for index in range(7):
        b.rod(
            base,
            (-1.01, -0.7 + index * 0.19, 0.06),
            (-0.78, -0.77 + index * 0.19, 0.06),
            0.004,
            "rubber",
        )
    b.metadata["capabilities"] = [
        "inert_coupon_translation",
        "optic_rotation",
        "visible_inert_alignment_target",
    ]
    b.metadata["limitations"].append(
        "Laser covers and optical placements are reference-informed approximations. No emitted beams, optical alignment metric, ultrafast physics or brand precision is simulated."
    )


def _cuhk_features(world, definition):
    metal = (0.62, 0.65, 0.66, 1)
    for x in (-1.78, 0.58):
        for y in (-2.07, 2.07):
            box(
                world,
                f"mpil_frame_post_{x}_{y}",
                (0.025, 0.025, 0.835),
                (x, y, 1.715),
                metal,
            )
        box(world, f"mpil_frame_long_{x}", (0.032, 2.10, 0.025), (x, 0, 2.55), metal)
    for y in (-2.07, 0, 2.07):
        box(
            world,
            f"mpil_frame_cross_{y}",
            (1.21, 0.032, 0.025),
            (-0.60, y, 2.55),
            metal,
        )
    box(
        world,
        "mpil_power_rail",
        (1.08, 0.035, 0.045),
        (-0.60, 1.85, 2.47),
        (0.15, 0.17, 0.18, 1),
        collision=False,
    )
    for index in range(9):
        box(
            world,
            f"mpil_power_socket_{index}",
            (0.032, 0.003, 0.024),
            (-1.47 + index * 0.21, 1.812, 2.47),
            (0.8, 0.81, 0.77, 1),
            collision=False,
        )
    # Background items share the same measured-height bench support; no unsupported props.
    for index, x in enumerate((-0.95, -0.20)):
        box(
            world,
            f"mpil_rear_laser_{index}",
            (0.26, 0.34, 0.12),
            (x, 1.40, 1.00),
            (0.87, 0.85, 0.76, 1),
        )
        box(
            world,
            f"mpil_rear_laser_foot_{index}",
            (0.23, 0.30, 0.025),
            (x, 1.40, 0.895),
            (0.07, 0.08, 0.09, 1),
        )
    for index in range(5):
        box(
            world,
            f"mpil_service_conduit_{index}",
            (2.8, 0.025, 0.025),
            (0, 2.7 - index * 0.11, 2.78),
            (0.80, 0.81, 0.79, 1),
            collision=False,
        )


BUILDERS["mpil_ultrafast_alignment_workspace"] = _cuhk_ultrafast
SOURCES["mpil_ultrafast_alignment_workspace"] = dict(
    reference="CUHK MPIL Room 303 actual optical-bay photograph",
    url=CUHK,
    dimensions_m=[2.16, 1.70, 0.89],
    dimension_basis="Undimensioned installed photo: elongated covers, optical partitions and table frames observed; positions and dimensions estimated. Motion stages are original non-emitting alignment proxies.",
)
SAMPLE_INTERFACES["mpil_ultrafast_alignment_workspace"] = (
    "alignment_coupon",
    (0.028, 0.028, 0.006),
    "clamped",
    "Visible inert alignment coupon on coarse translation carrier",
)
FEATURES["cuhk_mpil_ultrafast_optics"] = _cuhk_features
