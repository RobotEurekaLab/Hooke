"""Source-informed AAPF exterior with supported cold imaging mechanics.

The existing three authored axes, specimen scale and camera optics are retained.
The official tour establishes visible arrangement, not dimensions or OEM CAD.
Cable-carrier context is fixed; no flexible chain, electrical or optical model.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount
from real_labs.ntu_sgsr_geometry import Parts as InstrumentParts


class Parts(InstrumentParts):
    def name(self, key):
        return f"{self.b.name}__purdue_{key}"

    def ring(self, parent, key, center, inner, outer, half):
        annular_mount(
            self.b,
            parent,
            key,
            center,
            inner,
            outer,
            half,
            axis="z",
            proxies=True,
            namespace="purdue",
            material="metal",
        )


def build_booth(b, base, params):
    p = Parts(b)
    _frame(p, base)
    _transport(b, p, base)
    _camera(b, p, base)
    _lighting(p, base)
    b.metadata["capabilities"] = [
        "guided retained-carrier translation",
        "supported specimen indexing",
        "contact-capable camera-height positioning",
        "mounted-camera inert-witness inspection",
    ]
    b.metadata["inspection_interface"] = dict(
        witness_sites=[b.metadata["sites"][f"inert_witness_{i}"] for i in range(2)],
        witness_geoms=[p.name(f"witness_{i}") for i in range(2)],
        camera=b.name + "__plant_camera",
        guide_clearance_m=0.0006,
        target_attachment="Rigid retained inert holder; decorative specimen is not manipulated",
        scale_basis="Preserved authored booth, carrier and specimen dimensions; no measured facility scale",
    )
    b.metadata["limitations"].extend(
        [
            "Fixed specimen/canopy remains illustrative. No organism manipulation, growth, irrigation or biological process.",
            "Inert witness inspection uses uncalibrated perspective optics, not hyperspectral data or phenotyping metrics.",
            "Carrier translation is guided ideal-joint motion above fixed rollers; no powered belt or rolling-contact transport claim.",
            "Camera cable-carrier channel and supported local connectors are exterior context; no flexible chain deformation or live circuit.",
            "Open guides, masses, gravity compensation and original drives are authored estimates; no rated bearing or structural capacity.",
        ]
    )


def _frame(p, base):
    for x in (-0.66, 0.66):
        for y in (-0.6, 0.6):
            p.box(base, f"foot_{x}_{y}", (x, y, 0.005), (0.049, 0.049, 0.005), "rubber")
            p.box(base, f"post_{x}_{y}", (x, y, 1.18), (0.045, 0.045, 1.17), "shell")
            p.box(
                base, f"panel_tab_{x}_{y}", (x, y * 0.925, 1.29), (0.015, 0.014, 0.84)
            )
        p.box(base, f"top_beam_{x}", (x, 0, 2.29), (0.05, 0.65, 0.06), "shell")
        p.box(base, f"side_panel_{x}", (x, 0, 1.29), (0.012, 0.55, 0.84), "dark")
    p.box(base, "backdrop", (0, 0.62, 1.48), (0.66, 0.025, 0.76), "dark")
    p.box(base, "roof", (0, 0, 2.305), (0.66, 0.64, 0.045), "shell")
    # Original external control location, now with a real supporting standoff.
    p.box(base, "display_mount", (0.665, -0.628, 1.55), (0.065, 0.025, 0.13))
    p.box(base, "display_case", (0.665, -0.662, 1.55), (0.095, 0.015, 0.07), "dark")
    p.box(
        base, "inactive_display", (0.665, -0.678, 1.55), (0.086, 0.002, 0.060), "screen"
    )


def _transport(b, p, base):
    for x in (-0.4, 0.4):
        p.box(base, f"transport_frame_{x}", (x, 0, 0.58), (0.024, 0.92, 0.07))
        for y in (-0.84, 0.84):
            p.cylinder(
                base, f"transport_foot_{x}_{y}", (x, y, 0.005), 0.033, 0.005, "rubber"
            )
            p.box(base, f"transport_leg_{x}_{y}", (x, y, 0.27), (0.025, 0.025, 0.26))
    for i in range(19):
        y = -0.84 + i * 0.09
        p.cylinder(
            base,
            f"roller_{i}",
            (0, y, 0.6074),
            0.027,
            0.365,
            "bright",
            euler=(0, math.pi / 2, 0),
        )
        for side in (-1, 1):
            p.cylinder(
                base,
                f"roller_journal_{i}_{side}",
                (side * 0.385, y, 0.6074),
                0.012,
                0.025,
                "metal",
                euler=(0, math.pi / 2, 0),
            )
    for x in (-0.45, 0.45):
        p.box(base, f"carrier_rail_{x}", (x, 0, 0.66), (0.006, 0.92, 0.006), "bright")
        for y in (-0.87, 0.87):
            p.box(base, f"rail_pedestal_{x}_{y}", (x, y, 0.648), (0.012, 0.025, 0.006))
            p.box(
                base,
                f"rail_outrigger_{x}_{y}",
                (x * 0.95, y, 0.636),
                (0.042, 0.025, 0.006),
            )
    carrier = b.moving(
        base, "conveyor", (0, -0.54, 0.66), (0, 1, 0), (0, 0.80), mass=3, kp=300
    )
    p.box(carrier, "carrier_plate", (0, 0, 0), (0.255, 0.255, 0.025), "dark")
    for x in (-0.45, 0.45):
        for y in (-0.18, 0.18):
            p.sleeve(
                carrier,
                f"carrier_shoe_{x}_{y}",
                (x, y, 0),
                1,
                0.030,
                (0.0066, 0.0066),
                wall=0.006,
            )
            p.box(
                carrier,
                f"carrier_arm_{x}_{y}",
                (x * 0.78, y, 0.0176),
                (0.099, 0.024, 0.005),
            )
    p.cylinder(carrier, "spindle", (0, 0, 0.050), 0.025, 0.025, "bright")
    p.cylinder(carrier, "thrust_support", (0, 0, 0.0267), 0.055, 0.0017)
    turn = b.moving(
        carrier,
        "plant_turntable",
        (0, 0, 0.047),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=1,
    )
    p.ring(turn, "turntable", (0, 0, 0), 0.0256, 0.2, 0.018)
    # Seat reaches the unchanged decorative pot's bottom z=.040 in this frame.
    p.ring(turn, "pot_seat", (0, 0, 0.029), 0.0256, 0.132, 0.011)
    b.site(turn, "plant_pot", (0, 0, 0.025))
    p.box(turn, "witness_foot", (0.17, 0, 0.022), (0.03, 0.035, 0.004))
    p.box(turn, "witness_post", (0.195, 0, 0.248), (0.005, 0.009, 0.226))
    p.box(turn, "witness_board", (0.205, 0, 0.443), (0.005, 0.04, 0.07), "dark")
    for i, z in enumerate((0.403, 0.483)):
        p.cylinder(
            turn,
            f"witness_{i}",
            (0.211, 0, z),
            0.008,
            0.001,
            "cream",
            euler=(0, math.pi / 2, 0),
        )
        b.site(turn, f"inert_witness_{i}", (0.212, 0, z), 0.001)


def _camera(b, p, base):
    for z in (0.76, 1.98):
        p.box(base, f"camera_rail_anchor_{z}", (0.616, 0.05, z), (0.044, 0.028, 0.018))
    p.box(base, "camera_rail", (0.57, 0.05, 1.37), (0.006, 0.012, 0.61), "bright")
    camera = b.moving(
        base, "camera_height", (0.46, 0.05, 1.13), (0, 0, 1), (-0.18, 0.52), mass=0.6
    )
    p.box(camera, "camera_main", (0, 0, 0), (0.063, 0.066, 0.064), "orange")
    p.box(camera, "camera_backplate", (0.067, 0, 0.035), (0.012, 0.07, 0.185), "dark")
    for z in (-0.10, 0.10):
        p.sleeve(
            camera,
            f"camera_shoe_{z}",
            (0.11, 0, z),
            2,
            0.035,
            (0.0066, 0.0126),
            wall=0.008,
        )
        p.box(camera, f"camera_bracket_{z}", (0.086, 0, z), (0.01, 0.025, 0.015))
    for z in (0.115,):
        p.box(camera, "stacked_camera", (0, 0, z), (0.062, 0.062, 0.051), "orange")
        p.box(camera, "stack_separator", (0, 0, 0.065), (0.065, 0.065, 0.006), "dark")
    for z in (0.115, 0):
        p.box(camera, f"camera_face_{z}", (-0.064, 0, z), (0.003, 0.052, 0.044), "dark")
        for y in (-0.044, 0.044):
            p.cylinder(
                camera,
                f"face_screw_{z}_{y}",
                (-0.068, y, z + 0.034),
                0.003,
                0.002,
                "bright",
                euler=(0, math.pi / 2, 0),
            )
    p.cylinder(
        camera,
        "barrel",
        (-0.08, 0, 0),
        0.033,
        0.031,
        "metal",
        euler=(0, math.pi / 2, 0),
    )
    p.cylinder(
        camera, "lens", (-0.112, 0, 0), 0.029, 0.003, "lens", euler=(0, math.pi / 2, 0)
    )
    p.cylinder(
        camera,
        "upper_port",
        (-0.076, 0, 0.115),
        0.018,
        0.012,
        "dark",
        euler=(0, math.pi / 2, 0),
    )
    b.site(camera, "camera_optical_center", (-0.116, 0, 0))
    b.site(camera, "camera_pinhole", (-0.118, 0, 0))
    ET.SubElement(
        camera,
        "camera",
        name=b.name + "__plant_camera",
        pos="-.118 0 0",
        quat=".7071067811865476 0 .7071067811865476 0",
        fovy="65",
    )
    b.metadata["cameras"] = {"plant_side": b.name + "__plant_camera"}
    b.metadata["mounted_cameras"] = [
        dict(
            name=b.name + "__plant_camera",
            body=camera.get("name"),
            local_position_m=[-0.118, 0, 0],
            local_quaternion_wxyz=[math.sqrt(0.5), 0, math.sqrt(0.5), 0],
            fovy_deg=65,
            site=b.metadata["sites"]["camera_pinhole"],
            purpose="Actual moving camera for inert-witness inspection; uncalibrated perspective optics",
        )
    ]
    # Connected stationary carrier guideway; local camera connectors stay capped.
    p.box(base, "chain_backbone", (0.62, 0.20, 1.37), (0.012, 0.022, 0.61), "dark")
    for z in (0.76, 1.98):
        p.box(base, f"chain_anchor_{z}", (0.64, 0.20, z), (0.02, 0.025, 0.018))
    for i in range(30):
        z = 0.78 + i * 0.040
        p.box(base, f"chain_link_{i}", (0.607, 0.20, z), (0.004, 0.026, 0.018), "dark")
    for z in (-0.035, 0.11):
        p.cylinder(
            camera,
            f"connector_{z}",
            (0, 0.074, z),
            0.011,
            0.012,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
        p.rod(
            camera,
            f"local_strain_relief_{z}",
            (0, 0.083, z),
            (0.04, 0.083, z + 0.025),
            0.007,
            "rubber",
        )
        p.box(
            camera,
            f"lead_dock_{z}",
            (0.047, 0.072, z + 0.025),
            (0.009, 0.014, 0.01),
            "dark",
        )


def _lighting(p, base):
    for side in (-1, 1):
        for z in (0.90, 1.35, 1.80, 2.12):
            p.rod(
                base,
                f"light_arm_{side}_{z}",
                (side * 0.65, -0.38, z),
                (side * 0.53, -0.35, z),
                0.012,
            )
            p.box(
                base,
                f"light_yoke_{side}_{z}",
                (side * 0.525, -0.35, z),
                (0.012, 0.063, 0.053),
            )
            p.box(
                base,
                f"reflector_{side}_{z}",
                (side * 0.501, -0.35, z),
                (0.013, 0.055, 0.045),
                "bright",
            )
            p.box(
                base,
                f"lamp_face_{side}_{z}",
                (side * 0.486, -0.35, z),
                (0.002, 0.045, 0.035),
                "cream",
            )
            # Folded side cheeks explain the source's open reflector shape;
            # the lamp remains an uncalibrated visual fixture.
            for sign in (-1, 1):
                p.box(base, f"reflector_cheek_{side}_{z}_{sign}",
                      (side * .480, -.35 + sign * .055, z),
                      (.025, .003, .045), "bright")
            p.rod(base, f"lamp_lead_{side}_{z}",
                  (side * .525, -.288, z), (side * .65, -.288, z), .004, "rubber")
        p.rod(base, f"lighting_service_trunk_{side}",
              (side * .65, -.288, .90), (side * .65, -.288, 2.12), .005, "rubber")
        p.rod(base, f"lighting_top_route_{side}",
              (side * .65, -.288, 2.12), (side * .65, .30, 2.12), .005, "rubber")
    p.box(base, "upper_scanner_mount", (0, 0.30, 2.21), (0.08, 0.07, 0.05))
    p.box(base, "upper_scanner", (0, 0.30, 2.105), (0.066, 0.08, 0.058), "orange")
    p.cylinder(base, "upper_scanner_port", (0, 0.30, 2.033), 0.025, 0.014, "dark")
    for index in range(6):
        p.box(base, f"upper_scanner_vent_{index}", (.067, .30, 2.065 + index * .014),
              (.001, .058, .003), "dark")
    p.rod(base, "upper_scanner_service", (.066, .30, 2.105), (.65, .30, 2.105),
          .005, "rubber")
