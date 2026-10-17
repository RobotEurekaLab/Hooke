"""COEB photograph-informed microscope and original dry registration mechanics.

The official small photograph supports exterior organization, not dimensions,
OEM internals, probe geometry or scientific performance. All solid instrument
parts have enabled contact geometry; open guide clearances are authored.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.ntu_sgsr_geometry import Parts as InstrumentParts


class Parts(InstrumentParts):
    def name(self, key):
        return f"{self.b.name}__coeb_{key}"


def build_station(b, base, params):
    p = Parts(b)
    p.box(base, "board", (0, 0, 0.025), (0.58, 0.36, 0.025), "dark")
    for x in range(-5, 6):
        for y in range(-3, 4):
            p.cylinder(
                base,
                f"board_bore_{x}_{y}",
                (x * 0.1, y * 0.1, 0.0502),
                0.0025,
                0.0002,
                "metal",
            )
    scope = ET.SubElement(
        base, "body", name=b.name + "__scope_frame", pos="-0.20 0 0.05"
    )
    _frame(b, p, scope)
    _stages(b, p, scope)
    _probe(b, p, scope)
    _display(b, p, base)
    _optical_context(b, p, base)
    b.metadata["capabilities"] = [
        "supported_retained_coupon_xy",
        "guided_full_contact_probe_height",
        "measured_dry_registration_and_return",
    ]
    b.metadata["inspection_interface"] = dict(
        tip_site=b.metadata["sites"]["probe_tip"],
        target_site=b.metadata["sites"]["registration_mark"],
        target_geom=p.name("registration_mark"),
        guide_clearance_m=0.0006,
        coupon_attachment="Rigid retained specimen; original ideal joints, no loading/release",
        probe_description="Authored 125 micrometre outer fibre with 50 micrometre blunt terminal, not a verified NSOM aperture",
        scope="Geometric dry registration at millimetre standoff; no near-field coupling or acquisition",
    )
    b.metadata["limitations"].extend(
        [
            "Source photograph establishes an installed microscope/optical-stack/display arrangement, not OEM internals or measured dimensions.",
            "The fibre surrogate, guides, retained coupon, masses, ideal gravity compensation and original gains are authored; no nanometre precision claim.",
            "Probe stays above the retained 12 mm inert coupon. No near-field light coupling, optical image, force feedback or spectrum is simulated.",
            "The display is inactive and service stubs terminate locally; no fabricated scientific result or complete electrical topology.",
        ]
    )
    b.metadata["source_visible_context"] = {
        "reference": "temp/research/qs_asia_visual_checks/ntu_nsnom.jpg",
        "features": ["Left-side mounted optical modules", "Supported rear service leads", "Dark optical table and red indicator faces"],
        "scope": "Exterior photographic estimates. Red faces represent static indicators, not a calibrated laser field; the monitor remains inactive.",
    }


def _optical_context(b, p, base):
    """Source-visible side optics on connected mounts outside the moving bay."""
    ET.SubElement(b.root.find("asset"), "material", name=b.name + "__mat_indicator_red",
                  rgba=".65 .018 .01 1", specular=".2", emission=".25")
    # Each module sits on a post and foot on the breadboard. Exact component
    # identity and dimensions cannot be recovered from the small photograph.
    for index, (y, z) in enumerate(((.04, .28), (.25, .43))):
        x = -.50
        p.box(base, f"side_optic_foot_{index}", (x, y, .060), (.042, .040, .010), "dark")
        p.cylinder(base, f"side_optic_post_{index}", (x, y, (.07+z)/2), .012, (z-.07)/2, "metal")
        p.box(base, f"side_optic_mount_{index}", (x, y, z+.012), (.036, .035, .012), "dark")
        p.box(base, f"side_optic_body_{index}", (x, y, z+.052), (.031, .028, .028), "dark")
        p.cylinder(base, f"side_optic_port_{index}", (x, y-.030, z+.052), .014, .003, "metal", euler=(math.pi/2, 0, 0))
        p.cylinder(base, f"side_optic_indicator_{index}", (x, y-.034, z+.052), .009, .001, "indicator_red", euler=(math.pi/2, 0, 0))
        points = ((x, y+.028, z+.052), (x-.025, y+.035, z+.025),
                  (x-.035, y+.04, .075), (x-.035, .32, .075), (-.37, .32, .075))
        for segment, (a, c) in enumerate(zip(points, points[1:])):
            p.rod(base, f"side_service_{index}_{segment}", a, c, .003, "rubber")
    p.box(base, "rear_service_box", (-.38, .31, .087), (.035, .035, .037), "dark")


def _frame(b, p, scope):
    p.housing(
        scope,
        "cast_base",
        [
            (0, 0.17, 0.20, 0.01),
            (0.012, 0.18, 0.22, 0.01),
            (0.056, 0.18, 0.21, 0.01),
            (0.070, 0.155, 0.19, 0.02),
        ],
        "cream",
        0.025,
    )
    p.housing(
        scope,
        "rear_column",
        [
            (0.058, 0.145, 0.075, 0.18),
            (0.43, 0.11, 0.065, 0.185),
            (0.54, 0.09, 0.06, 0.185),
        ],
        "cream",
        0.025,
    )
    # Two connected, slim cheeks leave the specimen/optical bay open.
    for side in (-1, 1):
        p.box(
            scope,
            f"deck_cheek_{side}",
            (side * 0.118, -0.04, 0.133),
            (0.017, 0.091, 0.068),
            "cream",
        )
        p.cylinder(
            scope,
            f"focus_hub_{side}",
            (side * 0.15, 0.10, 0.125),
            0.025,
            0.022,
            "metal",
            euler=(0, math.pi / 2, 0),
        )
        p.cylinder(
            scope,
            f"coarse_focus_{side}",
            (side * 0.178, 0.10, 0.125),
            0.046,
            0.017,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
        p.cylinder(
            scope,
            f"fine_focus_{side}",
            (side * 0.20, 0.10, 0.125),
            0.022,
            0.008,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
        for j in range(16):
            angle = j * math.tau / 16
            p.box(
                scope,
                f"focus_grip_{side}_{j}",
                (
                    side * 0.178,
                    0.10 + 0.045 * math.cos(angle),
                    0.125 + 0.045 * math.sin(angle),
                ),
                (0.014, 0.002, 0.002),
                "metal",
                euler=(angle, 0, 0),
            )
    p.box(scope, "fixed_deck", (0, -0.04, 0.201), (0.145, 0.12, 0.012), "dark")
    # Lower binocular branch follows the visible low front-side optical port.
    p.rod(scope, "viewing_port", (0, 0.13, 0.145), (0, -0.155, 0.145), 0.028, "cream")
    for side in (-1, 1):
        p.rod(
            scope,
            f"ocular_bridge_{side}",
            (0, -0.145, 0.145),
            (side * 0.035, -0.21, 0.170),
            0.023,
            "cream",
        )
        p.rod(
            scope,
            f"ocular_{side}",
            (side * 0.035, -0.205, 0.169),
            (side * 0.040, -0.245, 0.185),
            0.017,
            "dark",
        )
    for z in (0.095, 0.31, 0.48):
        cover_x = 0.100 if z > 0.4 else 0.113
        p.box(
            scope, f"side_cover_{z}", (cover_x, 0.17, z), (0.003, 0.050, 0.022), "metal"
        )
        for y in (0.135, 0.205):
            p.cylinder(
                scope,
                f"side_fastener_{z}_{y}",
                (cover_x + 0.004, y, z),
                0.003,
                0.002,
                "dark",
                euler=(0, math.pi / 2, 0),
            )


def _stages(b, p, scope):
    # X guide: fixed pedestals -> rails -> open running shoes -> carriage.
    for y in (-0.055, 0.055):
        for x in (-0.128, 0.128):
            p.box(
                scope,
                f"x_pedestal_{x}_{y}",
                (x, y - 0.04, 0.21525),
                (0.006, 0.006, 0.00225),
            )
        p.box(
            scope,
            f"x_rail_{y}",
            (0, y - 0.04, 0.220),
            (0.134, 0.0025, 0.0025),
            "bright",
        )
    xstage = b.moving(
        scope,
        "sample_x",
        (0, -0.04, 0.24),
        (1, 0, 0),
        (-0.018, 0.018),
        mass=0.08,
        kp=220,
    )
    p.box(xstage, "x_carriage", (0, 0, 0), (0.05, 0.075, 0.012))
    for x in (-0.035, 0.035):
        for y in (-0.055, 0.055):
            p.sleeve(
                xstage,
                f"x_shoe_{x}_{y}",
                (x, y, -0.020),
                0,
                0.012,
                (0.0031, 0.0031),
                wall=0.0025,
            )
            p.box(
                xstage,
                f"x_shoe_mount_{x}_{y}",
                (x, y, -0.0132),
                (0.012, 0.0056, 0.0012),
            )
    # Y rails sit outboard of the Y carrier, leaving full 30 mm travel.
    for x in (-0.063, 0.063):
        for y in (-0.070, 0.070):
            p.box(
                xstage,
                f"y_outtrigger_{x}_{y}",
                (x * 0.8, y, 0.009),
                (0.020, 0.006, 0.003),
            )
            p.box(
                xstage, f"y_pedestal_{x}_{y}", (x, y, 0.01525), (0.006, 0.006, 0.00325)
            )
        p.box(xstage, f"y_rail_{x}", (x, 0, 0.021), (0.0025, 0.080, 0.0025), "bright")
    ystage = b.moving(
        xstage,
        "sample_y",
        (0, 0, 0.021),
        (0, 1, 0),
        (-0.015, 0.015),
        mass=0.045,
        kp=180,
    )
    p.box(ystage, "y_carriage", (0, 0, 0), (0.048, 0.044, 0.008), "dark")
    for x in (-0.063, 0.063):
        for y in (-0.021, 0.021):
            p.sleeve(
                ystage,
                f"y_shoe_{x}_{y}",
                (x, y, 0),
                1,
                0.010,
                (0.0031, 0.0031),
                wall=0.0025,
            )
            p.box(
                ystage,
                f"y_shoe_bridge_{x}_{y}",
                (x * 0.897, y, 0.0068),
                (0.012, 0.010, 0.0012),
            )
    p.box(ystage, "inert_coupon", (0, 0, 0.0085), (0.006, 0.006, 0.0005), "sample")
    # Four pads touch and retain the specimen edge; no invisible specimen constraint.
    for side in (-1, 1):
        p.box(
            ystage,
            f"retainer_{side}",
            (side * 0.0068, 0, 0.0084),
            (0.0008, 0.003, 0.0004),
            "bright",
        )
        p.box(
            ystage,
            f"retainer_top_{side}",
            (side * 0.0058, 0, 0.0092),
            (0.0018, 0.003, 0.0002),
            "bright",
        )
    p.box(
        ystage,
        "registration_mark",
        (0.001, -0.001, 0.009025),
        (0.0005, 0.0005, 0.000025),
        "bright",
    )
    b.site(ystage, "optical_coupon", (0, 0, 0.009), 0.0003)
    b.site(ystage, "registration_mark", (0.001, -0.001, 0.00905), 0.00003)


def _probe(b, p, scope):
    # Fixed lower bridge is physically seated on the rear column.
    for key, z in (("lower", 0.525), ("upper", 0.763)):
        p.box(scope, f"head_{key}_bridge", (0, 0.145, z), (0.108, 0.055, 0.007), "dark")
    for x in (-0.073, 0.073):
        p.box(
            scope, f"head_rail_{x}", (x, 0.13, 0.644), (0.004, 0.004, 0.112), "bright"
        )
    head = b.moving(
        scope,
        "probe_clearance",
        (0, -0.04, 0.61),
        (0, 0, 1),
        (-0.010, 0.014),
        mass=0.10,
        kp=220,
    )
    # Head coordinates are relative to original y=-40 mm axis frame.
    for x in (-0.073, 0.073):
        p.sleeve(
            head,
            f"head_shoe_{x}",
            (x, 0.17, 0.015),
            2,
            0.035,
            (0.0046, 0.0046),
            wall=0.0044,
        )
    p.box(head, "head_rear_tie", (0, 0.184, 0.015), (0.082, 0.005, 0.030), "metal")
    p.box(head, "probe_housing", (0, 0.1475, 0.015), (0.057, 0.035, 0.060), "dark")
    p.box(head, "optics_top_cover", (0, .1475, .078), (.057, .035, .003), "dark", density=0)
    p.cylinder(head, "optics_top_adapter", (0, .145, .096), .019, .015, "dark", density=0)
    p.box(head, "probe_reach", (0, 0.05, -0.025), (0.037, 0.09, 0.015), "metal")
    p.box(head, "optics_cube", (0, 0, -0.080), (0.042, 0.036, 0.045), "dark")
    p.cylinder(head, "objective_upper", (0, 0, -0.143), 0.026, 0.027, "dark")
    p.cylinder(head, "objective_lower", (0, 0, -0.210), 0.020, 0.040, "metal")
    p.cylinder(head, "objective_shoulder", (0, 0, -0.256), 0.026, 0.007, "dark")
    p.cylinder(head, "fibre_holder", (0.007, -0.004, -0.2795), 0.012, 0.017, "dark")
    p.cylinder(head, "ferrule", (0.007, -0.004, -0.3055), 0.003, 0.011, "bright")
    p.cylinder(head, "fibre", (0.007, -0.004, -0.3225), 0.0000625, 0.006, "lens")
    p.cylinder(head, "terminal", (0.007, -0.004, -0.32875), 0.000025, 0.00025, "bright")
    b.site(head, "probe_tip", (0.007, -0.004, -0.329), 0.00001)
    for z in (-0.065, -0.094):
        p.cylinder(
            head,
            f"optics_adjuster_{z}",
            (0.048, 0, z),
            0.011,
            0.010,
            "metal",
            euler=(0, math.pi / 2, 0),
        )
    for x in (-0.025, 0.025):
        p.cylinder(head, f"top_adjuster_{x}", (x, 0.145, 0.089), 0.008, 0.014, "metal")
        p.cylinder(head, f"top_knob_{x}", (x, 0.145, 0.108), 0.013, 0.006, "dark")
    # Short connected service ends move with the head; no stretching cable model.
    for x in (-0.022, 0.022):
        p.cylinder(
            head, f"service_socket_{x}", (x, 0.164, 0.082), 0.006, 0.007, "metal"
        )
        p.rod(
            head,
            f"service_stub_{x}",
            (x, 0.164, 0.089),
            (x + 0.012, 0.17, 0.124),
            0.003,
            "rubber",
        )


def _display(b, p, base):
    p.box(base, "display_foot", (0.33, 0.21, 0.064), (0.14, 0.10, 0.014), "dark")
    p.box(base, "display_post", (0.33, 0.258, 0.27), (0.028, 0.025, 0.20), "metal")
    p.box(base, "display_shell", (0.33, 0.216, 0.48), (0.26, 0.018, 0.245), "dark")
    # An inactive face avoids presenting the source's result image as simulated data.
    p.box(base, "display_face", (0.33, 0.197, 0.48), (0.248, 0.001, 0.231), "screen")
    p.box(base, "keyboard", (0.33, -0.23, 0.061), (0.21, 0.061, 0.011), "dark")
    for row in range(4):
        for col in range(13):
            p.box(
                base,
                f"key_{row}_{col}",
                (0.15 + col * 0.029, -0.268 + row * 0.023, 0.073),
                (0.010, 0.008, 0.001),
                "metal",
            )
