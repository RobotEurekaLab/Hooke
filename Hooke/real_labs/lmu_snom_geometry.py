"""Photo-informed LMU enclosure with explicitly authored positioning mechanics.

The open bay, flanking housings and raised lid follow the official commercial
S-SNOM photograph. Guides, travels and hidden attachments are original designs.
The cantilever uses a generic documented probe scale, not an identified LMU part.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

FINE_X_LIMITS = (-150e-6, 150e-6)
FINE_Z_LIMITS = (-95e-6, 95e-6)
TIP_RADIUS = 20e-9
TIP_LOCAL_BOTTOM = -0.0409
CANTILEVER_SIZE = (125e-6, 30e-6, 4e-6)


def _named(b, key):
    return f"{b.name}__{key}"


def _collar(b, body, key, x):
    """Open split bearing with a real 0.6 mm radial running clearance."""
    inner, outer, half = 0.0086, 0.016, 0.026
    angles = [math.pi / 2 + 0.12 + (math.tau - 0.24) * i / 40 for i in range(41)]
    n = len(angles)
    vertices = [
        (x + r * math.cos(a), 0.21 + r * math.sin(a), z)
        for z in (-half, half)
        for r in (outer, inner)
        for a in angles
    ]
    faces = []
    for i in range(n - 1):
        for q in (
            (i, i + 1, 2 * n + i + 1, 2 * n + i),
            (n + i + 1, n + i, 3 * n + i, 3 * n + i + 1),
            (i + 1, i, n + i, n + i + 1),
            (2 * n + i, 2 * n + i + 1, 3 * n + i + 1, 3 * n + i),
        ):
            faces.extend(((q[0], q[1], q[2]), (q[0], q[2], q[3])))
    for i in (0, n - 1):
        q = (i, 2 * n + i, 3 * n + i, n + i)
        faces.extend(((q[0], q[1], q[2]), (q[0], q[2], q[3])))
    mesh = _named(b, key + "_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for p in vertices for v in p]),
        face=" ".join(str(v) for f in faces for v in f),
    )
    b.geom(
        body, "mesh", (), material="metal", mesh=mesh, name=_named(b, key + "_surface")
    )
    for i, (a, c) in enumerate(zip(angles, angles[1:])):
        angle = (a + c) / 2
        radius = (inner + outer) / 2
        b.box(
            body,
            (x + radius * math.cos(angle), 0.21 + radius * math.sin(angle), 0),
            ((outer - inner) / 2, outer * math.tan((c - a) / 2), half),
            "metal",
            euler=(0, 0, angle),
            collision=True,
            group=3,
            rgba=(0, 0, 0, 0),
            name=_named(b, f"{key}_contact_{i}"),
        )
    # The neck joins the front half of the collar, outside its open bore.
    b.box(
        body,
        (x, 0.174, 0),
        (0.018, 0.020, 0.020),
        "dark",
        collision=True,
        name=_named(b, key + "_neck"),
    )


def _fine_probe(b, parent):
    """Two ideal fine axes carry a rigid generic cantilever and geometric tip."""
    xstage = b.moving(
        parent,
        "fine_scan_x",
        (0, -0.005, -0.110),
        (1, 0, 0),
        FINE_X_LIMITS,
        mass=0.012,
        kp=1200,
        force=2,
    )
    b.box(
        xstage,
        (0, 0, 0),
        (0.014, 0.013, 0.006),
        "metal",
        collision=True,
        name=_named(b, "fine_x_carriage"),
    )
    zstage = b.moving(
        xstage,
        "fine_approach_z",
        (0, 0, 0),
        (0, 0, 1),
        FINE_Z_LIMITS,
        mass=0.004,
        kp=1000,
        force=1,
    )
    # The moving plunger clears the plate through a real rectangular opening.
    # Its lateral guides are on the x carriage and cannot close that opening.
    b.cylinder(
        zstage,
        (0, 0, -0.013),
        0.004,
        0.016,
        "bright",
        collision=True,
        name=_named(b, "probe_plunger"),
    )
    tip_top = TIP_LOCAL_BOTTOM + TIP_RADIUS + 12e-6
    chip_top = tip_top + 0.00035
    b.box(
        zstage,
        (-0.0015, 0, (-0.029 + chip_top) / 2),
        (0.003, 0.002, (-0.029 - chip_top) / 2),
        "metal",
        collision=True,
        name=_named(b, "probe_chip_mount"),
    )
    # Chip and 125 um cantilever share a continuous underside interface.
    beam_bottom = tip_top
    b.box(
        zstage,
        (-0.00158, 0, beam_bottom + 0.000175),
        (0.0015, 0.0008, 0.000175),
        "metal",
        collision=True,
        name=_named(b, "probe_chip"),
    )
    b.box(
        zstage,
        (-0.0000425, 0, beam_bottom + 2e-6),
        (CANTILEVER_SIZE[0] / 2, CANTILEVER_SIZE[1] / 2, CANTILEVER_SIZE[2] / 2),
        "metal",
        collision=True,
        name=_named(b, "cantilever"),
    )
    # Shell inertia allows the true tiny visual mesh to compile; parent mass is
    # explicitly authored. This does not validate contact curvature or optics.
    vertices = []
    for z, r in ((tip_top, 2e-6), (TIP_LOCAL_BOTTOM + TIP_RADIUS, TIP_RADIUS)):
        vertices.extend(
            (r * math.cos(i * math.tau / 32), r * math.sin(i * math.tau / 32), z)
            for i in range(32)
        )
    mesh = _named(b, "fine_tip_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        inertia="shell",
        vertex=numbers([v for p in vertices for v in p]),
    )
    b.geom(
        zstage,
        "mesh",
        (),
        material="bright",
        mesh=mesh,
        collision=True,
        name=_named(b, "fine_tip_cone"),
    )
    b.geom(
        zstage,
        "sphere",
        (TIP_RADIUS,),
        pos=(0, 0, TIP_LOCAL_BOTTOM + TIP_RADIUS),
        material="bright",
        collision=True,
        name=_named(b, "snom_probe_tip"),
    )
    b.site(zstage, "probe_apex", (0, 0, TIP_LOCAL_BOTTOM), size=1e-6)
    b.site(
        zstage, "cantilever_reference", (-0.0000425, 0, beam_bottom + 2e-6), size=1e-6
    )
    return xstage, zstage


def _exterior_details(b, base, head, stage):
    """Resolvable commercial enclosure details; dimensions remain estimates."""
    # The official front photograph shows a raised circular left cover, a large
    # selector, black connector recesses and an unequal vertical module bank.
    b.cylinder(base, (0.565, 0.29, 1.285), 0.070, 0.035, "metal",
               collision=True, name=_named(b, "left_top_cover"))
    b.cylinder(base, (0.565, 0.29, 1.323), 0.075, 0.003, "dark",
               name=_named(b, "left_top_cover_lip"))
    b.cylinder(base, (0.565, -0.195, 1.133), 0.022, 0.014, "dark",
               euler=(math.pi / 2, 0, 0), collision=True,
               name=_named(b, "left_selector"))
    b.box(base, (1.215, 0.27, 1.265), (0.095, 0.19, 0.015), "metal",
          collision=True, name=_named(b, "module_bank_foot"))
    for i in range(6):
        y = 0.115 + i * 0.062
        b.box(base, (1.215, y, 1.385), (0.081, 0.022, 0.105), "dark",
              collision=True, name=_named(b, f"module_bank_{i}"))
        b.cylinder(base, (1.185, y, 1.502), 0.006, 0.012, "bright",
                   collision=True, name=_named(b, f"module_socket_{i}"))
        points = [(1.185, y, 1.514), (1.17, y - .02, 1.55),
                  (1.10, y - .025, 1.50), (1.105, y, 1.13)]
        for j, (a, c) in enumerate(zip(points, points[1:])):
            b.rod(base, a, c, 0.002, "dark",
                  name=_named(b, f"module_lead_{i}_{j}"))
    for label, x in (("left", 0.565), ("right", 1.215)):
        for dx in (-0.065, 0.065):
            for z in (0.90, 1.225):
                b.cylinder(base, (x + dx, -.183, z), .003, .002, "dark",
                           euler=(math.pi / 2, 0, 0),
                           name=_named(b, f"fascia_screw_{label}_{dx}_{z}"))
    for side in (-1, 1):
        b.cylinder(head, (side * .098, .065, 0), .014, .023, "metal",
                   euler=(0, math.pi / 2, 0), collision=True,
                   name=_named(b, f"head_adjuster_{side}"))
        b.cylinder(head, (side * .121, .065, 0), .018, .006, "dark",
                   euler=(0, math.pi / 2, 0), collision=True,
                   name=_named(b, f"head_knob_{side}"))
        for y in (-.09, .09):
            b.cylinder(stage, (side * .09, y, .017), .004, .001, "metal",
                       name=_named(b, f"carriage_screw_{side}_{y}"))


def add_inspection_station(b, base):
    """Enclosed bay, supported coarse carriage, and scale-separated fine probe."""
    from real_labs.optical_stages import aperture_plate

    b.box(
        base,
        (0.89, 0.19, 0.85),
        (0.45, 0.37, 0.03),
        "dark",
        collision=True,
        name=_named(b, "station_foundation"),
    )
    for x in (0.61, 1.17):
        b.box(
            base,
            (x, 0.39, 1.02),
            (0.023, 0.045, 0.14),
            "metal",
            collision=True,
            name=_named(b, f"coarse_frame_post_{x}"),
        )
    b.box(
        base,
        (0.89, 0.39, 1.18),
        (0.31, 0.055, 0.025),
        "dark",
        collision=True,
        name=_named(b, "coarse_frame_beam"),
    )
    b.box(
        base,
        (0.89, 0.425, 1.14),
        (0.15, 0.025, 0.14),
        "dark",
        collision=True,
        name=_named(b, "guide_backplate"),
    )
    for z in (1.04, 1.24):
        b.box(
            base,
            (0.89, 0.325, z),
            (0.13, 0.10, 0.01),
            "metal",
            collision=True,
            name=_named(b, f"guide_fixed_bracket_{z}"),
        )
    for x in (0.84, 0.94):
        b.cylinder(
            base,
            (x, 0.25, 1.14),
            0.008,
            0.09,
            "bright",
            collision=True,
            name=_named(b, f"coarse_guide_post_{x}"),
        )
    # Carriage glides over the two real external rail surfaces.
    for y in (-0.05, 0.12):
        b.box(
            base,
            (0.89, y, 0.8995),
            (0.23, 0.014, 0.0195),
            "bright",
            collision=True,
            name=_named(b, f"snom_x_guide_{y}"),
        )
    stage = b.moving(
        base,
        "specimen_translation",
        (0.89, 0.035, 0.935),
        (1, 0, 0),
        (-0.055, 0.055),
        mass=0.13,
        kp=350,
    )
    b.box(
        stage,
        (0, 0, 0),
        (0.12, 0.12, 0.016),
        "dark",
        collision=True,
        name=_named(b, "sample_carriage"),
    )
    b.cylinder(
        stage,
        (0, 0, 0.023),
        0.04,
        0.007,
        "metal",
        collision=True,
        name=_named(b, "coupon_puck"),
    )
    b.box(
        stage,
        (0, 0, 0.032),
        (0.009, 0.008, 0.002),
        "orange",
        collision=True,
        name=_named(b, "snom_inert_coupon"),
    )
    b.site(stage, "snom_coupon", (0, 0, 0.034), size=1e-6)
    head = b.moving(
        base,
        "inspection_head_height",
        (0.89, 0.04, 1.14),
        (0, 0, 1),
        (-0.02, 0.02),
        mass=0.15,
        kp=360,
    )
    b.box(
        head,
        (0, 0.065, 0),
        (0.077, 0.095, 0.028),
        "dark",
        collision=True,
        name=_named(b, "coarse_head"),
    )
    for x in (-0.05, 0.05):
        _collar(b, head, "coarse_collar_" + str(x), x)
    b.cylinder(
        head,
        (0, -0.005, -0.057),
        0.019,
        0.029,
        "dark",
        collision=True,
        name=_named(b, "scanner_housing"),
    )
    b.box(
        head,
        (0, -0.005, -0.0926),
        (0.025, 0.022, 0.011),
        "dark",
        collision=True,
        name=_named(b, "fine_stage_upper_mount"),
    )
    # Horizontal x carriage has a genuine 0.4 mm running gap below its fixed
    # cover and is supported by retained side flanges/ideal slide joint.
    for x in (-0.019, 0.019):
        b.box(
            head,
            (x, -0.005, -0.11),
            (0.004, 0.016, 0.010),
            "metal",
            collision=True,
            name=_named(b, f"fine_x_outer_guide_{x}"),
        )
    xstage, zstage = _fine_probe(b, head)
    # Replace the initial solid x carriage plate with an open one for z travel.
    solid = xstage.find(f'./geom[@name="{_named(b,"fine_x_carriage")}"]')
    xstage.remove(solid)
    aperture_plate(
        b,
        xstage,
        "fine_x_carriage",
        (0, 0, 0),
        (0.014, 0.013, 0.006),
        (0.005, 0.005),
        "metal",
    )
    for x in (-0.007, 0.007):
        b.box(
            xstage,
            (x, 0, -0.01),
            (0.0015, 0.007, 0.004),
            "metal",
            collision=True,
            name=_named(b, f"fine_z_guide_{x}"),
        )
    # Plunger shoulder connects to an original flexure-like cross member.
    b.box(
        zstage,
        (0, 0, -0.015),
        (0.005, 0.009, 0.002),
        "metal",
        collision=True,
        name=_named(b, "fine_z_shoulder"),
    )
    # Official commercial apparatus: two low housings flank an open centre.
    for label, x, w in (("left", 0.565, 0.095), ("right", 1.215, 0.105)):
        b.box(
            base,
            (x, 0.16, 1.065),
            (w, 0.33, 0.185),
            "shell",
            collision=True,
            name=_named(b, label + "_instrument_housing"),
        )
        b.box(
            base,
            (x, -0.175, 1.065),
            (w - 0.012, 0.006, 0.172),
            "shell",
            name=_named(b, label + "_fascia"),
        )
        b.box(
            base,
            (x, -0.183, 0.956),
            (0.042, 0.004, 0.039),
            "dark",
            name=_named(b, label + "_connector_recess"),
        )
        for dx in (-0.014, 0.014):
            b.cylinder(
                base,
                (x + dx, -0.190, 0.956),
                0.006,
                0.007,
                "metal",
                euler=(math.pi / 2, 0, 0),
                name=_named(b, label + "_port_" + str(dx)),
            )
    b.box(
        base,
        (0.89, 0.51, 1.245),
        (0.33, 0.04, 0.015),
        "dark",
        collision=True,
        name=_named(b, "lid_hinge_support"),
    )
    for x in (0.72, 1.06):
        b.cylinder(
            base,
            (x, 0.49, 1.255),
            0.009,
            0.034,
            "metal",
            collision=True,
            euler=(0, math.pi / 2, 0),
            name=_named(b, "lid_hinge_" + str(x)),
        )
    lid = ET.SubElement(base, "body", pos=".89 .49 1.255", euler="-1.45 0 0")
    b.box(
        lid,
        (0, -0.22, 0),
        (0.235, 0.22, 0.009),
        "dark",
        collision=True,
        name=_named(b, "raised_open_lid"),
    )
    b.box(
        lid,
        (0, -0.22, 0.0095),
        (0.221, 0.206, 0.001),
        "rubber",
        name=_named(b, "lid_inner_liner"),
    )
    # Supported external interface block and flexible service loops. They are
    # visual service routing estimates, not an inferred optical beam path.
    b.box(
        base,
        (1.29, -0.38, 0.86),
        (0.075, 0.10, 0.04),
        "shell",
        collision=True,
        name=_named(b, "front_service_block"),
    )
    b.box(
        base,
        (0.98, -0.02, 0.895),
        (0.06, 0.025, 0.015),
        "dark",
        collision=True,
        name=_named(b, "stage_service_connector"),
    )
    for i in range(4):
        points = []
        for j in range(13):
            t = j / 12
            points.append(
                (
                    1.25 + i * 0.017 - 0.30 * t,
                    -0.40 + 0.38 * t,
                    0.90 + 0.15 * math.sin(math.pi * t),
                )
            )
        for j, (a, c) in enumerate(zip(points, points[1:])):
            b.rod(base, a, c, 0.0025, "metal", name=_named(b, f"service_loop_{i}_{j}"))
    _exterior_details(b, base, head, stage)
    b.metadata["capabilities"] = [
        "registered coarse specimen alignment",
        "micrometre-scale geometric approach and line scan",
        "measured surface-relative clearance and physical retraction",
    ]
    b.metadata["positioning_contract"] = {
        "operation": "real_labs.lmu_snom_motion.inspect_lmu",
        "surface_site": b.metadata["sites"]["snom_coupon"],
        "probe_site": b.metadata["sites"]["probe_apex"],
        "minimum_authored_envelope_gap_m": 5e-6,
        "nominal_scan_clearance_m": 10e-6,
        "science": "rigid geometry only; no near-field or optical response",
    }
    b.metadata["limitations"].append(
        "Commercial open-bay exterior follows LMU photos. Hidden guides and all "
        "travel/room dimensions are authored. Generic 125 x 30 x 4 micrometre "
        "cantilever and 12 micrometre tip use NANOSENSORS PPP-NCHPt sizing only; "
        "LMU probe identity is unknown. Primitive/mesh contact and ideal servos "
        "do not validate tip curvature, nanometre accuracy, flexure, force "
        "feedback, AFM tapping, optics or near-field imaging."
    )
