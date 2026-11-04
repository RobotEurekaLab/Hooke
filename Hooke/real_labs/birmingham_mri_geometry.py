"""Source-informed scanner exterior and open inert-phantom table mechanics.

Dimensions, hidden bearings and retention hardware are authored estimates. This
is a dry rigid-body fixture, not OEM CAD, an MRI safety model or a patient task.
Original lift/insertion frames, travel, drives and phantom envelope are retained.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.mounting import annular_mount

PREFIX = "scanner__mri_access_"


def box(b, body, name, pos, half, material="shell"):
    return b.box(
        body, pos, half, material, collision=True, density=0, name=PREFIX + name
    )


def _prism(b, body, name, section, y0, y1, material="shell"):
    """Convex X/Z section extruded along Y; collision and visible mesh agree."""
    vertices = [(x, y, z) for y in (y0, y1) for x, z in section]
    n = len(section)
    faces = []
    for i in range(1, n - 1):
        faces.extend(((0, i, i + 1), (n, n + i + 1, n + i)))
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((i, n + j, j), (i, n + i, n + j)))
    mesh = PREFIX + name + "_mesh"
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for p in vertices for v in p]),
        face=" ".join(str(v) for f in faces for v in f),
    )
    return b.geom(
        body,
        "mesh",
        (),
        material=material,
        mesh=mesh,
        collision=True,
        density=0,
        name=PREFIX + name,
    )


def _pedestal(b, base):
    # Four separate contact-capable walls leave a genuine rectangular bore.
    # Preserve the moving stem's 302x380 mm envelope with 1 mm side clearance.
    for side in (-1, 1):
        box(
            b,
            base,
            f"lift_wall_x_{side}",
            (side * 0.166, -1.46, 0.43),
            (0.014, 0.22, 0.21),
        )
        box(
            b,
            base,
            f"lift_wall_y_{side}",
            (0, -1.46 + side * 0.2055, 0.43),
            (0.152, 0.0145, 0.21),
        )
    # Lower tapered aprons reproduce the source's broad cast support without
    # filling the upper opening. The moving stem never reaches these aprons.
    for side in (-1, 1):
        # A triangular convex side cheek is attached to the actual wall.
        section = [(side * 0.179, 0.22), (side * 0.220, 0.22), (side * 0.179, 0.49)]
        if side < 0:
            section.reverse()
        _prism(b, base, f"lower_side_apron_{side}", section, -1.70, -1.22)
    for side in (-1, 1):
        box(
            b,
            base,
            f"lower_end_apron_{side}",
            (0, -1.46 + side * 0.241, 0.305),
            (0.18, 0.025, 0.085),
        )


def _bearing_y(b, carrier, key, x, y):
    # Fixed transformation, no new DOF or inertia: existing parent moving mass
    # remains the complete authored assembly mass. The ring keeps its true bore.
    before = set(carrier.findall("geom"))
    annular_mount(
        b,
        carrier,
        key,
        (0, 0, 0),
        0.0146,
        0.024,
        0.045,
        axis="x",
        proxies=True,
        namespace="mri_access",
        material="metal",
    )
    for geom in carrier.findall("geom"):
        if geom in before:
            continue
        geom.set("density", "0")
        px, py, pz = map(float, geom.get("pos", "0 0 0").split())
        geom.set("pos", numbers((x - py, y + px, -0.085 + pz)))
        # Rz(90 degrees) Rx(angle), expressed directly on each shape so no
        # mass-bearing child body changes the original aggregate inertia.
        angle = float(geom.get("euler", "0 0 0").split()[0])
        c, s = math.cos(angle / 2) * math.sqrt(0.5), math.sin(angle / 2) * math.sqrt(
            0.5
        )
        geom.attrib.pop("euler", None)
        geom.set("quat", numbers((c, s, s, c)))
    # Crown brackets attach each bearing to the unchanged table underside.
    box(b, carrier, key + "_crown", (x, y, -0.043), (0.027, 0.047, 0.021), "metal")


def add_table(b, base):
    """Open pedestal and Y bearings, with the original two control contracts."""
    # Authored 4 mm cosmetic cap above the physical foot top: two visible
    # surfaces must not occupy the same plane. Enabled support stays unchanged.
    b.housing(
        base, ((0.09, 0.37, 0.54, -1.48), (0.224, 0.32, 0.48, -1.46)), radius=0.06
    )
    box(b, base, "table_foot", (0, -1.46, 0.155), (0.31, 0.45, 0.065))
    for i, x in enumerate((-0.31, 0.31)):
        for j, y in enumerate((-1.85, -1.10)):
            b.cylinder(
                base,
                (x, y, 0.070),
                0.07,
                0.029,
                "rubber",
                collision=True,
                density=0,
                euler=(0, math.pi / 2, 0),
                name=PREFIX + f"wheel_{i}_{j}",
            )
            b.cylinder(
                base,
                (x * 1.06, y, 0.070),
                0.037,
                0.011,
                "bright",
                euler=(0, math.pi / 2, 0),
                name=PREFIX + f"wheel_hub_{i}_{j}",
            )
            # Explicit caster-to-foot attachment; wheels remain fixed scenery.
            box(
                b,
                base,
                f"caster_fork_{i}_{j}",
                (x, y, 0.105),
                (0.025, 0.025, 0.035),
                "metal",
            )
    _pedestal(b, base)
    lift = b.moving(
        base, "table_height", (0, 0, 0.72), (0, 0, 1), (0, 0.06), mass=1.5, kp=300
    )
    box(b, lift, "lift_stem", (0, -1.46, -0.065), (0.151, 0.19, 0.095), "metal")
    box(b, lift, "lift_deck", (0, -1.27, 0), (0.28, 0.66, 0.026))
    for i, x in enumerate((-0.21, 0.21)):
        # Long low ledges support both rail ends without filling bearing bores.
        box(b, lift, f"rail_bed_{i}", (x, -1.02, 0.009), (0.034, 0.89, 0.010), "metal")
        b.rod(
            lift,
            (x, -1.86, 0.065),
            (x, -0.18, 0.065),
            0.014,
            "bright",
            collision=True,
            density=0,
            name=PREFIX + f"rail_{i}",
        )
        for j, y in enumerate((-1.83, -0.21)):
            box(
                b,
                lift,
                f"rail_end_mount_{i}_{j}",
                (x, y, 0.0475),
                (0.030, 0.020, 0.0285),
                "metal",
            )
    carriage = b.moving(
        lift, "table_insert", (0, -0.92, 0.15), (0, 1, 0), (0, 0.85), mass=1.2, kp=300
    )
    box(b, carriage, "table_board", (0, 0, 0), (0.20, 1.11, 0.025))
    for i, x in enumerate((-0.21, 0.21)):
        for j, y in enumerate((-0.50, -0.26)):
            _bearing_y(b, carriage, f"carriage_bearing_{i}_{j}", x, y)
    for i, x in enumerate((-0.196, 0.196)):
        b.rod(
            carriage,
            (x, -1.06, 0.027),
            (x, 0.99, 0.027),
            0.012,
            "bright",
            collision=True,
            density=0,
            name=PREFIX + f"table_edge_{i}",
        )
    box(b, carriage, "table_pad", (0, -0.21, 0.049), (0.18, 0.76, 0.022), "rubber")
    # Curved source table appearance: narrow edge fascia below the real deck,
    # leaving longitudinal rail/bearing access visible from either side.
    for side in (-1, 1):
        box(
            b,
            carriage,
            f"edge_fascia_{side}",
            (side * 0.191, -0.30, -0.009),
            (0.009, 0.72, 0.016),
        )
    # Inert sample geometry and original target are unchanged. A lower cradle
    # and tangent V saddles replace the old flat slab penetrating the cylinder.
    box(b, carriage, "cradle_base", (0, 0.38, 0.037), (0.15, 0.19, 0.012), "cream")
    for side in (-1, 1):
        box(
            b,
            carriage,
            f"cradle_side_{side}",
            (side * 0.127, 0.38, 0.106),
            (0.021, 0.19, 0.055),
        )
        # Tangent to the unchanged 96 mm-radius end caps; the central 93 mm
        # cylinder is retained between them, without penetrating a flat slab.
        c = 0.15 - 0.096 * math.sqrt(2)
        section = [(0.04, 0.049), (0.12, 0.049), (0.12, 0.12 + c), (0.04, 0.04 + c)]
        if side < 0:
            section = [(-x, z) for x, z in reversed(section)]
        _prism(b, carriage, f"phantom_saddle_{side}", section, 0.19, 0.57, "cream")
    b.cylinder(
        carriage,
        (0, 0.38, 0.150),
        0.093,
        0.147,
        "cyan",
        euler=(math.pi / 2, 0, 0),
        collision=True,
        density=0,
        name=PREFIX + "inert_phantom",
    )
    for i, y in enumerate((0.224, 0.536)):
        b.cylinder(
            carriage,
            (0, y, 0.150),
            0.096,
            0.006,
            "cream",
            euler=(math.pi / 2, 0, 0),
            collision=True,
            density=0,
            name=PREFIX + f"phantom_endcap_{i}",
        )
    for j, y in enumerate((0.27, 0.49)):
        box(
            b,
            carriage,
            f"retaining_bridge_{j}",
            (0, y, 0.249),
            (0.14, 0.012, 0.005),
            "cream",
        )
        for side in (-1, 1):
            box(
                b,
                carriage,
                f"retaining_post_{j}_{side}",
                (side * 0.127, y, 0.200),
                (0.007, 0.012, 0.049),
                "cream",
            )
            b.cylinder(
                carriage,
                (side * 0.127, y, 0.257),
                0.008,
                0.003,
                "metal",
                collision=True,
                density=0,
                name=PREFIX + f"retaining_fastener_{j}_{side}",
            )
    b.site(carriage, "inert_mri_phantom", (0, 0.38, 0.150), size=0.009)
    b.metadata["table_support"] = {
        "revision": 2,
        "dimension_basis": "Authored open bearing repair; not OEM internals",
        "lift_side_clearance_m": 0.001,
        "insertion_radial_clearance_m": 0.0006,
        "minimum_lift_stem_engagement_m": 0.020,
        "kinematics": "Original60 mm lift and850 mm insertion; ideal joint support",
        "scope": "Dry positioning of a restrained inert cylinder; no patient or MR science",
    }


def add_scanner_context(b, base):
    """Source-visible dark rear casing and supported front controls only."""
    for side in (-1, 1):
        box(
            b,
            base,
            f"rear_side_case_{side}",
            (side * 0.91, 0.79, 1.08),
            (0.13, 0.53, 0.90),
            "dark",
        )
        box(
            b,
            base,
            f"rear_case_edge_{side}",
            (side * 1.035, 0.79, 1.09),
            (0.015, 0.53, 0.89),
            "shell",
        )
        # Photograph supports vertical detailing on the rear side panel; these
        # are passive grooves rather than invented airflow or shielding parts.
        for k in range(7):
            b.box(
                base,
                (side * (0.82 + k * 0.026), 0.252, 1.72),
                (0.005, 0.008, 0.21),
                "metal",
                name=PREFIX + f"rear_panel_trim_{side}_{k}",
            )
    box(b, base, "rear_upper_case", (0, 0.79, 2.03), (0.90, 0.53, 0.075), "shell")
    for side in (-1, 1):
        # The rendered face sits ahead of the main shell contact sectors.
        # A fixed rear stem connects the enabled collar to those sectors.
        b.cylinder(
            base,
            (side * 0.69, 0.117, 1.28),
            0.021,
            0.053,
            "metal",
            collision=True,
            density=0,
            euler=(math.pi / 2, 0, 0),
            name=PREFIX + f"front_control_mount_{side}",
        )
        b.cylinder(
            base,
            (side * 0.69, 0.067, 1.28),
            0.031,
            0.014,
            "metal",
            collision=True,
            density=0,
            euler=(math.pi / 2, 0, 0),
            name=PREFIX + f"front_control_collar_{side}",
        )
        b.cylinder(
            base,
            (side * 0.69, 0.047, 1.28),
            0.019,
            0.006,
            "cream",
            euler=(math.pi / 2, 0, 0),
            name=PREFIX + f"front_control_button_{side}",
        )
    _add_fixed_table_services(b, base)


def _add_fixed_table_services(b, base):
    """Compact passive chassis line; the photo does not establish its function."""
    for i, y in enumerate((-1.76, -1.16)):
        # Both endpoint blocks grow from the existing fixed foot. Nothing is
        # tethered to the moving lift or insertion carriage.
        box(b, base, f"service_mount_{i}", (0.299, y, 0.247), (0.024, 0.029, 0.027))
        box(
            b,
            base,
            f"service_port_{i}",
            (0.327, y, 0.279),
            (0.018, 0.018, 0.012),
            "metal",
        )
        b.rod(
            base,
            (0.329, y, 0.279),
            (0.349, y, 0.279),
            0.010,
            "cream",
            collision=True,
            density=0,
            name=PREFIX + f"service_end_{i}",
        )
    # A short rounded U-shaped route follows the side of the stationary foot,
    # rather than making a floor loop or crossing the insertion/access lane.
    points = [
        (0.349, -1.76, 0.279),
        (0.349, -1.751, 0.262),
        (0.349, -1.730, 0.248),
        (0.349, -1.690, 0.244),
        (0.349, -1.250, 0.244),
        (0.349, -1.205, 0.248),
        (0.349, -1.175, 0.261),
        (0.349, -1.16, 0.279),
    ]
    for i, (a, c) in enumerate(zip(points, points[1:])):
        b.rod(
            base,
            a,
            c,
            0.009,
            "cream",
            collision=True,
            density=0,
            name=PREFIX + f"service_line_{i}",
        )
    for i, y in enumerate((-1.64, -1.30)):
        box(
            b,
            base,
            f"service_clip_{i}",
            (0.324, y, 0.228),
            (0.025, 0.018, 0.008),
            "metal",
        )
    b.metadata["source_context"] = {
        "revision": 1,
        "reference": "Cached Birmingham MRI installed-room photograph",
        "visible_features": "Varied white shelf accessories, pale blue insert and compact lower-chassis service detail",
        "dimension_basis": "Photographic estimates; accessory identities and service function are unresolved",
        "service_scope": "Passive fixed-chassis line with supported endpoints; no connection to the moving table or MR field model",
    }


def add_room_storage(p):
    """Replace generic cubes with the visible accessory silhouettes only.

    The single photo does not identify coils or clinical functions. These are
    fixed, inert stored shapes with estimated sizes; the white circular piece
    has a real opening instead of a painted dark disk.
    """
    generic = [
        g
        for g in p.world.findall("geom")
        if g.get("name", "").startswith("europe3_mri_stored_phantom_pad_")
    ]
    if len(generic) != 6:
        raise ValueError(
            "MRI storage replacement expects the six original generic pads"
        )
    for geom in generic:
        p.world.remove(geom)
    white = (0.90, 0.90, 0.87, 1)
    blue = (0.48, 0.69, 0.72, 1)
    # Middle shelf: a pale blue seat with one open circular white accessory.
    p.box(
        "source_blue_insert",
        (-1.53, 0.49, 1.172),
        (0.20, 0.155, 0.014),
        blue,
        collision=True,
    )
    for layer, x in enumerate((-1.58, -1.53, -1.48)):
        for i in range(32):
            a, c = i * math.tau / 32, (i + 1) * math.tau / 32
            p.rod(
                "source_open_white_accessory",
                (x, 0.49 + 0.060 * math.cos(a), 1.272 + 0.060 * math.sin(a)),
                (x, 0.49 + 0.060 * math.cos(c), 1.272 + 0.060 * math.sin(c)),
                0.026,
                white,
                collision=True,
            )
    # An adjacent rounded white pad lies directly on the same shelf.
    p.rod(
        "source_white_pad",
        (-1.64, 0.97, 1.204),
        (-1.44, 0.97, 1.204),
        0.046,
        white,
        collision=True,
    )
    # Upper shelf: stacked folded/rounded padding, matching the visible mixed
    # white storage instead of assigning unobserved instrument identities.
    p.box(
        "source_folded_pad_base",
        (-1.53, 0.49, 1.690),
        (0.20, 0.17, 0.022),
        white,
        collision=True,
    )
    p.box(
        "source_folded_pad_top",
        (-1.53, 0.48, 1.729),
        (0.17, 0.145, 0.017),
        white,
        collision=True,
    )
    p.rod(
        "source_rounded_upper_pad",
        (-1.66, 0.94, 1.718),
        (-1.43, 0.94, 1.718),
        0.050,
        white,
        collision=True,
    )
