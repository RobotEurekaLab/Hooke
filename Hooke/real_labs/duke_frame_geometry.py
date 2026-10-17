"""Source-informed 830LE-AT exterior with inferred open mechanical interfaces.

The installed Duke photograph constrains appearance, not hidden bearing design.
The two original unloaded controls remain intact. The additional guard hinge is
an authored access proxy, not an identified OEM motor or safety interlock.
"""

import math

from real_labs.mounting import annular_mount

PREFIX = "biaxial_frame__duke_"


def _box(b, body, key, pos, half, material="metal", **kwargs):
    return b.box(
        body,
        pos,
        half,
        material,
        collision=True,
        name=f"{b.name}__duke_{key}",
        **kwargs,
    )


def _cylinder(b, body, key, pos, radius, half, material="bright", **kwargs):
    return b.cylinder(
        body,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=f"{b.name}__duke_{key}",
        **kwargs,
    )


def _bearing(b, body, key, pos, inner, outer, half, material="metal"):
    annular_mount(
        b,
        body,
        key,
        pos,
        inner,
        outer,
        half,
        axis="z",
        proxies=True,
        namespace="duke",
        material=material,
    )


def _open_frame(b, base):
    for x in (-0.34, 0.34):
        for y in (-0.28, 0.28):
            _box(
                b, base, f"foot_{x}_{y}", (x, y, 0.035), (0.065, 0.075, 0.035), "rubber"
            )
    _box(b, base, "base_cabinet", (0, 0, 0.365), (0.43, 0.35, 0.295), "dark")
    _box(
        b, base, "front_service_panel", (0, -0.355, 0.355), (0.35, 0.008, 0.235), "dark"
    )
    for side in (-1, 1):
        _box(
            b,
            base,
            f"panel_edge_{side}",
            (side * 0.36, -0.36, 0.355),
            (0.012, 0.008, 0.235),
        )
    for index in range(8):
        _box(
            b,
            base,
            f"base_vent_{index}",
            (-0.17, -0.365, 0.20 + index * 0.019),
            (0.115, 0.004, 0.003),
            "rubber",
        )
    _box(b, base, "base_deck", (0, 0, 0.70), (0.45, 0.37, 0.04))
    # 550 mm clear space, 70 mm diameter: 620 mm centre spacing. This is a
    # declared interpretation of a family option, not an installed measurement.
    for x in (-0.31, 0.31):
        _cylinder(b, base, f"column_{x}", (x, 0, 1.56), 0.035, 0.82)
        for z in (0.78, 1.66, 2.35):
            _cylinder(
                b, base, f"column_collar_{x}_{z}", (x, 0, z), 0.055, 0.032, "metal"
            )
    # Connected crosshead perimeter and open central guide. No solid block
    # occupies the ram's bore, even in parent-filtered collision queries.
    for y in (-0.18, 0.18):
        _box(b, base, f"crosshead_y_{y}", (0, y, 1.66), (0.37, 0.10, 0.055), "dark")
    for x in (-0.225, 0.225):
        _box(b, base, f"crosshead_x_{x}", (x, 0, 1.66), (0.145, 0.08, 0.055), "dark")
    _bearing(b, base, "ram_crosshead", (0, 0, 1.66), 0.0336, 0.081, 0.055)
    _bearing(b, base, "ram_neck", (0, 0, 1.735), 0.0336, 0.105, 0.055)
    _bearing(b, base, "drive_floor", (0, 0, 1.805), 0.0336, 0.19, 0.015, "dark")
    for side in (-1, 1):
        _box(
            b,
            base,
            f"drive_side_{side}",
            (side * 0.187, 0.02, 2.045),
            (0.018, 0.205, 0.255),
            "dark",
        )
        _box(
            b,
            base,
            f"drive_face_{side}",
            (0, 0.02 + side * 0.187, 2.045),
            (0.17, 0.018, 0.255),
            "dark",
        )
    _box(b, base, "drive_top", (0, 0.02, 2.285), (0.205, 0.205, 0.015), "dark")
    _box(b, base, "top_bridge", (0, 0, 2.35), (0.365, 0.22, 0.05))
    for index in range(6):
        _box(
            b,
            base,
            f"upper_vent_{index}",
            (0.206, 0.02, 2.09 + index * 0.021),
            (0.002, 0.105, 0.003),
            "rubber",
        )


def _original_axes(b, base):
    ram = b.moving(
        base,
        "axial_chuck_position",
        (0, 0, 1.47),
        (0, 0, -1),
        (0, 0.20),
        mass=0.6,
        kp=500,
    )
    _cylinder(b, ram, "ram", (0, 0, 0.30), 0.033, 0.30)
    _cylinder(b, ram, "upper_chuck", (0, 0, 0), 0.066, 0.06, "metal")
    for x in (-0.048, 0.048):
        _box(b, ram, f"upper_jaw_{x}", (x, -0.018, -0.05), (0.015, 0.04, 0.03), "dark")
    for x in (-0.045, 0.045):
        _cylinder(
            b,
            ram,
            f"upper_fastener_{x}",
            (x, -0.056, 0.01),
            0.006,
            0.010,
            euler=(math.pi / 2, 0, 0),
        )
    # The motor cavity, bearing and actual rotating shaft share a real bore.
    _bearing(b, base, "torsion_motor", (0, 0, 0.80), 0.0356, 0.125, 0.06, "dark")
    _bearing(b, base, "torsion_bearing", (0, 0, 0.8797), 0.0356, 0.09, 0.0197)
    lower = b.moving(
        base,
        "lower_chuck_rotation",
        (0, 0, 0.934),
        (0, 0, 1),
        (-0.60, 0.60),
        kind="hinge",
        mass=0.35,
        kp=160,
    )
    _cylinder(b, lower, "torsion_spindle", (0, 0, -0.085), 0.035, 0.06)
    _cylinder(b, lower, "lower_chuck", (0, 0, 0), 0.085, 0.034, "metal")
    for x in (-0.042, 0.042):
        _box(b, lower, f"lower_jaw_{x}", (x, 0, 0.057), (0.014, 0.04, 0.040), "dark")
    _box(b, lower, "inert_coupon", (0, 0, 0.131), (0.024, 0.012, 0.07), "orange")
    for z in (0.080, 0.182):
        _box(b, lower, f"coupon_end_{z}", (0, 0, z), (0.038, 0.014, 0.019), "orange")
    b.site(lower, "inert_coupon", (0, -0.015, 0.135))
    b.site(lower, "specimen_surface", (0, -0.012, 0.131))


def _guard(b, base):
    # Photograph-supported transparent enclosure; side/rear construction and
    # the powered front hinge are explicitly authored, unverified details.
    for x in (-0.25, 0.25):
        for y in (-0.26, 0.20):
            _box(b, base, f"guard_post_{x}_{y}", (x, y, 1.17), (0.010, 0.010, 0.43))
        for z in (0.86, 1.60):
            _box(
                b, base, f"guard_side_edge_{x}_{z}", (x, -0.03, z), (0.010, 0.24, 0.010)
            )
        _box(b, base, f"side_pane_{x}", (x, -0.03, 1.23), (0.003, 0.22, 0.35), "glass")
    for z in (0.86, 1.60):
        _box(b, base, f"guard_rear_edge_{z}", (0, 0.20, z), (0.25, 0.010, 0.010))
    _box(b, base, "rear_pane", (0, 0.20, 1.23), (0.24, 0.003, 0.35), "glass")
    # Offset the hinge ahead of the fixed front posts: closed panel and frame
    # have a 4 mm face clearance, rather than overlapping through parent masks.
    pivot = (-0.25, -0.285, 1.23)
    door = b.moving(
        base,
        "guard_opening",
        pivot,
        (0, 0, -1),
        (0, math.radians(100)),
        kind="hinge",
        mass=0.8,
        kp=90,
    )
    _box(b, door, "door_pane", (0.25, 0, 0), (0.23, 0.003, 0.35), "glass")
    for x in (0.03, 0.48):
        _box(b, door, f"door_vertical_{x}", (x, 0, 0), (0.008, 0.008, 0.36))
    for z in (-0.36, 0.36):
        _box(b, door, f"door_horizontal_{z}", (0.25, 0, z), (0.24, 0.008, 0.010))
    for z in (-0.30, 0.30):
        _bearing(b, door, f"door_hinge_{z}", (0, 0, z), 0.0046, 0.010, 0.022)
        _cylinder(
            b, base, f"door_pin_{z}", (pivot[0], pivot[1], pivot[2] + z), 0.004, 0.029
        )
        for side in (-1, 1):
            _box(
                b,
                base,
                f"hinge_lug_{z}_{side}",
                (-0.25, -0.2725, pivot[2] + z + side * 0.026),
                (0.014, 0.0195, 0.0034),
            )
        _box(b, door, f"hinge_leaf_{z}", (0.020, 0, z), (0.014, 0.007, 0.019))
    for z in (-0.065, 0.065):
        _cylinder(
            b,
            door,
            f"handle_post_{z}",
            (0.48, -0.025, z),
            0.005,
            0.017,
            euler=(math.pi / 2, 0, 0),
        )
    _cylinder(b, door, "door_handle", (0.48, -0.047, 0), 0.008, 0.075, "dark")


def _service_bundle(b, base, key, points, radius=0.006):
    for index, (start, end) in enumerate(zip(points, points[1:])):
        b.rod(
            base,
            start,
            end,
            radius,
            "rubber",
            collision=True,
            name=f"{b.name}__duke_{key}_segment_{index}",
        )


def _controller(b, base):
    center = 1.08
    for x in (center - 0.21, center + 0.21):
        for y in (-0.26, 0.21):
            _cylinder(
                b,
                base,
                f"cart_wheel_{x}_{y}",
                (x, y, 0.05),
                0.05,
                0.025,
                "rubber",
                euler=(0, math.pi / 2, 0),
            )
            _box(b, base, f"caster_fork_{x}_{y}", (x, y, 0.105), (0.014, 0.028, 0.055))
    _box(b, base, "cart_floor", (center, -0.025, 0.15), (0.27, 0.31, 0.02))
    _box(
        b,
        base,
        "controller_cabinet",
        (center, -0.025, 0.59),
        (0.25, 0.29, 0.42),
        "dark",
    )
    for x in (center - 0.26, center + 0.26):
        _box(b, base, f"cart_rail_{x}", (x, -0.025, 0.59), (0.01, 0.31, 0.44))
    _box(b, base, "cart_top", (center, -0.025, 1.045), (0.27, 0.31, 0.015))
    # No unsupported large monitor: the installed photo shows small stacked
    # tabletop control boxes and a largely black cart front.
    for x in (center - 0.14, center + 0.14):
        for y in (-0.08, 0.10):
            _box(
                b,
                base,
                f"control_foot_{x}_{y}",
                (x, y, 1.064),
                (0.018, 0.018, 0.004),
                "rubber",
            )
    _box(
        b, base, "small_controller", (center, 0.01, 1.113), (0.19, 0.13, 0.045), "cream"
    )
    _box(
        b,
        base,
        "small_display_bezel",
        (center - 0.045, -0.122, 1.121),
        (0.084, 0.003, 0.016),
        "dark",
    )
    _box(
        b,
        base,
        "small_display",
        (center - 0.045, -0.126, 1.121),
        (0.074, 0.001, 0.010),
        "screen",
    )
    for x in (center + 0.085, center + 0.14):
        _cylinder(
            b,
            base,
            f"controller_knob_{x}",
            (x, -0.131, 1.111),
            0.010,
            0.012,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    _box(
        b,
        base,
        "stop_bezel",
        (center - 0.16, -0.322, 0.92),
        (0.027, 0.008, 0.027),
        "orange",
    )
    _cylinder(
        b,
        base,
        "stop_button",
        (center - 0.16, -0.336, 0.92),
        0.017,
        0.014,
        "guard_red",
        euler=(math.pi / 2, 0, 0),
    )
    for z in (0.84, 0.93):
        _cylinder(
            b,
            base,
            f"cart_lamp_{z}",
            (center + 0.17, -0.32, z),
            0.010,
            0.006,
            "orange",
            euler=(math.pi / 2, 0, 0),
        )
    for i in range(7):
        _box(
            b,
            base,
            f"cart_vent_{i}",
            (center - 0.03, -0.317, 0.51 + i * 0.016),
            (0.10, 0.004, 0.003),
            "rubber",
        )
    for i in range(4):
        z = 0.30 + i * 0.07
        _cylinder(
            b,
            base,
            f"base_socket_{i}",
            (0.435, 0.17, z),
            0.015,
            0.012,
            "metal",
            euler=(0, math.pi / 2, 0),
        )
        _cylinder(
            b,
            base,
            f"cart_socket_{i}",
            (center - 0.257, 0.16, z),
            0.015,
            0.014,
            "metal",
            euler=(0, math.pi / 2, 0),
        )
        points = [
            (0.445, 0.17, z),
            (0.50, 0.17, z - 0.015),
            (0.56, 0.17, z - 0.055),
            (0.63, 0.165, z - 0.075),
            (0.71, 0.16, z - 0.055),
            (0.77, 0.16, z - 0.015),
            (center - 0.27, 0.16, z),
        ]
        _service_bundle(b, base, f"service_{i}", points)


def build(b, base, params):
    if params.get("mechanical_revision", 2) != 2:
        raise ValueError("Duke guarded frame requires mechanical_revision=2")
    _open_frame(b, base)
    _original_axes(b, base)
    _guard(b, base)
    _controller(b, base)
    b.metadata["mechanical_revision"] = 2
    b.metadata["capabilities"] = [
        "bounded unloaded upper-chuck positioning",
        "lower-chuck inert specimen indexing",
        "authored clear guard opening and closure",
    ]
    b.metadata["duke_dimensions"] = {
        "column_clear_width_m": 0.55,
        "column_diameter_m": 0.07,
        "column_center_spacing_m": 0.62,
        "basis": "550 mm nominal family option interpreted as clear working width; exact installed option not measured",
        "ram_radial_clearance_m": 0.0006,
        "torsion_radial_clearance_m": 0.0006,
        "bearing_basis": "Authored ideal-joint running gaps, not OEM tolerance or bearing stiffness",
    }
    b.metadata["limitations"].extend(
        [
            "Installed300x400 photograph and830L family brochure inform appearance; exact installed option and hidden interfaces are unverified.",
            "Powered guard is an authored access proxy; no certified interlock or OEM actuator is claimed.",
            "Original unloaded chuck remains at least55mm above the rigid coupon; no physical material loading, torque, stress-strain, fatigue or fracture model.",
        ]
    )
