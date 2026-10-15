"""Photo-informed dry coater assembly; original metre-scale geometry.

Only 300 mm maximum specimen capacity is documented. Enclosure, open guide
clearances, door hinge, lower head and 50 mm dry standoff are authored estimates.
The parked service loop and capped moving-head lines carry no fluid simulation.
"""

import math
import xml.etree.ElementTree as ET

DRY_WAFER_STANDOFF_M = 0.05


def _box(b, parent, key, pos, half, material="metal"):
    return b.box(
        parent, pos, half, material, collision=True, name=f"{b.name}__coater_{key}"
    )


def _cylinder(b, parent, key, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=f"{b.name}__coater_{key}",
        **kwargs,
    )


def _rod(b, parent, key, a, c, radius=0.008, material="metal"):
    return b.rod(
        parent, a, c, radius, material, collision=True, name=f"{b.name}__coater_{key}"
    )


def _sleeve(b, parent, key, center, axis, half_length, inner, wall=0.006):
    """Four real box walls leave a rectangular bore; never a solid guide block."""
    transverse = [i for i in range(3) if i != axis]
    a, c = transverse
    for sign in (-1, 1):
        pos = list(center)
        half = [0.0, 0.0, 0.0]
        pos[a] += sign * (inner[0] + wall / 2)
        half[axis] = half_length
        half[a] = wall / 2
        half[c] = inner[1] + wall
        _box(b, parent, f"{key}_side_{sign}", pos, half)
        pos = list(center)
        half = [0.0, 0.0, 0.0]
        pos[c] += sign * (inner[1] + wall / 2)
        half[axis] = half_length
        half[a] = inner[0]
        half[c] = wall / 2
        _box(b, parent, f"{key}_cap_{sign}", pos, half)


def _annulus(b, parent, key, z):
    """Watertight hinge sleeve surface plus independent convex contact sectors."""
    n = 32
    vertices = []
    faces = []
    for height in (z - 0.028, z + 0.028):
        for radius in (0.009, 0.018):
            vertices.extend(
                (
                    radius * math.cos(math.tau * i / n),
                    radius * math.sin(math.tau * i / n),
                    height,
                )
                for i in range(n)
            )
    quads = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )

    def mesh(name, points, triangles):
        name = f"{b.name}__coater_{name}_mesh"
        ET.SubElement(
            b.root.find("asset"),
            "mesh",
            name=name,
            vertex=" ".join(str(v) for p in points for v in p),
            face=" ".join(str(i) for t in triangles for i in t),
        )
        return name

    for i in range(n):
        j = (i + 1) % n
        ids = [i, n + i, n + j, j, 2 * n + i, 3 * n + i, 3 * n + j, 2 * n + j]
        triangles = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
        name = mesh(f"{key}_contact_{i}", [vertices[k] for k in ids], triangles)
        b.geom(
            parent,
            "mesh",
            (),
            mesh=name,
            material="dark",
            collision=True,
            group="3",
            name=f"{b.name}__coater_{key}_contact_{i}",
        )
        for q in (quads[0], quads[1], quads[3], quads[5]):
            p = [ids[k] for k in q]
            faces.extend(((p[0], p[1], p[2]), (p[0], p[2], p[3])))
    name = mesh(key + "_surface", vertices, faces)
    b.geom(
        parent,
        "mesh",
        (),
        mesh=name,
        material="dark",
        name=f"{b.name}__coater_{key}_surface",
    )


def _door(b, base):
    """Inferred physical hinge, pin clearance and bonded glass; not a seal model."""
    pivot = (-0.525, -0.49, 1.5)
    for z in (1.105, 1.895):
        _cylinder(
            b, base, f"hinge_pin_{z}", (pivot[0], pivot[1], z), 0.008, 0.065, "bright"
        )
        for sign in (-1, 1):
            h = z + sign * 0.05
            _cylinder(
                b,
                base,
                f"hinge_fixed_knuckle_{z}_{sign}",
                (pivot[0], pivot[1], h),
                0.018,
                0.012,
                "dark",
            )
            _box(
                b,
                base,
                f"hinge_fixed_stay_{z}_{sign}",
                (-0.515, -0.460, h),
                (0.008, 0.018, 0.012),
                "dark",
            )
            _box(
                b,
                base,
                f"hinge_fixed_leaf_{z}_{sign}",
                (-0.496, -0.446, h),
                (0.021, 0.012, 0.012),
                "dark",
            )
    door = b.moving(
        base,
        "access_door",
        pivot,
        (0, 0, -1),
        (0, 1.25),
        kind="hinge",
        mass=0.65,
        kp=420,
        force=100,
    )
    for z in (-0.395, 0.395):
        _annulus(b, door, f"hinge_sleeve_{z}", z)
        _box(
            b,
            door,
            f"hinge_moving_leaf_{z}",
            (0.0415, 0, z),
            (0.0265, 0.009, 0.023),
            "dark",
        )
    for x in (0.05, 0.9):
        _box(b, door, f"door_vertical_{x}", (x, 0, 0), (0.019, 0.014, 0.456), "cream")
    for z in (-0.475, 0.475):
        _box(
            b,
            door,
            f"door_horizontal_{z}",
            (0.475, 0, z),
            (0.425, 0.014, 0.019),
            "cream",
        )
    _box(b, door, "door_contact_glass", (0.475, 0, 0), (0.406, 0.005, 0.456), "glass")
    _rod(b, door, "door_handle", (0.9, -0.048, -0.1), (0.9, -0.048, 0.1), 0.012, "dark")
    for z in (-0.10, 0.10):
        _rod(
            b,
            door,
            f"handle_mount_{z}",
            (0.9, -0.014, z),
            (0.9, -0.048, z),
            0.01,
            "dark",
        )
    return door


def build_coater(b, base, params):
    b.metadata['inspection_softbox_visible_in_glass'] = False
    for x in (-0.57, 0.57):
        for y in (-0.40, 0.40):
            _box(
                b, base, f"foot_{x}_{y}", (x, y, 0.025), (0.065, 0.065, 0.025), "rubber"
            )
            _box(b, base, f"leg_{x}_{y}", (x, y, 0.435), (0.035, 0.035, 0.385), "cream")
        _box(b, base, f"table_side_{x}", (x, 0, 0.30), (0.035, 0.40, 0.025), "cream")
    _box(b, base, "table_rear", (0, 0.40, 0.30), (0.57, 0.025, 0.025), "cream")
    _box(b, base, "tabletop", (0, 0, 0.85), (0.67, 0.51, 0.03), "dark")
    _box(b, base, "plinth", (0, 0, 0.925), (0.55, 0.43, 0.045), "cream")
    _box(b, base, "rear_wall", (0, 0.40, 1.50), (0.55, 0.03, 0.53), "cream")
    for x in (-0.525, 0.525):
        _box(b, base, f"side_wall_{x}", (x, 0, 1.50), (0.025, 0.40, 0.53), "cream")
    b.housing(base, [(2.03, 0.55, 0.43, 0), (2.16, 0.47, 0.37, 0.035)], 0.025, "cream")
    _box(b, base, "roof", (0, 0, 2.045), (0.53, 0.40, 0.015), "cream")
    for x in (-0.49, 0.38):
        _box(
            b,
            base,
            f"fixed_front_side_{x}",
            (x, -0.45, 1.50),
            (0.018, 0.018, 0.49),
            "cream",
        )
    for z in (1.0, 2.0):
        _box(
            b,
            base,
            f"fixed_front_edge_{z}",
            (-0.055, -0.45, z),
            (0.435, 0.018, 0.015),
            "cream",
        )
    _box(
        b, base, "control_fascia", (0.465, -0.445, 1.50), (0.060, 0.025, 0.50), "cream"
    )
    for z, r, mat in (
        (1.86, 0.027, "guard_red"),
        (1.71, 0.017, "plant"),
        (1.57, 0.02, "dark"),
    ):
        _cylinder(
            b,
            base,
            f"panel_button_{z}",
            (0.465, -0.482, z),
            r,
            0.014,
            mat,
            euler=(math.pi / 2, 0, 0),
        )
    # One prominent controller and a separate smaller connector, as in the photo.
    _box(
        b, base, "hand_controller", (0.458, -0.503, 1.20), (0.058, 0.033, 0.079), "dark"
    )
    _box(
        b,
        base,
        "controller_display",
        (0.458, -0.54, 1.224),
        (0.039, 0.004, 0.027),
        "screen",
    )
    _box(
        b, base, "lead_connector", (0.510, -0.472, 1.24), (0.012, 0.015, 0.025), "metal"
    )
    for index, x in enumerate((0.444, 0.510)):
        points = []
        for i in range(45):
            t = i / 44
            points.append(
                (
                    x + 0.012 * math.cos(t * math.tau * 10),
                    -0.525 + 0.012 * math.sin(t * math.tau * 10),
                    1.125 - 0.40 * math.sin(math.pi * t),
                )
            )
        # Both endpoints return to fixed supported connectors; no hanging cut end.
        anchor = (x, -0.503, 1.14) if index == 0 else (x, -0.472, 1.215)
        _rod(b, base, f"lead_top_{index}", anchor, points[0], 0.004, "dark")
        for i, (a, c) in enumerate(zip(points, points[1:])):
            _rod(b, base, f"coiled_lead_{index}_{i}", a, c, 0.004, "dark")
        _box(
            b,
            base,
            f"lead_end_socket_{index}",
            (x, -0.484, 1.125),
            (0.013, 0.016, 0.012),
            "dark",
        )
        _rod(
            b, base, f"lead_end_{index}", points[-1], (x, -0.484, 1.125), 0.004, "dark"
        )
    _box(
        b, base, "floor_power_unit", (-0.37, 0.12, 0.235), (0.115, 0.235, 0.235), "dark"
    )
    for z in (0.26, 0.30, 0.34):
        _box(
            b,
            base,
            f"power_unit_vent_{z}",
            (-0.37, -0.118, z),
            (0.085, 0.003, 0.01),
            "metal",
        )
    for x in (-0.425, 0.425):
        _box(b, base, f"rear_post_{x}", (x, 0.26, 1.335), (0.025, 0.025, 0.365))
    for z in (1.51, 1.63):
        _box(b, base, f"x_rail_{z}", (0, 0.245, z), (0.44, 0.012, 0.018), "bright")
    slide = b.moving(
        base,
        "dry_head_x",
        (0, 0.215, 1.57),
        (1, 0, 0),
        (-0.21, 0.21),
        mass=0.35,
        kp=450,
    )
    _box(b, slide, "x_bridge", (0, 0.0546, 0), (0.13, 0.006, 0.105), "dark")
    for x in (-0.045, 0.045):
        for z in (-0.06, 0.06):
            _sleeve(
                b, slide, f"x_shoe_{x}_{z}", (x, 0.030, z), 0, 0.018, (0.0126, 0.0186)
            )
    for x in (-0.105, 0.105):
        _box(
            b,
            slide,
            f"y_rail_{x}",
            (x, -0.175, -0.025),
            (0.012, 0.225, 0.015),
            "bright",
        )
    carriage = b.moving(
        slide,
        "dry_head_y",
        (0, -0.19, -0.04),
        (0, 1, 0),
        (-0.13, 0.13),
        mass=0.22,
        kp=420,
    )
    _box(b, carriage, "y_plate", (0, 0, 0), (0.074, 0.07, 0.028))
    for x in (-0.105, 0.105):
        sign = 1 if x > 0 else -1
        for y in (-0.035, 0.035):
            _sleeve(
                b,
                carriage,
                f"y_shoe_{x}_{y}",
                (x, y, 0.015),
                1,
                0.025,
                (0.0126, 0.0156),
            )
            _box(
                b,
                carriage,
                f"y_shoe_lug_{x}_{y}",
                (sign * 0.0802, y, 0.006),
                (0.0062, 0.025, 0.0126),
            )
    # Layered supported head is inspired by the visible installed assembly.
    # Its lower endpoint has an authored 50 mm dry wafer clearance, not an OEM
    # nozzle working height; fixed original joint frames/ranges are unchanged.
    _box(
        b,
        carriage,
        "head_backplate",
        (-0.047, 0, -0.23),
        (0.012, 0.033, 0.205),
        "bright",
    )
    _cylinder(b, carriage, "upper_head", (0, 0, -0.105), 0.034, 0.075)
    _cylinder(b, carriage, "head_coupling", (0, 0, -0.1825), 0.039, 0.0075, "dark")
    _cylinder(b, carriage, "lower_head", (0, 0, -0.2675), 0.022, 0.0775, "bright")
    _cylinder(b, carriage, "lower_stem", (0, 0, -0.3925), 0.012, 0.0475, "metal")
    _cylinder(
        b, carriage, "coater_dry_nozzle", (0, 0, -0.4475), 0.006, 0.0075, "bright"
    )
    for z in (-0.08, -0.245, -0.39):
        _box(b, carriage, f"head_clamp_{z}", (-0.026, 0, z), (0.022, 0.025, 0.01))
    for sign in (-1, 1):
        x = sign * 0.032
        _box(
            b,
            carriage,
            f"capped_connector_{sign}",
            (x, -0.028, -0.21),
            (0.009, 0.011, 0.014),
            "dark",
        )
        _rod(
            b,
            carriage,
            f"connector_stand_{sign}",
            (sign * 0.012, -0.016, -0.21),
            (x, -0.028, -0.21),
            0.008,
        )
        points = [
            (x, -0.037, -0.21),
            (x * 1.5, -0.045, -0.18),
            (x * 1.5, -0.045, -0.08),
            (x, -0.028, -0.06),
        ]
        for i, (a, c) in enumerate(zip(points, points[1:])):
            _rod(b, carriage, f"local_capped_line_{sign}_{i}", a, c, 0.003, "cream")
        _box(
            b,
            carriage,
            f"upper_capped_port_{sign}",
            (x, -0.028, -0.06),
            (0.008, 0.010, 0.012),
            "dark",
        )
    # The installed head has an asymmetric stack of fittings and small loops.
    # These are attached, capped exterior services; no flexible supply between
    # fixed and moving bodies or coating process is implied.
    for index, (x, z, height) in enumerate(((0.046, -0.105, 0.018), (-0.042, -0.24, 0.014), (0.034, -0.335, 0.012))):
        _rod(b, carriage, f"valve_mount_{index}", (0, 0, z), (x, -0.045, z), 0.006, "bright")
        _box(b, carriage, f"valve_body_{index}", (x, -0.045, z), (0.013, 0.013, height), "metal")
        _cylinder(b, carriage, f"valve_cap_{index}", (x, -0.045, z + height + 0.004), 0.009, 0.004, "dark")
        _rod(b, carriage, f"valve_elbow_{index}", (x, -0.045, z - height), (x, -0.065, z - height), 0.004, "bright")
        end = (0.015 - 0.015 * index, -0.02, -0.37)
        points = [(x, -0.065, z - height), (x * 1.55, -0.070, z - 0.035), (x * 1.50, -0.060, -0.36), end]
        for segment, (a, c) in enumerate(zip(points, points[1:])):
            _rod(b, carriage, f"capillary_{index}_{segment}", a, c, 0.0022, "cream")
        _rod(b, carriage, f"capillary_terminal_{index}", end, (0, 0, -0.37), 0.0035, "bright")
    for z in (-0.075, -0.29):
        for sign in (-1, 1):
            b.cylinder(carriage, (-0.047, sign * 0.025, z), 0.004, 0.0015,
                       "dark", euler=(math.pi / 2, 0, 0), density=0)
    _cylinder(b, base, "platen", (0, 0, 0.997), 0.19, 0.027)
    _cylinder(b, base, "supported_300mm_wafer", (0, 0, 1.0245), 0.15, 0.0005, "lens")
    _box(
        b,
        base,
        "physical_marker",
        (0.07, -0.06, 1.026),
        (0.012, 0.009, 0.001),
        "orange",
    )
    b.site(base, "coater_wafer", (0.07, -0.06, 1.027))
    b.site(carriage, "dry_nozzle_tip", (0, 0, -0.455), size=0.002)
    # Source-visible curved cable density is represented by a supported PARKED
    # service loop. It has no pretend fixed-to-moving flexible connection.
    _box(b, base, "parked_loop_beam", (0, 0.32, 1.735), (0.425, 0.025, 0.02), "dark")
    for x in (-0.425, 0.425):
        _rod(b, base, f"parked_loop_stay_{x}", (x, 0.26, 1.67), (x, 0.32, 1.735), 0.014)
    points = [
        (-0.28, 0.32, 1.755),
        (-0.30, 0.32, 1.80),
        (-0.30, 0.32, 1.88),
        (-0.26, 0.32, 1.94),
        (-0.19, 0.32, 1.96),
        (0.19, 0.32, 1.96),
        (0.26, 0.32, 1.92),
        (0.28, 0.32, 1.84),
        (0.28, 0.32, 1.755),
    ]
    for i, (a, c) in enumerate(zip(points, points[1:])):
        _rod(b, base, f"parked_loop_spine_{i}", a, c, 0.018, "dark")
    for i in range(13):
        x = -0.18 + i * 0.03
        _box(
            b,
            base,
            f"parked_chain_rib_{i}",
            (x, 0.32, 1.96),
            (0.009, 0.025, 0.025),
            "dark",
        )
    _cylinder(b, base, "exhaust_stub", (0.28, 0.16, 2.475), 0.085, 0.325)
    for z in (2.2, 2.32, 2.44, 2.56, 2.68, 2.79):
        b.ring(base, (0.28, 0.16, z), 0.087, 0.003, "dark")
    _door(b, base)
    b.metadata["inspection_interface"] = {
        "surface_site": b.metadata["sites"]["coater_wafer"],
        "nozzle_tip_site": b.metadata["sites"]["dry_nozzle_tip"],
        "wafer_standoff_m": DRY_WAFER_STANDOFF_M,
        "marker_standoff_m": 0.048,
        "standoff_basis": "Authored dry geometry, not a calibrated spray height",
        "marker_thickness_m": 0.002,
        "specimen_attachment": "fixed inert wafer",
        "service_state": "parked fixed cable loop; capped local head lines; no moving fluid feed",
    }
    b.metadata["capabilities"] = [
        "dry transverse/fore-aft head positioning",
        "supported hinged physical access door",
        "measured dry marker registration and return",
    ]
    b.metadata["limitations"].append(
        "Installed photo informs the enclosure, head layering and parked services. Only 300 mm maximum specimen capacity is documented; other dimensions, guide clearances and door travel are authored. Original XY frames/ranges retained. Fifty millimetre dry wafer standoff is uncalibrated, not a spray specification. No liquid, aerosol, ultrasonic atomization, deposition, coating uniformity, process chemistry or safety interlock model. The wafer stays rigidly fixed; no loading/removal or grasp operation is simulated."
    )
