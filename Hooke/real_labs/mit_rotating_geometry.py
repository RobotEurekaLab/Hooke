"""Source-informed MIT room cues and authored dry inspection mechanics.

The public room photograph supplies color, apparatus and furniture cues, not
surveyed dimensions. Bearings, motorization and internal construction are
engineering estimates. This is not a rotating-fluid solver or an OEM replica.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers
from real_labs.mounting import annular_mount
from real_labs.optical_stages import aperture_plate


CAMERA = "rotating_table__inspection_camera"


def _name(b, key):
    return f"{b.name}__mit_{key}"


def _box(b, parent, key, pos, half, material="metal", **kwargs):
    return b.box(
        parent, pos, half, material, collision=True, name=_name(b, key), **kwargs
    )


def _cylinder(b, parent, key, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=_name(b, key),
        **kwargs,
    )


def _ring(b, parent, key, pos, inner, outer, half, material="bright"):
    annular_mount(
        b,
        parent,
        key,
        pos,
        inner,
        outer,
        half,
        axis="z",
        proxies=True,
        material=material,
        namespace="mit",
    )


def build_station(b, base, params, *, frustum):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=f"{b.name}__mat_paoc_green",
        rgba=".13 .25 .22 1",
        specular=".18",
        shininess=".12",
    )
    _cylinder(b, base, "foundation", (0, 0, 0.04), 0.49, 0.04, "rubber")
    frustum(b, base, 0.08, 0.78, 0.455, 0.23, "paoc_green")
    # A visibly cast pedestal, not a floating tabletop. The access panel and
    # ribs are fixed exterior context and do not imply a calibrated drive.
    for sign in (-1, 1):
        b.rod(
            base,
            (sign * 0.255, -0.15, 0.13),
            (sign * 0.19, -0.11, 0.69),
            0.021,
            "paoc_green",
            collision=True,
            name=_name(b, f"casting_rib_{sign}"),
        )
    _box(
        b,
        base,
        "service_panel",
        (0, -0.361, 0.37),
        (0.10, 0.008, 0.14),
        "dark",
        euler=(-0.31, 0, 0),
    )
    _ring(b, base, "spindle_bearing", (0, 0, 0.804), 0.0356, 0.12, 0.024)
    rotor = b.moving(
        base,
        "table_rotation",
        (0, 0, 0.855),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=3,
        kp=350,
        force=500,
    )
    _cylinder(b, rotor, "spindle", (0, 0, -0.049), 0.035, 0.025)
    _cylinder(b, rotor, "rotating_platter", (0, 0, 0), 0.46, 0.025, "dark")
    _cylinder(b, rotor, "tank_floor", (0, 0, 0.034), 0.401, 0.009, "bright")
    # Continuous transparent wall plus explicit contact sectors, all leaving
    # the cavity open. It contains no fluid or pressure-tight seal model.
    # The photographed vessel is tall relative to its diameter. These exterior
    # proportions replace the shallow dish; absolute dimensions remain inferred.
    _ring(b, rotor, "tank_glass", (0, 0, 0.293), 0.400, 0.412, 0.250, "glass")
    for index, z in enumerate((0.043, 0.543)):
        _ring(b, rotor, f"tank_rim_{index}", (0, 0, z), 0.399, 0.413, 0.006)
    _cylinder(b, rotor, "calibration_disc", (0.22, 0, 0.047), 0.045, 0.004, "orange")
    # A physically attached fiducial, with a thin visible face and fasteners.
    for index, (x, y) in enumerate(((-0.025, 0), (0.025, 0))):
        _cylinder(
            b,
            rotor,
            f"target_fastener_{index}",
            (0.22 + x, y, 0.0513),
            0.0025,
            0.0003,
            "dark",
        )
    for i in range(8):
        a = i * math.tau / 8
        b.rod(
            rotor,
            (0.16 * math.cos(a), 0.16 * math.sin(a), 0.045),
            (0.37 * math.cos(a), 0.37 * math.sin(a), 0.045),
            0.001,
            "dark",
            name=_name(b, f"graduation_{i}"),
        )
    b.site(rotor, "calibration_target", (0.22, 0, 0.051))
    # The rear apparatus uprights and green cross members echo the visible
    # photograph. Their positions leave the complete original rotor envelope.
    for sign in (-1, 1):
        x = sign * 0.88
        _box(
            b,
            base,
            f"portal_foot_{sign}",
            (x, 0.64, 0.025),
            (0.17, 0.20, 0.025),
            "paoc_green",
        )
        _box(
            b,
            base,
            f"portal_post_{sign}",
            (x, 0.64, 0.95),
            (0.032, 0.032, 0.925),
            "paoc_green",
        )
        _box(
            b,
            base,
            f"portal_arm_{sign}",
            (x, 0.3025, 1.85),
            (0.032, 0.3375, 0.032),
            "bright",
        )
        _box(
            b,
            base,
            f"rear_bracket_{sign}",
            (x, 0.695, 0.75),
            (0.032, 0.023, 0.032),
            "paoc_green",
        )
    _box(b, base, "rear_tie", (0, 0.74, 0.75), (0.912, 0.022, 0.032), "paoc_green")
    _box(b, base, "camera_rail", (0, 0, 1.85), (0.91, 0.035, 0.035), "bright")
    for sign in (-1, 1):
        _box(
            b,
            base,
            f"camera_stop_{sign}",
            (sign * 0.432, 0, 1.85),
            (0.010, 0.060, 0.055),
            "dark",
        )
    _box(
        b, base, "rear_upper_tie", (0, 0.64, 1.48), (0.848, 0.032, 0.024), "paoc_green"
    )
    _box(b, base, "drive_support", (0, 0.79, 0.774), (0.10, 0.05, 0.008), "paoc_green")
    _box(b, base, "drive_pedestal", (0, 0.84, 0.830), (0.10, 0.07, 0.048), "paoc_green")
    _cylinder(b, base, "drive_housing", (0, 0.84, 0.978), 0.090, 0.10, "dark")
    _box(b, base, "drive_cover", (0, 0.84, 1.098), (0.09, 0.09, 0.02), "paoc_green")
    # The photograph shows multiple dressed services on the tall frame.
    # These fixed routes connect drive-side fittings to the rear support and
    # stay behind the open tank and the full moving camera envelope.
    for index, dx in enumerate((-.07, -.025, .025, .07)):
        _cylinder(b, base, f"drive_socket_{index}", (dx, .86, 1.13),
                  .009, .012, "bright")
        points = [(dx, .86, 1.142), (dx, .91, 1.27),
                  (.76 + dx, .91, 1.40), (.88, .67, 1.52 + index * .055)]
        for segment, (a, c) in enumerate(zip(points, points[1:])):
            b.rod(base, a, c, .0045, "cream" if index % 2 else "dark",
                  name=_name(b, f"fixed_service_{index}_{segment}"))
        _box(b, base, f"frame_service_clip_{index}",
             (.88, .67, 1.52 + index * .055), (.04, .015, .012), "dark")
    camera = b.moving(
        base,
        "camera_traverse",
        (0, 0, 1.85),
        (1, 0, 0),
        (-0.34, 0.34),
        mass=0.4,
        kp=300,
    )
    sleeve = ET.SubElement(
        camera, "body", name=_name(b, "camera_sleeve"), euler=f"0 {math.pi/2} 0"
    )
    aperture_plate(
        b,
        sleeve,
        "mit_camera_bearing",
        (0, 0, 0),
        (0.050, 0.055, 0.080),
        (0.0356, 0.0356),
        "metal",
    )
    _box(b, camera, "camera_case", (0, 0, -0.10), (0.06, 0.045, 0.05), "dark")
    _cylinder(b, camera, "camera_lens", (0, 0, -0.167), 0.026, 0.017, "lens")
    b.site(camera, "camera_center", (0, 0, -0.184))
    ET.SubElement(
        camera,
        "camera",
        name=CAMERA,
        pos="0 0 -.184",
        quat="1 0 0 0",
        fovy="45",
        mode="fixed",
    )
    b.metadata["mounted_cameras"] = [
        dict(
            name=CAMERA,
            body=camera.get("name"),
            site=b.metadata["sites"]["camera_center"],
            local_position_m=[0, 0, -0.184],
            local_quaternion_wxyz=[1, 0, 0, 0],
            fovy_deg=45,
            purpose="dry calibration target inspection",
        )
    ]
    b.metadata["inspection_interface"] = dict(
        camera=CAMERA,
        target_geom=_name(b, "calibration_disc"),
        target_radius_m=0.045,
        target_orbit_radius_m=0.22,
        bearing_clearance_m=0.0006,
        mounting="rigid fastened calibration disc; no free sample or transfer",
        reachable_centered_rotation_rad=[0, math.pi],
        observation="Mounted camera pose and visible target projection, with offline CPU snapshots; no optical calibration",
    )
    b.metadata["capabilities"] = [
        "rotating-table dry target indexing",
        "mounted camera lateral positioning",
        "reachable endpoint alignment and geometric image capture",
    ]
    b.metadata["limitations"].append(
        "Photo-informed exterior and color cues; dimensions, internal bearings and motorization are authored. Ideal joints carry bearing loads with0.6 mm running clearance. No water, Coriolis, image formation or geophysical dynamics model; X-only camera cannot center the target at arbitrary rotor angles."
    )


def room_features(world, definition):
    """Private-photo cues: wood computer cabinet, asymmetry and overhead rails."""
    center = (1.55, -1.5)
    wood = (0.55, 0.40, 0.22, 1)
    cream = (0.72, 0.72, 0.64, 1)
    dark = (0.045, 0.055, 0.055, 1)

    def part(key, pos, half, color=wood):
        return box(world, "mit_" + key, half, pos, color)

    # Cabinet is supported by four actual wheel/fork assemblies. These are
    # static furnishings, not an authored rolling or articulated caster task.
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        x, y = center[0] + 0.51 * math.cos(a), center[1] + 0.51 * math.sin(a)
        ET.SubElement(
            world,
            "geom",
            name=f"mit_cart_wheel_{i}",
            type="cylinder",
            size=".08 .025",
            pos=numbers((x, y, 0.08)),
            euler="1.5707963268 0 0",
            rgba=numbers(dark),
        )
        for sign in (-1, 1):
            part(
                f"cart_fork_{i}_{sign}",
                (x, y + sign * 0.040, 0.125),
                (0.020, 0.010, 0.045),
                (0.30, 0.31, 0.29, 1),
            )
        part(
            f"cart_fork_top_{i}",
            (x, y, 0.175),
            (0.035, 0.05, 0.005),
            (0.30, 0.31, 0.29, 1),
        )
        ET.SubElement(
            world,
            "geom",
            name=f"mit_cart_axle_{i}",
            type="cylinder",
            size=".009 .05",
            pos=numbers((x, y, 0.08)),
            euler="1.5707963268 0 0",
            rgba=".3 .31 .29 1",
        )
    for z in (0.205, 0.86):
        ET.SubElement(
            world,
            "geom",
            name=f"mit_cart_round_shelf_{z}",
            type="cylinder",
            size=".69 .025",
            pos=numbers((*center, z)),
            rgba=numbers(dark if z > 0.8 else wood),
        )
    for sign in (-1, 1):
        part(
            f"cart_side_{sign}",
            (center[0] + sign * 0.52, center[1], 0.5325),
            (0.016, 0.38, 0.3025),
        )
    part("cart_back", (center[0], center[1] + 0.38, 0.5325), (0.504, 0.016, 0.3025))
    part("cart_divider", (center[0], center[1], 0.5325), (0.014, 0.364, 0.3025))
    part(
        "cart_door", (center[0] + 0.267, center[1] - 0.38, 0.51), (0.239, 0.016, 0.280)
    )
    part(
        "cart_door_handle",
        (center[0] + 0.080, center[1] - 0.407, 0.56),
        (0.007, 0.011, 0.055),
        dark,
    )
    part("cart_front_rail", (center[0], center[1] - 0.38, 0.809), (0.504, 0.016, 0.026))
    part(
        "cart_computer",
        (center[0] - 0.26, center[1] - 0.04, 0.4625),
        (0.105, 0.245, 0.2325),
        cream,
    )
    part(
        "cart_computer_face",
        (center[0] - 0.26, center[1] - 0.287, 0.4625),
        (0.095, 0.002, 0.220),
        (0.61, 0.63, 0.61, 1),
    )
    for k in range(7):
        part(
            f"cart_computer_vent_{k}",
            (center[0] - 0.26, center[1] - 0.290, 0.32 + k * 0.018),
            (0.064, 0.001, 0.002),
            dark,
        )
    part(
        "monitor_base",
        (center[0] - 0.17, center[1] + 0.10, 0.899),
        (0.17, 0.12, 0.014),
        dark,
    )
    part(
        "monitor_stem",
        (center[0] - 0.17, center[1] + 0.12, 0.976),
        (0.033, 0.032, 0.063),
        cream,
    )
    part(
        "monitor_case",
        (center[0] - 0.17, center[1] + 0.10, 1.19),
        (0.245, 0.073, 0.165),
        cream,
    )
    part(
        "monitor_screen",
        (center[0] - 0.17, center[1] + 0.026, 1.19),
        (0.211, 0.001, 0.13),
        (0.025, 0.065, 0.08, 1),
    )
    part(
        "keyboard",
        (center[0] - 0.13, center[1] - 0.22, 0.897),
        (0.22, 0.07, 0.012),
        (0.12, 0.13, 0.12, 1),
    )
    for j in range(4):
        for i in range(12):
            part(
                f"key_{j}_{i}",
                (center[0] - 0.325 + i * 0.034, center[1] - 0.272 + j * 0.032, 0.911),
                (0.013, 0.011, 0.002),
                cream,
            )
    # Tall right-hand blinds remain estimated architectural context.
    for y in (-1.7, -0.1, 1.5):
        box(
            world,
            f"mit_window_{y}",
            (0.015, 0.65, 0.8),
            (3.24, y, 1.8),
            (0.55, 0.72, 0.82, 0.3),
            collision=False,
        )
        for i in range(17):
            box(
                world,
                f"mit_blind_{y}_{i}",
                (0.06, 0.66, 0.01),
                (3.20, y, 1.02 + i * 0.095),
                (0.87, 0.85, 0.79, 1),
                collision=False,
            )
    # Heterogeneous wall cabinets and a supported open equipment shelf, drawn
    # from visible room categories, without pretending to identify each unit.
    for x, width in ((-2.1, 0.45), (0.10, 0.48), (1.18, 0.35)):
        part(
            f"wall_cabinet_{x}",
            (x, 2.46, 1.88),
            (width, 0.24, 0.38),
            (0.75, 0.71, 0.56, 1),
        )
        part(
            f"wall_cabinet_handle_{x}",
            (x + width - 0.055, 2.206, 1.84),
            (0.006, 0.014, 0.085),
            dark,
        )
    for x in (-1.53, -0.82):
        part(f"rack_post_{x}", (x, 2.42, 1.52), (0.024, 0.19, 0.62), cream)
    for z in (0.924, 1.36, 1.79, 2.116):
        part(f"rack_shelf_{z}", (-1.175, 2.42, z), (0.331, 0.19, 0.024), cream)
    for k, (x, z, h) in enumerate(
        ((-1.33, 1.11, 0.162), (-1.0, 1.09, 0.142), (-1.25, 1.53, 0.146))
    ):
        part(
            f"shelf_instrument_{k}",
            (x, 2.405, z),
            (0.11, 0.16, h),
            (0.31, 0.34, 0.33, 1),
        )
        part(
            f"shelf_display_{k}",
            (x, 2.243, z + 0.04),
            (0.068, 0.002, 0.032),
            (0.025, 0.075, 0.07, 1),
        )
    part("bench_monitor_base", (0.45, 2.23, 0.922), (0.18, 0.15, 0.022), dark)
    part("bench_monitor_stem", (0.45, 2.26, 1.034), (0.035, 0.04, 0.09), cream)
    part("bench_monitor", (0.45, 2.30, 1.285), (0.22, 0.13, 0.161), cream)
    part(
        "bench_monitor_face",
        (0.45, 2.167, 1.285),
        (0.188, 0.003, 0.13),
        (0.045, 0.055, 0.06, 1),
    )
    # Photo shows suspended rails and a projector. All supports terminate on
    # the ceiling; this is static context, not an extra moving camera.
    for x in (-0.85, 0.15):
        part(
            f"overhead_rail_{x}",
            (x, 0.1, 2.72),
            (0.035, 1.2, 0.035),
            (0.44, 0.45, 0.42, 1),
        )
        for y in (-0.8, 1.0):
            part(
                f"rail_hanger_{x}_{y}",
                (x, y, 2.9275),
                (0.017, 0.017, 0.1725),
                (0.59, 0.60, 0.57, 1),
            )
    part(
        "overhead_bridge",
        (-0.35, -0.55, 2.66),
        (0.535, 0.09, 0.025),
        (0.27, 0.28, 0.26, 1),
    )
    part(
        "projector_mount",
        (-0.35, -0.55, 2.565),
        (0.035, 0.045, 0.070),
        (0.25, 0.26, 0.25, 1),
    )
    part("projector", (-0.35, -0.55, 2.425), (0.15, 0.18, 0.07), (0.28, 0.30, 0.28, 1))
    ET.SubElement(
        world,
        "geom",
        name="mit_projector_lens",
        type="cylinder",
        size=".041 .012",
        pos="-.35 -.742 2.425",
        euler="1.5707963268 0 0",
        rgba=".025 .045 .05 1",
    )
