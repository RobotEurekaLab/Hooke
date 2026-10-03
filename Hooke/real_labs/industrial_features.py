"""Reference-informed infrastructure for three larger experimental facilities.

Room envelopes and coordinates are design estimates, not surveyed floor plans.
NASA soil-bin dimensions and Waterloo's maglev footprint use published sizes.
Environmental processes and tracking equipment remain visual references.
"""

import math
import random
import xml.etree.ElementTree as ET


METAL = (0.47, 0.51, 0.52, 1)
DARK = (0.11, 0.14, 0.16, 1)
YELLOW = (0.91, 0.65, 0.06, 1)
PANEL = (0.80, 0.81, 0.76, 1)


def _numbers(values):
    return " ".join(f"{value:.6g}" for value in values)


def _body(parent, name, position=(0, 0, 0), rotation=(0, 0, 0)):
    return ET.SubElement(
        parent,
        "body",
        name=f"facility_{name}",
        pos=_numbers(position),
        euler=_numbers(rotation),
    )


def _geom(parent, name, shape, size, position, color, collision=False, **attributes):
    return ET.SubElement(
        parent,
        "geom",
        name=f"facility_{name}",
        type=shape,
        size=_numbers(size),
        pos=_numbers(position),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
        **attributes,
    )


def _box(parent, name, size, position, color, collision=False, **attributes):
    return _geom(parent, name, "box", size, position, color, collision, **attributes)


def _rod(parent, name, start, end, radius, color, collision=False):
    return ET.SubElement(
        parent,
        "geom",
        name=f"facility_{name}",
        type="capsule",
        size=str(radius),
        fromto=_numbers((*start, *end)),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
    )


def _monitor(parent, key, position):
    screen = _body(parent, key, position)
    _box(screen, f"{key}_foot", (0.16, 0.09, 0.012), (0, 0, 0.012), DARK)
    _box(screen, f"{key}_stand", (0.025, 0.025, 0.15), (0, 0.04, 0.16), METAL)
    _box(screen, f"{key}_housing", (0.27, 0.025, 0.17), (0, 0, 0.34), DARK)
    _box(
        screen,
        f"{key}_display",
        (0.251, 0.003, 0.15),
        (0, -0.028, 0.34),
        (0.06, 0.15, 0.22, 1),
    )
    for row in range(5):
        _box(
            screen,
            f"{key}_trace_{row}",
            (0.17 - row * 0.021, 0.001, 0.003),
            (-0.025, -0.032, 0.43 - row * 0.043),
            (0.25, 0.65, 0.65, 1),
        )


def _conveyor(parent, key, position, length, yaw=0):
    """Static twin-profile transport route, following the inspected AAPF still."""
    rail = _body(parent, key, position, (0, 0, yaw))
    for side in (-1, 1):
        _box(
            rail,
            f"{key}_edge_{side}",
            (0.042, length / 2, 0.065),
            (side * 0.25, 0, 0.40),
            METAL,
            collision=True,
        )
        # Extrusion slots and narrow drive strips leave the central service gap open.
        _box(
            rail,
            f"{key}_slot_{side}",
            (0.004, length / 2 - 0.02, 0.007),
            (side * 0.294, 0, 0.415),
            DARK,
        )
        _box(
            rail,
            f"{key}_belt_{side}",
            (0.038, length / 2, 0.012),
            (side * 0.16, 0, 0.457),
            DARK,
            collision=True,
        )
        _box(
            rail,
            f"{key}_guide_{side}",
            (0.012, length / 2, 0.008),
            (side * 0.222, 0, 0.474),
            (0.72, 0.75, 0.74, 1),
        )
    count = max(2, int(length / 0.95))
    for index in range(count):
        y = -length / 2 + 0.20 + index * (length - 0.40) / (count - 1)
        for side in (-1, 1):
            _box(
                rail,
                f"{key}_leg_{index}_{side}",
                (0.023, 0.023, 0.18),
                (side * 0.22, y, 0.205),
                METAL,
                collision=True,
            )
            _geom(
                rail,
                f"{key}_leveler_{index}_{side}",
                "cylinder",
                (0.012, 0.027),
                (side * 0.22, y, 0.030),
                (0.68, 0.70, 0.69, 1),
            )
            _box(
                rail,
                f"{key}_shoe_{index}_{side}",
                (0.065, 0.05, 0.005),
                (side * 0.22, y, 0.005),
                METAL,
            )
        _box(
            rail,
            f"{key}_crossbrace_{index}",
            (0.245, 0.018, 0.017),
            (0, y, 0.27),
            METAL,
        )
    for side in (-1, 1):
        y = side * (length / 2 - 0.09)
        _geom(
            rail, f"{key}_idler_{side}", "cylinder", (0.085, 0.03), (0, y, 0.441), DARK
        )
        _geom(
            rail,
            f"{key}_pulley_cap_{side}",
            "cylinder",
            (0.066, 0.008),
            (0, y, 0.48),
            (0.69, 0.71, 0.69, 1),
        )
    _box(
        rail,
        f"{key}_gearbox",
        (0.075, 0.075, 0.065),
        (0.32, -length / 2 + 0.20, 0.37),
        METAL,
    )
    _geom(
        rail,
        f"{key}_motor",
        "cylinder",
        (0.072, 0.085),
        (0.32, -length / 2 + 0.20, 0.505),
        (0.59, 0.63, 0.62, 1),
    )
    for rib in range(5):
        _geom(
            rail,
            f"{key}_motor_fin_{rib}",
            "cylinder",
            (0.078, 0.004),
            (0.32, -length / 2 + 0.20, 0.45 + rib * 0.022),
            METAL,
        )
    return rail


def _inline_mesh(asset, name, vertices, faces):
    ET.SubElement(
        asset,
        "mesh",
        name=name,
        vertex=_numbers(value for point in vertices for value in point),
        face=" ".join(str(index) for face in faces for index in face),
    )


def _crop_assets(root):
    """Original closed leaf ribbons and hollow tapered pot; no imported media."""
    asset = root.find("asset")
    if asset.find("./mesh[@name='facility_crop_pot_mesh']") is not None:
        return
    vertices, faces = [], []
    rings = ((0.13, 0.015), (0.177, 0.32), (0.170, 0.32), (0.125, 0.025))
    count = 32
    for radius, z in rings:
        vertices.extend(
            (
                radius * math.cos(i * 2 * math.pi / count),
                radius * math.sin(i * 2 * math.pi / count),
                z,
            )
            for i in range(count)
        )
    for ring in range(4):
        next_ring = (ring + 1) % 4
        for i in range(count):
            j = (i + 1) % count
            a, b, c, d = (
                ring * count + i,
                ring * count + j,
                next_ring * count + j,
                next_ring * count + i,
            )
            faces.extend(((a, b, c), (a, c, d)))
    _inline_mesh(asset, "facility_crop_pot_mesh", vertices, faces)
    for variety in range(3):
        for leaf in range(10):
            length = (0.75 - abs(leaf - 3) * 0.045) * (0.88 + variety * 0.08)
            width = (0.046 - leaf * 0.0018) * (0.88 + variety * 0.07)
            rise = 0.23 + leaf * 0.018
            drop = 0.25 - leaf * 0.020
            vertices, faces = [], []
            segments = 18
            for layer in (-1, 1):
                for step in range(segments + 1):
                    t = step / segments
                    half_width = 0.0008 + width * math.sin(math.pi * t) ** 0.75
                    center_y = 0.018 * math.sin(t * math.pi * 2) * t
                    center_z = rise * math.sin(math.pi * 0.84 * t) - drop * t * t
                    for side in (-1, 0, 1):
                        ridge = 0.006 * math.sin(math.pi * t) * (1 - abs(side))
                        vertices.append(
                            (
                                length * t,
                                center_y + side * half_width,
                                center_z + ridge + layer * 0.0006,
                            )
                        )
            surface = (segments + 1) * 3
            for step in range(segments):
                for col in range(2):
                    a = step * 3 + col
                    b, c, d = a + 1, a + 3, a + 4
                    faces.extend(
                        (
                            (a, b, c),
                            (b, d, c),
                            (a + surface, c + surface, b + surface),
                            (b + surface, c + surface, d + surface),
                        )
                    )
                for edge in (0, 2):
                    a, b = step * 3 + edge, (step + 1) * 3 + edge
                    if edge == 2:
                        a, b = b, a
                    faces.extend(((a, b, b + surface), (a, b + surface, a + surface)))
            for row in (0, segments):
                for col in range(2):
                    a, b = row * 3 + col, row * 3 + col + 1
                    if row == 0:
                        a, b = b, a
                    faces.extend(((a, b, b + surface), (a, b + surface, a + surface)))
            _inline_mesh(
                asset, f"facility_crop_leaf_mesh_{variety}_{leaf}", vertices, faces
            )
            if variety == 0:
                _inline_mesh(
                    asset,
                    f"facility_young_leaf_mesh_{leaf}",
                    [tuple(value * 0.40 for value in point) for point in vertices],
                    faces,
                )
    # Closed eight-sided reflector shell, open below around the recessed bulb.
    vertices, faces = [], []
    rings = ((0.25, -0.10), (0.115, 0.045), (0.108, 0.045), (0.243, -0.10))
    for radius, z in rings:
        vertices.extend(
            (radius * math.cos(i * math.pi / 4), radius * math.sin(i * math.pi / 4), z)
            for i in range(8)
        )
    for ring in range(4):
        for i in range(8):
            j = (i + 1) % 8
            a, b, c, d = (
                ring * 8 + i,
                ring * 8 + j,
                ((ring + 1) % 4) * 8 + j,
                ((ring + 1) % 4) * 8 + i,
            )
            faces.extend(((a, b, c), (a, c, d)))
    _inline_mesh(asset, "facility_growth_reflector_mesh", vertices, faces)


def _maize(parent, key, position, variety=0, yaw=0, collision=True, young=False):
    """Representative vegetative maize, with continuous arching tapered blades."""
    plant = _body(parent, key, position, (0, 0, yaw))
    _geom(
        plant,
        f"{key}_pot",
        "mesh",
        (),
        (0, 0, 0),
        (0.36, 0.37, 0.32, 1),
        mesh="facility_crop_pot_mesh",
        mass="0",
    )
    if collision:
        _geom(
            plant,
            f"{key}_pot_contact",
            "cylinder",
            (0.15, 0.16),
            (0, 0, 0.16),
            (0, 0, 0, 0),
            collision=True,
        )
    for z, radius in ((0.305, 0.178), (0.323, 0.182)):
        _geom(
            plant,
            f"{key}_rim_{z}",
            "cylinder",
            (radius, 0.008),
            (0, 0, z),
            (0.42, 0.43, 0.37, 1),
            mass="0",
        )
    _geom(
        plant,
        f"{key}_soil",
        "cylinder",
        (0.168, 0.006),
        (0, 0, 0.336),
        (0.22, 0.15, 0.075, 1),
        mass="0",
    )
    _box(
        plant,
        f"{key}_identity_tag",
        (0.025, 0.0015, 0.009),
        (0, -0.168, 0.28),
        (0.91, 0.88, 0.74, 1),
        mass="0",
    )
    height = (1.15 + variety * 0.15) * (0.60 if young else 1)
    _rod(
        plant,
        f"{key}_stem",
        (0, 0, 0.33),
        (0.025, 0, 0.33 + height),
        0.009,
        (0.31, 0.43, 0.085, 1),
    )
    for leaf in range(10):
        angle = (leaf % 2) * math.pi + 0.22 * math.sin(leaf * 1.3)
        z = 0.38 + leaf * height / 10
        _geom(
            plant,
            f"{key}_leaf_{leaf}",
            "mesh",
            (),
            (0.018 * leaf / 10, 0, z),
            (0.16 + 0.017 * (leaf % 3), 0.31 + 0.018 * variety, 0.035, 1),
            mesh=(
                f"facility_young_leaf_mesh_{leaf}"
                if young
                else f"facility_crop_leaf_mesh_{variety}_{leaf}"
            ),
            euler=_numbers((0, 0, angle)),
            mass="0",
        )
    return plant


def _growth_lamp(parent, row, column, x, y):
    key = f"growth_lamp_{row}_{column}"
    z = 3.95
    # Reflector geometry follows the dense recessed lamp array in the still;
    # lamp technology, output and spacing are not a calibrated lighting model.
    _box(
        parent,
        key + "_housing",
        (0.27, 0.26, 0.017),
        (x, y, z + 0.07),
        (0.72, 0.73, 0.67, 1),
    )
    _geom(
        parent,
        key + "_reflector",
        "mesh",
        (),
        (x, y, z),
        (0.83, 0.85, 0.82, 1),
        mesh="facility_growth_reflector_mesh",
    )
    _geom(
        parent,
        key + "_socket",
        "cylinder",
        (0.034, 0.026),
        (x, y, z + 0.014),
        (0.46, 0.43, 0.34, 1),
    )
    _geom(
        parent,
        f"growth_lamp_face_{row}_{column}",
        "ellipsoid",
        (0.032, 0.032, 0.043),
        (x, y, z - 0.036),
        (1, 0.91, 0.68, 1),
    )


def decorate_purdue_equipment(root):
    """Replace visual default specimen after equipment assembly, retaining axes."""
    turn = root.find(".//body[@name='phenotyping_booth__body_plant_turntable']")
    if turn is None:
        return
    for geom in list(turn.findall("geom")):
        # Keep the first collision-enabled turntable; specimen shapes are visual.
        if geom.get("contype", "0") == "0":
            turn.remove(geom)
    _maize(
        turn,
        "imaging_specimen",
        (0, 0, 0.025),
        variety=0,
        yaw=math.pi / 2,
        collision=False,
        young=True,
    )
    base = root.find(".//body[@name='phenotyping_booth']")
    if base is None:
        return
    for side in (-1, 1):
        for z in (0.83, 1.27, 1.71, 2.06):
            _box(
                base,
                f"imaging_light_bracket_{side}_{z}",
                (0.06, 0.036, 0.012),
                (side * 0.54, 0.34, z),
                METAL,
            )
            _box(
                base,
                f"imaging_light_housing_{side}_{z}",
                (0.027, 0.070, 0.055),
                (side * 0.51, 0.27, z),
                (0.56, 0.58, 0.55, 1),
            )
            _box(
                base,
                f"imaging_light_face_{side}_{z}",
                (0.005, 0.059, 0.045),
                (side * 0.48, 0.27, z),
                (0.94, 0.88, 0.72, 1),
            )
        _rod(
            base,
            f"imaging_light_cable_{side}",
            (side * 0.55, 0.39, 0.71),
            (side * 0.55, 0.39, 2.14),
            0.006,
            DARK,
        )
    for z in (0.82, 1.12, 1.42, 1.72, 2.02):
        _box(
            base, f"camera_chain_link_{z}", (0.025, 0.036, 0.065), (0.58, 0.13, z), DARK
        )


def _purdue(world, definition):
    # A cutaway growth chamber occupies the left half. Its front wall is omitted
    # for inspection; real enclosed chambers, rather than greenhouses, are shown.
    _box(
        world,
        "growth_chamber_divider",
        (0.07, 1.35, 2.0),
        (-0.45, 2.75, 2.0),
        PANEL,
        collision=True,
    )
    _box(
        world,
        "growth_door_header",
        (0.07, 0.80, 0.60),
        (-0.45, 0.60, 3.40),
        PANEL,
        collision=True,
    )
    _box(
        world,
        "growth_divider_low",
        (0.07, 0.72, 0.50),
        (-0.45, -0.93, 0.50),
        PANEL,
        collision=True,
    )
    for z in (0.10, 3.95):
        _box(
            world,
            f"chamber_east_rail_{z}",
            (0.08, 2.88, 0.022),
            (-0.45, 1.25, z),
            METAL,
        )
    for index, y in enumerate((-1.4, -0.3, 0.8, 1.9, 3.0, 4.1)):
        _box(
            world,
            f"insulated_panel_seam_{index}",
            (0.006, 0.015, 1.96),
            (-6.43, y, 2.0),
            (0.63, 0.65, 0.63, 1),
        )
    for row, x in enumerate((-5.40, -4.14, -2.88, -1.62)):
        _conveyor(world, f"growth_lane_{row}", (x, 1.05, 0), 5.5)
        for column in range(7):
            y = -1.10 + column * 0.74
            key = row * 7 + column
            _geom(
                world,
                f"pot_carrier_{key}",
                "cylinder",
                (0.21, 0.040),
                (x, y, 0.508),
                DARK,
                collision=True,
            )
            _box(
                world,
                f"carrier_red_guide_{key}",
                (0.043, 0.095, 0.016),
                (x + 0.335, y, 0.545),
                (0.70, 0.055, 0.025, 1),
            )
            _geom(
                world,
                f"carrier_red_post_{key}",
                "cylinder",
                (0.021, 0.052),
                (x + 0.335, y, 0.49),
                (0.70, 0.055, 0.025, 1),
            )
            _box(
                world,
                f"carrier_handle_slot_{key}",
                (0.022, 0.050, 0.002),
                (x + 0.335, y, 0.562),
                DARK,
            )
            _maize(
                world,
                f"crop_{key}",
                (x, y, 0.548),
                variety=(row + column) % 3,
                yaw=0.15 * math.sin(key * 1.7),
            )
    # Denser reflector array and suspension channels reproduce the dominant
    # ceiling rhythm seen in the growth-room photograph.
    for row in range(7):
        x = -5.88 + row * 0.76
        _box(
            world,
            f"growth_lighting_channel_{row}",
            (0.022, 2.87, 0.04),
            (x, 1.16, 4.07),
            METAL,
        )
        for column in range(8):
            _growth_lamp(world, row, column, x, -1.40 + column * 0.72)
    for y in (-1.58, 1.05, 3.69):
        _box(
            world,
            f"growth_light_crossmember_{y}",
            (2.70, 0.025, 0.025),
            (-3.50, y, 4.14),
            METAL,
        )
    _box(
        world,
        "growth_reflective_wall_band",
        (0.010, 2.77, 0.42),
        (-6.41, 1.25, 3.43),
        (0.72, 0.74, 0.71, 1),
    )
    ceiling = _box(
        world,
        "growth_chamber_ceiling",
        (2.94, 2.88, 0.025),
        (-3.47, 1.25, 4.22),
        (0.71, 0.73, 0.70, 1),
    )
    ceiling.set("name", "arch_roof_growth_chamber")
    # Low transfer rail outside the chamber; a deliberate aisle separates it
    # from the imaging station instead of a fake continuous powered transfer.
    _conveyor(world, "plant_loading_lane", (-3.60, -2.37, 0), 5.2, math.pi / 2)
    _box(
        world,
        "conveyor_aisle_stripe",
        (2.72, 0.028, 0.001),
        (-3.60, -2.84, 0.003),
        YELLOW,
    )
    _box(world, "floor_drain", (0.075, 2.25, 0.006), (0.65, 0.25, 0.007), DARK)
    for i in range(40):
        _box(
            world,
            f"drain_bar_{i}",
            (0.067, 0.012, 0.006),
            (0.65, -1.92 + i * 0.112, 0.014),
            METAL,
        )
    # Fertigation hardware is observed in the hallway photograph; the locations
    # below are an estimate, separate from the small pH-meter service bench.
    for index, x in enumerate((4.30, 5.42)):
        _geom(
            world,
            f"fertigation_tank_{index}",
            "cylinder",
            (0.40, 0.66),
            (x, 3.35, 0.68),
            (0.86, 0.87, 0.81, 1),
            collision=True,
        )
        _geom(
            world,
            f"fertigation_lid_{index}",
            "cylinder",
            (0.42, 0.035),
            (x, 3.35, 1.36),
            PANEL,
        )
        _rod(
            world, f"water_feed_{index}", (x, 3.35, 1.38), (x, 3.35, 2.25), 0.025, METAL
        )
        _rod(
            world,
            f"water_branch_{index}",
            (x, 3.35, 2.25),
            (x, 4.34, 2.25),
            0.025,
            METAL,
        )
    _box(
        world,
        "tank_bund_base",
        (1.11, 0.70, 0.010),
        (4.86, 3.35, 0.010),
        (0.41, 0.42, 0.36, 1),
    )
    for side in (-1, 1):
        _box(
            world,
            f"tank_bund_front_{side}",
            (1.15, 0.04, 0.075),
            (4.86, 3.35 + side * 0.70, 0.075),
            YELLOW,
        )
        _box(
            world,
            f"tank_bund_side_{side}",
            (0.04, 0.70, 0.075),
            (4.86 + side * 1.11, 3.35, 0.075),
            YELLOW,
        )
    _purdue_services(world)
    for x in (1.5, 3.2, 4.9):
        _box(world, f"overhead_service_tray_{x}", (0.08, 3.6, 0.06), (x, 0, 4.6), METAL)
    _rod(
        world,
        "hall_supply_duct",
        (-5.8, -3.65, 4.55),
        (5.8, -3.65, 4.55),
        0.20,
        (0.79, 0.80, 0.78, 1),
    )


def _purdue_services(world):
    """Observed service categories, with estimated pipe routes and positions."""
    for index in range(4):
        x = 5.75 + index * 0.14
        _rod(
            world,
            f"hall_service_conduit_{index}",
            (x, -3.15, 4.42),
            (x, 4.13, 4.42),
            0.016,
            (0.28, 0.30, 0.29, 1),
        )
        _rod(
            world,
            f"hall_service_drop_{index}",
            (x, 3.70, 4.42),
            (x, 3.70, 1.73 + index * 0.13),
            0.016,
            (0.28, 0.30, 0.29, 1),
        )
    for y in (-2.8, -1.2, 0.4, 2.0, 3.6):
        _box(
            world, f"conduit_bracket_{y}", (0.40, 0.017, 0.020), (5.95, y, 4.45), METAL
        )
    # Wall process panel, distribution manifold and valves are visual only.
    _box(
        world,
        "irrigation_wall_panel",
        (0.065, 0.37, 0.43),
        (6.36, 2.10, 1.98),
        (0.62, 0.66, 0.64, 1),
    )
    for index in range(4):
        y = 1.83 + index * 0.18
        _rod(
            world,
            f"irrigation_manifold_{index}",
            (6.16, y, 1.12),
            (6.16, y, 1.72),
            0.024,
            METAL,
        )
        _box(
            world,
            f"irrigation_valve_lever_{index}",
            (0.022, 0.065, 0.012),
            (6.12, y, 1.54),
            (0.67, 0.09, 0.045, 1),
        )
    station = _body(world, "wash_station", (5.95, 0.30, 0), (0, 0, -math.pi / 2))
    for x in (-0.48, 0.48):
        for y in (-0.26, 0.26):
            _box(
                station,
                f"wash_leg_{x}_{y}",
                (0.022, 0.022, 0.45),
                (x, y, 0.45),
                METAL,
                collision=True,
            )
    _box(station, "sink_rim_back", (0.55, 0.035, 0.035), (0, 0.30, 0.91), METAL)
    _box(station, "sink_rim_front", (0.55, 0.035, 0.035), (0, -0.30, 0.91), METAL)
    for side in (-1, 1):
        _box(
            station,
            f"sink_end_{side}",
            (0.065, 0.30, 0.12),
            (side * 0.485, 0, 0.825),
            METAL,
            collision=True,
        )
        _box(
            station,
            f"sink_bowl_side_{side}",
            (0.42, 0.013, 0.12),
            (0, side * 0.267, 0.825),
            METAL,
            collision=True,
        )
    _box(
        station,
        "sink_bowl_bottom",
        (0.42, 0.27, 0.01),
        (0, 0, 0.71),
        (0.47, 0.51, 0.49, 1),
        collision=True,
    )
    _geom(station, "sink_drain", "cylinder", (0.042, 0.003), (0, 0, 0.723), DARK)
    _rod(station, "tap_upright", (0, 0.30, 0.94), (0, 0.30, 1.20), 0.014, METAL)
    _rod(station, "tap_outlet", (0, 0.30, 1.20), (0, 0.08, 1.20), 0.014, METAL)
    _rod(station, "tap_downturn", (0, 0.08, 1.20), (0, 0.08, 1.14), 0.014, METAL)
    _rod(station, "sink_drain_down", (0, 0, 0.71), (0, 0, 0.40), 0.025, METAL)
    _rod(station, "sink_drain_trap", (0, 0, 0.40), (0, 0.23, 0.40), 0.025, METAL)
    _box(
        station,
        "sink_splashback",
        (0.55, 0.01, 0.17),
        (0, 0.335, 1.11),
        (0.76, 0.79, 0.76, 1),
    )
    _box(
        world,
        "growth_controller_housing",
        (0.032, 0.33, 0.40),
        (-0.365, 2.70, 1.87),
        (0.09, 0.18, 0.27, 1),
    )
    _box(
        world,
        "growth_controller_display",
        (0.006, 0.17, 0.105),
        (-0.327, 2.70, 1.97),
        (0.16, 0.31, 0.34, 1),
    )
    _rod(
        world,
        "growth_controller_conduit",
        (-0.38, 2.70, 2.29),
        (-0.38, 2.70, 3.79),
        0.012,
        METAL,
    )


def _waterloo(world, definition):
    roof = _box(
        world, "robohub_diffuser", (6, 5, 0.03), (0, 0, 5.38), (0.89, 0.91, 0.89, 1)
    )
    # Match the renderer's explicit cutaway convention; the flat roof is an
    # approximation of the bright overhead enclosure seen in the tour.
    roof.set("name", "arch_roof_robohub_diffuser")
    # Authored review lighting below the roof; fixture locations are not surveyed.
    for index, (x, y) in enumerate(((-3, -2.5), (3, -2.5), (-3, 2.5), (3, 2.5))):
        fixture = _box(
            world,
            f"robohub_light_{index}",
            (0.9, 0.3, 0.018),
            (x, y, 5.32),
            (0.96, 0.97, 0.96, 1),
        )
        fixture.set("name", f"arch_luminaire_robohub_{index}")
        mount = _box(
            world,
            f"robohub_light_mount_{index}",
            (0.6, 0.2, 0.02),
            (x, y, 5.35),
            METAL,
        )
        mount.set("name", f"arch_roof_robohub_light_mount_{index}")
    # The third glass facade is cut away, with its base and end posts retained.
    _box(world, "third_glazed_facade_sill", (0.04, 5, 0.045), (5.98, 0, 0.045), METAL)
    for y in (-4.95, 4.95):
        _box(
            world,
            f"facade_end_post_{y}",
            (0.035, 0.035, 2.7),
            (5.98, y, 2.7),
            METAL,
            collision=True,
        )
    for side in (-1, 1):
        _box(
            world,
            f"gantry_runway_{side}",
            (0.08, 4.25, 0.13),
            (side * 5.25, 0, 4.88),
            METAL,
            collision=True,
        )
        for y in (-4.2, 4.2):
            _box(
                world,
                f"gantry_column_{side}_{y}",
                (0.09, 0.09, 2.4),
                (side * 5.25, y, 2.4),
                METAL,
                collision=True,
            )
    _box(
        world,
        "powered_gantry_bridge",
        (5.3, 0.16, 0.20),
        (0, 2.5, 4.82),
        (0.74, 0.58, 0.13, 1),
        collision=True,
    )
    _box(world, "gantry_carriage", (0.25, 0.23, 0.16), (0.7, 2.5, 4.48), DARK)
    # 2 x 3 m is published by Waterloo; the installation's placement is inferred.
    _box(
        world,
        "maglev_2x3m",
        (1, 1.5, 0.002),
        (-3.45, -1.5, 0.003),
        (0.20, 0.23, 0.26, 1),
    )
    for i in range(7):
        _box(
            world,
            f"maglev_seam_x_{i}",
            (0.994, 0.002, 0.001),
            (-3.45, -3.0 + i * 0.5, 0.006),
            (0.49, 0.52, 0.56, 1),
        )
    for i in range(5):
        _box(
            world,
            f"maglev_seam_y_{i}",
            (0.002, 1.5, 0.001),
            (-4.45 + i * 0.5, -1.5, 0.006),
            (0.49, 0.52, 0.56, 1),
        )
    for side in (-1, 1):
        _box(
            world,
            f"workcell_bound_x_{side}",
            (1.35, 0.025, 0.001),
            (2.4, 0.60 + side * 1.40, 0.004),
            YELLOW,
        )
        _box(
            world,
            f"workcell_bound_y_{side}",
            (0.025, 1.4, 0.001),
            (2.4 + side * 1.35, 0.60, 0.004),
            YELLOW,
        )
    # Camera housings denote motion capture infrastructure, not calibrated views.
    for i, x in enumerate((-4.8, -2.4, 0, 2.4, 4.8)):
        _box(world, f"tracking_mount_{i}", (0.12, 0.08, 0.045), (x, 4.78, 4.5), DARK)
        _geom(
            world,
            f"tracking_lens_{i}",
            "cylinder",
            (0.043, 0.025),
            (x, 4.66, 4.48),
            (0.07, 0.17, 0.20, 1),
            euler=_numbers((math.pi / 2, 0, 0)),
        )
    for index, x in enumerate((-4.60, -3.82)):
        _monitor(world, f"telemetry_monitor_{index}", (x, 3.72, 0.78))
    _box(
        world,
        "network_rack",
        (0.28, 0.34, 0.85),
        (-5.42, 3.70, 0.85),
        DARK,
        collision=True,
    )
    for i in range(9):
        _box(
            world,
            f"rack_equipment_{i}",
            (0.24, 0.02, 0.051),
            (-5.42, 3.34, 0.19 + i * 0.16),
            (0.26, 0.30, 0.32, 1),
        )
    # A light grid is a visible spatial cue in the public spherical tour.
    for x in (-3.7, -1.85, 0, 1.85, 3.7):
        _box(
            world,
            f"roof_light_grid_x_{x}",
            (0.020, 4.5, 0.025),
            (x, 0, 5.24),
            (0.60, 0.64, 0.66, 1),
        )
    for y in (-3.6, -1.8, 0, 1.8, 3.6):
        _box(
            world,
            f"roof_light_grid_y_{y}",
            (5.0, 0.020, 0.025),
            (0, y, 5.24),
            (0.60, 0.64, 0.66, 1),
        )


def _soil_bin(parent, key, position, width, length, color, rotation=(0, 0, 0)):
    bin_body = _body(parent, key, position, rotation)
    _box(
        bin_body,
        f"{key}_soil",
        (width / 2, length / 2, 0.15),
        (0, 0, 0.15),
        color,
        collision=True,
        friction="1.1 .02 .003",
    )
    for side in (-1, 1):
        _box(
            bin_body,
            f"{key}_side_{side}",
            (0.075, length / 2 + 0.07, 0.22),
            (side * (width / 2 + 0.075), 0, 0.22),
            METAL,
            collision=True,
        )
        _box(
            bin_body,
            f"{key}_end_{side}",
            (width / 2, 0.075, 0.22),
            (0, side * (length / 2 + 0.075), 0.22),
            METAL,
            collision=True,
        )
    # Deterministic low-relief clumps break up the surface without pretending to
    # simulate grains. The rigid collision top remains exactly z=0.30 m; every
    # visible clump rises less than 3 mm and has collision disabled.
    rng = random.Random(20260930 + sum(map(ord, key)))
    for i in range(150):
        tint = rng.uniform(0.94, 1.035)
        _geom(
            bin_body,
            f"{key}_surface_{i}_soil",
            "ellipsoid",
            (
                rng.uniform(0.025, 0.10),
                rng.uniform(0.020, 0.075),
                rng.uniform(0.0008, 0.0025),
            ),
            (
                rng.uniform(-width / 2 + 0.12, width / 2 - 0.12),
                rng.uniform(-length / 2 + 0.12, length / 2 - 0.12),
                0.3002,
            ),
            tuple(value * tint for value in color[:3]) + (1,),
            euler=_numbers((0, 0, rng.uniform(-math.pi, math.pi))),
        )
    return bin_body


def _nasa(world, definition):
    # NASA ARES publishes nominal 12 x 3 m fixed tanks and a 6 x 5 m tilt tank.
    for key, y, color in (
        ("grc1_lane", -0.3, (0.60, 0.55, 0.44, 1)),
        ("sink_lane", -4.8, (0.74, 0.69, 0.57, 1)),
    ):
        _soil_bin(world, key, (-3.2, y, 0), 3, 12, color, (0, 0, math.pi / 2))
    tilt = _soil_bin(
        world, "tiltbed", (6.0, 3.6, 1.04), 5, 6, (0.69, 0.63, 0.50, 1), (0.30, 0, 0)
    )
    for side in (-1, 1):
        for y in (-2.8, -1.4, 0, 1.4, 2.8):
            _rod(
                tilt,
                f"tilt_guard_post_{side}_{y}",
                (side * 2.66, y, 0.30),
                (side * 2.66, y, 1.16),
                0.027,
                YELLOW,
                collision=True,
            )
        _rod(
            tilt,
            f"tilt_guard_top_{side}",
            (side * 2.66, -2.9, 1.16),
            (side * 2.66, 2.9, 1.16),
            0.027,
            YELLOW,
            collision=True,
        )
    for x in (4.1, 7.9):
        _rod(
            world,
            f"tilt_lift_{x}",
            (x, 5.15, 0.08),
            (x, 5.5, 1.55),
            0.09,
            METAL,
            collision=True,
        )
        _box(
            world,
            f"tilt_foot_{x}",
            (0.3, 0.35, 0.07),
            (x, 5.15, 0.07),
            DARK,
            collision=True,
        )
    # The articulated rover stays in a visible service pose, wheels clear of soil.
    _box(
        world,
        "rover_service_stand",
        (0.18, 0.16, 0.1675),
        (-4.70, -0.3, 0.4675),
        DARK,
        collision=True,
    )
    for y in (-6.58, -2.65, 1.50):
        _box(
            world,
            f"soil_aisle_stripe_{y}",
            (6.1, 0.035, 0.001),
            (-3.2, y, 0.003),
            YELLOW,
        )
    # High bay structural members and tracking array locations are estimated.
    for i, x in enumerate((-8.8, -4.4, 0, 4.4, 8.8)):
        # The nominal central column would obstruct the inferred loading-door
        # opening (-0.35 to 2.85 m); put its support beyond the right jamb.
        column_x = 3.3 if x == 0 else x
        _box(
            world,
            f"highbay_column_{i}",
            (0.13, 0.17, 3.0),
            (column_x, 7.72, 3.0),
            (0.65, 0.67, 0.64, 1),
            collision=True,
        )
        _box(
            world,
            f"highbay_roof_beam_{i}",
            (0.10, 7.6, 0.15),
            (x, 0, 5.72),
            (0.69, 0.70, 0.66, 1),
        )
        _box(world, f"tracking_camera_{i}", (0.07, 0.11, 0.07), (x, 7.45, 4.30), DARK)
    _rod(
        world,
        "highbay_ventilation_trunk",
        (-8.7, 6.5, 5.45),
        (8.7, 6.5, 5.45),
        0.30,
        (0.81, 0.82, 0.78, 1),
    )
    for i, x in enumerate((-6.4, -0.4, 5.6)):
        _rod(
            world,
            f"duct_drop_{i}",
            (x, 6.5, 5.45),
            (x, 3.2, 5.45),
            0.20,
            (0.81, 0.82, 0.78, 1),
        )
    # Soil handling storage is separated from sample-analysis worktops.
    for i in range(3):
        x = -8.35 + i * 0.70
        _geom(
            world,
            f"soil_drum_{i}",
            "cylinder",
            (0.26, 0.42),
            (x, 3.30, 0.42),
            (0.34, 0.39, 0.39, 1),
            collision=True,
        )
        _geom(
            world,
            f"soil_drum_lid_{i}",
            "cylinder",
            (0.275, 0.02),
            (x, 3.30, 0.86),
            METAL,
        )


def add_features(root, world, definition):
    """Add a facility's distinct structures; no-op for other catalogue entries."""
    builders = {
        "purdue_phenotyping": _purdue,
        "waterloo_robohub": _waterloo,
        "nasa_planetary": _nasa,
    }
    builder = builders.get(definition["id"])
    if builder:
        if definition["id"] == "purdue_phenotyping":
            _crop_assets(root)
        builder(world, definition)
