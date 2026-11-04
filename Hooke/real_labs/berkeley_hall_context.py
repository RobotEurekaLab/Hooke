"""Fixed Richmond shed details inferred from the official Berkeley photograph.

The image resolves corrugated roof sheets, fluorescent fixtures, wall services,
an aisle cabinet and a stool. Their dimensions and hidden attachments are
estimates. Nothing here changes the repaired carriage, water or its mechanics;
fixed electrical exteriors do not implement power, hydraulic or fluid systems.
Revision two adds source-visible ribbed cladding, red-brown steelwork and
asymmetric stored equipment. Colors are image-informed estimates, not spectra.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

PREFIX = "berkeley_context_"
STEEL = (0.20, 0.23, 0.20, 1)
PORTAL = (0.30, 0.14, 0.105, 1)
SHEET = (0.25, 0.28, 0.23, 1)
CREAM = (0.74, 0.76, 0.69, 1)
DARK = (0.035, 0.043, 0.038, 1)
GREEN = (0.22, 0.33, 0.25, 1)


def _name(key, roof=False):
    # The unchanged renderer places estimated emitters below this explicit
    # fixture role. Hangers and opaque housings retain the roof-cutaway role.
    if key.startswith("lamp_tube_"):
        return "arch_luminaire_" + PREFIX + key
    if key.startswith("right_wall_"):
        return "arch_cutaway_wall_" + PREFIX + key
    return ("arch_roof_" if roof else "") + PREFIX + key


def _box(
    world,
    key,
    pos,
    half,
    color=STEEL,
    *,
    roof=False,
    angle=None,
    contact=True,
    rotation=None,
):
    kwargs = {} if angle is None else {"euler": numbers((0, angle, 0))}
    if rotation is not None:
        kwargs["euler"] = numbers(rotation)
    return box(world, _name(key, roof), half, pos, color, collision=contact, **kwargs)


def _rod(world, key, start, end, radius, color=STEEL, *, roof=False, contact=True):
    return ET.SubElement(
        world,
        "geom",
        name=_name(key, roof),
        type="cylinder",
        fromto=numbers((*start, *end)),
        size=str(radius),
        rgba=numbers(color),
        contype="1" if contact else "0",
        conaffinity="1" if contact else "0",
    )


def _roof_point(side, u, y, w):
    angle = side * 0.20
    return (
        side * 1.55 + math.cos(angle) * u + math.sin(angle) * w,
        y,
        5.28 - math.sin(angle) * u + math.cos(angle) * w,
    )


def _roof(world):
    # Estimated end portals support the roof extension to the authored room
    # boundary. Existing seventeen portals and all instrument parts stay put.
    for portal_id, y in ((-34.0, -34.6), (34.0, 34.0)):
        # The front pair must stay outside the existing door approach.
        for side in (-1, 1):
            _box(
                world,
                f"end_column_{portal_id}_{side}",
                (side * 3.1, y, 2.55),
                (0.060, 0.075, 2.55),
                PORTAL,
            )
            _box(
                world,
                f"end_rafter_{portal_id}_{side}",
                (side * 1.55, y, 5.28),
                (1.62, 0.065, 0.075),
                PORTAL,
                roof=True,
                angle=side * 0.20,
            )
    # Existing portal rafters have local normal half-height 75 mm. Purlins
    # overlap them by 1 mm; the roof underside meets the purlin tops. Small
    # sheet/ridge overlaps represent laps, not disconnected decorative lines.
    for side in (-1, 1):
        angle = side * 0.20
        for index, u in enumerate((-1.25, 0.0, 1.25)):
            _box(
                world,
                f"purlin_{side}_{index}",
                _roof_point(side, u, 0, 0.099),
                (0.040, 35.01, 0.025),
                PORTAL,
                roof=True,
                angle=angle,
            )
        _box(
            world,
            f"roof_sheet_{side}",
            _roof_point(side, 0, 0, 0.134),
            (1.765, 35.01, 0.011),
            SHEET,
            roof=True,
            angle=angle,
        )
        for i in range(389):
            # Finite inward ribs join the continuous sheet. The first version
            # placed sparse folds outside the sheet, invisible from the hall.
            # Estimated 180 mm pitch is visibly closer to the source roof.
            _box(
                world,
                f"roof_fold_{side}_{i}",
                _roof_point(side, 0, -34.92 + i * 0.18, 0.116),
                (1.765, 0.022, 0.010),
                SHEET,
                roof=True,
                angle=angle,
            )
    _rod(
        world,
        "ridge_cap",
        (0, -35.01, 5.717),
        (0, 35.01, 5.717),
        0.068,
        SHEET,
        roof=True,
    )
    # The local reference shows two irregular rows. These repetitions preserve
    # the observed relationship, not measured fixture counts or photometry.
    for side, u in ((-1, 0.95), (1, -0.25)):
        x = _roof_point(side, u, 0, 0.10)[0]
        for index, y in enumerate(range(-29, 32, 8)):
            z = 4.70 if side < 0 else 4.57
            for end in (-0.58, 0.58):
                anchor = _roof_point(side, u, y + end, 0.123)
                _rod(
                    world,
                    f"lamp_hanger_{side}_{index}_{end}",
                    anchor,
                    (x, y + end, z + 0.035),
                    0.006,
                    roof=True,
                )
                # Crosswise clamps connect the hanging point to the nearest
                # continuous roof sheet, rather than ending in open air.
            _box(
                world,
                f"lamp_case_{side}_{index}",
                (x, y, z),
                (0.135, 0.73, 0.035),
                CREAM,
                roof=True,
            )
            for offset in (-0.064, 0.064):
                _rod(
                    world,
                    f"lamp_tube_{side}_{index}_{offset}",
                    (x + offset, y - 0.67, z - 0.043),
                    (x + offset, y + 0.67, z - 0.043),
                    0.012,
                    (0.88, 0.90, 0.82, 1),
                    roof=True,
                )
                for end in (-0.66, 0.66):
                    _box(
                        world,
                        f"lamp_holder_{side}_{index}_{offset}_{end}",
                        (x + offset, y + end, z - 0.031),
                        (0.019, 0.018, 0.024),
                        CREAM,
                        roof=True,
                    )


def _services(world):
    # The original visible festoon spans remain byte-for-byte unchanged. These
    # standoffs now give their fixed ends explicit attachments to a wall rail.
    _rod(world, "left_service_rail", (-3.065, -32, 4.27), (-3.065, 34, 4.27), 0.018)
    for index, y in enumerate(range(-30, 35, 4)):
        _rod(
            world,
            f"festoon_standoff_{index}",
            (-3.065, y, 4.27),
            (-2.90, y, 4.25),
            0.012,
        )
        _rod(
            world,
            f"festoon_eye_{index}",
            (-2.90, y - 0.025, 4.25),
            (-2.90, y + 0.025, 4.25),
            0.020,
            DARK,
        )
    _rod(world, "right_conduit", (3.045, -32, 3.65), (3.045, 32, 3.65), 0.014, CREAM)
    for index, y in enumerate((-27.7, -25.3, -16.8)):
        _box(world, f"wall_mount_{index}", (3.030, y, 2.35), (0.012, 0.36, 0.44))
        _box(world, f"control_box_{index}", (2.838, y, 2.35), (0.18, 0.32, 0.42), CREAM)
        _box(
            world,
            f"control_door_{index}",
            (2.650, y, 2.35),
            (0.008, 0.301, 0.401),
            (0.58, 0.61, 0.57, 1),
        )
        _box(
            world,
            f"idle_display_{index}",
            (2.640, y - 0.08, 2.49),
            (0.004, 0.115, 0.069),
            DARK,
        )
        _box(
            world,
            f"door_handle_{index}",
            (2.628, y + 0.24, 2.30),
            (0.020, 0.012, 0.082),
            DARK,
        )
        for row in range(2):
            for column in range(3):
                _rod(
                    world,
                    f"button_{index}_{row}_{column}",
                    (2.634, y - 0.15 + column * 0.10, 2.25 - row * 0.09),
                    (2.647, y - 0.15 + column * 0.10, 2.25 - row * 0.09),
                    0.017,
                    GREEN if column else DARK,
                )
        _rod(
            world,
            f"conduit_branch_{index}",
            (3.045, y, 3.65),
            (2.990, y, 3.65),
            0.014,
            CREAM,
        )
        _rod(
            world,
            f"conduit_drop_{index}",
            (2.990, y, 3.65),
            (2.990, y, 2.765),
            0.014,
            CREAM,
        )
        _rod(
            world,
            f"case_gland_{index}",
            (2.990, y, 2.75),
            (2.990, y, 2.79),
            0.021,
            DARK,
        )


def _cabinet(world):
    # The original raised aisle top is 0.85 m. Feet rest on it; the cabinet
    # occupies its outer edge, leaving a continuous minimum 0.87 m passage.
    x, y, floor = 1.86, -26.8, 0.85
    for dx in (-0.255, 0.255):
        for dy in (-0.43, 0.43):
            _box(
                world,
                f"cabinet_foot_{dx}_{dy}",
                (x + dx, y + dy, floor + 0.045),
                (0.03, 0.035, 0.045),
                DARK,
            )
    _box(world, "cabinet_plinth", (x, y, floor + 0.105), (0.29, 0.47, 0.025))
    _box(world, "cabinet_body", (x, y, floor + 0.44), (0.29, 0.47, 0.31), CREAM)
    _box(
        world,
        "cabinet_top",
        (x, y, floor + 0.78),
        (0.31, 0.49, 0.03),
        (0.53, 0.57, 0.55, 1),
    )
    for side in (-1, 1):
        _box(
            world,
            f"cabinet_door_{side}",
            (x - 0.299, y + side * 0.225, floor + 0.43),
            (0.009, 0.212, 0.282),
            CREAM,
        )
        _box(
            world,
            f"cabinet_handle_{side}",
            (x - 0.315, y + side * 0.07, floor + 0.48),
            (0.018, 0.010, 0.06),
            DARK,
        )
    _box(world, "bench_case", (x, y + 0.17, floor + 0.85), (0.19, 0.24, 0.04), CREAM)
    _box(
        world,
        "bench_case_face",
        (x - 0.195, y + 0.17, floor + 0.85),
        (0.005, 0.20, 0.029),
        DARK,
    )


def _stool(world):
    center, floor = (1.73, -30.1), 0.85
    x, y = center
    _rod(world, "stool_seat", (x, y, floor + 0.655), (x, y, floor + 0.692), 0.168, DARK)
    for side_x in (-1, 1):
        for side_y in (-1, 1):
            bottom = (x + side_x * 0.21, y + side_y * 0.21, floor + 0.015)
            top = (x + side_x * 0.10, y + side_y * 0.10, floor + 0.67)
            _rod(world, f"stool_leg_{side_x}_{side_y}", bottom, top, 0.015)
            _rod(
                world,
                f"stool_foot_{side_x}_{side_y}",
                (bottom[0], bottom[1], floor),
                (bottom[0], bottom[1], floor + 0.026),
                0.020,
                DARK,
            )
    level, offset = floor + 0.25, 0.171
    corners = [
        (x + a * offset, y + b * offset, level)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    for i in range(4):
        _rod(world, f"stool_stretcher_{i}", corners[i], corners[(i + 1) % 4], 0.010)


APPEARANCE_PREFIXES = (
    "richmond_column_",
    "arch_roof_richmond_rafter_",
    "richmond_wall_rib_",
    "towing_facility__tank_sidewall_",
    "towing_facility__tank_endwall_",
    "towing_facility__service_walkway_",
)


def _appearance(world):
    """Change fixed exterior color assignments only; leave water and rails alone.

    No liquid profile in the frozen renderer expresses source turbidity. The
    existing glass surface is therefore preserved, not called corrected water.
    """
    for geom in world.iter("geom"):
        name = geom.get("name", "")
        if name.startswith(("richmond_column_", "arch_roof_richmond_rafter_")):
            color = PORTAL
        elif name.startswith("richmond_wall_rib_"):
            color = (0.17, 0.20, 0.16, 1)
        else:
            continue
        # Removing a metal/cream role here is appearance-only: a painted aged
        # tank wall is not metallic chrome. Geometry, friction and masks stay.
        geom.attrib.pop("material", None)
        geom.set("rgba", numbers(color))


def apply_tank_appearance(base):
    """Assign fixed tank finishes after its geometry exists, never to liquid.

    Hall features run before equipment construction. This explicit local hook
    avoids pretending an earlier world traversal changed a later instrument.
    """
    changed = []
    for geom in base.findall("geom"):
        name = geom.get("name", "")
        if name.startswith(
            ("towing_facility__tank_sidewall_", "towing_facility__tank_endwall_")
        ):
            color = (0.13, 0.19, 0.12, 1)
        elif name.startswith("towing_facility__service_walkway_"):
            color = (0.47, 0.44, 0.35, 1)
        else:
            continue
        geom.attrib.pop("material", None)
        geom.set("rgba", numbers(color))
        changed.append(name)
    if len(changed) != 5:
        raise ValueError(
            "Expected the four fixed tank walls and single service walkway"
        )


def _steelwork(world):
    """Visible red-brown portal faces, knee braces and continuous wall girts."""
    for index, y in enumerate(range(-32, 33, 4)):
        for side in (-1, 1):
            _box(
                world,
                f"column_face_{index}_{side}",
                (side * 3.022, y, 2.55),
                (0.022, 0.13, 2.55),
                PORTAL,
            )
            _box(
                world,
                f"rafter_face_{index}_{side}",
                _roof_point(side, 0, y, -0.071),
                (1.62, 0.12, 0.014),
                PORTAL,
                roof=True,
                angle=side * 0.20,
            )
            _rod(
                world,
                f"portal_knee_{index}_{side}",
                (side * 3.055, y, 4.55),
                (side * 2.35, y, 5.075),
                0.035,
                PORTAL,
                roof=True,
            )
    for side in (-1, 1):
        for level in (2.45, 4.55):
            _box(
                world,
                f"wall_girt_{side}_{level}",
                (side * 3.075, 0, level),
                (0.045, 35, 0.055),
                PORTAL,
            )


def _cladding(world):
    """Dense finite ribs contact continuous linings; door aperture stays open.

    Pitch/depth and color variation are photographic estimates. These fixed
    plates are exterior presentation, not a sheet-buckling/material model.
    """
    for side in (-1, 1):
        tag = "right_wall_" if side > 0 else "left_wall_"
        # The source-facing lining attaches to the original wall at |x|=3.3.
        intervals = [(-35.0, -34.1), (-33.1, 35.0)] if side < 0 else [(-35.0, 35.0)]
        for part, (low, high) in enumerate(intervals):
            _box(
                world,
                f"{tag}lining_{part}",
                (side * 3.276, (low + high) / 2, 2.55),
                (0.025, (high - low) / 2, 2.55),
                SHEET,
            )
        for i in range(389):
            y = -34.92 + i * 0.18
            if side < 0 and -34.13 < y < -33.07:
                continue
            # Broad narrow-rib combination makes a real trapezoid-like fold,
            # with finite laps into the backing. No normal-map decals.
            tone = (
                0.22 + 0.008 * (i % 5),
                0.25 + 0.006 * (i % 5),
                0.205 + 0.006 * (i % 5),
                1,
            )
            _box(
                world,
                f"{tag}fold_{i}",
                (side * 3.242, y, 2.55),
                (0.013, 0.025, 2.55),
                tone,
            )
    # Source aisle continues to the right wall. Its elevation is the preexisting
    # authored datum, not a claim that the real tank is above ground.
    _box(
        world,
        "outer_aisle_infill",
        (2.751, 0, 0.425),
        (0.551, 32.2, 0.425),
        (0.47, 0.44, 0.35, 1),
    )


def _source_storage(world):
    """Source-visible foreground stock, rolls and a supported wall service rack."""
    floor = 0.85
    # Service cases remain externally supported even between portal stations;
    # their conduit is not used as the mechanical load path.
    for index, y in enumerate((-27.7, -25.3, -16.8)):
        for dy in (-0.23, 0.23):
            for dz in (-0.31, 0.31):
                _rod(
                    world,
                    f"case_standoff_{index}_{dy}_{dz}",
                    (3.301, y + dy, 2.35 + dz),
                    (3.017, y + dy, 2.35 + dz),
                    0.015,
                    STEEL,
                )
    # Wood-like calibration/support blocks: a small fixed stack, not generic
    # crates duplicated down the room. Each upper block rests on a lower one.
    for row in range(3):
        for column in range(3):
            _box(
                world,
                f"stored_block_{row}_{column}",
                (2.64 + column * 0.14, -31.05, floor + 0.029 + row * 0.058),
                (0.061, 0.13, 0.029),
                (0.32 + 0.03 * column, 0.20, 0.11, 1),
            )
    for column in range(2):
        _box(
            world,
            f"stored_block_cross_{column}",
            (2.78, -31.13 + column * 0.16, floor + 0.198),
            (0.22, 0.054, 0.024),
            (0.41, 0.28, 0.16, 1),
        )
    # Leaning green stock actually meets both the floor and wall lining. The
    # original source resolves rectangular beams, not their material/section.
    for index, (y, half, angle) in enumerate(
        ((-30.10, 1.44, 0.14), (-29.55, 0.91, 0.24))
    ):
        width = 0.027
        x = 3.249 - half * math.sin(angle) - width * math.cos(angle)
        z = floor + half * math.cos(angle) + width * math.sin(angle)
        _box(
            world,
            f"stored_green_beam_{index}",
            (x, y, z),
            (width, 0.105, half),
            (0.28, 0.37, 0.19, 1),
            angle=angle,
        )
    for index, (y, height) in enumerate(((-31.68, 0.47), (-31.48, 0.39))):
        _rod(
            world,
            f"stored_roll_{index}",
            (3.03, y, floor),
            (3.03, y, floor + height),
            0.064,
            (0.31, 0.35, 0.33, 1),
        )
    # Wall rack with real standoffs and two open spool rims, as in the source.
    for end in (-0.52, 0.52):
        _rod(
            world,
            f"right_wall_rack_standoff_{end}",
            (3.301, -28.9 + end, 2.50),
            (2.975, -28.9 + end, 2.50),
            0.023,
        )
    for level in (2.31, 2.69):
        _box(
            world,
            f"right_wall_rack_rail_{level}",
            (2.997, -28.9, level),
            (0.025, 0.575, 0.028),
            (0.42, 0.45, 0.40, 1),
        )
    for end in (-0.52, 0.52):
        _box(
            world,
            f"right_wall_rack_post_{end}",
            (2.997, -28.9 + end, 2.50),
            (0.025, 0.028, 0.20),
            (0.42, 0.45, 0.40, 1),
        )
    for reel, y in enumerate((-29.14, -28.72)):
        center = (2.952, y, 2.50)
        _rod(
            world,
            f"right_wall_reel_axle_{reel}",
            (3.004, y, 2.50),
            (2.920, y, 2.50),
            0.015,
            DARK,
        )
        # A backing tie joins the axle to the two horizontal rack rails.
        _box(
            world,
            f"right_wall_reel_tie_{reel}",
            (2.997, y, 2.50),
            (0.026, 0.022, 0.181),
            (0.40, 0.43, 0.38, 1),
        )
        for i in range(24):
            theta = 2 * math.pi * i / 24
            radius = 0.104
            _box(
                world,
                f"right_wall_reel_rim_{reel}_{i}",
                (
                    center[0],
                    y + radius * math.cos(theta),
                    center[2] + radius * math.sin(theta),
                ),
                (0.012, 0.016, 0.017),
                CREAM,
                rotation=(theta, 0, 0),
            )
        for i in range(3):
            theta = 2 * math.pi * i / 3
            _rod(
                world,
                f"right_wall_reel_spoke_{reel}_{i}",
                center,
                (
                    center[0],
                    y + 0.104 * math.cos(theta),
                    center[2] + 0.104 * math.sin(theta),
                ),
                0.008,
                CREAM,
            )
    # One red wall canister is visible in the source; identity and contents are
    # deliberately unspecified. It is supported storage, not a working device.
    _box(
        world,
        "right_wall_canister_bracket",
        (3.185, -26.35, 1.40),
        (0.12, 0.070, 0.025),
        STEEL,
    )
    _rod(
        world,
        "right_wall_red_canister",
        (3.10, -26.35, 1.43),
        (3.10, -26.35, 1.86),
        0.062,
        (0.43, 0.035, 0.025, 1),
    )
    _box(
        world,
        "right_wall_canister_lower_seat",
        (3.10, -26.35, 1.42),
        (0.070, 0.070, 0.011),
        STEEL,
    )
    _rod(
        world,
        "right_wall_canister_top",
        (3.10, -26.35, 1.86),
        (3.10, -26.35, 1.905),
        0.024,
        DARK,
    )


def add_context(world, definition):
    """Append fixed exteriors and explicitly scoped appearance-only assignments."""
    _appearance(world)
    _roof(world)
    _steelwork(world)
    _cladding(world)
    _services(world)
    _cabinet(world)
    _stool(world)
    _source_storage(world)
