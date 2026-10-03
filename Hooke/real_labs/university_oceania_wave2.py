"""Three photographed Australian engineering facilities and bounded mechanics.

The machinery is independently authored from inspected exterior photographs.
Unreported dimensions and loads are estimates; no fluid or soil solver is implied.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

UWA = "https://ngcf.edu.au/"
UTS = "https://www.uts.edu.au/research/centres/robotics-institute/our-research/research-labs/infrastructure-robotics-lab"
UNSW = "https://www.unsw.edu.au/research/wrl/facilities-and-equipment"

TEAL = (0.015, 0.42, 0.49, 1)
BLUE = (0.055, 0.22, 0.48, 1)
STEEL = (0.53, 0.57, 0.59, 1)
YELLOW = (0.92, 0.66, 0.025, 1)
CONCRETE = (0.56, 0.56, 0.52, 1)


def _material(b, name, rgba):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=b.name + "__mat_" + name,
        rgba=numbers(rgba),
        specular="0.35",
        shininess="0.4",
    )


def _c72(b, base, params):
    _material(b, "teal", TEAL)
    # Four real tapered feet leave the photographed arch openings unobstructed.
    for x in (-0.95, 0.95):
        for y in (-0.76, 0.76):
            b.box(base, (x, y, 0.13), (0.40, 0.32, 0.13), "teal", collision=True)
            b.rod(
                base,
                (x, y, 0.25),
                (x * 0.49, y * 0.57, 1.72),
                0.25,
                "teal",
                collision=True,
            )
            for dx in (-0.22, 0.22):
                b.cylinder(base, (x + dx, y, 0.29), 0.035, 0.055, "bright")
    b.cylinder(base, (0, 0, 1.71), 0.69, 0.29, "teal", collision=True)
    b.cylinder(base, (0, 0, 2.035), 0.72, 0.035, "bright")
    # Photographed upright external motor and cooling fins, continuously supported.
    b.cylinder(base, (0, -1.11, 0.80), 0.28, 0.53, "teal", collision=True)
    b.box(base, (0, -1.11, 0.135), (0.34, 0.32, 0.135), "teal", collision=True)
    for i in range(18):
        a = i * math.tau / 18
        b.rod(
            base,
            (0.28 * math.cos(a), -1.11 + 0.28 * math.sin(a), 0.30),
            (0.28 * math.cos(a), -1.11 + 0.28 * math.sin(a), 1.22),
            0.015,
            "teal",
        )
    b.box(base, (0, -1.405, 0.92), (0.19, 0.04, 0.16), "teal")
    b.rod(base, (0, -1.11, 1.32), (0, -0.48, 1.62), 0.11, "dark")
    arm = b.moving(
        base,
        "beam_azimuth",
        (0, 0, 2.12),
        (0, 0, 1),
        (-0.16, 0.16),
        kind="hinge",
        mass=400,
        kp=16000,
        force=20000,
    )
    b.cylinder(arm, (0, 0, 0), 0.70, 0.05, "teal", collision=True)
    for y in (-0.58, 0.58):
        b.box(arm, (-1.13, y, 0.47), (3.80, 0.12, 0.38), "shell", collision=True)
        for x in (-3.5, -2.6, -1.7, -0.8, 0.1):
            b.cylinder(
                arm, (x, y, 0.50), 0.28, 0.09, "bright", euler=(math.pi / 2, 0, 0)
            )
            for dx in (-0.18, 0.18):
                b.cylinder(
                    arm,
                    (x + dx, y - 0.13, 0.75),
                    0.020,
                    0.018,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
    b.box(arm, (1.45, 0, 0.70), (0.32, 0.89, 0.70), "teal", collision=True)
    b.box(arm, (2.62, 0, 0.70), (0.09, 0.86, 0.56), "teal", collision=True)
    # Two stacks of separated counterweight discs along the arm direction.
    for y in (-0.47, 0.47):
        b.rod(arm, (1.70, y, 0.77), (2.72, y, 0.77), 0.10, "bright")
        for i in range(18):
            b.cylinder(
                arm,
                (1.79 + i * 0.045, y, 0.77),
                0.35,
                0.018,
                "metal",
                euler=(0, math.pi / 2, 0),
            )
    bucket = b.moving(
        arm,
        "basket_tilt",
        (-4.30, 0, 0.27),
        (0, 1, 0),
        (-0.12, 0.12),
        kind="hinge",
        mass=12,
        kp=1800,
        force=2500,
    )
    for y in (-0.61, 0.61):
        b.cylinder(bucket, (0, y, 0), 0.13, 0.07, "bright", euler=(math.pi / 2, 0, 0))
        b.box(bucket, (0, y, -0.63), (0.58, 0.055, 0.62), "shell", collision=True)
    b.box(bucket, (0, 0, -1.20), (0.58, 0.66, 0.055), "dark", collision=True)
    b.box(bucket, (-0.58, 0, -0.65), (0.055, 0.66, 0.55), "shell", collision=True)
    # Visible dry calibration block retained by opposing clamp pads.
    b.box(bucket, (0.03, 0, -0.99), (0.27, 0.27, 0.155), "cream", collision=True)
    for y in (-0.31, 0.31):
        b.box(bucket, (0.03, y, -1.02), (0.30, 0.04, 0.14), "metal")
    b.box(bucket, (0.03, 0, -0.832), (0.20, 0.025, 0.003), "dark")
    b.site(bucket, "inert_block", (0.03, 0, -0.99))
    # Upper service platform and mesh rails are carried by the rotating beam.
    for x in (-1.18, 0.36):
        for y in (-0.73, 0.73):
            b.rod(arm, (x, y, 0.84), (x, y, 2.37), 0.027, "teal")
    b.box(arm, (-0.41, 0, 1.20), (0.80, 0.78, 0.04), "teal")
    for y in (-0.75, 0.75):
        for z in (1.64, 2.06, 2.36):
            b.rod(arm, (-1.18, y, z), (0.36, y, z), 0.020, "teal")
        for x in (-0.94, -0.68, -0.42, -0.16, 0.1):
            b.rod(arm, (x, y, 1.25), (x, y, 2.36), 0.004, "metal")
    b.metadata["capabilities"] = [
        "bounded_beam_azimuth",
        "bounded_basket_tilt",
        "visible_clamped_inert_block",
    ]
    b.metadata["limitations"].append(
        "Quasistatic azimuth and basket attitude surrogates only. No rated spin speed, high-g soil model, balancing, drive dynamics or instrument certification. Published 10 m nominal rotor diameter guides the bay envelope; detailed lever lengths and travel are estimates."
    )


def _c72_room(world, definition):
    # Four ceiling panels retain a real square service hatch visible in the photo.
    for g in list(world.findall("geom")):
        if g.get("name", "").startswith("arch_roof") or g.get("name", "").startswith(
            "arch_luminaire"
        ):
            world.remove(g)
    for name, pos, size in (
        ("left", (-3.875, 0, 5.1), (2.125, 5.5, 0.07)),
        ("right", (3.875, 0, 5.1), (2.125, 5.5, 0.07)),
        ("front", (0, -3.625, 5.1), (1.75, 1.875, 0.07)),
        ("back", (0, 3.625, 5.1), (1.75, 1.875, 0.07)),
    ):
        box(world, "arch_roof_c72_" + name, size, pos, (0.88, 0.89, 0.86, 1))
    for x in (-3.6, 3.6):
        for y in (-3.1, 0, 3.1):
            box(
                world,
                f"arch_luminaire_c72_{x}_{y}",
                (0.55, 0.15, 0.025),
                (x, y, 5.0),
                (1, 0.98, 0.91, 1),
                collision=False,
            )
    for x in (-1.7, 1.7):
        box(
            world,
            f"arch_roof_c72_hatch_edge_{x}",
            (0.04, 1.74, 0.10),
            (x, 0, 5.17),
            STEEL,
        )


def _crane(b, base, params):
    _material(b, "bridge_yellow", YELLOW)
    _material(b, "runway_blue", BLUE)
    # Floor-supported columns and two continuous runways.
    for x in (-5.2, 5.2):
        for y in (-3.8, 3.8):
            b.box(base, (x, y, 0.045), (0.25, 0.24, 0.045), "metal", collision=True)
            b.box(
                base, (x, y, 2.02), (0.095, 0.11, 1.975), "runway_blue", collision=True
            )
            b.rod(base, (x, y, 3.20), (x, y * 0.82, 3.99), 0.045, "runway_blue")
        b.box(base, (x, 0, 4.01), (0.10, 4.1, 0.18), "runway_blue", collision=True)
        b.box(base, (x, 0, 4.21), (0.045, 4.1, 0.018), "metal")
    bridge = b.moving(
        base,
        "bridge_y",
        (0, 0, 4.42),
        (0, 1, 0),
        (-1.8, 1.8),
        mass=100,
        kp=12000,
        force=15000,
    )
    for y in (-0.28, 0.28):
        b.box(bridge, (0, y, 0), (5.34, 0.065, 0.20), "bridge_yellow", collision=True)
    for x in (-5.2, 5.2):
        b.box(bridge, (x, 0, -0.055), (0.20, 0.48, 0.09), "bridge_yellow")
        for y in (-0.35, 0.35):
            b.cylinder(
                bridge, (x, y, -0.09), 0.095, 0.04, "dark", euler=(0, math.pi / 2, 0)
            )
    trolley = b.moving(
        bridge,
        "trolley_x",
        (-2.65, 0, 0.14),
        (1, 0, 0),
        (-1.05, 1.05),
        mass=8,
        kp=1600,
        force=3000,
    )
    b.box(trolley, (0, 0, 0), (0.37, 0.39, 0.07), "bridge_yellow")
    b.cylinder(
        trolley, (0, 0, -0.22), 0.18, 0.25, "bridge_yellow", euler=(math.pi / 2, 0, 0)
    )
    for x in (-0.095, 0.095):
        b.rod(trolley, (x, 0, -0.25), (x, 0, -2.78), 0.010, "dark")
    b.cylinder(
        trolley, (0, 0, -2.79), 0.15, 0.04, "bridge_yellow", euler=(math.pi / 2, 0, 0)
    )
    b.ring(trolley, (0, 0, -3.00), 0.085, 0.018, "bright", plane="xz")
    # An explicitly authored inspection dummy is rigidly retained for motion checks.
    for x in (-0.24, 0.24):
        b.rod(trolley, (0, 0, -3.07), (x, 0, -3.26), 0.009, "dark")
        b.box(trolley, (x, 0, -3.45), (0.025, 0.20, 0.19), "metal", collision=True)
    b.box(trolley, (0, 0, -3.62), (0.25, 0.22, 0.02), "metal", collision=True)
    b.box(trolley, (0, 0, -3.46), (0.16, 0.15, 0.14), "cream", collision=True)
    b.box(trolley, (0, -0.152, -3.46), (0.07, 0.002, 0.06), "dark")
    b.site(trolley, "inspection_dummy", (0, 0, -3.46))
    b.metadata["capabilities"] = [
        "bridge_translation",
        "trolley_translation",
        "visible_suspended_inspection_dummy",
    ]
    b.metadata["limitations"].append(
        "Two mechanically positioned crane axes with fixed hoist length; cable sway, lifting dynamics, infrastructure-inspection robots and water physics are not modeled. The retained dummy is an authored calibration aid, not a claimed UTS robot."
    )


def _infrastructure_room(world, definition):
    # Remove the generic floor before constructing a genuine recessed test tank.
    for g in list(world.findall("geom")):
        if g.get("name", "").startswith("arch_floor"):
            world.remove(g)
    # Authored 5 x 6 x 1.5 m void has the published 45 kL capacity; dimensions are estimates.
    for name, pos, size in (
        ("left", (-5.625, 0, -0.10), (0.375, 5, 0.10)),
        ("right", (2.875, 0, -0.10), (3.125, 5, 0.10)),
        ("front", (-2.75, -4, -0.10), (2.50, 1, 0.10)),
        ("back", (-2.75, 4, -0.10), (2.50, 1, 0.10)),
        ("tank", (-2.75, 0, -1.60), (2.50, 3, 0.10)),
    ):
        box(world, "arch_floor_uts_" + name, size, pos, CONCRETE)
    for x in (-5.32, -0.18):
        box(world, f"uts_tank_wall_x_{x}", (0.07, 3.07, 0.79), (x, 0, -0.71), CONCRETE)
    for y in (-3.07, 3.07):
        box(
            world,
            f"uts_tank_wall_y_{y}",
            (2.50, 0.07, 0.79),
            (-2.75, y, -0.71),
            CONCRETE,
        )
    box(
        world,
        "uts_static_water",
        (2.495, 2.995, 0.32),
        (-2.75, 0, -0.88),
        (0.14, 0.33, 0.34, 0.28),
        collision=False,
    )
    # Rails bound the deep tank while leaving the dry right-side circulation route.
    for y in (-2.8, -1.4, 0, 1.4, 2.8):
        box(world, f"uts_guard_post_{y}", (0.025, 0.025, 0.55), (-0.04, y, 0.55), STEEL)
    for z in (0.55, 1.1):
        box(world, f"uts_guard_rail_{z}", (0.026, 2.88, 0.026), (-0.04, 0, z), STEEL)
    for x in (0.75, 1.58, 2.41):
        box(
            world,
            f"uts_tool_cart_{x}",
            (0.36, 0.29, 0.39),
            (x, 4.18, 0.48),
            (0.39, 0.40, 0.40, 1),
        )
        box(
            world,
            f"uts_tool_cart_base_{x}",
            (0.35, 0.28, 0.045),
            (x, 4.18, 0.045),
            STEEL,
        )
        for z in (0.27, 0.42, 0.57, 0.72):
            box(
                world,
                f"uts_tool_drawer_{x}_{z}",
                (0.32, 0.010, 0.065),
                (x, 3.879, z),
                (0.71, 0.72, 0.70, 1),
            )
            box(
                world,
                f"uts_tool_handle_{x}_{z}",
                (0.23, 0.018, 0.012),
                (x, 3.86, z),
                STEEL,
            )
    # Exposed industrial wall channels and supported inspection pipe sections.
    for y in (-1.7, 0, 1.7):
        for x in (3.7, 4.5):
            box(
                world,
                f"uts_fixture_foot_{x}_{y}",
                (0.04, 0.31, 0.42),
                (x, y, 0.42),
                STEEL,
            )
        box(world, f"uts_fixture_tray_{y}", (0.55, 0.34, 0.04), (4.1, y, 0.88), STEEL)
        ET.SubElement(
            world,
            "geom",
            name=f"uts_inspection_pipe_{y}",
            type="cylinder",
            size="0.18 0.47",
            pos=numbers((4.1, y, 1.10)),
            euler="0 1.5707963 0",
            rgba="0.67 0.67 0.62 1",
            contype="0",
            conaffinity="0",
        )


def _paddles(b, base, params):
    _material(b, "paddle_blue", BLUE)
    # All twenty independent piston paddles are mechanically controllable.
    b.box(base, (0, 0.27, 1.19), (5.2, 0.055, 0.085), "metal", collision=True)
    b.box(base, (0, 0.98, 1.19), (5.2, 0.055, 0.085), "metal", collision=True)
    for x in (-5.05, -2.525, 0, 2.525, 5.05):
        b.box(base, (x, 0.99, 0.60), (0.045, 0.045, 0.59), "metal", collision=True)
        b.box(base, (x, 1.04, 0.035), (0.12, 0.22, 0.035), "metal", collision=True)
    for i in range(20):
        x = (i - 9.5) * 0.50
        for dx in (-0.17, 0.17):
            b.rod(base, (x + dx, 0.22, 1.34), (x + dx, 0.98, 1.34), 0.012, "bright")
        b.cylinder(base, (x, 0.95, 1.58), 0.055, 0.12, "shell")
        b.box(base, (x, 0.85, 1.39), (0.09, 0.15, 0.07), "metal")
        b.rod(base, (x, 0.88, 1.67), (x, 1.18, 1.85), 0.006, "dark")
        paddle = b.moving(
            base,
            f"paddle_{i:02d}",
            (x, 0.47, 0),
            (0, 1, 0),
            (-0.10, 0.10),
            mass=2.5,
            kp=800,
            force=1600,
        )
        b.box(paddle, (0, 0, 0.54), (0.241, 0.035, 0.53), "metal", collision=True)
        b.box(paddle, (-0.17, -0.038, 0.51), (0.018, 0.004, 0.45), "paddle_blue")
        b.box(paddle, (-0.17, -0.043, 0.75), (0.004, 0.001, 0.08), "shell")
        b.box(paddle, (0, 0.04, 1.10), (0.24, 0.12, 0.065), "metal")
        for dx in (-0.17, 0.17):
            b.box(paddle, (dx, 0.21, 1.29), (0.04, 0.26, 0.055), "metal")
        b.rod(paddle, (0, 0.10, 1.38), (0, 0.58, 1.38), 0.014, "bright")
    # Static water occupies the front local view; its surface is below the guides.
    _material(b, "basin_water", (0.17, 0.27, 0.25, 0.37))
    b.box(base, (0, -1.96, 0.27), (5.30, 2.08, 0.265), "basin_water")
    # A supported visible reference gauge makes the local apparatus inspectable.
    b.box(base, (5.10, -0.48, 0.055), (0.10, 0.10, 0.055), "metal", collision=True)
    b.rod(base, (5.10, -0.48, 0.11), (5.10, -0.48, 1.1), 0.009, "bright")
    b.box(base, (5.10, -0.48, 0.89), (0.025, 0.005, 0.16), "cream")
    for i in range(8):
        b.box(base, (5.10, -0.487, 0.75 + i * 0.04), (0.018, 0.001, 0.002), "dark")
    b.site(base, "reference_gauge", (5.10, -0.48, 0.89))
    b.metadata["capabilities"] = [
        "twenty_independent_paddle_positions",
        "visible_reference_gauge",
    ]
    b.metadata["limitations"].append(
        "Piston motion is a bounded independent position surrogate. Real drive linkage and stroke are unverified. Static water does not generate waves or reproduce hydrodynamic forces, spectra or coastal measurements."
    )


def _wave_bay(world, definition):
    for i in range(94):
        x = -5.92 + i * 0.126
        box(
            world,
            f"wrl_wall_corrugation_{i}",
            (0.035, 0.022, 1.74),
            (x, 3.77, 1.78),
            (0.69, 0.69, 0.65, 1),
            collision=False,
        )
    for x in (-5.5, -2.75, 0, 2.75, 5.5):
        box(
            world,
            f"wrl_column_{x}",
            (0.075, 0.09, 1.8),
            (x, 3.65, 1.8),
            (0.33, 0.26, 0.24, 1),
        )
    for z in (1.4, 2.9):
        box(
            world,
            f"wrl_purlin_{z}",
            (5.9, 0.06, 0.055),
            (0, 3.64, z),
            (0.34, 0.28, 0.26, 1),
        )
    for x in (-5.5, 5.5):
        box(
            world,
            f"wrl_local_basin_edge_{x}",
            (0.13, 2.50, 0.39),
            (x, -0.10, 0.39),
            CONCRETE,
        )


BUILDERS = {
    "uwa_c72_beam_centrifuge": _c72,
    "uts_inspection_bridge_crane": _crane,
    "unsw_wrl_piston_array": _paddles,
}
SOURCES = {
    "uwa_c72_beam_centrifuge": dict(
        reference="UWA NGCF C72 installed centrifuge",
        url=UWA,
        dimensions_m=[9.5, 3.0, 4.6],
        dimension_basis="Published nominal rotor diameter 10 m; detailed lever lengths, bay, drive and motion limits estimated from C72 photos, not C61.",
    ),
    "uts_inspection_bridge_crane": dict(
        reference="UTS Infrastructure Robotics Laboratory panorama",
        url=UTS,
        dimensions_m=[11.2, 8.3, 4.7],
        dimension_basis="All crane geometry estimated from panorama. Published tank volume 45 kL does not establish dimensions; the chosen 5 x 6 x 1.5 m tank is an authored approximation.",
    ),
    "unsw_wrl_piston_array": dict(
        reference="UNSW WRL photographed segmented wave basin",
        url=UNSW,
        dimensions_m=[10.8, 5.3, 1.95],
        dimension_basis="Local paddle-bank dimensions and 0.20 m stroke estimated. No dimensions borrowed from the separate WRL flumes.",
    ),
}
SAMPLE_INTERFACES = {
    "uwa_c72_beam_centrifuge": (
        "inert_block",
        (0.54, 0.54, 0.31),
        "clamped",
        "Visible inert dry calibration block retained inside the hanging basket",
    ),
    "uts_inspection_bridge_crane": (
        "inspection_dummy",
        (0.32, 0.30, 0.28),
        "clamped",
        "Visible inert inspection dummy rigidly retained beneath fixed-length hoist cables",
    ),
    "unsw_wrl_piston_array": (
        "reference_gauge",
        (0.05, 0.01, 0.32),
        "clamped",
        "Visible static reference gauge fixed to a supported post beside the paddle bank",
    ),
}
FEATURES = {
    "uwa_ngcf_c72": _c72_room,
    "uts_infrastructure_robotics": _infrastructure_room,
    "unsw_wrl_wave_basin": _wave_bay,
}
