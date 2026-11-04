"""Authored open carriage and mounting interfaces for the Richmond tank.

The official photograph establishes the tank and its rails, not the distant
carriage's hidden construction. These are explicitly estimated mechanics.
Published tank dimensions and the existing motion/inertia contracts are kept.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.optical_stages import aperture_plate


def _name(b, key):
    return f"{b.name}__berkeley_{key}"


def _box(b, body, key, pos, half, material="metal"):
    return b.box(body, pos, half, material, collision=True, name=_name(b, key))


def _rolling_wheel(b, carriage, side, end, x, y):
    """Passive integral wheel/shaft, supported by two open bearing housings.

    Authored mass is 90 g for the tire/hub cylinder and 10 g for the shaft.
    Their analytic cylinder inertias are summed about the shared center. These
    are explicit surrogate values, not surveyed carriage specifications.
    """
    key = f"wheel_{side}_{end}"
    wheel = ET.SubElement(
        carriage,
        "body",
        name=_name(b, key),
        pos=f"{x} {y} -0.15",
        gravcomp="0",
    )
    axial = 0.5 * (0.09 * 0.10**2 + 0.01 * 0.012**2)
    transverse = (
        0.09 * (3 * 0.10**2 + 0.124**2) + 0.01 * (3 * 0.012**2 + 0.174**2)
    ) / 12
    ET.SubElement(
        wheel,
        "joint",
        name=_name(b, key + "_radial_play"),
        type="slide",
        axis="0 0 1",
        limited="true",
        range="-0.0003 0.0003",
        damping="0.02",
        armature="0",
    )
    ET.SubElement(
        wheel,
        "inertial",
        pos="0 0 0",
        mass="0.1",
        diaginertia=f"{axial:.16g} {transverse:.16g} {transverse:.16g}",
    )
    ET.SubElement(
        wheel,
        "joint",
        name=_name(b, key + "_spin"),
        type="hinge",
        axis="1 0 0",
        limited="false",
        damping="0.0001",
        armature="0",
    )
    # Exact axis quaternion avoids the shared primitive helper's rounded Euler
    # serialization. Rotation about X preserves the cylindrical rail envelope.
    orientation = "0.7071067811865476 0 0.7071067811865476 0"
    b.cylinder(
        wheel,
        (0, 0, 0),
        0.10,
        0.062,
        "dark",
        quat=orientation,
        collision=True,
        priority="1",
        solref="0.004 1",
        margin="0.000005",
        name=b.unique("carriage_wheel"),
    )
    b.cylinder(
        wheel,
        (0, 0, 0),
        0.012,
        0.087,
        "bright",
        quat=orientation,
        collision=True,
        name=_name(b, key + "_shaft"),
    )
    for sign in (-1, 1):
        bearing_key = f"{key}_bearing_{sign}"
        housing = ET.Element("body")
        aperture_plate(
            b,
            housing,
            "berkeley_" + bearing_key,
            (0, 0, 0),
            (0.028, 0.028, 0.012),
            (0.0126, 0.0126),
            "metal",
        )
        # Fixed housing mass remains represented by the unchanged carriage
        # inertia; only the four independent wheel/shaft inertias are added.
        for geom in housing.findall("geom"):
            local_x, local_y, local_z = map(float, geom.get("pos").split())
            geom.set(
                "pos", f"{x + sign * 0.075 + local_z} {y + local_y} {-0.15 - local_x}"
            )
            geom.set("quat", orientation)
            geom.set("density", "0")
            carriage.append(geom)
        _box(
            b,
            carriage,
            bearing_key + "_stalk",
            (x + sign * 0.075, y, -0.0835),
            (0.012, 0.020, 0.0385),
        )


def build_carriage(b, base):
    """Keep the original two body frames, controls, mass and inertia."""
    carriage = b.moving(
        base,
        "carriage_travel",
        (0, -22, 2.22),
        (0, 1, 0),
        (0, 8.0),
        mass=5,
        kp=1100,
        force=1200,
    )
    aperture_plate(
        b,
        carriage,
        "berkeley_platform",
        (0, 0, 0.0025),
        (1.49, 0.43, 0.0475),
        (0.075, 0.075),
        "metal",
    )
    # A 5 mm relieved underside clears rotating wheel crowns. The full-depth
    # perimeter retains the original 100 mm exterior deck envelope.
    aperture_plate(
        b,
        carriage,
        "berkeley_platform_lower_rim",
        (0, 0, -0.0475),
        (1.49, 0.43, 0.0025),
        (1.43, 0.37),
        "metal",
    )
    # Four passive wheel spins add actual rail-driven rotation. The original
    # ideal carriage guide still constrains all motion except translation.
    for side, x in enumerate((-1.3, 1.3)):
        for end, y in enumerate((-0.28, 0.28)):
            _rolling_wheel(b, carriage, side, end, x, y)
        _box(
            b,
            carriage,
            f"guard_post_{side}",
            (x, -0.32, 0.28),
            (0.025, 0.025, 0.23),
            "warning",
        )
    b.rod(
        carriage,
        (-1.3, -0.32, 0.51),
        (1.3, -0.32, 0.51),
        0.025,
        "warning",
        collision=True,
        name=_name(b, "supported_guard_handle"),
    )
    b.site(carriage, "supported_carriage_handle", (0, -0.32, 0.51), size=0.035)
    for index, x in enumerate((-0.16, 0.16)):
        b.cylinder(
            carriage,
            (x, 0, 0.062),
            0.030,
            0.012,
            "dark",
            collision=True,
            name=_name(b, f"guide_foot_{index}"),
        )
        b.cylinder(
            carriage,
            (x, 0, 0.367),
            0.014,
            0.293,
            "bright",
            collision=True,
            name=_name(b, f"lift_guide_{index}"),
        )
    _box(b, carriage, "guide_top_bridge", (0, 0, 0.682), (0.20, 0.060, 0.022), "blue")
    specimen = b.moving(
        carriage,
        "model_immersion",
        (0, 0, -0.32),
        (0, 0, -1),
        (0, 0.35),
        mass=0.4,
        kp=600,
    )
    b.geom(
        specimen,
        "ellipsoid",
        (0.14, 0.55, 0.12),
        material="orange",
        collision=True,
        name=_name(b, "inert_hull"),
    )
    b.rod(
        specimen,
        (0, 0, 0.11),
        (0, 0, 0.94),
        0.018,
        "bright",
        collision=True,
        name=_name(b, "mounting_sting"),
    )
    # The upper clamp position leaves the crosshead above the platform over
    # all 350 mm of immersion. Both guide housings have genuine open bores.
    for index, x in enumerate((-0.16, 0.16)):
        aperture_plate(
            b,
            specimen,
            f"berkeley_guide_housing_{index}",
            (x, 0, 0.88),
            (0.032, 0.035, 0.040),
            (0.0146, 0.0146),
            "dark",
        )
    aperture_plate(
        b,
        specimen,
        "berkeley_crosshead",
        (0, 0, 0.88),
        (0.128, 0.035, 0.035),
        (0.018, 0.018),
        "metal",
    )
    # Estimated clamp screws terminate at the retained 36 mm sting envelope.
    for sign in (-1, 1):
        b.cylinder(
            specimen,
            (sign * 0.042, 0, 0.88),
            0.005,
            0.024,
            "bright",
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=_name(b, f"sting_clamp_screw_{sign}"),
        )
    b.site(specimen, "tow_model", (0, 0, 0.12), size=0.03)
    b.site(carriage, "carriage_handle", (0, 0, 0.51), size=0.035)
    b.metadata["mounting_interface"] = dict(
        revision=2,
        platform_opening_m=[0.15, 0.15],
        platform_underside_relief_m=0.005,
        sting_diameter_m=0.036,
        guide_radius_m=0.014,
        square_bearing_half_opening_m=0.0146,
        minimum_radial_running_clearance_m=0.0006,
        legacy_carriage_handle="Preserved reference datum; supported_carriage_handle identifies the actual offset guard grip",
        physical_scope="Contact-capable open mounting, four passive rolling wheels, and ideal prismatic guidance; no hydrodynamics",
        passive_wheels=dict(
            added_hinge_dofs=4,
            added_vertical_play_dofs=4,
            vertical_play_range_m=[-0.0003, 0.0003],
            added_total_mass_kg=0.4,
            tire_hub_mass_kg_each=0.09,
            shaft_mass_kg_each=0.01,
            inertia_basis="Sum of two coaxial solid-cylinder inertias; authored values",
            shaft_radius_m=0.012,
            bearing_half_opening_m=0.0126,
            damping_Nm_s_per_rad=0.0001,
            vertical_damping_N_s_per_m=0.02,
            contact_response_s=0.004,
            contact_activation_margin_m=0.000005,
            contact_priority=1,
            load_basis="Each wheel bears its own uncompensated gravity load; the original ideal carriage guide carries the carriage mass",
            actuation="None; rail contact drives wheel rotation",
            limitations="Ideal hinge and carriage constraints; no calibrated bearing friction or rail-load distribution",
        ),
        source_basis="Official photo establishes tank/rail arrangement; lift, clamp, brackets and bearings are authored estimates",
    )
