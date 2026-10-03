"""Four source-specific engineering facilities with bounded inert inspection tasks."""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box


ETH = "https://baug.ethz.ch/en/news-and-events/news/2023/11/big-blue.html"
EPFL = "https://www.epfl.ch/research/domains/swiss-plasma-center/introduction-to-tcv/"
DELFT = "https://radar.tudelft.nl/Facilities/ducat.php"
TUM = "https://www.mos.ed.tum.de/en/nma/research/infrastructure-methods/"


def _material(b, name, color):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=f"{b.name}__mat_{name}",
        rgba=" ".join(map(str, color)),
    )


def _cradle(b, parent):
    b.box(parent, (0, 0, -0.95), (0.56, 0.59, 0.055), "eth_blue", collision=True)
    for y in (-0.58, 0.58):
        b.box(parent, (0, y, -0.53), (0.57, 0.025, 0.39), "eth_blue", collision=True)
        for side in (-1, 1):
            b.rod(parent, (0, y, 0), (side * 0.52, y, -0.90), 0.055, "eth_blue")
        b.cylinder(
            parent, (0, y, 0), 0.12, 0.045, "eth_blue", euler=(math.pi / 2, 0, 0)
        )
    b.box(parent, (0, 0, -0.88), (0.41, 0.42, 0.025), "metal", collision=True)
    for x in (-0.43, 0.43):
        b.box(parent, (x, 0, -0.70), (0.022, 0.44, 0.20), "metal")
    for y in (-0.43, 0.43):
        b.box(parent, (0, y, -0.70), (0.42, 0.020, 0.20), "glass")
    b.box(parent, (0, 0, -0.79), (0.39, 0.40, 0.062), "sample", collision=True)


def _big_blue(b, base, params):
    _material(b, "eth_blue", (0.065, 0.11, 0.46, 1))
    # The 9 m diameter is published. This low-angle display never produces a high-g field.
    b.cylinder(base, (0, 0, 0.075), 0.82, 0.075, "metal", collision=True)
    b.cylinder(base, (0, 0, 0.74), 0.46, 0.59, "eth_blue", collision=True)
    b.cylinder(base, (0, 0, 1.45), 0.32, 0.12, "metal")
    for angle in range(16):
        a = angle * math.tau / 16
        b.cylinder(
            base, (0.69 * math.cos(a), 0.69 * math.sin(a), 0.16), 0.025, 0.018, "bright"
        )
    rotor = b.moving(
        base,
        "beam_azimuth",
        (0, 0, 1.72),
        (0, 0, 1),
        (-0.035, 0.035),
        kind="hinge",
        mass=25,
        kp=9000,
        force=30000,
    )
    # Explicit proxy inertia prevents decorative fixed-child panels from being
    # interpreted as tonnes of solid metal by automatic volume inference.
    rotor.find("inertial").set("diaginertia", "0.8 110 110")
    b.actuator.find(f"./position[@name='{b.name}__actuator_beam_azimuth']").set(
        "kv", "2800"
    )
    b.box(rotor, (0, 0, 0), (3.92, 0.27, 0.17), "eth_blue", collision=True)
    for y in (-0.30, 0.30):
        b.box(rotor, (0, y, 0), (3.96, 0.035, 0.19), "eth_blue")
        for index in range(30):
            b.cylinder(
                rotor,
                (-3.65 + index * 0.25, y * 1.13, 0.06),
                0.012,
                0.005,
                "metal",
                euler=(math.pi / 2, 0, 0),
            )
    for side in (-1, 1):
        x = side * 3.91
        for y in (-0.59, 0.59):
            b.rod(rotor, (side * 3.45, 0, 0.03), (x, y, 0), 0.065, "eth_blue")
        if side < 0:
            cradle = ET.SubElement(
                rotor,
                "body",
                name=f"{b.name}__fixed_counter_cradle",
                pos=f"{x} 0 0",
                gravcomp="1",
            )
            ET.SubElement(
                cradle, "inertial", pos="0 0 -0.60", mass="3", diaginertia="0.3 0.3 0.3"
            )
        else:
            cradle = b.moving(
                rotor,
                "model_cradle_tilt",
                (x, 0, 0),
                (0, 1, 0),
                (-0.075, 0.075),
                kind="hinge",
                mass=3,
                kp=2000,
                force=5000,
            )
            cradle.find("inertial").set("pos", "0 0 -0.60")
            cradle.find("inertial").set("diaginertia", "0.3 0.3 0.3")
        _cradle(b, cradle)
        if side > 0:
            b.site(cradle, "retained_model_box", (0, 0, -0.79), 0.005)
    # Service cabinet stays with the central support, clear of the moving beam.
    b.box(base, (0, 0.95, 1.37), (0.42, 0.25, 0.52), "dark", collision=True)
    for x in (-0.29, 0.29):
        b.box(base, (x, 0.95, 0.43), (0.045, 0.20, 0.43), "metal", collision=True)
    for i in range(5):
        b.screen(base, (-0.18, 0.685, 1.06 + i * 0.19), 0.23, 0.095)
        b.box(base, (0.17, 0.68, 1.06 + i * 0.19), (0.115, 0.009, 0.06), "metal")
    b.metadata["capabilities"] = [
        "bounded_beam_azimuth",
        "retained_model_cradle_tilt",
        "visible_inert_model_box",
    ]
    b.metadata["limitations"].append(
        "Low-angle inspection surrogate only. No continuous centrifugation, elevated gravity, soil constitutive law, earthquake shaking, hydraulic drive or OEM interlock model; cradles and mass properties are estimates."
    )


def _eth_features(world, definition):
    for x in (-2.3, 2.3):
        for y in (-2.6, 2.5):
            box(
                world,
                f"gcc_grille_recess_{x}_{y}",
                (1.1, 0.65, 0.003),
                (x, y, 0.003),
                (0.17, 0.20, 0.21, 1),
                collision=False,
            )
            for index in range(18):
                box(
                    world,
                    f"gcc_grille_long_{x}_{y}_{index}",
                    (0.010, 0.64, 0.004),
                    (x - 1.04 + index * 0.122, y, 0.008),
                    (0.57, 0.60, 0.60, 1),
                    collision=False,
                )
            for index in range(9):
                box(
                    world,
                    f"gcc_grille_cross_{x}_{y}_{index}",
                    (1.06, 0.008, 0.004),
                    (x, y - 0.59 + index * 0.148, 0.008),
                    (0.57, 0.60, 0.60, 1),
                    collision=False,
                )
    for i in range(5):
        for j in range(4):
            box(
                world,
                f"arch_roof_gcc_yellow_{i}_{j}",
                (0.91, 0.72, 0.025),
                (-3.68 + i * 1.84, -2.16 + j * 1.44, 3.66),
                (0.80, 0.69, 0.045, 1),
                collision=False,
            )


BUILDERS = {"gcc_beam_inspection_workspace": _big_blue}
SOURCES = {
    "gcc_beam_inspection_workspace": dict(
        reference="ETH Geotechnical Centrifuge Center Big Blue installed photograph",
        url=ETH,
        dimensions_m=[9.04, 2.20, 1.94],
        dimension_basis="Published 9 m beam centrifuge diameter anchors radial extent; cradle, column, yellow ceiling and room dimensions estimated from the installed photo.",
    )
}
SAMPLE_INTERFACES = {
    "gcc_beam_inspection_workspace": (
        "retained_model_box",
        (0.78, 0.80, 0.124),
        "clamped",
        "Inert retained model box on the mechanically supported right cradle",
    )
}
FEATURES = {"eth_gcc_big_blue": _eth_features}


def _bolted_panel(b, parent, x, y, z, width=0.82, height=0.55):
    b.box(parent, (x, y, z), (width / 2, 0.026, height / 2), "metal", collision=True)
    for side in (-1, 1):
        for i in range(8):
            b.cylinder(
                parent,
                (
                    x - width * 0.43 + i * width * 0.86 / 7,
                    y - 0.037,
                    z + side * height * 0.43,
                ),
                0.014,
                0.010,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
        for i in range(3):
            b.cylinder(
                parent,
                (
                    x + side * width * 0.43,
                    y - 0.037,
                    z - height * 0.25 + i * height * 0.25,
                ),
                0.014,
                0.010,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )


def _tcv_sector(b, base, params):
    _material(b, "viewport", (0.37, 0.10, 0.56, 1))
    for x in (-1.12, 0.14, 1.15):
        for y in (-0.20, 0.42):
            b.box(base, (x, y, 0.04), (0.14, 0.15, 0.04), "metal", collision=True)
        b.box(base, (x, 0.18, 1.36), (0.065, 0.19, 1.28), "shell", collision=True)
    for z in (0.14, 2.64):
        b.box(base, (0.01, 0.20, z), (1.20, 0.20, 0.055), "metal")
    for z in (0.56, 1.40, 2.24):
        # Horizontal diagnostic exterior with repeated foil-wrap collars.
        b.cylinder(
            base, (-0.65, 0, z), 0.205, 0.59, "bright", euler=(0, math.pi / 2, 0)
        )
        for i in range(11):
            b.cylinder(
                base,
                (-0.87 + i * 0.066, 0, z),
                0.212 + (i % 3) * 0.003,
                0.012,
                "metal",
                euler=(0, math.pi / 2, 0),
            )
        for x in (-1.26, -0.10):
            b.cylinder(base, (x, 0, z), 0.245, 0.03, "metal", euler=(0, math.pi / 2, 0))
            for i in range(12):
                a = i * math.tau / 12
                b.cylinder(
                    base,
                    (x - 0.035, 0.211 * math.cos(a), z + 0.211 * math.sin(a)),
                    0.016,
                    0.016,
                    "bright",
                    euler=(0, math.pi / 2, 0),
                )
        b.cylinder(
            base, (-1.01, 0, z), 0.221, 0.051, "copper", euler=(0, math.pi / 2, 0)
        )
        b.cylinder(base, (-1.46, 0, z), 0.094, 0.17, "metal", euler=(0, math.pi / 2, 0))
        # Brackets are authored exterior supports for unseen service structure.
        b.box(base, (-0.60, 0.28, z), (0.52, 0.10, 0.04), "dark")
        b.rod(base, (-0.87, 0.28, z - 0.03), (-1.08, 0.20, z - 0.34), 0.022, "metal")
    for z in (0.18, 0.29, 0.40, 0.80, 0.91, 1.02, 1.81, 1.92, 2.03, 2.53):
        b.box(base, (0.64, 0.05, z), (0.46, 0.08, 0.035), "copper")
    _bolted_panel(b, base, 0.65, -0.18, 0.58)
    _bolted_panel(b, base, 0.65, -0.18, 2.23)
    # An explicitly authored dry access hinge makes this a mechanical inspection proxy.
    for z in (1.11, 1.70):
        b.cylinder(base, (0.21, -0.28, z), 0.035, 0.045, "dark")
        b.box(base, (0.17, -0.12, z), (0.05, 0.16, 0.023), "metal")
    panel = b.moving(
        base,
        "inspection_panel_open",
        (0.21, -0.28, 1.40),
        (0, 0, -1),
        (0, 0.22),
        kind="hinge",
        mass=0.4,
        kp=480,
    )
    _bolted_panel(b, panel, 0.44, 0, 0)
    b.cylinder(
        panel, (0.55, -0.045, 0.02), 0.075, 0.020, "bright", euler=(math.pi / 2, 0, 0)
    )
    b.cylinder(
        panel, (0.55, -0.067, 0.02), 0.056, 0.003, "viewport", euler=(math.pi / 2, 0, 0)
    )
    b.box(panel, (0.19, -0.057, -0.09), (0.046, 0.027, 0.10), "dark")
    datum = b.moving(
        panel,
        "datum_height",
        (0.19, -0.096, -0.08),
        (0, 0, 1),
        (-0.035, 0.035),
        mass=0.035,
        kp=160,
    )
    b.box(datum, (0, 0, 0), (0.035, 0.009, 0.029), "metal", collision=True)
    b.cylinder(datum, (0, -0.012, 0), 0.020, 0.003, "sample", euler=(math.pi / 2, 0, 0))
    b.site(datum, "inspection_datum", (0, -0.015, 0), 0.0005)
    # Fixed service routes run between exterior brackets, with no simulated signals.
    for i in range(6):
        z = 0.35 + i * 0.39
        b.rod(base, (-1.12, -0.025, z + 0.12), (-0.45, -0.42, z), 0.018, "rubber")
        b.rod(base, (-0.45, -0.42, z), (0.14, -0.025, z + 0.13), 0.018, "rubber")
    b.rod(base, (1.16, 0, 0.22), (1.31, -0.27, 1.1), 0.045, "metal")
    b.rod(base, (1.31, -0.27, 1.1), (1.12, 0, 2.44), 0.045, "metal")
    b.metadata["capabilities"] = [
        "dry_inspection_panel_hinge",
        "inert_datum_translation",
        "visible_retained_inspection_datum",
    ]
    b.metadata["limitations"].append(
        "Only the photographed access-sector exterior; dry hinge and datum carriage are authored additions. No torus/hall reconstruction, plasma, magnetic fields, heating, nuclear process, RF beam or diagnostic signal is implemented."
    )


def _tcv_features(world, definition):
    box(
        world,
        "tcv_local_backplane",
        (1.80, 0.04, 1.35),
        (0, 1.3, 1.35),
        (0.19, 0.22, 0.23, 1),
    )
    for i in range(8):
        box(
            world,
            f"tcv_backplane_service_{i}",
            (1.7, 0.012, 0.007),
            (0, 1.248, 0.40 + i * 0.27),
            (0.11, 0.14, 0.15, 1),
            collision=False,
        )


BUILDERS["tcv_sector_inspection_workspace"] = _tcv_sector
SOURCES["tcv_sector_inspection_workspace"] = dict(
    reference="EPFL TCV official photographed diagnostic-access sector",
    url=EPFL,
    dimensions_m=[3.10, 1.12, 2.72],
    dimension_basis="No dimensions published for the pictured local sector. Bolted rectangular panels, wrapped horizontal cylinders, viewport and hoses are observed; geometry and inspection mechanisms are estimated.",
)
SAMPLE_INTERFACES["tcv_sector_inspection_workspace"] = (
    "inspection_datum",
    (0.040, 0.006, 0.040),
    "clamped",
    "Inert external datum fixed to an original dry inspection slider",
)
FEATURES["epfl_tcv_diagnostic_sector"] = _tcv_features


def _pyramid_mesh(b):
    name = b.unique("rf_pyramid_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex="-.105 -.105 0 .105 -.105 0 .105 .105 0 -.105 .105 0 0 0 .23",
        face="0 2 1 0 3 2 0 1 4 1 2 4 2 3 4 3 0 4",
    )
    return name


def _ducat(b, base, params):
    _material(b, "rf_blue", (0.012, 0.105, 0.23, 1))
    _material(b, "wood", (0.63, 0.49, 0.29, 1))
    pyramid = _pyramid_mesh(b)
    # Published chamber dimensions are 6 x 3 x 3 m; individual absorber dimensions are estimated.
    for i in range(27):
        x = -2.86 + i * 0.22
        for j in range(13):
            z = 0.15 + j * 0.22
            b.geom(
                base,
                "mesh",
                (),
                (x, 1.485, z),
                "rf_blue",
                mesh=pyramid,
                euler=(math.pi / 2, 0, 0),
            )
            roof = b.geom(
                base,
                "mesh",
                (),
                (x, -1.32 + j * 0.22, 2.98),
                "rf_blue",
                mesh=pyramid,
                euler=(math.pi, 0, 0),
            )
            roof.set("name", f"arch_roof_ducat_absorber_{i}_{j}")
        for y in (-1.32, -1.10, -0.88, -0.66, 0.66, 0.88, 1.10, 1.32):
            b.geom(base, "mesh", (), (x, y, 0.025), "rf_blue", mesh=pyramid)
    for i in range(13):
        for j in range(13):
            # Keep the authored entry clear; the source does not locate its door.
            if -0.90 < -1.32 + i * 0.22 < 0.12 and 0.15 + j * 0.22 < 2.20:
                continue
            b.geom(
                base,
                "mesh",
                (),
                (-2.985, -1.32 + i * 0.22, 0.15 + j * 0.22),
                "rf_blue",
                mesh=pyramid,
                euler=(0, math.pi / 2, 0),
            )
    # Raised wooden target stand, visible in the near-field configuration.
    b.box(base, (-1.10, 0, 0.12), (0.53, 0.46, 0.12), "rf_blue", collision=True)
    b.box(base, (-1.10, 0, 0.255), (0.42, 0.35, 0.015), "wood", collision=True)
    for y in (-0.25, 0.25):
        b.box(base, (-1.10, y, 0.68), (0.11, 0.018, 0.41), "wood", collision=True)
        b.rod(base, (-1.37, y, 0.28), (-1.10, y, 1.05), 0.023, "wood")
    b.box(base, (-1.10, 0, 1.105), (0.44, 0.37, 0.018), "wood", collision=True)
    b.box(base, (-1.1, 0, 1.38), (0.016, 0.25, 0.255), "cream", collision=True)
    b.cylinder(base, (-1.08, 0, 1.43), 0.14, 0.004, "dark", euler=(0, math.pi / 2, 0))
    for i in range(8):
        a = i * math.tau / 8
        b.rod(
            base,
            (-1.073, 0, 1.43),
            (-1.073, 0.125 * math.cos(a), 1.43 + 0.125 * math.sin(a)),
            0.008,
            "cream",
        )
    b.site(base, "antenna_reference", (-1.074, 0, 1.43), 0.001)
    # Cream scanner mast with an independent moving near-field head.
    b.box(base, (1.1, 0, 0.055), (0.49, 0.36, 0.055), "cream", collision=True)
    b.box(base, (1.1, 0.14, 1.36), (0.13, 0.14, 1.25), "cream", collision=True)
    b.rod(base, (0.94, -0.012, 0.15), (0.94, -0.012, 2.6), 0.015, "bright")
    b.box(base, (1.22, 0.005, 1.36), (0.018, 0.011, 1.22), "dark")
    for i in range(25):
        b.box(base, (1.239, -0.016, 0.19 + i * 0.093), (0.024, 0.015, 0.030), "dark")
    zslide = b.moving(
        base,
        "scanner_height",
        (0.86, 0, 1.40),
        (0, 0, 1),
        (-0.34, 0.34),
        mass=0.40,
        kp=600,
    )
    b.box(zslide, (0, 0, 0), (0.063, 0.12, 0.075), "cream", collision=True)
    b.box(zslide, (-0.20, 0, 0), (0.20, 0.055, 0.038), "metal")
    for x in (-0.27, -0.37):
        b.rod(zslide, (x, -0.37, 0.02), (x, 0.37, 0.02), 0.009, "bright")
    scan = b.moving(
        zslide,
        "scanner_lateral",
        (-0.32, 0, 0.03),
        (0, 1, 0),
        (-0.23, 0.23),
        mass=0.10,
        kp=230,
    )
    b.box(scan, (0, 0, 0), (0.073, 0.055, 0.035), "dark", collision=True)
    b.rod(scan, (-0.07, 0, 0), (-0.26, 0, 0), 0.012, "metal")
    b.box(scan, (-0.29, 0, 0), (0.03, 0.075, 0.065), "cream")
    # The network instrument shelf is tied to the fixed mast, not floating.
    b.box(base, (1.31, -0.32, 1.03), (0.34, 0.25, 0.025), "metal", collision=True)
    for x in (1.02, 1.57):
        b.rod(base, (1.12, 0.04, 0.76), (x, -0.49, 1.00), 0.018, "metal")
    b.box(base, (1.32, -0.30, 1.18), (0.28, 0.19, 0.125), "shell")
    b.screen(base, (1.19, -0.496, 1.18), 0.24, 0.16)
    for i in range(4):
        for j in range(4):
            b.box(
                base,
                (1.39 + j * 0.037, -0.497, 1.1 + i * 0.055),
                (0.008, 0.003, 0.010),
                "metal",
            )
    b.metadata["capabilities"] = [
        "near_field_head_height",
        "near_field_head_lateral_position",
        "fixed_visible_antenna_reference",
    ]
    b.metadata["limitations"].append(
        "Only the near-field arrangement is counted and modeled. Original mechanical scanning proxy; no RF propagation, antenna response, network-analyser signals, absorber performance or calibrated scanner precision."
    )


def _ducat_features(world, definition):
    box(
        world,
        "ducat_clear_floor_strip",
        (2.94, 0.28, 0.008),
        (0, 0, 0.010),
        (0.39, 0.43, 0.43, 1),
        collision=False,
    )


BUILDERS["ducat_near_field_workspace"] = _ducat
SOURCES["ducat_near_field_workspace"] = dict(
    reference="TU Delft DUCAT installed near-field configuration",
    url=DELFT,
    dimensions_m=[6.0, 3.0, 3.0],
    dimension_basis="Official chamber 6 x 3 x 3 m. Scanner, wooden fixture, pyramidal absorber dimensions and mechanical travel estimated from installed near-field photograph.",
)
SAMPLE_INTERFACES["ducat_near_field_workspace"] = (
    "antenna_reference",
    (0.008, 0.28, 0.28),
    "clamped",
    "Visible passive circular antenna-reference surrogate on the wooden support",
)
FEATURES["delft_ducat_antenna"] = _ducat_features


def _powertrain(b, base, params):
    # Closed load machine and supporting plinth reproduce the installed exterior only.
    b.box(base, (-0.68, 0.18, 0.075), (0.58, 0.70, 0.075), "dark", collision=True)
    b.box(base, (-0.68, 0.18, 0.53), (0.52, 0.62, 0.38), "shell", collision=True)
    b.box(base, (-0.68, -0.452, 0.91), (0.49, 0.028, 0.48), "metal", collision=True)
    b.cylinder(
        base, (-0.68, 0.18, 1.48), 0.245, 0.50, "metal", euler=(math.pi / 2, 0, 0)
    )
    for y in (-0.34, 0.03, 0.36, 0.65):
        b.cylinder(
            base, (-0.68, y, 1.48), 0.262, 0.015, "bright", euler=(math.pi / 2, 0, 0)
        )
    for i in range(16):
        a = i * math.tau / 16
        b.rod(
            base,
            (-0.68 + 0.245 * math.cos(a), -0.30, 1.48 + 0.245 * math.sin(a)),
            (-0.68 + 0.245 * math.cos(a), 0.60, 1.48 + 0.245 * math.sin(a)),
            0.006,
            "bright",
        )
    b.box(base, (-0.68, 0.18, 1.20), (0.31, 0.47, 0.05), "metal")
    # External fixture travel is a dry inspection proxy, not a powered dyno shaft.
    for x in (-0.93, -0.43):
        b.box(base, (x, -0.69, 0.64), (0.046, 0.29, 0.032), "metal")
        b.box(base, (x, -0.54, 0.44), (0.042, 0.11, 0.19), "metal")
    slide = b.moving(
        base,
        "inspection_carriage",
        (-0.68, -0.70, 0.93),
        (0, 1, 0),
        (-0.045, 0.045),
        mass=0.40,
        kp=420,
    )
    b.box(slide, (0, 0, -0.224), (0.28, 0.12, 0.032), "metal", collision=True)
    b.box(slide, (0, 0.035, -0.12), (0.18, 0.04, 0.08), "dark")
    b.cylinder(slide, (0, 0.065, 0), 0.19, 0.045, "metal", euler=(math.pi / 2, 0, 0))
    turn = b.moving(
        slide,
        "coupon_index_angle",
        (0, 0, 0),
        (0, 1, 0),
        (-0.28, 0.28),
        kind="hinge",
        mass=0.14,
        kp=240,
    )
    b.cylinder(
        turn, (0, 0, 0), 0.182, 0.020, "dark", euler=(math.pi / 2, 0, 0), collision=True
    )
    b.cylinder(turn, (0, -0.024, 0), 0.147, 0.005, "metal", euler=(math.pi / 2, 0, 0))
    for i in range(12):
        a = i * math.tau / 12
        b.cylinder(
            turn,
            (0.162 * math.cos(a), -0.027, 0.162 * math.sin(a)),
            0.009,
            0.009,
            "bright",
            euler=(math.pi / 2, 0, 0),
        )
    b.box(turn, (0, -0.035, 0.04), (0.048, 0.007, 0.033), "copper", collision=True)
    b.site(turn, "motor_test_coupon", (0, -0.043, 0.04), 0.001)
    for i in range(3):
        x = -0.30 + i * 0.043
        b.rod(base, (x, -0.48, 1.12), (x, -0.92, 0.57), 0.013, "orange")
        b.rod(base, (x, -0.92, 0.57), (x + 0.22, -1.01, 0.05), 0.013, "orange")
        b.rod(
            base,
            (x + 0.22, -1.01, 0.05),
            (0.58 + i * 0.03, -0.73, 0.045),
            0.013,
            "orange",
        )
        b.rod(
            base,
            (0.58 + i * 0.03, -0.73, 0.045),
            (0.58 + i * 0.03, -0.52, 0.15),
            0.013,
            "orange",
        )
    # Separate electronics enclosure on a floor-standing aluminium rack.
    for x in (0.47, 1.13):
        for y in (-0.52, 0.14):
            b.box(base, (x, y, 0.035), (0.07, 0.07, 0.035), "dark", collision=True)
            b.box(base, (x, y, 0.44), (0.026, 0.026, 0.37), "metal", collision=True)
    for z in (0.14, 0.45, 0.79):
        b.box(base, (0.80, -0.19, z), (0.36, 0.36, 0.025), "metal")
    for i in range(3):
        z = 0.20 + i * 0.17
        b.box(base, (0.80, -0.19, z), (0.30, 0.29, 0.07), "dark")
        b.box(base, (0.80, -0.488, z), (0.275, 0.008, 0.054), "metal")
        for j in range(7):
            b.cylinder(
                base,
                (0.58 + j * 0.075, -0.50, z),
                0.012,
                0.004,
                "dark",
                euler=(math.pi / 2, 0, 0),
            )
    b.box(base, (0.80, -0.19, 1.29), (0.36, 0.36, 0.475), "shell", collision=True)
    b.box(base, (0.89, -0.558, 1.38), (0.22, 0.009, 0.36), "dark")
    b.box(base, (0.63, -0.566, 1.38), (0.028, 0.009, 0.37), "blue")
    b.cylinder(
        base, (1.05, -0.58, 1.43), 0.025, 0.008, "shell", euler=(math.pi / 2, 0, 0)
    )
    b.metadata["capabilities"] = [
        "unpowered_inspection_carriage",
        "inert_coupon_indexing",
        "visible_retained_test_coupon",
    ]
    b.metadata["limitations"].append(
        "Ratings identify the photographed facility only. Geometry and motion are unpowered proxies; no voltage, current, torque, speed rating, heat, coolant, energy conversion or real dynamometer operating process is simulated."
    )


def _tum_features(world, definition):
    for index, x in enumerate((-0.8, 0.05)):
        box(
            world,
            f"tum_rear_cabinet_{index}",
            (0.40, 0.30, 1.02),
            (x, 1.27, 1.02),
            (0.80, 0.81, 0.76, 1),
        )
        box(
            world,
            f"tum_rear_cabinet_trim_{index}",
            (0.40, 0.015, 0.035),
            (x, 0.955, 1.94),
            (0.43, 0.10, 0.07, 1),
            collision=False,
        )
        box(
            world,
            f"tum_rear_cabinet_handle_{index}",
            (0.012, 0.015, 0.11),
            (x + 0.24, 0.95, 1.06),
            (0.15, 0.17, 0.18, 1),
            collision=False,
        )


BUILDERS["nma_powertrain_inspection_workspace"] = _powertrain
SOURCES["nma_powertrain_inspection_workspace"] = dict(
    reference="TUM NMA December 2024 installed electric powertrain test-bench photograph",
    url=TUM,
    dimensions_m=[2.55, 2.10, 1.82],
    dimension_basis="Photographed load machine, circular test face, orange cables and white/blue control enclosure are observed. All dimensions and unpowered fixture travel estimated; published engineering ratings are not implemented.",
)
SAMPLE_INTERFACES["nma_powertrain_inspection_workspace"] = (
    "motor_test_coupon",
    (0.096, 0.014, 0.066),
    "clamped",
    "Inert metal coupon retained on the unpowered inspection disc",
)
FEATURES["tum_nma_electric_powertrain"] = _tum_features
