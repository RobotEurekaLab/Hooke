"""Reference-informed fluid-laboratory mechanisms, without process solvers.

Metre-scale original geometry uses public photographs as private references.
Only explicitly documented dimensions are measured claims; room envelopes,
actuated inspection strokes and hidden mechanical connections are estimates.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

MIT = "https://paocweb.mit.edu/about/facilities/the-fluids-lab"
PRINCETON = "https://bamlab.princeton.edu/facilities"
UCSD = "https://mccartney.ucsd.edu/facilities"
MCGILL = "https://www.mcgill.ca/metals-processing-centre/about-us/research-facilities-process-metallurgy"


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _frustum(b, parent, z0, z1, r0, r1, material="dark"):
    """Circular taper; a compact solid mesh keeps the pedestal recognizably round."""
    vertices = [(0, 0, z0), (0, 0, z1)]
    for z, radius in ((z0, r0), (z1, r1)):
        vertices.extend(
            (
                radius * math.cos(i * math.tau / 40),
                radius * math.sin(i * math.tau / 40),
                z,
            )
            for i in range(40)
        )
    faces = []
    for i in range(40):
        j = (i + 1) % 40
        faces.extend(
            (
                (0, 2 + j, 2 + i),
                (1, 42 + i, 42 + j),
                (2 + i, 2 + j, 42 + j),
                (2 + i, 42 + j, 42 + i),
            )
        )
    name = b.unique("taper_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=" ".join(str(v) for p in vertices for v in p),
        face=" ".join(str(v) for p in faces for v in p),
    )
    ET.SubElement(
        parent,
        "geom",
        name=b.unique("pedestal_taper"),
        type="mesh",
        mesh=name,
        material=f"{b.name}__mat_{material}",
        contype="1",
        conaffinity="1",
    )


def _rotating_fluids(b, base, params):
    from real_labs.mit_rotating_geometry import build_station

    build_station(b, base, params, frustum=_frustum)


def _mit_features(world, definition):
    from real_labs.mit_rotating_geometry import room_features

    room_features(world, definition)


BUILDERS = {"paoc_rotating_table": _rotating_fluids}
SOURCES = {
    "paoc_rotating_table": {
        "reference": "MIT PAOC rotating-fluid table and overhead imaging arrangement",
        "url": MIT,
        "dimensions_m": [2.10, 1.70, 1.95],
        "dimension_basis": "Estimated photo-informed exterior; source supplies no calibrated apparatus measurements",
    }
}
SAMPLE_INTERFACES = {
    "paoc_rotating_table": (
        "calibration_target",
        (0.09, 0.09, 0.008),
        "clamped",
        "rigid calibration disc attached to the rotating tank floor",
    )
}
FEATURES = {"mit_paoc_rotating_fluids": _mit_features}


def _wind_tunnel(b, base, params):
    """Large return circuit with a genuinely hollow 4 ft square test section."""
    from real_labs.princeton_tunnel_geometry import add_probe_roof, add_return_casings
    # Three interchangeable test-section frames; 4 ft is the clear internal bore.
    half = 0.6096
    y, z = -1.8, 2.15
    for x in (-1.5, -0.5, 0.5, 1.5):
        for dy in (-half - 0.055, half + 0.055):
            b.box(
                base,
                (x, y + dy, z),
                (0.040, 0.040, half + 0.095),
                "dark",
                collision=True,
            )
        for dz in (-half - 0.055, half + 0.055):
            b.box(
                base,
                (x, y, z + dz),
                (0.040, half + 0.095, 0.040),
                "dark",
                collision=True,
            )
        for dy in (-half - 0.055, half + 0.055):
            for dz in (-0.46, -0.23, 0, 0.23, 0.46):
                b.cylinder(
                    base,
                    (x + 0.046, y + dy, z + dz),
                    0.014,
                    0.007,
                    "bright",
                    euler=(0, math.pi / 2, 0),
                )
    for x in (-1, 0, 1):
        for side in (-1, 1):
            _tag(
                b,
                b.box(
                    base,
                    (x, y + side * (half + 0.012), z),
                    (0.46, 0.012, half),
                    "glass",
                    collision=True,
                ),
                "test_side_panel",
            )
            if x == 1 and side == 1:
                add_probe_roof(b, base)
            else:
                _tag(
                    b,
                    b.box(
                        base,
                        (x, y, z + side * (half + 0.012)),
                        (0.46, half, 0.012),
                        "glass",
                        collision=True,
                    ),
                    "test_horizontal_panel",
                )
    for x in (-1.45, 1.45):
        for dy in (-0.72, 0.72):
            b.box(base, (x, y + dy, 0.05), (0.20, 0.18, 0.05), "blue", collision=True)
            b.box(
                base, (x, y + dy, 0.7677), (0.05, 0.05, 0.6677), "blue", collision=True
            )
        b.box(base, (x, y, 1.4664), (0.085, 0.78, 0.05), "blue", collision=True)
        b.rod(base, (x, y - 0.70, 0.12), (x, y + 0.70, 1.42), 0.029, "blue")
    # Photo-informed curved exteriors and connected open underframes; this
    # is apparatus geometry, not a CFD domain or a validated turning-vane model.
    add_return_casings(b, base)
    # Fan barrel and motor sit in the return leg; support piers reach its casing.
    b.cylinder(base, (1.8, 2.6, z), 1.02, 0.44, "cyan", euler=(0, math.pi / 2, 0))
    for x in (1.34, 2.26):
        b.cylinder(base, (x, 2.6, z), 1.06, 0.025, "metal", euler=(0, math.pi / 2, 0))
    b.box(base, (0, 2.6, 0.65), (0.48, 0.60, 0.65), "blue", collision=True)
    b.box(base, (1.8, 2.6, 0.565), (0.42, 0.50, 0.565), "cyan", collision=True)
    b.box(base, (1.8, 3.58, z), (0.30, 0.36, 0.27), "dark", collision=True)
    # Rigid airfoil coupon with a visible trunnion and grounded support pedestal.
    b.box(base, (-0.35, y, 1.5704), (0.17, 0.17, 0.030), "metal", collision=True)
    b.cylinder(base, (-0.35, y, 1.8104), 0.035, 0.21, "metal")
    b.cylinder(base, (-0.35, y, 2.04), 0.052, 0.09, "metal", euler=(math.pi / 2, 0, 0))
    model = b.moving(
        base,
        "model_pitch",
        (-0.35, y, 2.04),
        (0, 1, 0),
        (-0.32, 0.32),
        kind="hinge",
        mass=0.45,
        kp=300,
    )
    b.geom(
        model, "ellipsoid", (0.38, 0.34, 0.045), (0, 0, 0.025), "orange", collision=True
    )
    b.site(model, "airfoil_coupon", (0, 0, 0.025))
    for dy in (-0.95, 0.95):
        b.box(base, (0.90, y + dy, 0.05), (0.15, 0.15, 0.05), "metal", collision=True)
        b.box(base, (0.90, y + dy, 1.65), (0.025, 0.025, 1.55), "metal", collision=True)
    b.box(base, (0.90, y, 3.2), (0.025, 1.0, 0.025), "metal", collision=True)
    probe = b.moving(
        base,
        "probe_traverse",
        (0.90, y, 3.2),
        (0, 1, 0),
        (-0.36, 0.36),
        mass=0.3,
        kp=300,
    )
    b.box(probe, (0, 0, 0), (0.05, 0.06, 0.045), "dark")
    # The unchanged traverse passes through a real, inferred roof slot;
    # shaft and adjacent roof both retain genuine collision surfaces.
    b.cylinder(probe, (0, 0, -0.33), 0.010, 0.30, "bright", collision=True)
    b.cylinder(probe, (0, 0, -0.66), 0.018, 0.035, "orange", collision=True)
    b.site(probe, "probe_tip", (0, 0, -0.695))
    b.metadata["capabilities"] = [
        "inert model pitch adjustment",
        "test-section diagnostic traverse",
    ]
    b.metadata["limitations"].append(
        "Published 4 ft square test-section bore retained; loop geometry, model mount and inspection strokes are estimated. No airflow, fan, aerodynamic forces or PIV simulation."
    )
    b.metadata["limitations"].append(
        "Rounded turning casings, seam thicknesses and support frames are exterior estimates. "
        "The 44 by 770 mm probe roof slot is an authored mechanical interface, not a "
        "verified OEM feedthrough; no internal turning vanes or CFD solution are represented."
    )


def _princeton_features(world, definition):
    for x in (-6, 6):
        for y in (-4, 0, 4):
            box(
                world,
                f"bam_hall_column_{x}_{y}",
                (0.10, 0.12, 2.9),
                (x, y, 2.9),
                (0.42, 0.43, 0.42, 1),
            )
    for y in (-4, 0, 4):
        box(
            world,
            f"bam_roof_tie_{y}",
            (6.1, 0.06, 0.07),
            (0, y, 5.7),
            (0.34, 0.36, 0.37, 1),
        )
        for sign in (-1, 1):
            ET.SubElement(
                world,
                "geom",
                name=f"bam_rafter_{y}_{sign}",
                type="capsule",
                size=".05",
                fromto=f"{sign*6} {y} 5.7 0 {y} 6.8",
                rgba=".42 .44 .45 1",
                contype="0",
                conaffinity="0",
            )
    box(
        world,
        "bam_service_cabinet",
        (0.46, 0.34, 0.86),
        (4.8, -3.6, 0.86),
        (0.72, 0.73, 0.69, 1),
    )
    box(
        world,
        "bam_service_console",
        (0.30, 0.035, 0.20),
        (4.8, -3.951, 1.24),
        (0.035, 0.13, 0.16, 1),
        collision=False,
    )


BUILDERS["bam_closed_return_tunnel"] = _wind_tunnel
SOURCES["bam_closed_return_tunnel"] = {
    "reference": "Princeton BAM closed-loop tunnel at Gas Dynamics Laboratory",
    "url": PRINCETON,
    "dimensions_m": [9.9, 7.2, 3.4],
    "dimension_basis": "Published test-section clear bore 4 ft by 4 ft = 1.2192 m square; loop envelope, axial section lengths and mechanism travels estimated",
}
SAMPLE_INTERFACES["bam_closed_return_tunnel"] = (
    "airfoil_coupon",
    (0.76, 0.68, 0.09),
    "clamped",
    "inert ellipsoidal airfoil coupon on a pitch trunnion",
)
FEATURES["princeton_bam_closed_loop_wind_tunnel"] = _princeton_features


def _centrifuge(b, base, params):
    """Inspection-speed surrogate of a swinging-basket centrifuge, not a high-g model."""
    from real_labs.ucsd_centrifuge_geometry import (
        add_basket_exterior, add_counterweight, add_pedestal_surface, add_basket_bearings,
    )

    b.cylinder(base, (0, 0, 0.05), 0.79, 0.05, "blue", collision=True)
    # Tapered shell sectors leave a real front access arch; upper ring ties the
    # sectors together. Hidden wall thickness is inferred from the photograph.
    for i in range(32):
        a0, a1 = i * math.tau / 32, (i + 1) * math.tau / 32
        if 21 <= i <= 26:
            continue
        verts = []
        for z, outer, inner in ((0.10, 0.74, 0.51), (1.04, 0.41, 0.25)):
            for radius, a in ((outer, a0), (outer, a1), (inner, a1), (inner, a0)):
                verts.append((radius * math.cos(a), radius * math.sin(a), z))
        faces = (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        )
        name = b.unique("pedestal_sector_mesh")
        ET.SubElement(
            b.root.find("asset"),
            "mesh",
            name=name,
            vertex=" ".join(str(v) for p in verts for v in p),
            face=" ".join(str(v) for p in faces for v in p),
        )
        ET.SubElement(
            base,
            "geom",
            name=b.unique("pedestal_sector"),
            type="mesh",
            mesh=name,
            material=f"{b.name}__mat_blue",
            contype="1",
            conaffinity="1",
        )
    add_pedestal_surface(b, base)
    b.cylinder(base, (0, 0, 1.075), 0.42, 0.035, "blue", collision=True)
    _frustum(b, base, 1.11, 1.25, 0.42, 0.25, "blue")
    _tag(b, b.cylinder(base, (0, 0, 1.285), 0.27, 0.035, "metal"), "rotary_bearing")
    for i in range(10):
        a = i * math.tau / 10
        b.cylinder(
            base,
            (0.68 * math.cos(a), 0.68 * math.sin(a), 0.111),
            0.018,
            0.012,
            "bright",
        )
    arm = b.moving(
        base,
        "arm_index",
        (0, 0, 1.40),
        (0, 0, 1),
        (-0.65, 0.65),
        kind="hinge",
        mass=12,
        kp=1500,
        force=2000,
    )
    _tag(
        b,
        b.box(arm, (0.13, 0, 0), (1.55, 0.18, 0.08), "blue", collision=True),
        "rotary_arm",
    )
    for z in (-0.085, 0.085):
        b.box(arm, (0.13, 0, z), (1.55, 0.235, 0.012), "blue", collision=True)
    add_counterweight(b, arm)
    add_basket_bearings(b, arm)
    b.box(arm, (1.45, 0, 0.065), (0.095, 0.50, 0.045), "blue", collision=True)
    basket = b.moving(
        arm,
        "basket_tilt",
        (1.45, 0, -0.05),
        (0, 1, 0),
        (0, 0.40),
        kind="hinge",
        mass=2,
        kp=550,
        force=500,
    )
    add_basket_exterior(b, basket)
    # Footprint is exactly the published maximum supported container size;
    # thickness and interior arrangement are estimates. The coupon stays rigid.
    _tag(
        b,
        b.box(basket, (0, 0, -0.49), (0.30, 0.35, 0.105), "orange", collision=True),
        "container_max_footprint",
    )
    b.box(basket, (0, 0, -0.379), (0.275, 0.325, 0.006), "copper", collision=True)
    b.site(basket, "soil_analogue", (0, 0, -0.373), size=0.025)
    b.metadata["capabilities"] = [
        "bounded inspection indexing of rotary arm",
        "commanded basket attitude",
    ]
    b.metadata["limitations"].append(
        "Maximum container footprint 0.6 by 0.7 m is documented; all machine dimensions and inspection strokes estimated. Basket actuator is an authored inspection surrogate, not the real passive high-g dynamics; soil and centrifuge similitude absent."
    )
    b.metadata["limitations"].append(
        "Blue counterweight and curved white basket follow the low-resolution exterior photograph. "
        "Split mounting clamp, shell thickness, support saddles and open-top inspection state are "
        "estimated assembly details, not OEM CAD or a verified high-g load path."
    )


def _ucsd_features(world, definition):
    # Only the visible rear portion of the circular enclosure is reconstructed;
    # the front remains an access cutaway rather than a claimed building plan.
    for i in range(18):
        a = (10 + i * 160 / 18) * math.pi / 180
        radius = 2.70
        box(
            world,
            f"ucsd_curved_shield_{i}",
            (0.18, 0.215, 1.4),
            (radius * math.cos(a), radius * math.sin(a), 1.4),
            (0.66, 0.64, 0.59, 1),
            euler=f"0 0 {a}",
        )
    box(
        world,
        "ucsd_daq_cabinet",
        (0.38, 0.32, 0.70),
        (-2.3, -1.55, 0.70),
        (0.69, 0.72, 0.74, 1),
    )
    box(
        world,
        "ucsd_daq_display",
        (0.25, 0.02, 0.14),
        (-2.3, -1.879, 1.12),
        (0.035, 0.13, 0.16, 1),
        collision=False,
    )
    for i in range(8):
        box(
            world,
            f"ucsd_daq_slit_{i}",
            (0.25, 0.012, 0.006),
            (-2.3, -1.876, 0.35 + i * 0.03),
            (0.08, 0.10, 0.11, 1),
            collision=False,
        )


BUILDERS["mccartney_swinging_basket"] = _centrifuge
SOURCES["mccartney_swinging_basket"] = {
    "reference": "UC San Diego McCartney centrifuge and swinging specimen basket",
    "url": UCSD,
    "dimensions_m": [4.3, 2.7, 1.9],
    "dimension_basis": "Published maximum container footprint 0.6 by 0.7 m retained; centrifuge envelope, basket depth and inspection travel estimated",
}
SAMPLE_INTERFACES["mccartney_swinging_basket"] = (
    "soil_analogue",
    (0.60, 0.70, 0.21),
    "clamped",
    "inert rigid soil-analogue block fixed in the supported container",
)
FEATURES["ucsd_mccartney_geotechnical_centrifuge"] = _ucsd_features


def _tundish(b, base, params):
    """Bolted transparent water-model exterior and two dry inspection mechanisms."""
    for x in (-2.12, 2.12):
        for y in (-0.86, 0.86):
            b.box(base, (x, y, 0.035), (0.16, 0.16, 0.035), "metal", collision=True)
            b.box(base, (x, y, 1.47), (0.055, 0.055, 1.435), "dark", collision=True)
    for z in (0.80, 1.93, 2.91):
        for y in (-0.86, 0.86):
            b.box(base, (0, y, z), (2.175, 0.055, 0.055), "dark", collision=True)
        for x in (-2.12, 2.12):
            b.box(base, (x, 0, z), (0.055, 0.86, 0.055), "dark", collision=True)
    _tag(
        b,
        b.box(base, (0, 0, 0.865), (2.05, 0.81, 0.035), "metal", collision=True),
        "tank_base",
    )
    # Four visible front bays and a shallow tank surround follow the image.
    # The hidden delta-shaped plan cannot be recovered from a single frontal
    # view; this inspectable rectangular local envelope is disclosed explicitly.
    for side in (-1, 1):
        for x in (-1.5, -0.5, 0.5, 1.5):
            b.box(
                base,
                (x, side * 0.805, 1.405),
                (0.47, 0.015, 0.505),
                "glass",
                collision=True,
            )
        for x in (-2, -1, 0, 1, 2):
            b.box(
                base,
                (x, side * 0.84, 1.405),
                (0.030, 0.024, 0.54),
                "dark",
                collision=True,
            )
            for z in (1.0, 1.16, 1.32, 1.48, 1.64, 1.8):
                b.cylinder(
                    base,
                    (x, side * 0.871, z),
                    0.014,
                    0.009,
                    "bright",
                    euler=(math.pi / 2, 0, 0),
                )
    for x in (-2.03, 2.03):
        b.box(base, (x, 0, 1.405), (0.020, 0.79, 0.505), "glass", collision=True)
    for y in (-0.84, 0.84):
        for z in (0.90, 1.91):
            b.box(base, (0, y, z), (2.03, 0.025, 0.032), "dark", collision=True)
            for i in range(21):
                b.cylinder(
                    base,
                    (-1.95 + i * 0.195, y - 0.033, z),
                    0.011,
                    0.008,
                    "bright",
                    euler=(math.pi / 2, 0, 0),
                )
    # Four outlets are external routing geometry only, with capped hose ends.
    for x in (-1.50, -0.50, 0.50, 1.50):
        b.cylinder(base, (x, 0, 0.65), 0.043, 0.215, "bright")
        b.cylinder(base, (x, 0, 0.45), 0.074, 0.018, "metal")
        b.rod(base, (x, 0, 0.45), (x, 0.68, 0.24), 0.033, "cream")
        b.box(base, (x, 0.73, 0.22), (0.075, 0.06, 0.075), "blue", collision=True)
    # Elevated feed reservoir is seated on an independently supported crossbar.
    for x in (-1.28, -0.32):
        b.box(base, (x, 0.35, 2.37), (0.035, 0.035, 0.425), "metal", collision=True)
    b.box(base, (-0.8, 0.35, 2.56), (0.52, 0.40, 0.035), "metal", collision=True)
    b.cylinder(base, (-0.8, 0.35, 2.79), 0.29, 0.195, "cream", collision=True)
    for z in (2.60, 2.98):
        b.cylinder(base, (-0.8, 0.35, z), 0.315, 0.018, "metal")
    # Reservoir supports extend to the tank's rear crossmember via a lower tie.
    b.box(base, (-0.8, 0.59, 1.975), (0.52, 0.295, 0.025), "metal", collision=True)
    for x in (-1.30, -0.30):
        b.cylinder(
            base, (x, -0.035, 2.63), 0.07, 0.020, "cream", euler=(math.pi / 2, 0, 0)
        )
        b.rod(base, (x - 0.025, -0.06, 2.60), (x + 0.025, -0.06, 2.66), 0.003, "dark")
    b.box(base, (-0.8, 0, 2.23), (0.11, 0.10, 0.27), "metal")
    b.rod(base, (-0.8, 0.35, 2.61), (-0.8, 0, 2.55), 0.05, "bright")
    b.cylinder(base, (-0.8, 0, 2.49), 0.045, 0.07, "bright")
    feed = b.moving(
        base,
        "feed_nozzle_height",
        (-0.8, 0, 2.05),
        (0, 0, 1),
        (-0.18, 0.12),
        mass=0.6,
        kp=400,
    )
    b.box(feed, (0, 0, 0), (0.12, 0.12, 0.045), "blue")
    b.cylinder(feed, (0, 0, -0.27), 0.027, 0.245, "bright", collision=True)
    b.cylinder(feed, (0, 0, -0.52), 0.043, 0.025, "cream", collision=True)
    b.site(feed, "feed_nozzle", (0, 0, -0.545))
    b.box(base, (0, 0.02, 2.91), (2.1, 0.035, 0.035), "metal", collision=True)
    probe = b.moving(
        base,
        "probe_traverse",
        (0.8, 0.02, 2.91),
        (1, 0, 0),
        (-0.55, 0.55),
        mass=0.5,
        kp=350,
    )
    b.box(probe, (0, 0, 0), (0.10, 0.065, 0.065), "blue")
    b.cylinder(probe, (0, 0, -0.88), 0.012, 0.82, "bright")
    b.cylinder(probe, (0, 0, -1.73), 0.048, 0.025, "orange", collision=True)
    b.site(probe, "calibration_target", (0, 0, -1.755))
    b.metadata["capabilities"] = [
        "representative feed-nozzle height adjustment",
        "diagnostic coupon traverse",
    ]
    b.metadata["limitations"].append(
        "Local rectangular viewing envelope approximates only the photographed front of a documented delta-tundish water model; hidden plan and all dimensions are unverified. Dry mechanical controls only, no water, residence time, mixing, metallurgy or casting model."
    )


def _mcgill_features(world, definition):
    # Dense low service zone and floor grating, not a generic office room.
    for i in range(19):
        box(
            world,
            f"mcgill_grating_long_{i}",
            (0.018, 0.43, 0.02),
            (-2.25 + i * 0.25, -1.40, 0.020),
            (0.19, 0.21, 0.22, 1),
        )
    for y in (-1.83, -1.4, -0.97):
        box(
            world,
            f"mcgill_grating_tie_{y}",
            (2.29, 0.020, 0.018),
            (0, y, 0.018),
            (0.19, 0.21, 0.22, 1),
        )
    box(
        world,
        "mcgill_overhead_duct",
        (2.55, 0.30, 0.16),
        (0, 1.8, 3.30),
        (0.65, 0.64, 0.59, 1),
    )
    for x in (-2.5, -2.2):
        ET.SubElement(
            world,
            "geom",
            name=f"mcgill_service_pipe_{x}",
            type="cylinder",
            size=".045 1.40",
            pos=f"{x} 1.68 1.4",
            rgba=".73 .75 .74 1",
        )
    for x in (-2.5, 2.5):
        box(
            world,
            f"mcgill_duct_hanger_{x}",
            (0.025, 0.025, 0.20),
            (x, 1.8, 3.5),
            (0.33, 0.35, 0.34, 1),
        )
    box(
        world,
        "mcgill_daq_case",
        (0.22, 0.08, 0.20),
        (2.2, -2.25, 1.18),
        (0.20, 0.22, 0.23, 1),
    )
    box(
        world,
        "mcgill_daq_screen",
        (0.19, 0.012, 0.15),
        (2.2, -2.339, 1.18),
        (0.025, 0.14, 0.17, 1),
        collision=False,
    )
    box(
        world,
        "mcgill_daq_stand",
        (0.045, 0.05, 0.065),
        (2.2, -2.25, 0.935),
        (0.26, 0.28, 0.28, 1),
    )
    box(
        world,
        "mcgill_daq_keyboard",
        (0.22, 0.095, 0.012),
        (2.2, -2.48, 0.912),
        (0.19, 0.22, 0.23, 1),
    )


BUILDERS["mmpc_tundish_water_model"] = _tundish
SOURCES["mmpc_tundish_water_model"] = {
    "reference": "McGill Metals Processing Centre full-scale four-strand delta-tundish water model photograph",
    "url": MCGILL,
    "dimensions_m": [4.6, 2.1, 3.1],
    "dimension_basis": "All dimensions estimated; source identifies full-scale delta tundish but does not provide calibrated dimensions or recoverable hidden plan",
}
SAMPLE_INTERFACES["mmpc_tundish_water_model"] = (
    "calibration_target",
    (0.096, 0.096, 0.05),
    "clamped",
    "rigid diagnostic target attached to the traversing shaft",
)
FEATURES["mcgill_mmpc_water_modelling"] = _mcgill_features
