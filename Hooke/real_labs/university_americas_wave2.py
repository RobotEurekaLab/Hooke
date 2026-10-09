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
    b.cylinder(base, (0, 0, 0.04), 0.53, 0.04, "rubber", collision=True)
    _frustum(b, base, 0.08, 0.78, 0.49, 0.26)
    _tag(b, b.cylinder(base, (0, 0, 0.805), 0.29, 0.025, "metal"), "turntable_bearing")
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
    _tag(
        b,
        b.cylinder(rotor, (0, 0, 0), 0.69, 0.025, "dark", collision=True),
        "rotating_platter",
    )
    b.cylinder(rotor, (0, 0, 0.034), 0.585, 0.009, "bright", collision=True)
    # Thin tangential panes retain an open tank rather than a solid glass plug.
    for i in range(48):
        a = i * math.tau / 48
        b.box(
            rotor,
            (0.59 * math.cos(a), 0.59 * math.sin(a), 0.205),
            (0.006, 0.039, 0.162),
            "glass",
            collision=True,
            euler=(0, 0, a),
        )
    for z in (0.043, 0.367):
        b.ring(rotor, (0, 0, z), 0.59, 0.006, "bright", segments=48)
    # Rigid calibration target represents a specimen only; there is no water.
    b.cylinder(rotor, (0.22, 0, 0.047), 0.045, 0.004, "orange", collision=True)
    for i in range(8):
        a = i * math.tau / 8
        b.rod(
            rotor,
            (0.16 * math.cos(a), 0.16 * math.sin(a), 0.045),
            (0.49 * math.cos(a), 0.49 * math.sin(a), 0.045),
            0.002,
            "dark",
        )
    b.site(rotor, "calibration_target", (0.22, 0, 0.051))
    # Separate floor-mounted imaging portal preserves a continuous load path.
    for x in (-0.88, 0.88):
        b.box(base, (x, 0.64, 0.025), (0.17, 0.20, 0.025), "metal", collision=True)
        b.box(base, (x, 0.64, 0.95), (0.032, 0.032, 0.925), "bright", collision=True)
        b.rod(base, (x, 0.64, 1.85), (x, 0, 1.85), 0.032, "bright")
    b.box(base, (0, 0, 1.85), (0.91, 0.035, 0.035), "bright", collision=True)
    camera = b.moving(
        base,
        "camera_traverse",
        (0, 0, 1.85),
        (1, 0, 0),
        (-0.34, 0.34),
        mass=0.4,
        kp=300,
    )
    # Bearing sleeve is visual because the prismatic joint represents contact.
    b.box(camera, (0, 0, 0), (0.08, 0.055, 0.050), "metal")
    b.box(camera, (0, 0, -0.10), (0.06, 0.045, 0.05), "dark", collision=True)
    b.cylinder(camera, (0, 0, -0.167), 0.026, 0.017, "lens", collision=True)
    b.site(camera, "camera_center", (0, 0, -0.184))
    b.metadata["capabilities"] = [
        "rotating-table inspection indexing",
        "overhead camera lateral positioning",
    ]
    b.metadata["limitations"].append(
        "Tank dimensions, indexing range and camera motorization are estimates; no fluid, Coriolis, optical-image or geophysical dynamics model."
    )


def _mit_features(world, definition):
    # Separate round rolling workstation visible in the foreground of the source.
    center = (1.55, -1.5)
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        x, y = center[0] + 0.51 * math.cos(a), center[1] + 0.51 * math.sin(a)
        ET.SubElement(
            world,
            "geom",
            name=f"mit_cart_wheel_{i}",
            type="cylinder",
            size=".08 .025",
            pos=f"{x} {y} .08",
            euler="1.5707963 0 0",
            rgba=".045 .05 .05 1",
        )
        box(
            world,
            f"mit_cart_post_{i}",
            (0.025, 0.025, 0.34),
            (x, y, 0.5),
            (0.70, 0.57, 0.37, 1),
        )
    for z in (0.18, 0.45, 0.86):
        ET.SubElement(
            world,
            "geom",
            name=f"mit_cart_round_shelf_{z}",
            type="cylinder",
            size=".69 .025",
            pos=f"{center[0]} {center[1]} {z}",
            rgba=".16 .15 .13 1" if z > 0.8 else ".73 .63 .46 1",
        )
    box(
        world,
        "mit_cart_camera_column",
        (0.028, 0.028, 0.42),
        (center[0], center[1] + 0.48, 1.305),
        (0.20, 0.22, 0.23, 1),
    )
    box(
        world,
        "mit_cart_camera_boom",
        (0.025, 0.27, 0.025),
        (center[0], center[1] + 0.21, 1.70),
        (0.20, 0.22, 0.23, 1),
    )
    box(
        world,
        "mit_cart_camera",
        (0.07, 0.055, 0.045),
        (center[0], center[1] - 0.03, 1.63),
        (0.05, 0.07, 0.08, 1),
    )
    # Tall window/blind wall at the right, as in the room photograph.
    for y in (-1.7, -0.1, 1.5):
        box(
            world,
            f"mit_window_{y}",
            (0.015, 0.65, 0.80),
            (3.24, y, 1.8),
            (0.55, 0.72, 0.82, 0.30),
            collision=False,
        )
        for i in range(17):
            box(
                world,
                f"mit_blind_{y}_{i}",
                (0.06, 0.66, 0.010),
                (3.20, y, 1.02 + i * 0.095),
                (0.87, 0.85, 0.79, 1),
                collision=False,
            )
    for x in (-2, -0.6, 0.8):
        box(
            world,
            f"mit_wall_cabinet_{x}",
            (0.61, 0.24, 0.38),
            (x, 2.42, 1.88),
            (0.83, 0.79, 0.66, 1),
        )
        box(
            world,
            f"mit_cabinet_seam_{x}",
            (0.006, 0.008, 0.35),
            (x, 2.171, 1.88),
            (0.30, 0.29, 0.25, 1),
            collision=False,
        )


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
    teal = (0.13, 0.46, 0.49, 1)
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
            _tag(
                b,
                b.box(
                    base,
                    (x, y, z + side * (half + 0.012)),
                    (0.46, half, 0.012),
                    "glass",
                    collision=side < 0,
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
    # Tapered casings are original external shells, not a flow mesh. Their axes
    # follow the photographed full return circuit, not a short tabletop tunnel.
    for sign in (-1, 1):
        body = ET.SubElement(
            base, "body", pos=f"{sign*1.55} {y} {z}", euler=f"0 {sign*math.pi/2} 0"
        )
        b.housing(
            body,
            [(0, 0.65, 0.65, 0), (2.30, 1.05, 1.05, 0)],
            radius=0.06,
            material="cyan",
        )
        for t in (0, 0.55, 1.10, 1.65, 2.3):
            w = 0.65 + t * 0.4 / 2.3
            for side in (-1, 1):
                b.box(
                    body, (side * (w + 0.015), 0, t), (0.025, w + 0.025, 0.025), "dark"
                )
                b.box(
                    body, (0, side * (w + 0.015), t), (w + 0.025, 0.025, 0.025), "dark"
                )
    for x in (-4.25, 4.25):
        b.box(base, (x, 0.40, z), (0.68, 2.8, 1.05), "cyan", collision=True, rgba=teal)
        for y2 in (-1.8, 2.6):
            b.box(base, (x, y2, 0.51), (0.46, 0.46, 0.51), "blue", collision=True)
            for dx in (-0.5, 0.5):
                b.rod(base, (x + dx, y2, 0.08), (x + dx, y2, 1.10), 0.032, "bright")
        for y2 in (-2.32, -1.0, 0.6, 2.20, 3.1):
            b.box(base, (x, y2, z), (0.70, 0.025, 1.08), "dark")
    b.box(base, (0, 2.6, z), (3.57, 0.82, 0.85), "cyan", collision=True, rgba=teal)
    for x in (-3.25, -1.8, 0, 1.8, 3.25):
        for dy in (-0.85, 0.85):
            b.box(base, (x, 2.6 + dy, z), (0.025, 0.026, 0.89), "dark")
    # Fan barrel and motor sit in the return leg; support piers reach its casing.
    b.cylinder(base, (1.8, 2.6, z), 1.02, 0.44, "cyan", euler=(0, math.pi / 2, 0))
    for x in (1.34, 2.26):
        b.cylinder(base, (x, 2.6, z), 1.06, 0.025, "metal", euler=(0, math.pi / 2, 0))
    b.box(base, (0, 2.6, 0.65), (0.48, 0.60, 0.65), "blue", collision=True)
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
    # Shaft passes a representative roof feedthrough; window/shaft bearing
    # contact is omitted, while the exposed measuring tip remains collidable.
    b.cylinder(probe, (0, 0, -0.33), 0.010, 0.30, "bright")
    b.cylinder(probe, (0, 0, -0.66), 0.018, 0.035, "orange", collision=True)
    b.site(probe, "probe_tip", (0, 0, -0.695))
    b.metadata["capabilities"] = [
        "inert model pitch adjustment",
        "test-section diagnostic traverse",
    ]
    b.metadata["limitations"].append(
        "Published 4 ft square test-section bore retained; loop geometry, model mount and inspection strokes are estimated. No airflow, fan, aerodynamic forces or PIV simulation."
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
    for i in range(5):
        b.box(
            arm,
            (-1.08, 0, 0.14 + i * 0.068),
            (0.34, 0.29, 0.030),
            "metal",
            collision=True,
        )
    for y in (-0.50, 0.50):
        b.box(arm, (1.45, y, -0.015), (0.085, 0.055, 0.105), "blue", collision=True)
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
    for y in (-0.46, 0.46):
        b.rod(basket, (0, y, 0), (0, y, -0.58), 0.028, "bright")
        b.cylinder(basket, (0, y, 0), 0.05, 0.026, "metal", euler=(math.pi / 2, 0, 0))
    b.box(basket, (0, 0, -0.63), (0.42, 0.46, 0.035), "cream", collision=True)
    for y in (-0.445, 0.445):
        b.box(basket, (0, y, -0.48), (0.42, 0.015, 0.115), "cream", collision=True)
        for i in range(17):
            a = math.pi + i * math.pi / 16
            b.rod(
                basket,
                (0.42 * math.cos(a), y, -0.39 + 0.24 * math.sin(a)),
                (
                    0.42 * math.cos(a + math.pi / 16),
                    y,
                    -0.39 + 0.24 * math.sin(a + math.pi / 16),
                ),
                0.024,
                "cream",
            )
    for x in (-0.405, 0.405):
        b.box(basket, (x, 0, -0.48), (0.015, 0.445, 0.115), "cream", collision=True)
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
