"""Original mechanical reconstructions of four documented European facilities.

Public images constrain exterior composition, not a surveyed digital twin.
Unknown internals, unmeasured dimensions and original inspection mechanics are
identified in equipment and scene metadata. Scientific process solvers omitted.
"""

import math
import xml.etree.ElementTree as ET
from real_labs.architecture import box

GLASGOW = "https://www.gla.ac.uk/research/az/jwnc/"
KTH = "https://www.kth.se/en/tekmek/infrastruktur/vindtunnel-l2000-1.1244292"
ETH = "https://highpressure.ethz.ch/facilities/experimental.html"
EPFL = "https://www.epfl.ch/labs/lch/infrastructure/hydraulic-laboratory/"
BUILDERS = {}
SOURCES = {}
SAMPLE_INTERFACES = {}
FEATURES = {}


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _wafer_loader(b, base, params):
    b.box(base, (0, 0, 0.014), (0.22, 0.17, 0.014), "metal", collision=True)
    for x in (-0.213, 0.213):
        b.box(base, (x, 0, 0.056), (0.007, 0.17, 0.028), "bright", collision=True)
    b.box(base, (0, 0.163, 0.062), (0.206, 0.007, 0.034), "bright", collision=True)
    for x in (-0.186, 0.186):
        for y in (-0.13, 0.12):
            b.cylinder(base, (x, y, 0.033), 0.007, 0.005, "dark")
    for x, key in ((-0.087, "left_carrier_y"), (0.087, "right_carrier_y")):
        for dx in (-0.048, 0.048):
            _tag(
                b,
                b.box(base, (x + dx, 0, 0.0415), (0.006, 0.128, 0.0135), "bright"),
                "wafer_carrier_guide",
            )
        carrier = b.moving(
            base, key, (x, 0, 0.064), (0, 1, 0), (-0.025, 0.025), mass=0.055, kp=300
        )
        _tag(
            b,
            b.box(carrier, (0, 0, 0), (0.073, 0.075, 0.009), "metal", collision=True),
            f"{key}_platform",
        )
        b.cylinder(carrier, (0, 0, 0.015), 0.062, 0.006, "bright", collision=True)
        _tag(
            b,
            b.cylinder(carrier, (0, 0, 0.0215), 0.050, 0.0005, "lens", collision=True),
            "retained_100mm_estimated_wafer",
        )
        for a in (0.15, 2.25, 4.35):
            clamp = ET.SubElement(
                carrier,
                "body",
                pos=f"{.059*math.cos(a)} {.059*math.sin(a)} 0",
                euler=f"0 0 {a}",
            )
            b.cylinder(clamp, (0, 0, 0.017), 0.006, 0.008, "copper")
            b.box(
                clamp,
                (-0.008, 0, 0.026),
                (0.014, 0.005, 0.004),
                "bright",
                collision=True,
            )
            b.cylinder(clamp, (0.002, 0, 0.031), 0.003, 0.001, "metal")
        b.box(carrier, (0.017, -0.012, 0.0227), (0.0025, 0.0018, 0.0007), "orange")
        b.site(
            carrier, "left_wafer" if x < 0 else "right_wafer", (0.017, -0.012, 0.0235)
        )
    # Only the photographed neighbouring guard edge is represented.
    b.box(base, (-0.295, 0.025, 0.085), (0.047, 0.215, 0.085), "dark", collision=True)
    b.box(base, (-0.246, 0.025, 0.119), (0.004, 0.20, 0.045), "metal")
    for y in (-0.12, 0.15):
        b.cylinder(
            base, (-0.238, y, 0.126), 0.008, 0.004, "bright", euler=(0, math.pi / 2, 0)
        )
    b.metadata["capabilities"] = [
        "left retained wafer carrier positioning",
        "right retained wafer carrier positioning",
    ]
    b.metadata["limitations"].append(
        "Photo-evidenced paired wafers, machined bed and clamp fingers; independent dry carrier drives are original and not an identified OEM loading mechanism. Estimated 100 mm disks are not a verified wafer diameter. Full cleanroom, lithography, etching, vacuum and fabrication response are omitted."
    )


BUILDERS["glasgow_jwnc_dual_wafer_carrier"] = _wafer_loader
SOURCES["glasgow_jwnc_dual_wafer_carrier"] = {
    "reference": "Glasgow JWNC official paired-wafer loading photograph",
    "url": GLASGOW,
    "dimensions_m": [0.562, 0.43, 0.17],
    "dimension_basis": "All dimensions estimated, including nominal 100 mm inert disks; apparatus model and full room plan unverified.",
}
SAMPLE_INTERFACES["glasgow_jwnc_dual_wafer_carrier"] = (
    "left_wafer",
    (0.10, 0.10, 0.001),
    "clamped",
    "left inert disk retained by three clamp fingers; second retained disk and site also provided",
)


def _l2000(b, base, params):
    # Published clear 2 m by 2 m octagon; axial length is an estimate.
    b.box(base, (0, 0, 0.16), (0.66, 2.80, 0.16), "metal", collision=True)
    q = math.sqrt(2) - 1
    vertices = [(-q, 1), (q, 1), (1, q), (1, -q), (q, -1), (-q, -1), (-1, -q), (-1, q)]
    for i, (x, z) in enumerate(vertices):
        xx, zz = vertices[(i + 1) % 8]
        dx, dz = xx - x, zz - z
        length = math.hypot(dx, dz)
        nx, nz = -dz / length, dx / length
        center = ((x + xx) / 2 + 0.04 * nx, 0, 1.40 + (z + zz) / 2 + 0.04 * nz)
        face = ET.SubElement(
            base,
            "body",
            pos=" ".join(map(str, center)),
            euler=f"0 {-math.atan2(dz,dx)} 0",
        )
        if i == 6:
            for y, h in ((-2.30, 0.50), (1.80, 1.0)):
                _tag(
                    b,
                    b.box(
                        face, (0, y, 0), (length / 2, h, 0.04), "shell", collision=True
                    ),
                    "octagon_face_6",
                )
            b.box(
                face, (0, -0.50, 0), (length / 2, 1.30, 0.016), "glass", collision=True
            )
            for y in (-1.80, 0.80):
                b.box(
                    face,
                    (0, y, -0.025),
                    (length / 2, 0.018, 0.016),
                    "metal",
                    collision=True,
                )
            for u in (-length / 2, length / 2):
                b.box(
                    face,
                    (u, -0.50, -0.025),
                    (0.020, 1.30, 0.016),
                    "metal",
                    collision=True,
                )
        else:
            _tag(
                b,
                b.box(
                    face, (0, 0, 0), (length / 2, 2.80, 0.040), "shell", collision=True
                ),
                f"octagon_face_{i}",
            )
        if i in (1, 3, 5, 7):
            for j in range(7):
                y = -2.38 + j * 0.72
                b.box(face, (0, y, -0.043), (0.19, 0.25, 0.006), "metal")
                b.box(face, (0, y, -0.051), (0.155, 0.21, 0.003), "cream")
    b.box(base, (0, 2.855, 1.4), (1.055, 0.020, 1.055), "dark", collision=True)
    # Thin triangulated rig stays inside the octagonal section.
    b.box(base, (0, 1.35, 0.422), (0.34, 0.37, 0.022), "metal", collision=True)
    for x, y in ((-0.30, 1.08), (0.30, 1.08), (0, 1.65)):
        b.geom(
            base,
            "capsule",
            (0.008,),
            material="metal",
            fromto=(x, y, 0.444, x * 0.25, 1.35 + (y - 1.35) * 0.25, 1.18),
            collision=True,
        )
        b.geom(
            base,
            "capsule",
            (0.006,),
            material="metal",
            fromto=(x, y, 0.444, -x * 0.25, 1.35 - (y - 1.35) * 0.25, 1.18),
            collision=True,
        )
    b.cylinder(base, (0, 1.35, 1.19), 0.11, 0.018, "metal", collision=True)
    yaw = b.moving(
        base,
        "model_yaw",
        (0, 1.35, 1.226),
        (0, 0, 1),
        (-0.28, 0.28),
        kind="hinge",
        mass=0.08,
        kp=105,
    )
    b.cylinder(yaw, (0, 0, 0), 0.082, 0.018, "dark", collision=True)
    for x in (-0.07, 0.07):
        b.box(yaw, (x, 0, 0.047), (0.014, 0.03, 0.03), "metal", collision=True)
        b.cylinder(yaw, (x, 0, 0.085), 0.018, 0.014, "metal", euler=(0, math.pi / 2, 0))
    pitch = b.moving(
        yaw,
        "model_pitch",
        (0, 0, 0.085),
        (1, 0, 0),
        (-0.17, 0.17),
        kind="hinge",
        mass=0.055,
        kp=95,
    )
    _tag(
        b,
        b.geom(
            pitch, "ellipsoid", (0.63, 0.13, 0.011), (0, 0, 0), "metal", collision=True
        ),
        "l2000_inert_wing",
    )
    b.geom(
        pitch,
        "ellipsoid",
        (0.065, 0.29, 0.040),
        (0, 0.045, 0.027),
        "dark",
        collision=True,
    )
    b.box(pitch, (0.40, -0.025, 0.014), (0.035, 0.012, 0.005), "orange")
    b.site(pitch, "wind_model", (0.40, -0.025, 0.020))
    # Adjacent cabinet is visible through the source's long observation window.
    b.box(base, (-1.65, -0.50, 0.04), (0.30, 0.38, 0.04), "metal", collision=True)
    b.box(base, (-1.65, -0.50, 0.91), (0.30, 0.38, 0.83), "cream", collision=True)
    b.box(base, (-1.341, -0.50, 1.17), (0.009, 0.23, 0.20), "screen")
    b.metadata["capabilities"] = [
        "dry wing-model yaw alignment",
        "dry wing-model pitch alignment",
    ]
    b.metadata["limitations"].append(
        "Official 2 m by 2 m octagonal clear section represented; 5.6 m axial segment, linkage and inert model are estimates. Only local tunnel section and adjacent cabinet; no airflow, pressure, aerodynamic forces, balance output or aeroelastic response."
    )


BUILDERS["kth_l2000_octagonal_section"] = _l2000
SOURCES["kth_l2000_octagonal_section"] = {
    "reference": "KTH official L2000 test-section photograph and 2 m by 2 m octagonal cross-section specification",
    "url": KTH,
    "dimensions_m": [3.03, 5.79, 2.48],
    "dimension_basis": "Clear 2 m width and height documented. Axial length, adjacent cabinet, model and pitch/yaw rig dimensions estimated.",
}
SAMPLE_INTERFACES["kth_l2000_octagonal_section"] = (
    "wind_model",
    (1.26, 0.58, 0.10),
    "clamped",
    "inert marked wing model held by an authored dry pitch/yaw fixture",
)


def _multianvil(b, base, params):
    for x in (-0.63, 0.63):
        for y in (-0.45, 0.45):
            b.box(base, (x, y, 0.04), (0.12, 0.12, 0.04), "metal", collision=True)
    b.box(base, (0, 0, 0.15), (0.69, 0.54, 0.07), "blue", collision=True)
    for x in (-0.46, 0.46):
        b.box(base, (x, 0.25, 1.205), (0.065, 0.16, 0.985), "blue", collision=True)
    for x in (-0.67, 0.67):
        brace = ET.SubElement(
            base, "body", pos=f"{x*.82} -.25 .69", euler=f"0 {-.21 if x>0 else .21} 0"
        )
        b.box(brace, (0, 0, 0), (0.045, 0.055, 0.49), "blue", collision=True)
    b.box(base, (0, 0.25, 2.05), (0.525, 0.24, 0.18), "blue", collision=True)
    b.cylinder(base, (0, 0.08, 0.86), 0.265, 0.64, "warning", collision=True)
    b.cylinder(base, (0, 0.08, 1.53), 0.34, 0.03, "metal", collision=True)
    for x in (-0.24, 0.24):
        _tag(
            b,
            b.box(base, (x, -0.22, 1.575), (0.018, 0.42, 0.015), "bright"),
            "multianvil_access_rail",
        )
    stage = b.moving(
        base,
        "specimen_drawer",
        (0, -0.15, 1.615),
        (0, -1, 0),
        (0, 0.22),
        mass=0.40,
        kp=450,
    )
    _tag(
        b,
        b.box(stage, (0, 0, 0), (0.30, 0.20, 0.025), "dark", collision=True),
        "multianvil_drawer_plate",
    )
    b.cylinder(stage, (0, 0, 0.040), 0.060, 0.015, "metal", collision=True)
    _tag(
        b,
        b.box(stage, (0, 0, 0.070), (0.024, 0.023, 0.015), "orange", collision=True),
        "multianvil_inert_cube",
    )
    b.site(stage, "multianvil_coupon", (0, -0.030, 0.085))
    # Side service access cannot energize the static press exterior.
    b.box(base, (0.60, 0.30, 0.72), (0.14, 0.05, 0.47), "blue", collision=True)
    hinge = b.moving(
        base,
        "service_panel_angle",
        (0.76, 0.30, 0.72),
        (0, 0, 1),
        (0, 0.65),
        kind="hinge",
        mass=0.16,
        kp=115,
    )
    b.box(hinge, (0, -0.27, 0), (0.012, 0.27, 0.44), "metal", collision=True)
    b.rod(hinge, (0.020, -0.45, -0.08), (0.020, -0.45, 0.08), 0.010, "dark")
    for z in (0.32, 1.12):
        b.cylinder(base, (0.76, 0.30, z), 0.024, 0.045, "metal")
    for x in (-0.31, 0.31):
        b.rod(base, (x, 0.43, 2.15), (x + 0.12, 0.50, 1.80), 0.016, "orange")
        b.rod(base, (x + 0.12, 0.50, 1.80), (x + 0.09, 0.41, 1.0), 0.016, "orange")
    b.metadata["capabilities"] = [
        "unpowered inert-specimen drawer access",
        "dry exterior service-panel opening",
    ]
    b.metadata["limitations"].append(
        "Blue/yellow front press exterior follows the room panorama. Drawer, side access and hidden mechanisms are authored dry proxies. No pressure generation, force calibration, hydraulic actuation, heating, phase change or high-pressure operating sequence."
    )


def _eth_features(world, definition):
    # The orange press/handling frame is a separate source-visible static shape.
    for x in (-3.1, -1.65):
        for y in (0.15, 1.75):
            box(
                world,
                f"eth_orange_post_{x}_{y}",
                (0.055, 0.055, 1.25),
                (x, y, 1.25),
                (0.85, 0.34, 0.035, 1),
            )
    for y in (0.15, 1.75):
        box(
            world,
            f"eth_orange_cross_{y}",
            (0.88, 0.075, 0.085),
            (-2.375, y, 2.52),
            (0.85, 0.34, 0.035, 1),
        )
    for x in (-3.1, -1.65):
        box(
            world,
            f"eth_orange_side_{x}",
            (0.075, 0.88, 0.085),
            (x, 0.95, 2.52),
            (0.85, 0.34, 0.035, 1),
        )
    box(
        world,
        "eth_orange_bottom",
        (0.85, 0.90, 0.07),
        (-2.375, 0.95, 0.07),
        (0.85, 0.34, 0.035, 1),
    )
    ET.SubElement(
        world,
        "geom",
        name="eth_red_static_ram",
        type="cylinder",
        size=".38 .59",
        pos="-2.375 .95 .73",
        rgba=".62 .035 .035 1",
    )
    box(
        world,
        "eth_red_ram_cap",
        (0.42, 0.42, 0.045),
        (-2.375, 0.95, 1.365),
        (0.50, 0.55, 0.57, 1),
    )
    # Right four-column press, with bracing and a large upper platen.
    for x in (2.15, 3.25):
        for y in (0.67, 1.83):
            box(
                world,
                f"eth_right_foot_{x}_{y}",
                (0.11, 0.11, 0.025),
                (x, y, 0.025),
                (0.35, 0.42, 0.46, 1),
            )
            ET.SubElement(
                world,
                "geom",
                name=f"eth_right_column_{x}_{y}",
                type="cylinder",
                size=".052 1.335",
                pos=f"{x} {y} 1.385",
                rgba=".62 .67 .70 1",
            )
    for z in (0.20, 1.22, 2.72):
        box(
            world,
            f"eth_right_platen_{z}",
            (0.67, 0.68, 0.07),
            (2.70, 1.25, z),
            (0.07, 0.12, 0.32, 1),
        )
    ET.SubElement(
        world,
        "geom",
        name="eth_right_upper_ram",
        type="cylinder",
        size=".29 .38",
        pos="2.70 1.25 2.27",
        rgba=".50 .57 .60 1",
    )
    # Inferred storage spindle joins the middle platen and every stored disk;
    # this load path is authored, not a measured internal press component.
    ET.SubElement(
        world,
        "geom",
        name="eth_right_inferred_storage_spindle",
        type="cylinder",
        size=".065 .3075",
        pos="2.33 1.23 1.5975",
        rgba=".46 .51 .55 1",
    )
    for i in range(5):
        ET.SubElement(
            world,
            "geom",
            name=f"eth_right_static_disc_{i}",
            type="cylinder",
            size=".30 .025",
            pos=f"2.33 1.23 {1.36+i*.13}",
            rgba=".06 .10 .28 1",
        )
    for x in (2.15, 3.25):
        ET.SubElement(
            world,
            "geom",
            name=f"eth_right_lower_brace_{x}",
            type="capsule",
            size=".032",
            fromto=f"{x} .67 .27 {x} 1.83 1.15",
            rgba=".06 .10 .28 1",
        )
    # Red overhead structural rails attach to ceiling hangers.
    for x in (-3.55, 0.0, 3.55):
        box(
            world,
            f"eth_red_rail_{x}",
            (0.055, 3.20, 0.085),
            (x, 0, 3.15),
            (0.67, 0.035, 0.035, 1),
        )
        for y in (-2.5, 0, 2.5):
            box(
                world,
                f"eth_rail_hanger_{x}_{y}",
                (0.025, 0.025, 0.14),
                (x, y, 3.375),
                (0.40, 0.43, 0.43, 1),
            )
    box(
        world,
        "eth_red_bridge",
        (3.55, 0.070, 0.080),
        (0, 0.65, 3.15),
        (0.70, 0.04, 0.035, 1),
    )
    # Computer table in the panorama, right of the front press.
    for x in (1.38, 1.95):
        box(
            world,
            f"eth_monitor_foot_{x}",
            (0.16, 0.11, 0.018),
            (x, -1.48, 0.868),
            (0.11, 0.13, 0.15, 1),
        )
        box(
            world,
            f"eth_monitor_stem_{x}",
            (0.021, 0.025, 0.095),
            (x, -1.42, 0.981),
            (0.35, 0.38, 0.38, 1),
        )
        box(
            world,
            f"eth_monitor_{x}",
            (0.225, 0.022, 0.14),
            (x, -1.42, 1.216),
            (0.055, 0.07, 0.09, 1),
        )
        box(
            world,
            f"eth_monitor_screen_{x}",
            (0.204, 0.003, 0.12),
            (x, -1.447, 1.216),
            (0.025, 0.10, 0.14, 1),
        )
    box(
        world,
        "eth_yellow_service_cabinet",
        (0.46, 0.32, 0.95),
        (3.20, 2.95, 0.95),
        (0.75, 0.65, 0.065, 1),
    )
    for i in range(3):
        box(
            world,
            f"eth_rear_step_{i}",
            (0.50, 0.16, 0.06 * (i + 1)),
            (3.25, 2.12 + i * 0.32, 0.06 * (i + 1)),
            (0.17, 0.20, 0.19, 1),
        )


BUILDERS["eth_multianvil_dry_access"] = _multianvil
SOURCES["eth_multianvil_dry_access"] = {
    "reference": "ETH Experimental Petrology official multianvil laboratory panorama",
    "url": ETH,
    "dimensions_m": [1.9, 1.72, 2.30],
    "dimension_basis": "All dimensions and original dry drawer/service-access travel estimated; panoramic perspective is not a survey.",
}
SAMPLE_INTERFACES["eth_multianvil_dry_access"] = (
    "multianvil_coupon",
    (0.048, 0.046, 0.030),
    "clamped",
    "inert solid coupon retained on an original unpowered access drawer",
)
FEATURES["eth_experimental_petrology_multianvil"] = _eth_features


def _lch_probe_rig(b, base, params):
    for x in (-0.94, 0.94):
        for y in (-1.52, 1.52):
            b.box(base, (x, y, 0.025), (0.11, 0.11, 0.025), "metal", collision=True)
            b.box(base, (x, y, 0.405), (0.035, 0.035, 0.355), "dark", collision=True)
    _tag(
        b,
        b.box(base, (0, 0, 0.81), (0.98, 1.58, 0.05), "metal", collision=True),
        "lch_bed_support",
    )
    for x in (-0.965, 0.965):
        b.box(base, (x, 0, 1.35), (0.015, 1.58, 0.49), "glass", collision=True)
        for y in (-1.56, -0.52, 0.52, 1.56):
            b.box(base, (x, y, 1.36), (0.035, 0.030, 0.50), "metal", collision=True)
    for y in (-1.56, 1.56):
        b.box(base, (0, y, 1.35), (0.965, 0.016, 0.49), "glass", collision=True)
    for x in (-0.96, 0.96):
        b.box(base, (x, 0, 1.84), (0.045, 1.59, 0.030), "metal", collision=True)
    # Narrow side lids preserve the source enclosure while exposing the proxy.
    for x in (-0.66, 0.66):
        b.box(base, (x, 0, 1.92), (0.30, 1.55, 0.04), "cream", collision=True)
    for x in (-1.05, 1.05):
        for y in (-1.70, 1.70):
            b.box(base, (x, y, 1.04), (0.032, 0.032, 1.04), "bright", collision=True)
        _tag(
            b,
            b.box(
                base,
                (x, 0, 2.02),
                (0.035, 1.73, 0.040),
                "metal",
                collision=True,
                friction="0 0 0",
                condim=1,
            ),
            "lch_traverse_rail",
        )
    # The ideal linear bearing has zero tangential friction but retains normal contact.
    traverse = b.moving(
        base, "probe_traverse", (0, 0, 2.085), (0, 1, 0), (-1.0, 1.0), mass=0.60, kp=620
    )
    _tag(
        b,
        b.box(
            traverse,
            (0, 0, 0),
            (1.085, 0.065, 0.025),
            "bright",
            collision=True,
            friction="0 0 0",
            condim=1,
        ),
        "lch_traverse_bridge",
    )
    lift = b.moving(
        traverse,
        "probe_height",
        (0, 0, -0.04),
        (0, 0, 1),
        (-0.25, 0),
        mass=0.20,
        kp=410,
    )
    b.box(lift, (0, 0, 0), (0.055, 0.05, 0.075), "dark")
    b.box(lift, (-0.018, 0, -0.43), (0.012, 0.022, 0.35), "bright", collision=True)
    _tag(
        b,
        b.cylinder(lift, (0, 0, -0.83), 0.035, 0.05, "metal", collision=True),
        "lch_dry_probe",
    )
    b.cylinder(
        lift, (0.045, 0, -0.80), 0.018, 0.045, "metal", euler=(0, math.pi / 2, 0)
    )
    b.box(base, (0, -0.05, 0.869), (0.030, 0.022, 0.009), "orange", collision=True)
    b.site(base, "lch_bed_coupon", (0, -0.05, 0.880))
    # Orange controller is a static exterior adjacent to the glass-sided rig.
    for x in (0.18, 1.12):
        for y in (-2.42, -1.98):
            b.box(base, (x, y, 0.055), (0.065, 0.060, 0.055), "dark", collision=True)
    b.box(base, (0.65, -2.20, 0.81), (0.56, 0.30, 0.70), "orange", collision=True)
    b.box(base, (0.65, -2.51, 0.99), (0.51, 0.012, 0.39), "orange")
    b.box(base, (0.80, -2.529, 1.10), (0.055, 0.004, 0.105), "shell")
    b.box(base, (0.80, -2.536, 1.13), (0.034, 0.003, 0.051), "screen")
    b.box(base, (0.65, -2.524, 0.45), (0.45, 0.015, 0.14), "dark")
    for i in range(7):
        b.cylinder(
            base,
            (0.29 + i * 0.12, -2.546, 0.43),
            0.018,
            0.005,
            "metal",
            euler=(math.pi / 2, 0, 0),
        )
    b.metadata["capabilities"] = [
        "dry longitudinal probe traverse",
        "bounded noncontact probe-height alignment",
    ]
    b.metadata["limitations"].append(
        "Selected LCH bay, not the full two-level 1775 square metre facility. Separate probe close-up informs style only; exact placement is unregistered. Open inspection strip, bridge and motions are original estimates; traverse bearing idealized as frictionless normal contact. Inert dry bed with no water, sediment transport, cavitation, erosion, pump or hydraulic response."
    )


def _lch_features(world, definition):
    # Long elevated left channel, independent from the active glass-rig proxy.
    for y in (-3.4, -1.9, -0.4, 1.1, 2.6, 4.1):
        for x in (-2.88, -2.32):
            box(
                world,
                f"lch_flume_leg_{x}_{y}",
                (0.025, 0.035, 0.625),
                (x, y, 0.625),
                (0.42, 0.43, 0.42, 1),
            )
    box(
        world,
        "lch_long_channel_bed",
        (0.34, 4.10, 0.05),
        (-2.60, 0.4, 1.30),
        (0.47, 0.40, 0.31, 1),
    )
    for x in (-2.93, -2.27):
        box(
            world,
            f"lch_long_glass_{x}",
            (0.008, 4.1, 0.21),
            (x, 0.4, 1.56),
            (0.40, 0.64, 0.68, 0.14),
        )
        box(
            world,
            f"lch_long_cap_{x}",
            (0.020, 4.12, 0.018),
            (x, 0.4, 1.788),
            (0.48, 0.45, 0.36, 1),
        )
    for i in range(96):
        y = -3.55 + (i % 32) * 0.253
        x = -2.60 + 0.245 * math.sin(i * 2.399)
        ET.SubElement(
            world,
            "geom",
            name=f"lch_inert_bed_stone_{i}",
            type="ellipsoid",
            size=f"{.010+(i%3)*.003} {.013+(i%4)*.002} .004",
            pos=f"{x} {y} 1.354",
            rgba=".38 .35 .29 1",
        )
    for y in (-3.3, -1.0, 1.3, 3.6):
        for x in (-2.20, -1.50):
            box(
                world,
                f"lch_walkway_leg_{x}_{y}",
                (0.025, 0.035, 0.525),
                (x, y, 0.525),
                (0.45, 0.48, 0.48, 1),
            )
        box(
            world,
            f"lch_rail_post_{y}",
            (0.022, 0.022, 0.54),
            (-1.48, y, 1.66),
            (0.62, 0.65, 0.65, 1),
        )
    box(
        world,
        "lch_walkway_platform",
        (0.38, 4.10, 0.05),
        (-1.88, 0.4, 1.10),
        (0.42, 0.45, 0.45, 1),
    )
    for z in (1.68, 2.20):
        box(
            world,
            f"lch_walkway_guard_{z}",
            (0.022, 4.10, 0.018),
            (-1.48, 0.4, z),
            (0.64, 0.67, 0.67, 1),
        )
    for i in range(7):
        box(
            world,
            f"lch_walkway_step_{i}",
            (0.38, 0.16, 0.075 * (i + 1)),
            (-1.88, -5.78 + i * 0.32, 0.075 * (i + 1)),
            (0.42, 0.46, 0.46, 1),
        )
    # Source-visible inclined carrier; geometry is static and fully supported.
    for x in (-0.52, -0.10):
        ET.SubElement(
            world,
            "geom",
            name=f"lch_inclined_beam_{x}",
            type="capsule",
            size=".025",
            fromto=f"{x} 1.0 .12 {x} 2.8 2.30",
            rgba=".58 .63 .60 1",
        )
        box(
            world,
            f"lch_inclined_rear_post_{x}",
            (0.035, 0.035, 1.15),
            (x, 2.8, 1.15),
            (0.45, 0.49, 0.47, 1),
        )
    for i in range(12):
        y = 1.0 + i * 1.8 / 11
        z = 0.12 + i * 2.18 / 11
        box(
            world,
            f"lch_carrier_rung_{i}",
            (0.23, 0.025, 0.016),
            (-0.31, y, z),
            (0.68, 0.70, 0.63, 1),
        )
    # Yellow elevated operator booth and yellow bridge crane, not articulated.
    for x in (1.10, 2.90):
        for y in (3.35, 5.05):
            box(
                world,
                f"lch_booth_post_{x}_{y}",
                (0.075, 0.075, 1.15),
                (x, y, 1.15),
                (0.42, 0.43, 0.34, 1),
            )
    box(
        world,
        "lch_booth_floor",
        (1.06, 1.02, 0.06),
        (2.0, 4.20, 2.36),
        (0.75, 0.64, 0.08, 1),
    )
    box(
        world,
        "lch_booth_lower_front",
        (1.0, 0.035, 0.40),
        (2.0, 3.25, 2.82),
        (0.77, 0.68, 0.13, 1),
    )
    box(
        world,
        "lch_booth_back",
        (1.0, 0.035, 0.86),
        (2.0, 5.15, 3.28),
        (0.62, 0.61, 0.32, 1),
    )
    for x in (1.0, 3.0):
        box(
            world,
            f"lch_booth_side_{x}",
            (0.035, 0.95, 0.40),
            (x, 4.20, 2.82),
            (0.77, 0.68, 0.13, 1),
        )
        box(
            world,
            f"lch_booth_side_glass_{x}",
            (0.012, 0.90, 0.37),
            (x, 4.20, 3.61),
            (0.17, 0.27, 0.30, 0.7),
        )
        for y in (3.25, 4.20, 5.15):
            box(
                world,
                f"lch_booth_mullion_{x}_{y}",
                (0.026, 0.026, 0.39),
                (x, y, 3.61),
                (0.47, 0.46, 0.28, 1),
            )
    box(
        world,
        "lch_booth_front_glass",
        (0.95, 0.012, 0.37),
        (2.0, 3.25, 3.61),
        (0.17, 0.27, 0.30, 0.7),
    )
    box(
        world,
        "lch_booth_roof",
        (1.06, 1.02, 0.055),
        (2.0, 4.20, 4.045),
        (0.77, 0.68, 0.13, 1),
    )
    for x in (-5.5, 5.5):
        for y in (-3.0, 4.8):
            box(
                world,
                f"lch_hall_crane_column_{x}_{y}",
                (0.13, 0.15, 3.15),
                (x, y, 3.15),
                (0.25, 0.28, 0.27, 1),
            )
        box(
            world,
            f"lch_hall_runway_{x}",
            (0.16, 5.75, 0.20),
            (x, 0, 6.5),
            (0.72, 0.56, 0.04, 1),
        )
    box(
        world,
        "lch_yellow_bridge",
        (5.50, 0.22, 0.25),
        (0, 3.7, 6.50),
        (0.80, 0.64, 0.055, 1),
    )


BUILDERS["epfl_lch_dry_probe_bay"] = _lch_probe_rig
SOURCES["epfl_lch_dry_probe_bay"] = {
    "reference": "EPFL LCH official hydraulic hall overview and separate SedMix probe detail",
    "url": EPFL,
    "dimensions_m": [2.30, 4.24, 2.16],
    "dimension_basis": "All local apparatus and selected-bay dimensions estimated. Published 1775 square metre total two-level area is not used as a local footprint.",
}
SAMPLE_INTERFACES["epfl_lch_dry_probe_bay"] = (
    "lch_bed_coupon",
    (0.060, 0.044, 0.018),
    "clamped",
    "fixed inert bed coupon below the dry traverse-and-lift probe",
)
FEATURES["epfl_lch_hydraulic_flume"] = _lch_features
