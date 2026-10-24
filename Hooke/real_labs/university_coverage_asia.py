"""Six bounded university workcells with original, source-informed geometry.

Installed photographs constrain visible composition only. All hidden mechanisms,
travel and dimensions are estimates unless the individual source says otherwise.
None of these dry inspection proxies simulates the associated scientific process.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

BUILDERS = {}
SOURCES = {}
SAMPLE_INTERFACES = {}
FEATURES = {}


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _register(kind, builder, url, dimensions, site, size, basis):
    BUILDERS[kind] = builder
    SOURCES[kind] = {
        "reference": "Official installed laboratory photograph; original dry inspection mechanism",
        "url": url,
        "dimensions_m": dimensions,
        "dimension_basis": basis,
    }
    SAMPLE_INTERFACES[kind] = (
        site,
        size,
        "clamped",
        "Inert calibration target rigidly retained by the depicted fixture; no scientific response.",
    )


def _legs(b, base, half_x, half_y, height, material="metal"):
    for x in (-half_x, half_x):
        for y in (-half_y, half_y):
            b.box(base, (x, y, 0.025), (0.075, 0.075, 0.025), "dark", collision=True)
            b.box(
                base,
                (x, y, (height + 0.05) / 2),
                (0.035, 0.035, (height - 0.05) / 2),
                material,
                collision=True,
            )


def _akta(b, base, params):
    # The installed picture establishes collector-left / valve-panel-right adjacency.
    b.box(base, (0.21, 0.05, 0.028), (0.275, 0.22, 0.028), "guard_red", collision=True)
    b.box(base, (0.21, 0.16, 0.36), (0.275, 0.11, 0.304), "shell", collision=True)
    for x in (-0.075, 0.495):
        b.box(base, (x, 0.05, 0.365), (0.018, 0.23, 0.31), "guard_red", collision=True)
    for z in (0.17, 0.33, 0.49, 0.64):
        b.box(base, (0.21, 0.037, z), (0.258, 0.006, 0.006), "metal")
    for i, (x, z) in enumerate(
        (
            (-0.01, 0.54),
            (0.15, 0.54),
            (0.36, 0.37),
            (0.02, 0.24),
            (0.20, 0.14),
            (0.39, 0.14),
        )
    ):
        b.cylinder(base, (x, 0.01, z), 0.037, 0.025, "dark", euler=(math.pi / 2, 0, 0))
        for a in range(5):
            b.geom(
                base,
                "sphere",
                (0.005,),
                (
                    x + 0.026 * math.cos(a * 1.257),
                    -0.018,
                    z + 0.026 * math.sin(a * 1.257),
                ),
                "bright",
            )
    for x, top in ((0.09, 0.89), (0.27, 0.78), (0.36, 0.77)):
        b.cylinder(base, (x, -0.065, 0.66), 0.015, 0.12, "glass")
        b.cylinder(base, (x, -0.065, top), 0.020, 0.025, "guard_red")
        b.box(base, (x, 0.015, 0.65), (0.027, 0.09, 0.015), "dark")
    for i in range(7):
        x = 0.02 + i * 0.052
        b.rod(base, (x, -0.025, 0.58), (x + 0.035, -0.075, 0.38), 0.002, "cream")
        b.rod(base, (x + 0.035, -0.075, 0.38), (x - 0.02, -0.025, 0.15), 0.002, "cream")
    b.cylinder(base, (-0.35, 0, 0.055), 0.185, 0.055, "shell", collision=True)
    carousel = b.moving(
        base,
        "collector_angle",
        (-0.35, 0, 0.145),
        (0, 0, 1),
        (-1.15, 1.15),
        kind="hinge",
        mass=0.12,
        kp=65,
    )
    _tag(
        b,
        b.cylinder(carousel, (0, 0, 0), 0.17, 0.029, "dark", collision=True),
        "collector_seat",
    )
    for ring, n in ((0.08, 12), (0.135, 20)):
        for i in range(n):
            a = 2 * math.pi * i / n
            b.cylinder(
                carousel,
                (ring * math.cos(a), ring * math.sin(a), 0.043),
                0.009,
                0.014,
                "cream",
                collision=True,
            )
    b.site(carousel, "retained_fraction_tube", (0.135, 0, 0.058))
    # Lift is an authored inspection surrogate, not an asserted OEM control axis.
    b.box(base, (-0.35, 0.20, 0.21), (0.025, 0.025, 0.21), "metal", collision=True)
    arm = b.moving(
        base,
        "nozzle_height",
        (-0.35, 0.14, 0.38),
        (0, 0, 1),
        (0, 0.06),
        mass=0.04,
        kp=220,
    )
    b.box(arm, (0, -0.08, 0), (0.022, 0.11, 0.018), "dark", collision=True)
    b.cylinder(arm, (0, -0.14, -0.045), 0.005, 0.027, "bright", collision=True)
    b.site(arm, "nozzle_tip", (0, -0.14, -0.073))
    b.metadata["capabilities"] = [
        "retained fraction-tube indexing",
        "dry collection-nozzle inspection lift",
    ]
    b.metadata["limitations"].append(
        "Kurumizaka AKTA pure photograph anchors the local workstation; no CKX53 adjacency is inferred. Collector motion and nozzle lift are estimated original drives. No chromatography, pressure, liquid transport or fraction chemistry."
    )


_register(
    "tokyo_akta_dry_collector",
    _akta,
    "https://www.iqb.u-tokyo.ac.jp/kurumizakalab/eng/research_kizai.html",
    [1.08, 0.48, 0.92],
    "retained_fraction_tube",
    [0.018, 0.018, 0.028],
    "All dimensions and travel estimated from the installed AKTA pure workstation photograph; no OEM CAD or surveyed room.",
)


def _rotavap(b, base, params):
    b.box(base, (0, 0, 0.018), (0.30, 0.25, 0.018), "metal", collision=True)
    b.box(base, (-0.15, 0.12, 0.30), (0.055, 0.055, 0.264), "shell", collision=True)
    b.box(base, (-0.15, 0.12, 0.58), (0.065, 0.065, 0.016), "dark", collision=True)
    # Hollow unheated bath, with independent bottom and collision-bearing walls.
    b.cylinder(base, (0.16, -0.055, 0.075), 0.145, 0.04, "blue", collision=True)
    b.cylinder(base, (0.16, -0.055, 0.121), 0.143, 0.006, "metal", collision=True)
    for i in range(36):
        a = 2 * math.pi * i / 36
        b.box(
            base,
            (0.16 + 0.144 * math.cos(a), -0.055 + 0.144 * math.sin(a), 0.202),
            (0.013, 0.009, 0.075),
            "cream",
            euler=(0, 0, a),
            collision=True,
        )
    lift = b.moving(
        base, "glass_lift", (-0.15, 0.12, 0.38), (0, 0, 1), (0, 0.10), mass=0.20, kp=420
    )
    b.box(lift, (0.18, -0.055, 0.02), (0.11, 0.085, 0.065), "blue", collision=True)
    # Rotation axis follows the inclined flask neck; glass remains above bath base.
    axis = (math.sqrt(0.5), 0, -math.sqrt(0.5))
    spindle = b.moving(
        lift,
        "flask_rotation",
        (0.17, -0.175, -0.015),
        axis,
        (-1.2, 1.2),
        kind="hinge",
        mass=0.035,
        kp=55,
    )
    b.rod(spindle, (0, 0, 0), (0.10, 0, -0.10), 0.014, "glass")
    _tag(
        b,
        b.geom(spindle, "sphere", (0.057,), (0.14, 0, -0.14), "glass", collision=True),
        "retained_rotary_flask",
    )
    # A small dry solid makes rotation observable without inventing liquid motion.
    b.geom(
        spindle, "sphere", (0.008,), (0.158, 0.016, -0.145), "sample", collision=True
    )
    b.site(spindle, "dry_flask_marker", (0.158, 0.016, -0.145))
    b.cylinder(lift, (0.02, -0.07, 0.23), 0.032, 0.17, "glass")
    for j in range(16):
        b.ring(
            lift, (0.02, -0.07, 0.075 + j * 0.019), 0.021, 0.0025, "cyan", segments=12
        )
    b.cylinder(lift, (0.02, -0.07, 0.41), 0.036, 0.012, "guard_red")
    b.rod(lift, (0.02, -0.07, 0.05), (-0.06, -0.14, -0.04), 0.012, "glass")
    b.geom(lift, "sphere", (0.052,), (-0.06, -0.14, -0.09), "glass", collision=True)
    b.metadata["capabilities"] = ["dry flask assembly lift", "bounded flask rotation"]
    b.metadata["limitations"].append(
        "Zhao laboratory C805-6 photograph shows serial blue rotary evaporators beside a reagent shelf and window. Device identity, dimensions, travel and glass support details estimated. Dry bath and rigid glass: no heating, vacuum, cooling, evaporation or solvent chemistry."
    )


_register(
    "peking_zhao_rotary_evaporator",
    _rotavap,
    "https://www.chem.pku.edu.cn/zhaodh/2023/galleries.html",
    [0.68, 0.54, 0.94],
    "dry_flask_marker",
    [0.016, 0.016, 0.016],
    "Estimated envelope at full lift; no make/model or measured glassware dimensions inferred from the photograph.",
)


def _korea_tunnel(b, base, params):
    _legs(b, base, 1.06, 1.42, 0.40, "blue")
    b.box(base, (0, 0, 0.44), (1.13, 1.50, 0.04), "metal", collision=True)
    for x in (-1.10, 1.10):
        b.box(base, (x, 0, 1.26), (0.012, 1.50, 0.78), "glass", collision=True)
        for y in (-1.47, -0.49, 0.49, 1.47):
            b.box(base, (x, y, 1.26), (0.03, 0.03, 0.82), "metal", collision=True)
        b.box(base, (x, 0, 2.08), (0.035, 1.52, 0.035), "metal", collision=True)
    for y in (-1.51, 1.51):
        b.box(base, (0, y, 1.26), (1.13, 0.02, 0.78), "glass", collision=True)
    b.box(base, (0, 0, 2.12), (1.13, 1.52, 0.015), "glass", collision=True)
    # Paired circular alignment frames and under-wing brass counterweight cues.
    for x in (-0.95, 0.95):
        b.ring(base, (x, 0, 1.34), 0.43, 0.018, "bright", plane="yz", segments=40)
        b.cylinder(base, (x, 0.32, 0.69), 0.10, 0.21, "copper", collision=True)
        b.box(base, (x, 0, 0.83), (0.05, 0.06, 0.37), "bright", collision=True)
    lift = b.moving(
        base, "model_heave", (0, 0, 1.34), (0, 0, 1), (-0.055, 0.055), mass=0.12, kp=380
    )
    from real_labs.mounting import annular_mount

    for x in (-0.98, 0.98):
        annular_mount(
            b,
            lift,
            f"pitch_bearing_{x}",
            (x, 0, 0),
            0.021,
            0.034,
            0.018,
            axis="x",
            proxies=True,
            material="metal",
        )
    pitch = b.moving(
        lift,
        "model_pitch",
        (0, 0, 0),
        (1, 0, 0),
        (-0.16, 0.16),
        kind="hinge",
        mass=0.06,
        kp=75,
    )
    # Shaft and bridge section are one rotating rigid assembly.
    b.cylinder(
        pitch,
        (0, 0, 0),
        0.018,
        1.01,
        "metal",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    _tag(
        b,
        b.box(pitch, (0, 0, 0), (0.85, 0.19, 0.018), "shell", collision=True),
        "retained_bridge_section",
    )
    for x in (-0.83, 0.83):
        b.box(pitch, (x, 0, 0.10), (0.014, 0.21, 0.12), "shell", collision=True)
    b.box(pitch, (0.11, -0.07, 0.021), (0.012, 0.012, 0.003), "orange", collision=True)
    b.site(pitch, "model_marker", (0.11, -0.07, 0.025))
    b.metadata["capabilities"] = [
        "bounded bridge-section heave alignment",
        "bridge-section pitch alignment",
    ]
    b.metadata["limitations"].append(
        "Korea ACE photographs show glazed frame, circular alignment rings, suspended bridge section and brass cylindrical weights. Authored kinematic heave/pitch mount does not reproduce the IFSS spring system. No flow, aerodynamic forces, damping identification or aeroelastic response."
    )


_register(
    "korea_ace_bridge_section",
    _korea_tunnel,
    "https://ace.korea.ac.kr/sub4_4.php",
    [2.3, 3.1, 2.15],
    "model_marker",
    [0.024, 0.024, 0.006],
    "All room, section dimensions and heave/pitch ranges estimated; IFSS performance is not reproduced.",
)


def _gap_rig(b, base, params):
    # Unenergized electrode metrology proxy with red insulating columns.
    b.box(base, (0, 0, 0.055), (1.02, 0.53, 0.055), "metal", collision=True)
    for x in (-0.77, 0.77):
        b.cylinder(base, (x, 0.10, 0.61), 0.12, 0.50, "guard_red", collision=True)
        for z in (0.14, 0.38, 0.63, 0.88, 1.11):
            b.cylinder(base, (x, 0.10, z), 0.15, 0.025, "bright", collision=True)
    b.box(base, (0, 0.10, 1.16), (0.86, 0.09, 0.035), "bright", collision=True)
    b.box(base, (-0.64, -0.02, 1.31), (0.05, 0.16, 0.115), "metal", collision=True)
    b.rod(base, (-0.64, -0.12, 1.34), (-0.25, -0.12, 1.34), 0.025, "bright")
    b.geom(base, "sphere", (0.10,), (-0.25, -0.12, 1.34), "bright", collision=True)
    slide = b.moving(
        base,
        "electrode_gap",
        (0.42, -0.02, 1.238),
        (1, 0, 0),
        (0, 0.15),
        mass=0.11,
        kp=320,
    )
    b.box(slide, (0, 0.08, 0), (0.09, 0.13, 0.035), "dark", collision=True)
    b.box(slide, (0, -0.05, 0.11), (0.06, 0.01, 0.10), "bright", collision=True)
    lift = b.moving(
        slide,
        "electrode_height",
        (0, -0.10, 0.102),
        (0, 0, 1),
        (-0.04, 0.04),
        mass=0.05,
        kp=260,
    )
    b.box(lift, (0, 0, 0), (0.035, 0.035, 0.075), "metal", collision=True)
    b.rod(lift, (0, 0, 0), (-0.20, 0, 0), 0.025, "bright")
    _tag(
        b,
        b.geom(lift, "sphere", (0.10,), (-0.20, 0, 0), "bright", collision=True),
        "dry_sphere_electrode",
    )
    b.site(lift, "electrode_witness", (-0.30, 0, 0))
    b.metadata["capabilities"] = [
        "unenergized electrode gap alignment",
        "unenergized electrode height alignment",
    ]
    b.metadata["limitations"].append(
        "Official UMHVL images establish red cylindrical apparatus and yellow cage; this two-axis sphere-gap fixture is an original dry metrology surrogate. No energization, electrical discharge, capacitance, dielectric breakdown or certified electrical safety model."
    )


_register(
    "malaya_unenergized_gap_fixture",
    _gap_rig,
    "https://ee.um.edu.my/research-area",
    [2.08, 1.10, 1.52],
    "electrode_witness",
    [0.20, 0.20, 0.20],
    "All sizes and two-axis dry metrology fixture estimated; no electrical ratings or real clearances claimed.",
)


def _melbourne_tunnel(b, base, params):
    # Documented clear cross section; only an estimated 4 m local bay is modeled.
    _legs(b, base, 0.88, 1.80, 0.72)
    b.box(base, (0, 0, 0.74), (0.99, 2.02, 0.02), "shell", collision=True)
    for x in (-0.965, 0.965):
        b.box(base, (x, 0, 1.22), (0.02, 2.02, 0.46), "glass", collision=True)
        for y in (-2, -1, 0, 1, 2):
            b.box(
                base,
                (math.copysign(1.0, x), y, 1.22),
                (0.035, 0.025, 0.46),
                "metal",
                collision=True,
            )
    b.box(base, (0, 0, 1.70), (0.99, 2.02, 0.02), "shell", collision=True)
    # Fig. 8 of Rendon-Cardona et al. (2020) shows a right-hand vertical
    # traverse with black pads and a long upstream-pointing horizontal probe.
    for x in (0.30, 0.50):
        b.box(base, (x, 0.20, 0.78), (0.025, 1.10, 0.020), "bright", collision=True)
    traverse = b.moving(
        base,
        "probe_span",
        (0.40, 0.30, 0.83),
        (0, 1, 0),
        (-0.45, 0.45),
        mass=0.10,
        kp=300,
    )
    b.box(traverse, (0, 0, 0), (0.12, 0.12, 0.020), "metal", collision=True)
    b.box(traverse, (0, 0, 0.34), (0.06, 0.07, 0.32), "bright", collision=True)
    for z in (0.12, 0.24, 0.36, 0.48, 0.60):
        b.box(traverse, (0, -0.073, z), (0.032, 0.004, 0.035), "dark")
    lift = b.moving(
        traverse,
        "probe_height",
        (0, -0.11, 0.49),
        (0, 0, 1),
        (-0.20, 0),
        mass=0.035,
        kp=250,
    )
    b.box(lift, (0, 0, 0), (0.035, 0.025, 0.035), "metal", collision=True)
    b.rod(lift, (0, -0.025, 0), (0, -0.60, -0.015), 0.008, "bright")
    b.rod(lift, (0, -0.60, -0.015), (0, -0.77, -0.025), 0.002, "bright")
    _tag(
        b,
        b.geom(lift, "sphere", (0.004,), (0, -0.77, -0.025), "orange", collision=True),
        "dry_probe_tip",
    )
    b.site(lift, "probe_marker", (0, -0.77, -0.025))
    for j in range(8):
        b.box(base, (0.40, -0.54 + j * 0.025, 0.762), (0.020, 0.005, 0.002), "copper")
    b.metadata["capabilities"] = [
        "dry streamwise probe traverse",
        "dry probe-height positioning",
    ]
    b.metadata["limitations"].append(
        "Published HRNBLWT cross section is 1.89 m by 0.92 m; 27 m full working section is not modeled. The local 4 m bay, floor traverse support and travel are original estimates. No turbulence, air flow, heat transfer, hot-wire response or calibrated measurements."
    )


_register(
    "melbourne_hrnblwt_probe",
    _melbourne_tunnel,
    "https://mechanical.eng.unimelb.edu.au/research/fluid-mechanics/research/walter-bassett",
    [2.02, 4.10, 1.75],
    "probe_marker",
    [0.016, 0.016, 0.016],
    "Published clear cross-section 1.89 by 0.92 m; local 4 m length, supports, probe mechanism and travel estimated. Full 27 m tunnel omitted.",
)


def _seismic_table(b, base, params):
    # The photographed seismic platform is a fixed 2.4 m by 1.2 m table.
    # Only an authored specimen-positioning fixture moves; this is not a shaker.
    _legs(b, base, 1.12, 0.52, 0.70, "blue")
    for y in (-0.52, 0.52):
        for height in (0.26, 0.72):
            b.box(base, (0, y, height), (1.16, 0.04, 0.04), "blue", collision=True)
    for x in (-1.12, 1.12):
        b.box(base, (x, 0, 0.72), (0.04, 0.56, 0.04), "blue", collision=True)
    _tag(
        b,
        b.box(base, (0, 0, 0.80), (1.20, 0.60, 0.04), "bright", collision=True),
        "fixed_seismic_deck",
    )
    for x in range(-11, 12):
        for y in range(-5, 6):
            b.cylinder(base, (x * 0.10, y * 0.10, 0.841), 0.007, 0.001, "dark")
    for y in (-0.22, 0.22):
        b.box(base, (0.12, y, 0.86), (0.57, 0.025, 0.02), "metal", collision=True)
    travel = b.moving(
        base,
        "specimen_translation",
        (0.12, 0, 0.895),
        (1, 0, 0),
        (-0.08, 0.08),
        mass=0.25,
        kp=450,
    )
    b.box(travel, (0, 0, 0), (0.48, 0.27, 0.010), "metal", collision=True)
    yaw = b.moving(
        travel,
        "coupon_yaw",
        (0, 0, 0.030),
        (0, 0, 1),
        (-0.30, 0.30),
        kind="hinge",
        mass=0.10,
        kp=85,
    )
    b.cylinder(yaw, (0, 0, 0), 0.16, 0.010, "dark", collision=True)
    _tag(
        b,
        b.box(yaw, (0, 0, 0.024), (0.38, 0.22, 0.014), "metal", collision=True),
        "retained_rock_coupon",
    )
    for x in (-0.42, 0.42):
        b.box(yaw, (x, 0, 0.019), (0.035, 0.24, 0.014), "bright", collision=True)
    for i in range(9):
        b.box(yaw, (-0.28 + i * 0.06, 0.04, 0.040), (0.010, 0.016, 0.002), "copper")
    b.site(yaw, "rock_coupon", (0.04, 0.02, 0.041))
    # Source-visible launcher and narrow specimen rail are stationary exteriors.
    b.box(base, (-0.87, 0, 0.90), (0.22, 0.09, 0.060), "metal", collision=True)
    b.cylinder(base, (-0.87, 0, 1.00), 0.025, 0.22, "bright", euler=(0, math.pi / 2, 0))
    b.metadata["capabilities"] = [
        "slow retained-specimen translation",
        "retained plate indexing",
    ]
    b.metadata["limitations"].append(
        "The table is fixed. Only an original inspection fixture moves. "
        "The 2.4 by 1.2 m deck is published; table height, sensor pattern, fixture, mounting clearance and travel are estimated. "
        "No impact launch, seismic waveform, rock fracture, strain or damping model."
    )


_register(
    "monash_seismic_dry_platform",
    _seismic_table,
    "https://www.monash.edu/engineering/mge/facilities",
    [2.42, 1.22, 1.08],
    "rock_coupon",
    [0.76, 0.44, 0.028],
    "Published table footprint 2.4 by 1.2 m; height, hole pattern and original inspection fixture estimated from installed platform photograph.",
)


def _pku_features(world, definition):
    # Visible reagent shelving and a second stationary blue evaporator silhouette.
    for z in (1.46, 1.90):
        box(
            world,
            f"pku_shelf_{z}",
            (1.22, 0.16, 0.018),
            (0, 1.38, z),
            (0.78, 0.81, 0.80, 1),
        )
        for i in range(9):
            x = -1.04 + i * 0.25
            box(
                world,
                f"pku_bottle_{z}_{i}",
                (0.05, 0.05, 0.10),
                (x, 1.38, z + 0.118),
                (0.91, 0.94, 0.88, 1),
            )
            box(
                world,
                f"pku_cap_{z}_{i}",
                (0.035, 0.035, 0.015),
                (x, 1.38, z + 0.233),
                (0.03, 0.18, 0.55, 1),
            )
    box(
        world,
        "pku_neighbor_base",
        (0.24, 0.23, 0.02),
        (-0.80, 0.65, 0.92),
        (0.18, 0.22, 0.25, 1),
    )
    box(
        world,
        "pku_neighbor_tower",
        (0.05, 0.06, 0.24),
        (-0.94, 0.78, 1.18),
        (0.80, 0.84, 0.85, 1),
    )
    box(
        world,
        "pku_neighbor_motor",
        (0.10, 0.08, 0.065),
        (-0.85, 0.65, 1.36),
        (0.06, 0.25, 0.67, 1),
    )
    ET.SubElement(
        world,
        "geom",
        name="pku_neighbor_bath",
        type="cylinder",
        pos="-.68 .57 1.08",
        size=".15 .14",
        rgba=".87 .89 .86 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="pku_neighbor_condenser",
        type="cylinder",
        pos="-.85 .65 1.62",
        size=".034 .19",
        rgba=".33 .72 .79 .30",
    )


def _tokyo_features(world, definition):
    for i in range(4):
        box(
            world,
            f"tokyo_wall_note_{i}",
            (0.075, 0.002, 0.10),
            (0.30 + i * 0.14, 1.488, 1.65),
            (0.94, 0.94, 0.90, 1),
        )


def _korea_features(world, definition):
    for y in (-2, 0, 2):
        box(
            world,
            f"korea_overhead_beam_{y}",
            (2.9, 0.065, 0.10),
            (0, y, 3.05),
            (0.33, 0.39, 0.39, 1),
        )
    box(
        world,
        "korea_back_duct",
        (2.45, 0.36, 0.32),
        (0, 2.55, 1.72),
        (0.45, 0.55, 0.56, 1),
    )


def _malaya_features(world, definition):
    # Cage mesh is a coarse original geometric proxy, not certified guarding.
    for x in (-2.15, 2.15):
        for y in (-0.8, 0.6, 2.0):
            box(
                world,
                f"malaya_cage_post_{x}_{y}",
                (0.035, 0.035, 1.05),
                (x, y, 1.05),
                (0.93, 0.69, 0.05, 1),
            )
        for z in (0.10, 1.05, 2.10):
            box(
                world,
                f"malaya_cage_side_{x}_{z}",
                (0.025, 1.42, 0.018),
                (x, 0.6, z),
                (0.93, 0.69, 0.05, 1),
            )
        for i in range(17):
            box(
                world,
                f"malaya_cage_bar_{x}_{i}",
                (0.008, 0.008, 0.98),
                (x, -0.72 + i * 0.17, 1.06),
                (0.56, 0.53, 0.21, 1),
            )
    for x in (-1.4, -0.7, 0, 0.7, 1.4):
        box(
            world,
            f"malaya_cage_rear_post_{x}",
            (0.02, 0.02, 1.05),
            (x, 2, 1.05),
            (0.93, 0.69, 0.05, 1),
        )
    for z in (0.10, 1.05, 2.10):
        box(
            world,
            f"malaya_cage_rear_{z}",
            (2.15, 0.025, 0.018),
            (0, 2, z),
            (0.93, 0.69, 0.05, 1),
        )

    # The source gate is shown closed. The original cutaway leaves the central
    # foreground bay open for inspection; it is not a safety interlock model.
    for x in (-2.15, -0.95, 0.95, 2.15):
        box(
            world,
            f"malaya_front_post_{x}",
            (0.025, 0.025, 1.05),
            (x, -0.8, 1.05),
            (0.93, 0.69, 0.05, 1),
        )
    for side in (-1, 1):
        for j in range(12):
            x = side * (1.0 + j * 0.10)
            box(
                world,
                f"malaya_front_mesh_{side}_{j}",
                (0.006, 0.006, 1.0),
                (x, -0.8, 1.05),
                (0.15, 0.16, 0.14, 1),
            )
        for z in (0.08, 0.60, 1.10, 1.60, 2.10):
            box(
                world,
                f"malaya_front_cross_{side}_{z}",
                (0.60, 0.008, 0.008),
                (side * 1.55, -0.8, z),
                (0.15, 0.16, 0.14, 1),
            )
    box(
        world,
        "malaya_console",
        (0.55, 0.20, 0.14),
        (0, -1.72, 0.99),
        (0.72, 0.73, 0.69, 1),
    )
    box(
        world,
        "malaya_console_display",
        (0.15, 0.008, 0.065),
        (-0.20, -1.93, 1.03),
        (0.03, 0.10, 0.12, 1),
    )
    for j in range(5):
        box(
            world,
            f"malaya_console_switch_{j}",
            (0.012, 0.014, 0.012),
            (0.05 + j * 0.07, -1.93, 0.98),
            (0.18, 0.21, 0.21, 1),
        )


def _melbourne_features(world, definition):
    for x in (-1.72, 1.72):
        for y in (-1.8, 0, 1.8):
            box(
                world,
                f"melbourne_service_post_{x}_{y}",
                (0.025, 0.025, 0.56),
                (x, y, 0.56),
                (0.57, 0.61, 0.60, 1),
            )
        box(
            world,
            f"melbourne_service_rail_{x}",
            (0.025, 2, 0.025),
            (x, 0, 1.12),
            (0.57, 0.61, 0.60, 1),
        )


def _monash_features(world, definition):
    # Small monitor stand remains separate from the moving platform.
    box(
        world,
        "monash_monitor_foot",
        (0.15, 0.12, 0.014),
        (2.10, 0.95, 0.864),
        (0.18, 0.20, 0.21, 1),
    )
    box(
        world,
        "monash_monitor_stem",
        (0.025, 0.025, 0.12),
        (2.10, 1.0, 0.998),
        (0.34, 0.37, 0.38, 1),
    )
    box(
        world,
        "monash_monitor",
        (0.27, 0.025, 0.17),
        (2.10, 1.0, 1.27),
        (0.06, 0.08, 0.09, 1),
    )
    box(
        world,
        "monash_monitor_screen",
        (0.25, 0.003, 0.15),
        (2.10, 0.972, 1.27),
        (0.03, 0.10, 0.15, 1),
    )


FEATURES.update(
    {
        "peking_zhao_rotary_evaporation": _pku_features,
        "tokyo_kurumizaka_chromatography": _tokyo_features,
        "korea_ace_wind_engineering": _korea_features,
        "malaya_high_voltage_metrology": _malaya_features,
        "melbourne_hrnblwt_probe_alignment": _melbourne_features,
        "monash_geoextremes_seismic_platform": _monash_features,
    }
)
