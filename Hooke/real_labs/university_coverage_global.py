"""Six independently authored, source-informed laboratory workstations.

Institutional photographs establish visible composition, never hidden internals
or surveyed dimensions. Drives are bounded dry inspection surrogates. Original
procedural geometry is distributed; reference photographs are not.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers
from real_labs.mounting import annular_mount

BUILDERS = {}
SOURCES = {}
SAMPLE_INTERFACES = {}
FEATURES = {}

TORONTO = "https://www.labs.chem-eng.utoronto.ca/werber/facilities/"
ZURICH = "https://www.otobm.uzh.ch/en/Infrastructure.html"
ADELAIDE = "https://www.newswise.com/articles/extraterrestrial-surface-simulation-lab-launches-new-chapter-in-space-research"
UCD = "https://siriusxt.com/library/news/zeiss-microscopy-visit-with-siriusxt-to-see-soft-x-ray-tomography-in-action/"


def _tag(b, geom, key):
    geom.set("name", b.unique(key))
    return geom


def _register(kind, build, url, reference, dimensions, site, specimen, description):
    BUILDERS[kind] = build
    SOURCES[kind] = {
        "reference": reference,
        "url": url,
        "dimensions_m": dimensions,
        "dimension_basis": "Estimated geometry and travel from reviewed installed-laboratory photographs; no OEM CAD or measured floor plan.",
    }
    SAMPLE_INTERFACES[kind] = (site, specimen, "clamped", description)


def _post(b, parent, start, end, radius=0.012, material="metal"):
    return b.geom(
        parent,
        "capsule",
        (radius,),
        material=material,
        fromto=tuple(start) + tuple(end),
        collision=True,
    )


def _toronto_crossflow(b, base, params):
    """Six retained membrane cells; two original guided inspection lids."""
    b.box(base, (0, 0, 0.012), (0.64, 0.30, 0.012), "cream", collision=True)
    for row, y in enumerate((-0.09, 0.12)):
        for col, x in enumerate((-0.34, 0, 0.34)):
            name = f"cell_{row}_{col}"
            b.box(base, (x, y, 0.052), (0.10, 0.073, 0.028), "dark", collision=True)
            b.box(base, (x, y, 0.085), (0.055, 0.043, 0.005), "bright", collision=True)
            coupon = b.box(
                base, (x, y, 0.0905), (0.025, 0.018, 0.0005), "sample", collision=True
            )
            _tag(b, coupon, name + "_membrane_coupon")
            # Two short retaining fingers remain on the fixed lower block.
            for dx in (-0.031, 0.031):
                b.box(
                    base,
                    (x + dx, y, 0.092),
                    (0.008, 0.006, 0.002),
                    "metal",
                    collision=True,
                )
            if row == 0 and col in (0, 2):
                key = "left_lid_z" if col == 0 else "right_lid_z"
                for dx in (-0.070, 0.070):
                    b.box(
                        base,
                        (x + dx, y + 0.105, 0.027),
                        (0.010, 0.010, 0.003),
                        "metal",
                        collision=True,
                    )
                    b.cylinder(
                        base,
                        (x + dx, y + 0.105, 0.140),
                        0.004,
                        0.110,
                        "bright",
                        collision=True,
                    )
                # Open bearing bores preserve clearance around fixed guides.
                lid = b.moving(
                    base, key, (x, y, 0.113), (0, 0, 1), (0, 0.075), mass=0.15, kp=350
                )
                for dx in (-0.070, 0.070):
                    annular_mount(
                        b,
                        lid,
                        f"{key}_{dx}",
                        (dx, 0.105, 0.020),
                        0.0044,
                        0.012,
                        0.020,
                        axis="z",
                        proxies=True,
                    )
                b.box(
                    lid,
                    (0, 0.087, 0.020),
                    (0.075, 0.010, 0.009),
                    "dark",
                    collision=True,
                )
                b.box(
                    lid,
                    (0, 0.037, 0.020),
                    (0.026, 0.068, 0.009),
                    "metal",
                    collision=True,
                )
                b.box(lid, (0, 0, 0), (0.10, 0.073, 0.012), "bright", collision=True)
                b.site(lid, key + "_handle", (0, -0.060, 0.018))
            else:
                lid = ET.SubElement(base, "body", pos=numbers((x, y, 0.113)))
                b.box(lid, (0, 0, 0), (0.10, 0.073, 0.012), "bright", collision=True)
            for dx in (-0.073, 0.073):
                for dy in (-0.049, 0.049):
                    b.cylinder(lid, (dx, dy, 0.020), 0.006, 0.008, "metal")
            if row == 0 and col == 0:
                b.site(base, "membrane_coupon", (x, y, 0.091))
    # Actual photo cues: paired black filter housings, steel manifold and gauges.
    for x in (-0.13, 0.13):
        b.cylinder(base, (x, 0.23, 0.15), 0.035, 0.126, "dark", collision=True)
        b.cylinder(base, (x, 0.23, 0.28), 0.040, 0.008, "metal", collision=True)
    for y in (-0.09, 0.12):
        _post(b, base, (-0.54, y, 0.055), (0.54, y, 0.055), 0.005)
    for x in (-0.55, 0.55):
        _post(b, base, (x, -0.09, 0.06), (x, -0.09, 0.17), 0.006)
        b.cylinder(
            base, (x, -0.10, 0.20), 0.035, 0.011, "metal", euler=(math.pi / 2, 0, 0)
        )
        b.cylinder(
            base, (x, -0.112, 0.20), 0.029, 0.001, "cream", euler=(math.pi / 2, 0, 0)
        )
        _post(b, base, (x, -0.115, 0.20), (x + 0.016, -0.115, 0.219), 0.0012, "dark")
    b.metadata["capabilities"] = [
        "left retained membrane-cell inspection lid",
        "right retained membrane-cell inspection lid",
    ]
    b.metadata["limitations"].append(
        "Six-cell WB233 composition follows the actual installed apparatus. Guided lifts are original unpressurized inspection mechanisms; real bolted lids, pressure seals, membrane permeation, liquid flow and filtration are not simulated. Back guide mounting interfaces are estimated."
    )


_register(
    "toronto_membrane_crossflow_inspection",
    _toronto_crossflow,
    TORONTO,
    "Toronto Advanced Membranes Lab WB233 six-cell crossflow system",
    [1.28, 0.60, 0.30],
    "membrane_coupon",
    (0.05, 0.036, 0.001),
    "Inert thin coupon retained between fixed inspection fingers; no membrane transport model",
)


def _specimen_xy(b, parent, z, stroke=0.035):
    """Open guide stack with visible bearing housings and retained coupon."""
    b.box(parent, (0, 0, z + 0.012), (0.15, 0.12, 0.012), "metal", collision=True)
    # Guides terminate outside the moving blocks. Contact proxies represent the
    # exposed guide ends, while rails pass through open annular bearing bores.
    for y in (-0.085, 0.085):
        for x in (-0.135, 0.135):
            b.box(
                parent,
                (x, y, z + 0.030),
                (0.005, 0.008, 0.006),
                "metal",
                collision=True,
            )
        b.cylinder(
            parent,
            (0, y, z + 0.040),
            0.004,
            0.14,
            "bright",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
    xstage = b.moving(
        parent,
        "specimen_x",
        (0, 0, z + 0.040),
        (1, 0, 0),
        (-stroke, stroke),
        mass=0.08,
        kp=320,
    )
    b.box(xstage, (0, 0, 0.018), (0.095, 0.07, 0.012), "dark", collision=True)
    for y in (-0.085, 0.085):
        annular_mount(
            b, xstage, f"x_{y}", (0, y, 0), 0.0044, 0.012, 0.035, axis="x", proxies=True
        )
        b.box(
            xstage, (0, y * 0.86, 0.013), (0.035, 0.012, 0.006), "metal", collision=True
        )
    for x in (-0.065, 0.065):
        # y-axis rails are modelled as a rotated local subassembly.
        for y in (-0.09, 0.09):
            b.box(xstage, (x, y, 0.037), (0.006, 0.005, 0.007), "metal", collision=True)
        rail = ET.SubElement(
            xstage, "body", pos=numbers((x, 0, 0.047)), euler=f"0 0 {math.pi/2}"
        )
        b.cylinder(
            rail,
            (0, 0, 0),
            0.003,
            0.095,
            "bright",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
    ystage = b.moving(
        xstage,
        "specimen_y",
        (0, 0, 0.047),
        (0, 1, 0),
        (-stroke * 0.7, stroke * 0.7),
        mass=0.06,
        kp=300,
    )
    b.box(ystage, (0, 0, 0.016), (0.052, 0.060, 0.010), "metal", collision=True)
    for x in (-0.065, 0.065):
        bearing = ET.SubElement(
            ystage, "body", pos=numbers((x, 0, 0)), euler=f"0 0 {math.pi/2}"
        )
        annular_mount(
            b,
            bearing,
            f"y_{x}",
            (0, 0, 0),
            0.0034,
            0.009,
            0.025,
            axis="x",
            proxies=True,
        )
        b.box(
            ystage, (x * 0.85, 0, 0.010), (0.010, 0.025, 0.005), "metal", collision=True
        )
    b.box(ystage, (0, 0, 0.030), (0.030, 0.024, 0.004), "dark", collision=True)
    _tag(
        b,
        b.box(ystage, (0, 0, 0.035), (0.018, 0.014, 0.001), "sample", collision=True),
        "retained_dry_coupon",
    )
    for x in (-0.022, 0.022):
        b.box(ystage, (x, 0, 0.036), (0.006, 0.004, 0.002), "bright", collision=True)
    b.site(ystage, "dry_coupon", (0, 0, 0.036))
    return ystage


def _zurich_scope(b, base, params):
    b.box(base, (0, 0.06, 0.014), (0.32, 0.29, 0.014), "shell", collision=True)
    b.box(base, (0.25, 0.23, 0.220), (0.035, 0.035, 0.206), "shell", collision=True)
    b.box(base, (0.11, 0.23, 0.430), (0.175, 0.045, 0.036), "shell", collision=True)
    b.box(base, (-0.025, 0.10, 0.430), (0.065, 0.14, 0.036), "shell", collision=True)
    b.cylinder(base, (-0.025, -0.015, 0.363), 0.044, 0.032, "dark", collision=True)
    b.cylinder(base, (-0.025, -0.015, 0.330), 0.036, 0.006, "lens")
    # Paired binocular barrels and rectangular camera head reproduce the
    # photographed surgical-scope silhouette without asserting its OEM model.
    for x in (-0.062, 0.012):
        b.cylinder(
            base,
            (x, -0.070, 0.480),
            0.021,
            0.065,
            "shell",
            euler=(0.75, 0, 0),
            collision=True,
        )
        b.cylinder(base, (x, -0.117, 0.530), 0.023, 0.018, "dark", euler=(0.75, 0, 0))
    b.box(base, (0.005, 0.22, 0.526), (0.09, 0.075, 0.060), "metal", collision=True)
    _specimen_xy(b, base, 0.028)
    b.metadata["capabilities"] = [
        "dry specimen lateral alignment beneath a surgical microscope",
        "orthogonal specimen positioning and return",
    ]
    b.metadata["limitations"].append(
        "Photo establishes two surgical microscope stations and ceiling extraction in the UZH preparation room. The benchtop scope envelope, XY inspection fixture and fiducial coupon are original approximations. No tissue, cutting, acoustic transduction, microscope optics or clinical procedure."
    )


_register(
    "zurich_otology_dry_microscope",
    _zurich_scope,
    ZURICH,
    "UZH Otology and Biomechanics of Hearing preparation-room photograph",
    [0.64, 0.64, 0.59],
    "dry_coupon",
    (0.036, 0.028, 0.002),
    "Rigid calibration coupon retained by two fingers on the XY carrier",
)


def _adelaide_chamber(b, base, params):
    """Open-bore cylindrical rTVAC exterior, vented dry inspection only."""
    b.box(base, (0, 0, 0.045), (0.85, 0.69, 0.045), "metal", collision=True)
    for x in (-0.46, 0.46):
        for y in (-0.40, 0.40):
            b.box(base, (x, y, 0.275), (0.060, 0.060, 0.185), "metal", collision=True)
        b.box(base, (x, 0, 0.49), (0.075, 0.45, 0.035), "metal", collision=True)
    annular_mount(
        b, base, "vessel", (0, 0, 1.05), 0.53, 0.56, 0.59, axis="x", proxies=True
    )
    annular_mount(
        b,
        base,
        "mouth_flange",
        (-0.61, 0, 1.05),
        0.53,
        0.60,
        0.02,
        axis="x",
        proxies=True,
    )
    b.cylinder(
        base,
        (0.615, 0, 1.05),
        0.56,
        0.025,
        "metal",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    for i in range(16):
        a = math.tau * i / 16
        b.cylinder(
            base,
            (-0.64, 0.57 * math.cos(a), 1.05 + 0.57 * math.sin(a)),
            0.009,
            0.009,
            "bright",
            euler=(0, math.pi / 2, 0),
        )
    # Hinged service door remains vented, 35 mm outside the rim at home.
    for z in (0.76, 1.34):
        _post(b, base, (-0.61, -0.52, z), (-0.64, -0.66, z), 0.020)
        annular_mount(
            b,
            base,
            f"door_bearing_{z}",
            (-0.69, -0.66, z),
            0.013,
            0.030,
            0.032,
            axis="z",
            proxies=True,
        )
    door = b.moving(
        base,
        "door_yaw",
        (-0.69, -0.66, 1.05),
        (0, 0, 1),
        (0, 1.30),
        kind="hinge",
        mass=1.5,
        kp=350,
    )
    b.cylinder(door, (0, 0, 0), 0.012, 0.34, "metal", collision=True)
    _post(b, door, (0, 0, 0), (0, 0.15, 0), 0.025)
    b.cylinder(
        door,
        (0, 0.66, 0),
        0.60,
        0.025,
        "metal",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    b.cylinder(door, (-0.027, 0.66, 0), 0.065, 0.005, "dark", euler=(0, math.pi / 2, 0))
    b.site(door, "door_handle", (-0.035, 1.12, 0))
    # Supported straight rails sit well below the open vessel centre.
    for y in (-0.15, 0.15):
        b.cylinder(
            base,
            (0.08, y, 0.700),
            0.005,
            0.47,
            "bright",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
        b.box(base, (0.35, y, 0.58), (0.025, 0.025, 0.080), "metal", collision=True)
        b.box(base, (0.35, y, 0.6775), (0.025, 0.025, 0.0175), "metal", collision=True)
    tray = b.moving(
        base,
        "specimen_x",
        (0.08, 0, 0.735),
        (1, 0, 0),
        (-0.30, 0.05),
        mass=0.30,
        kp=400,
    )
    b.box(tray, (0, 0, 0), (0.19, 0.19, 0.022), "metal", collision=True)
    for y in (-0.15, 0.15):
        annular_mount(
            b,
            tray,
            f"tray_bearing_{y}",
            (0, y, -0.035),
            0.0055,
            0.014,
            0.06,
            axis="x",
            proxies=True,
        )
    b.box(tray, (0, 0, 0.031), (0.075, 0.060, 0.009), "dark", collision=True)
    _tag(
        b,
        b.box(tray, (0, 0, 0.045), (0.050, 0.035, 0.005), "sample", collision=True),
        "inert_rtvac_coupon",
    )
    for x in (-0.057, 0.057):
        b.box(tray, (x, 0, 0.048), (0.012, 0.009, 0.005), "bright", collision=True)
    b.site(tray, "rtvac_coupon", (0, 0, 0.050))
    b.metadata["capabilities"] = [
        "open vented chamber inspection door",
        "translate retained inert coupon within chamber and return",
    ]
    b.metadata["limitations"].append(
        "Exterior follows the university-authored 2022 EXTERRES photo with blue perforated wall and large side-hinged vessel. Internal rail tray, drive and dimensions are authored. Door home is deliberately vented, not a pressure seal. Earth gravity; no vacuum, thermal cycle, regolith, spacecraft qualification or opening a pressurized vessel."
    )


_register(
    "adelaide_exterres_vented_rtvac",
    _adelaide_chamber,
    ADELAIDE,
    "EXTERRES laboratory university-authored press-release photograph",
    [2.85, 2.15, 1.65],
    "rtvac_coupon",
    (0.10, 0.07, 0.01),
    "Rigid inert coupon retained on an authored internal inspection tray",
)


def _ucd_sxt(b, base, params):
    for x in (-0.60, 0.60):
        for y in (-0.40, 0.40):
            b.box(base, (x, y, 0.04), (0.075, 0.075, 0.04), "dark", collision=True)
            b.box(base, (x, y, 0.45), (0.035, 0.035, 0.37), "metal", collision=True)
    b.box(base, (0, 0, 0.85), (0.72, 0.52, 0.030), "metal", collision=True)
    annular_mount(
        b,
        base,
        "main_vessel",
        (0, 0.12, 1.10),
        0.27,
        0.30,
        0.22,
        axis="z",
        proxies=True,
    )
    annular_mount(
        b,
        base,
        "upper_flange",
        (0, 0.12, 1.335),
        0.27,
        0.35,
        0.015,
        axis="z",
        proxies=True,
    )
    b.cylinder(base, (0, 0.12, 0.890), 0.30, 0.010, "metal", collision=True)
    # Paired support columns and rigid lifting crossbeam, visible in the photo.
    for x in (-0.43, 0.43):
        b.cylinder(base, (x, 0.36, 1.48), 0.016, 0.60, "dark", collision=True)
    b.box(base, (0, 0.36, 2.06), (0.45, 0.028, 0.024), "dark", collision=True)
    lid = b.moving(
        base, "lid_z", (0, 0.12, 1.38), (0, 0, 1), (0, 0.22), mass=0.7, kp=400
    )
    b.cylinder(lid, (0, 0, 0), 0.35, 0.020, "bright", collision=True)
    for x in (-0.13, 0.13):
        b.cylinder(lid, (x, 0, 0.034), 0.055, 0.014, "dark", collision=True)
    b.box(lid, (0, 0.19, 0.18), (0.035, 0.035, 0.16), "dark", collision=True)
    b.box(lid, (0, 0.24, 0.34), (0.395, 0.020, 0.020), "metal", collision=True)
    for x in (-0.43, 0.43):
        annular_mount(
            b,
            lid,
            f"lift_bearing_{x}",
            (x, 0.24, 0.34),
            0.017,
            0.035,
            0.03,
            axis="z",
            proxies=True,
        )
    b.site(lid, "lid_handle", (0, -0.25, 0.025))
    for i in range(12):
        a = math.tau * i / 12
        b.cylinder(
            lid,
            (0.318 * math.cos(a), 0.318 * math.sin(a), 0.028),
            0.006,
            0.008,
            "metal",
        )
    b.cylinder(base, (0, 0.12, 0.98), 0.045, 0.08, "metal", collision=True)
    stage = b.moving(
        base,
        "specimen_theta",
        (0, 0.12, 1.075),
        (0, 0, 1),
        (-0.45, 0.45),
        kind="hinge",
        mass=0.08,
        kp=150,
    )
    b.cylinder(stage, (0, 0, 0), 0.11, 0.014, "dark", collision=True)
    b.box(stage, (0, 0, 0.026), (0.035, 0.025, 0.012), "bright", collision=True)
    _tag(
        b,
        b.box(stage, (0, 0, 0.040), (0.015, 0.010, 0.002), "sample", collision=True),
        "inert_sxt_alignment_coupon",
    )
    b.site(stage, "alignment_coupon", (0, 0, 0.042))
    # Foreground auxiliary chamber and boxed electronics in the installed view.
    b.box(base, (-0.36, -0.25, 1.02), (0.25, 0.20, 0.14), "bright", collision=True)
    b.cylinder(base, (0.34, -0.25, 0.99), 0.17, 0.11, "metal", collision=True)
    b.cylinder(base, (0.34, -0.25, 1.12), 0.19, 0.020, "bright", collision=True)
    b.cylinder(base, (0.34, -0.25, 1.15), 0.065, 0.01, "dark")
    b.metadata["capabilities"] = [
        "lift authored vented inspection cover",
        "rotate retained alignment coupon and return",
    ]
    b.metadata["limitations"].append(
        "Installed SXT-100 photograph in UCD Conway supports stainless chamber cluster and black overhead frame. Internal holder, simplified vertical cover lift and all dimensions are inferred dry surrogates, not OEM load-lock procedure. No X-rays, plasma, vacuum, cryogenics, tomography or biological sample imaging."
    )


_register(
    "ucd_conway_sxt_inspection",
    _ucd_sxt,
    UCD,
    "SiriusXT primary installed-site report at UCD Conway",
    [1.44, 1.04, 2.12],
    "alignment_coupon",
    (0.03, 0.02, 0.004),
    "Rigid inert alignment coupon retained on an original internal rotary platform",
)

UBA = "https://commons.wikimedia.org/wiki/File:Laboratorio_de_Qu%C3%ADmica_-_Facultad_de_Ingenier%C3%ADa_UBA.JPG"
TRINITY = "https://microscopy.ie/wp-content/uploads/2025/07/EM-UKI_2025_National-spotlight-on-Ireland.pdf"


def _uba_burette(b, base, params):
    b.box(base, (0, 0, 0.012), (0.25, 0.22, 0.012), "cream", collision=True)
    b.box(base, (0.13, 0.12, 0.035), (0.085, 0.080, 0.011), "dark", collision=True)
    b.cylinder(base, (0.13, 0.12, 0.37), 0.007, 0.324, "bright", collision=True)
    # Guide carriage is an open annulus around the photographed stand rod.
    carriage = b.moving(
        base,
        "clamp_z",
        (0.13, 0.12, 0.36),
        (0, 0, 1),
        (-0.040, 0.080),
        mass=0.06,
        kp=220,
    )
    annular_mount(
        b,
        carriage,
        "stand_clamp",
        (0, 0, 0),
        0.0075,
        0.020,
        0.020,
        axis="z",
        proxies=True,
    )
    _post(b, carriage, (-0.02, 0, 0), (-0.13, -0.12, 0), 0.006)
    b.cylinder(carriage, (-0.13, -0.12, 0), 0.010, 0.19, "glass", collision=True)
    # Clear glass tube contains no liquid. Thin dark graduation marks remain
    # presentation geometry only, not an invented measurement readout.
    for i in range(16):
        b.box(
            carriage,
            (-0.13, -0.130, -0.16 + i * 0.020),
            (0.006, 0.0006, 0.0005),
            "dark",
        )
    valve_housing = ET.SubElement(
        carriage,
        "body",
        pos="-.13 -.12 -.202",
        euler=f"0 0 {math.pi / 2}",
        gravcomp="1",
    )
    annular_mount(
        b,
        valve_housing,
        "stopcock_bearing",
        (0, 0, 0),
        0.0046,
        0.012,
        0.011,
        axis="x",
        proxies=True,
        material="shell",
    )
    valve = b.moving(
        carriage,
        "stopcock_theta",
        (-0.13, -0.12, -0.202),
        (0, 1, 0),
        (-0.60, 0.60),
        kind="hinge",
        mass=0.02,
        kp=60,
    )
    b.cylinder(
        valve,
        (0, -0.025, 0),
        0.004,
        0.025,
        "metal",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    b.box(valve, (0, -0.050, 0), (0.025, 0.006, 0.005), "guard_red", collision=True)
    b.site(valve, "stopcock_handle", (0.025, -0.050, 0))
    b.cylinder(carriage, (-0.13, -0.12, -0.235), 0.003, 0.022, "glass", collision=True)
    b.site(carriage, "burette_tip", (-0.13, -0.12, -0.257))
    # Retained inert receiver on a dry bath tray. No heater or gas ignition.
    b.box(base, (0, -0.015, 0.033), (0.12, 0.10, 0.009), "shell", collision=True)
    annular_mount(
        b,
        base,
        "receiver_bore",
        (0, 0, 0.065),
        0.019,
        0.022,
        0.023,
        axis="z",
        proxies=True,
        material="glass",
    )
    b.cylinder(base, (0, 0, 0.043), 0.022, 0.001, "glass", collision=True)
    for x in (-0.030, 0.030):
        b.box(base, (x, 0, 0.060), (0.010, 0.014, 0.018), "bright", collision=True)
    b.site(base, "receiving_vial", (0, 0, 0.088))
    b.metadata["capabilities"] = [
        "adjust burette stand height over retained receiver",
        "position dry stopcock and return",
    ]
    b.metadata["limitations"].append(
        "Historical appearance only: institutional FIUBA photograph dated 14 May 2008. Burette/stand mechanics and dimensions are original approximations. No fluid, titration, heating, flame, gas supply or reaction model; service rails are inactive geometry. Present-day arrangement is unverified."
    )


_register(
    "uba_historical_dry_burette",
    _uba_burette,
    UBA,
    "Comunicacion Institucional FIUBA installed chemistry laboratory photograph, 14 May 2008, CC BY-SA 4.0",
    [0.50, 0.44, 0.78],
    "receiving_vial",
    (0.044, 0.044, 0.046),
    "Rigid empty receiving vial retained by two support blocks on the dry tray",
)


def _trinity_titan(b, base, params):
    b.box(base, (0, 0, 0.10), (0.50, 0.43, 0.10), "shell", collision=True)
    b.cylinder(base, (0, 0.07, 0.47), 0.31, 0.27, "shell", collision=True)
    b.box(base, (0, 0.16, 1.39), (0.32, 0.25, 0.65), "shell", collision=True)
    b.box(base, (0, 0.16, 2.28), (0.37, 0.28, 0.24), "shell", collision=True)
    # Tall dark front enclosure and light side towers establish the Titan
    # silhouette in the installed-room image; no internal electron column.
    b.cylinder(base, (0, -0.11, 1.64), 0.205, 0.72, "dark", collision=True)
    b.cylinder(base, (0, -0.11, 2.37), 0.205, 0.03, "metal", collision=True)
    b.box(base, (0.38, 0.16, 1.78), (0.055, 0.25, 0.70), "metal", collision=True)
    b.box(base, (-0.36, 0.18, 1.54), (0.035, 0.23, 0.50), "shell", collision=True)
    b.box(base, (0.45, -0.14, 1.23), (0.08, 0.08, 0.14), "blue", collision=True)
    # An explicitly external dry alignment cradle avoids representing unknown
    # vacuum internals. It is mechanically connected to the column side.
    b.box(base, (0.48, 0.02, 0.95), (0.19, 0.15, 0.030), "metal", collision=True)
    for y in (-0.075, 0.115):
        for x in (0.46, 0.74):
            b.box(base, (x, y, 1.0005), (0.008, 0.008, 0.0205), "metal", collision=True)
        b.cylinder(
            base,
            (0.60, y, 1.025),
            0.004,
            0.15,
            "bright",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
    holder = b.moving(
        base,
        "holder_x",
        (0.60, 0.02, 1.025),
        (1, 0, 0),
        (-0.035, 0.065),
        mass=0.08,
        kp=280,
    )
    for y in (-0.095, 0.095):
        annular_mount(
            b,
            holder,
            f"holder_guide_{y}",
            (0, y, 0),
            0.0045,
            0.012,
            0.030,
            axis="x",
            proxies=True,
        )
        b.box(
            holder, (0, y * 0.85, 0.014), (0.030, 0.012, 0.006), "metal", collision=True
        )
    b.box(holder, (0, 0, 0.022), (0.055, 0.075, 0.012), "dark", collision=True)
    annular_mount(
        b,
        holder,
        "grid_rotation_bearing",
        (0, 0, 0.060),
        0.0085,
        0.020,
        0.026,
        axis="z",
        proxies=True,
    )
    turn = b.moving(
        holder,
        "holder_theta",
        (0, 0, 0.093),
        (0, 0, 1),
        (-0.35, 0.35),
        kind="hinge",
        mass=0.03,
        kp=85,
    )
    b.cylinder(turn, (0, 0, -0.023), 0.008, 0.026, "metal", collision=True)
    b.cylinder(turn, (0, 0, 0), 0.035, 0.006, "bright", collision=True)
    b.box(turn, (0, 0, 0.009), (0.012, 0.012, 0.003), "dark", collision=True)
    _tag(
        b,
        b.cylinder(turn, (0, 0, 0.0125), 0.008, 0.0005, "copper", collision=True),
        "retained_grid_surrogate",
    )
    for x in (-0.009, 0.009):
        b.box(turn, (x, 0, 0.013), (0.003, 0.002, 0.001), "bright", collision=True)
    b.site(turn, "grid_mount", (0, 0, 0.013))
    b.metadata["capabilities"] = [
        "external retained-grid carrier alignment",
        "rotate dry grid carrier and return",
    ]
    b.metadata["limitations"].append(
        "Installed Titan image on page 10 of the 2025 Irish EM facility presentation constrains the local bay, blue partition and tall column; stock Orion image is not layout evidence. The external grid cradle and drives are authored, not documented Titan loading hardware. No beam, vacuum, atomic resolution, tomography or full-room reconstruction."
    )


_register(
    "trinity_aml_titan_dry_alignment",
    _trinity_titan,
    TRINITY,
    "Trinity AML installed FEI Titan photograph, EMUKI 2025 presentation page 10",
    [1.48, 0.90, 2.52],
    "grid_mount",
    (0.016, 0.016, 0.001),
    "Inert visible grid surrogate retained on an authored external alignment cradle",
)


def _toronto_features(world, definition):
    # Source WB233: black-framed tall window, open metallic ductwork, blue-bin
    # rack and suspended utility bars; only the chosen membrane bay is mapped.
    for x in (-1.8, -0.6, 0.6, 1.8):
        box(
            world,
            f"toronto_window_mullion_{x}",
            (0.025, 0.035, 0.75),
            (x, 2.78, 1.85),
            (0.06, 0.08, 0.09, 1),
        )
    for z in (1.10, 2.60):
        box(
            world,
            f"toronto_window_header_{z}",
            (2.15, 0.04, 0.03),
            (0, 2.78, z),
            (0.06, 0.08, 0.09, 1),
        )
    for j, z in enumerate((0.30, 0.68, 1.06, 1.44, 1.82)):
        box(
            world,
            f"toronto_rack_shelf_{j}",
            (0.35, 0.26, 0.012),
            (-2.05, 0.3, z),
            (0.55, 0.59, 0.61, 1),
        )
        for i in range(3):
            box(
                world,
                f"toronto_blue_bin_{j}_{i}",
                (0.095, 0.20, 0.11),
                (-2.28 + i * 0.23, 0.3, z + 0.12),
                (0.04, 0.12, 0.55, 1),
            )
    for x in (-2.38, -1.72):
        for y in (0.06, 0.54):
            box(
                world,
                f"toronto_rack_post_{x}_{y}",
                (0.012, 0.012, 1.02),
                (x, y, 1.02),
                (0.6, 0.65, 0.67, 1),
            )
    for y in (0.4, 1.9):
        ET.SubElement(
            world,
            "geom",
            name=f"toronto_overhead_duct_{y}",
            type="cylinder",
            size="0.14 2.4",
            pos=f"0 {y} 2.70",
            euler=f"0 {math.pi/2} 0",
            rgba=".57 .60 .61 1",
        )


FEATURES["toronto_wb233_membrane_crossflow"] = _toronto_features


def _zurich_features(world, definition):
    # The second photographed scope provides room context only. The right-hand
    # instrument owns the declared manipulation controls and sample interface.
    body = ET.SubElement(
        world, "body", name="zurich_secondary_scope", pos="-1.12 .85 .90"
    )
    for name, size, pos, color in (
        ("base", (0.25, 0.23, 0.018), (0, 0, 0.018), (0.82, 0.84, 0.85, 1)),
        ("column", (0.035, 0.035, 0.21), (-0.18, 0.16, 0.246), (0.83, 0.85, 0.86, 1)),
        ("arm", (0.13, 0.045, 0.030), (-0.065, 0.16, 0.46), (0.84, 0.86, 0.87, 1)),
        ("head", (0.07, 0.13, 0.04), (0.035, 0.06, 0.46), (0.84, 0.86, 0.87, 1)),
    ):
        box(body, "zurich_secondary_" + name, size, pos, color)
    ET.SubElement(
        body,
        "geom",
        name="zurich_secondary_objective",
        type="cylinder",
        pos=".035 -.035 .39",
        size=".035 .03",
        rgba=".06 .08 .09 1",
    )
    for x in (0, 0.07):
        ET.SubElement(
            body,
            "geom",
            name=f"zurich_secondary_eyepiece_{x}",
            type="cylinder",
            pos=f"{x} -.10 .50",
            size=".017 .06",
            euler=".7 0 0",
            rgba=".12 .14 .16 1",
        )
    # Three ceiling extraction arms line two narrow preparation counters.
    for i, (x, y) in enumerate(((-1.12, 0.75), (1.12, 0.85), (1.12, -0.55))):
        p0 = (x, y, 2.76)
        p1 = (x, y, 2.13)
        p2 = (x * 0.72, y - 0.20, 1.78)
        p3 = (x * 0.65, y - 0.35, 1.51)
        for j, (a, c) in enumerate(zip((p0, p1, p2), (p1, p2, p3))):
            ET.SubElement(
                world,
                "geom",
                name=f"zurich_extract_{i}_{j}",
                type="capsule",
                fromto=numbers(a + c),
                size=".055",
                rgba=".82 .84 .85 1",
            )
        ET.SubElement(
            world,
            "geom",
            name=f"zurich_extract_hood_{i}",
            type="ellipsoid",
            size=".18 .18 .035",
            pos=numbers(p3),
            rgba=".65 .75 .79 .30",
        )
    box(
        world,
        "zurich_back_monitor",
        (0.50, 0.035, 0.30),
        (0, 2.12, 1.97),
        (0.025, 0.03, 0.045, 1),
    )
    # Upper cupboards photographed along both sides.
    for x in (-1.57, 1.57):
        for j, y in enumerate((-0.65, 0.25, 1.15)):
            box(
                world,
                f"zurich_upper_cabinet_{x}_{j}",
                (0.16, 0.42, 0.27),
                (x, y, 2.10),
                (0.90, 0.92, 0.93, 1),
            )
    box(
        world,
        "zurich_sterile_trays",
        (0.19, 0.26, 0.07),
        (1.12, -0.80, 1.01),
        (0.60, 0.65, 0.68, 1),
    )


FEATURES["zurich_otology_specimen_preparation"] = _zurich_features


def _adelaide_features(world, definition):
    # EXTERRES chamber photograph: blue perforated metal screen and tall glazing.
    for i, x in enumerate((-1.2, -0.4, 0.4, 1.2, 2.0)):
        box(
            world,
            f"adelaide_blue_panel_{i}",
            (0.385, 0.025, 1.10),
            (x, 2.55, 1.25),
            (0.035, 0.14, 0.40, 1),
        )
        for row in range(10):
            for col in range(6):
                box(
                    world,
                    f"adelaide_panel_slot_{i}_{row}_{col}",
                    (0.012, 0.002, 0.014),
                    (x - 0.30 + col * 0.12, 2.522, 0.34 + row * 0.20),
                    (0.018, 0.04, 0.09, 1),
                    collision=False,
                )
    box(
        world,
        "adelaide_window_frame",
        (0.025, 0.035, 1.20),
        (-2.0, 2.55, 1.35),
        (0.04, 0.05, 0.06, 1),
    )
    # Beside-vessel utility cabinet and free-standing access table.
    box(
        world,
        "adelaide_pump_cabinet",
        (0.23, 0.24, 0.44),
        (1.55, 1.62, 0.44),
        (0.81, 0.83, 0.85, 1),
    )
    box(
        world,
        "adelaide_pump_panel",
        (0.15, 0.008, 0.16),
        (1.55, 1.371, 0.59),
        (0.05, 0.08, 0.12, 1),
    )


FEATURES["adelaide_exterres_rtvac_access"] = _adelaide_features


def _ucd_features(world, definition):
    box(
        world,
        "ucd_service_shelf",
        (1.45, 0.18, 0.025),
        (0, 2.22, 1.40),
        (0.68, 0.64, 0.53, 1),
    )
    for i, x in enumerate((-0.9, -0.45, 0)):
        box(
            world,
            f"ucd_blue_storage_{i}",
            (0.19, 0.15, 0.10),
            (x, 2.22, 1.525),
            (0.08, 0.17, 0.48, 1),
        )
    box(
        world,
        "ucd_component_drawers",
        (0.22, 0.13, 0.27),
        (0.83, 2.20, 1.695),
        (0.06, 0.08, 0.10, 1),
    )
    for row in range(5):
        for col in range(3):
            box(
                world,
                f"ucd_drawer_label_{row}_{col}",
                (0.047, 0.002, 0.022),
                (0.685 + col * 0.145, 2.066, 1.49 + row * 0.095),
                (0.72, 0.75, 0.76, 1),
            )
    ET.SubElement(
        world,
        "geom",
        name="ucd_wall_service_pipe",
        type="cylinder",
        size=".055 .70",
        pos="0 2.24 2.35",
        euler=f"0 {math.pi/2} 0",
        rgba=".90 .89 .84 1",
    )


FEATURES["ucd_conway_sxt_chamber_inspection"] = _ucd_features


def _uba_features(world, definition):
    for k, y in enumerate((-0.55, 1.25)):
        # Black raised service rail above dark-topped teaching islands.
        box(
            world,
            f"uba_service_rail_{k}",
            (2.20, 0.040, 0.075),
            (0, y, 1.30),
            (0.05, 0.06, 0.06, 1),
        )
        for x in (-1.8, -0.6, 0.6, 1.8):
            box(
                world,
                f"uba_rail_support_{k}_{x}",
                (0.025, 0.025, 0.18),
                (x, y, 1.09),
                (0.08, 0.09, 0.08, 1),
            )
            box(
                world,
                f"uba_socket_{k}_{x}",
                (0.065, 0.016, 0.042),
                (x, y - 0.058, 1.30),
                (0.89, 0.87, 0.78, 1),
            )
        ET.SubElement(
            world,
            "geom",
            name=f"uba_inactive_yellow_service_{k}",
            type="cylinder",
            size=".017 2.20",
            pos=f"0 {y} 1.13",
            euler=f"0 {math.pi/2} 0",
            rgba=".90 .65 .08 1",
        )
    # A tiled column with upward service pipes is distinctive in the 2008 photo.
    box(
        world,
        "uba_tiled_column",
        (0.22, 0.23, 1.50),
        (2.0, 1.65, 1.50),
        (0.88, 0.87, 0.78, 1),
    )
    for z in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8):
        box(
            world,
            f"uba_tile_seam_{z}",
            (0.223, 0.002, 0.002),
            (2.0, 1.416, z),
            (0.67, 0.68, 0.62, 1),
            collision=False,
        )
    for x in (1.73, 1.81):
        ET.SubElement(
            world,
            "geom",
            name=f"uba_vertical_service_{x}",
            type="cylinder",
            size=".012 .85",
            pos=f"{x} 1.35 2.12",
            rgba=".85 .65 .08 1",
        )


FEATURES["uba_fiuba_chemistry_2008"] = _uba_features


def _trinity_features(world, definition):
    box(
        world,
        "trinity_blue_partition",
        (0.035, 1.05, 1.08),
        (-1.22, 0.70, 1.28),
        (0.04, 0.16, 0.36, 1),
    )
    for y in (-0.35, 1.75):
        box(
            world,
            f"trinity_partition_trim_{y}",
            (0.042, 0.022, 1.10),
            (-1.22, y, 1.28),
            (0.81, 0.82, 0.80, 1),
        )
    # Installed photo shows a separate operator console to the instrument right.
    box(
        world,
        "trinity_console_stand",
        (0.05, 0.05, 0.20),
        (1.23, 0.55, 1.05),
        (0.10, 0.11, 0.12, 1),
    )
    box(
        world,
        "trinity_console_monitor",
        (0.26, 0.035, 0.17),
        (1.23, 0.55, 1.38),
        (0.025, 0.04, 0.055, 1),
    )
    box(
        world,
        "trinity_console_keyboard",
        (0.22, 0.085, 0.012),
        (1.23, 0.25, 0.925),
        (0.16, 0.18, 0.18, 1),
    )


FEATURES["trinity_aml_titan_alignment"] = _trinity_features
