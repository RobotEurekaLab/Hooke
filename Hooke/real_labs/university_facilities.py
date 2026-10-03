"""Reference-informed cleanroom, chemistry, membrane and aquarium equipment.

Dimensions and mechanisms are authored estimates. External photographs provide
appearance and facility cues, not verified machine drawings or process models.
The aquarium bridge is an explicitly added sampling-control surrogate.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET


HARVARD = "https://cns1.rc.fas.harvard.edu/virtual-visit/"
CALTECH = "https://www.stoltz2.caltech.edu/labtour2.html"
NTU = "https://www.ntu.edu.sg/docs/librariesprovider88/newri-domains/smtc/smtc-brochure_compressede6c80970-b0ba-4414-90e7-bcb662e94af7.pdf?sfvrsn=87821fc1_3"
UQ = "https://moreton-bay.research.uq.edu.au/facilities/research-laboratories-aquariums"


def _bolts(b, parent, x_values, y, z, radius=0.006):
    for x in x_values:
        b.cylinder(parent, (x, y, z), radius, 0.002, "bright")


def _rail(b, parent, x, y0, y1, z):
    b.rod(parent, (x, y0, z), (x, y1, z), 0.012, "bright")
    for y in (y0, y1):
        b.box(parent, (x, y, z - 0.012), (0.025, 0.023, 0.017), "metal")


def _gauge(b, parent, pos):
    x, y, z = pos
    b.cylinder(parent, pos, 0.055, 0.018, "bright", euler=(math.pi / 2, 0, 0))
    b.cylinder(
        parent, (x, y - 0.02, z), 0.047, 0.002, "cream", euler=(math.pi / 2, 0, 0)
    )
    b.rod(parent, (x, y - 0.023, z), (x + 0.023, y - 0.023, z + 0.024), 0.002, "dark")
    for a in (-0.6, 0.0, 0.6):
        b.rod(
            parent,
            (x + 0.034 * math.sin(a), y - 0.023, z + 0.034 * math.cos(a)),
            (x + 0.040 * math.sin(a), y - 0.023, z + 0.040 * math.cos(a)),
            0.0015,
            "dark",
        )


def _mask_aligner(b, base, params):
    """Accessible XY specimen carriage and vertical alignment head."""
    b.feet(base, 0.96, 0.78)
    b.housing(
        base,
        [
            (0.025, 0.46, 0.37, 0),
            (0.055, 0.48, 0.39, 0),
            (0.19, 0.48, 0.39, 0),
            (0.22, 0.44, 0.35, 0),
        ],
        0.026,
    )
    b.box(base, (0, 0, 0.17), (0.45, 0.35, 0.055), "shell", collision=True)
    b.box(base, (0, 0.035, 0.233), (0.37, 0.29, 0.012), "metal")
    for x in (-0.29, 0.29):
        _rail(b, base, x, -0.26, 0.19, 0.263)
    carrier = b.moving(
        base, "wafer_y", (0, -0.01, 0.30), (0, 1, 0), (-0.09, 0.09), mass=0.55
    )
    b.box(carrier, (0, 0, 0), (0.33, 0.23, 0.018), "dark", collision=True)
    for y in (-0.145, 0.145):
        b.rod(carrier, (-0.25, y, 0.035), (0.25, y, 0.035), 0.01, "bright")
    stage = b.moving(
        carrier, "wafer_x", (0, 0, 0.067), (1, 0, 0), (-0.065, 0.065), mass=0.35
    )
    b.box(stage, (0, 0, 0), (0.16, 0.14, 0.017), "metal", collision=True)
    b.cylinder(stage, (0, 0, 0.031), 0.098, 0.014, "dark")
    b.cylinder(stage, (0, 0, 0.047), 0.074, 0.0015, "lens")
    b.ring(stage, (0, 0, 0.049), 0.076, 0.0018, "bright")
    b.site(stage, "wafer", (0, 0, 0.047))
    for x in (-0.24, 0.24):
        b.cylinder(base, (x, 0.35, 0.59), 0.025, 0.32, "bright", collision=True)
        b.box(base, (x, 0.35, 0.24), (0.046, 0.047, 0.025), "metal")
    b.box(base, (0, 0.35, 0.925), (0.295, 0.07, 0.035), "shell", collision=True)
    head = b.moving(
        base, "alignment_head", (0, 0.115, 0.735), (0, 0, 1), (0, 0.11), mass=0.8
    )
    b.housing(
        head,
        [
            (-0.06, 0.20, 0.20, 0.015),
            (0.07, 0.20, 0.20, 0.015),
            (0.105, 0.16, 0.15, 0.04),
        ],
        0.025,
    )
    for x in (-0.085, 0.085):
        b.cylinder(head, (x, -0.14, -0.12), 0.03, 0.06, "dark")
        b.cylinder(head, (x, -0.14, -0.183), 0.024, 0.003, "lens")
        b.rod(head, (x, -0.01, 0.075), (x, -0.16, 0.17), 0.027, "shell")
        b.rod(head, (x, -0.16, 0.17), (x, -0.19, 0.19), 0.03, "rubber")
    b.box(base, (0.32, -0.24, 0.36), (0.055, 0.025, 0.08), "shell")
    b.screen(base, (0.315, -0.273, 0.367), 0.083, 0.11)
    for x in (-0.31, -0.23, -0.15):
        b.cylinder(
            base, (x, -0.38, 0.12), 0.018, 0.018, "dark", euler=(math.pi / 2, 0, 0)
        )
    b.metadata["capabilities"] = [
        "XY wafer carrier positioning",
        "Alignment-head access adjustment",
    ]
    b.metadata["limitations"].append(
        "Generic mask-aligner architecture; neither AB-M CAD nor lithographic exposure simulation. The 150 mm inert wafer is an authored carrier."
    )


def _hood(b, base, params):
    """Floor-supported hood with independent sash and under-deck supply drawer."""
    w, d = 1.7, 0.94
    b.feet(base, w, d, 0.085)
    b.box(base, (0, 0.025, 0.37), (0.81, 0.415, 0.285), "cream", collision=True)
    # Toe, cabinet doors and a functional upper drawer; no implied ventilation.
    b.box(base, (0, -0.414, 0.115), (0.75, 0.012, 0.045), "dark")
    for x in (-0.41, 0.41):
        b.box(base, (x, -0.417, 0.39), (0.388, 0.014, 0.225), "shell")
        b.rod(base, (x - 0.12, -0.445, 0.54), (x + 0.12, -0.445, 0.54), 0.008, "bright")
    # The drawer slides within the open .65–.865 m under-deck cavity.
    drawer = b.moving(
        base,
        "supply_drawer",
        (0, -0.045, 0.754),
        (0, -1, 0),
        (0, 0.23),
        mass=1.1,
        kp=240,
    )
    b.box(drawer, (0, 0, 0), (0.72, 0.35, 0.012), "metal", collision=True)
    b.box(drawer, (0, -0.375, 0.017), (0.76, 0.018, 0.073), "cream", collision=True)
    b.rod(drawer, (-0.13, -0.408, 0.025), (0.13, -0.408, 0.025), 0.011, "bright")
    b.box(drawer, (0.22, 0.04, 0.055), (0.09, 0.065, 0.03), "blue")
    b.site(drawer, "sample_carrier", (0.22, 0.04, 0.055))
    b.box(base, (0, 0, 0.9), (0.85, 0.47, 0.035), "dark", collision=True)
    for x in (-0.805, 0.805):
        b.box(base, (x, 0.03, 1.545), (0.045, 0.44, 0.61), "shell", collision=True)
        b.box(base, (x, -0.418, 1.36), (0.032, 0.018, 0.36), "metal")
    b.box(base, (0, 0.44, 1.54), (0.79, 0.027, 0.605), "cream", collision=True)
    b.box(base, (0, 0.04, 2.13), (0.85, 0.43, 0.10), "shell", collision=True)
    b.box(base, (0, -0.395, 2.115), (0.64, 0.008, 0.042), "dark")
    for z in (1.04, 1.36, 1.81):
        b.box(base, (0, 0.409, z), (0.66, 0.006, 0.016), "dark")
    sash = b.moving(
        base,
        "sash",
        (0, -0.418, 1.35),
        (0, 0, 1),
        (0, 0.40),
        mass=2.0,
        kp=300,
        force=150,
    )
    b.box(sash, (0, 0, 0), (0.746, 0.009, 0.365), "glass")
    for x in (-0.76, 0.76):
        b.box(sash, (x, 0, 0), (0.009, 0.016, 0.38), "metal")
    for z in (-0.38, 0.38):
        b.box(sash, (0, 0, z), (0.767, 0.018, 0.012), "metal")
    b.rod(sash, (-0.17, -0.046, -0.35), (0.17, -0.046, -0.35), 0.008, "bright")
    # Glassware rests on hotplate surrogates; clamp poles terminate on the deck.
    for x in (-0.47, 0, 0.44):
        b.box(base, (x, 0.05, 0.986), (0.115, 0.12, 0.047), "blue")
        b.cylinder(base, (x, 0.05, 1.04), 0.104, 0.01, "bright")
        b.geom(base, "ellipsoid", (0.058, 0.058, 0.071), (x, 0.05, 1.12), "glass")
        b.cylinder(base, (x, 0.05, 1.23), 0.017, 0.055, "glass")
        b.rod(base, (x + 0.14, 0.23, 0.942), (x + 0.14, 0.23, 1.77), 0.009, "bright")
        b.rod(base, (x + 0.14, 0.23, 1.38), (x, 0.05, 1.38), 0.006, "metal")
        b.cylinder(base, (x, 0.05, 1.41), 0.023, 0.13, "glass")
        b.rod(base, (x, 0.05, 1.52), (x, 0.28, 1.7), 0.005, "cyan")
    b.screen(base, (0.799, -0.443, 1.15), 0.047, 0.095)
    for x in (-0.70, -0.61):
        b.cylinder(
            base, (x, -0.485, 0.91), 0.016, 0.012, "blue", euler=(math.pi / 2, 0, 0)
        )
    b.cylinder(base, (0, 0.18, 2.29), 0.13, 0.07, "metal")
    b.metadata["capabilities"] = [
        "Hood sash access",
        "Under-deck supply-cassette drawer positioning",
    ]
    b.metadata["limitations"].append(
        "Hood exterior and supported glassware are source-informed; ventilation, chemical synthesis and containment are not simulated. Drawer is an authored mechanism."
    )


def _membrane_skid(b, base, params):
    # Open steel frame follows the photographed campus industrial project area.
    for x in (-0.77, 0.77):
        for y in (-0.43, 0.43):
            b.cylinder(base, (x, y, 0.024), 0.048, 0.024, "rubber", collision=True)
            b.box(base, (x, y, 0.84), (0.022, 0.022, 0.79), "bright", collision=True)
    for z in (0.16, 0.69, 1.6):
        for y in (-0.43, 0.43):
            b.box(base, (0, y, z), (0.79, 0.023, 0.023), "bright", collision=True)
        for x in (-0.77, 0.77):
            b.box(base, (x, 0, z), (0.023, 0.44, 0.023), "bright", collision=True)
    b.box(base, (0, 0, 0.185), (0.73, 0.40, 0.012), "metal", collision=True)
    for x in (-0.44, 0.30):
        b.cylinder(base, (x, 0.14, 0.46), 0.16, 0.26, "shell")
        b.cylinder(base, (x, 0.14, 0.73), 0.17, 0.012, "metal")
        b.rod(base, (x, 0.14, 0.75), (x, 0.14, 1.34), 0.016, "bright")
    for z in (1.06, 1.36):
        b.cylinder(base, (0, 0.12, z), 0.071, 0.60, "shell", euler=(0, math.pi / 2, 0))
        for x in (-0.63, 0.63):
            b.cylinder(
                base, (x, 0.12, z), 0.081, 0.024, "bright", euler=(0, math.pi / 2, 0)
            )
            b.box(base, (x * 0.82, 0.14, z - 0.09), (0.048, 0.055, 0.035), "metal")
            b.rod(base, (x, 0.12, z), (x, -0.26, z), 0.014, "dark")
            b.rod(base, (x, -0.26, z), (x, -0.26, 0.84), 0.014, "dark")
    b.rod(base, (-0.62, -0.26, 0.84), (0.61, -0.26, 0.84), 0.02, "bright")
    for x in (-0.28, 0.26):
        _gauge(b, base, (x, -0.31, 1.11))
        b.rod(base, (x, -0.26, 0.85), (x, -0.26, 1.06), 0.012, "bright")
    b.box(base, (0.30, -0.33, 0.86), (0.04, 0.042, 0.04), "metal")
    valve = b.moving(
        base,
        "isolation_valve",
        (0.30, -0.405, 0.86),
        (0, 1, 0),
        (0, math.pi / 2),
        kind="hinge",
        mass=0.08,
        kp=50,
    )
    b.cylinder(valve, (0, 0, 0), 0.025, 0.012, "metal", euler=(math.pi / 2, 0, 0))
    b.rod(valve, (0, -0.015, 0), (0, -0.015, 0.15), 0.019, "blue")
    # Sampling vial carriage is an explicit bench-scale task fixture.
    for x in (-0.17, 0.17):
        _rail(b, base, x, -0.39, 0.05, 0.36)
    carriage = b.moving(
        base, "sample_carriage", (0, -0.16, 0.40), (0, -1, 0), (0, 0.15), mass=0.28
    )
    b.box(carriage, (0, 0, 0), (0.20, 0.115, 0.025), "blue", collision=True)
    for x in (-0.12, 0, 0.12):
        b.cylinder(carriage, (x, 0, 0.065), 0.032, 0.045, "glass")
        b.cylinder(carriage, (x, 0, 0.114), 0.03, 0.006, "dark")
    b.site(carriage, "water_vial", (0, 0, 0.065))
    b.box(base, (0.51, 0.27, 1.81), (0.22, 0.10, 0.15), "shell")
    b.rod(base, (0.51, 0.27, 1.6), (0.51, 0.27, 1.68), 0.024, "metal")
    b.screen(base, (0.51, 0.16, 1.81), 0.32, 0.20)
    b.metadata["capabilities"] = [
        "Isolation-valve handle positioning",
        "Rigid water-vial carriage access",
    ]
    b.metadata["limitations"].append(
        "Membrane module and piping exterior are a representative campus test skid; pressure, filtration, flow and contaminant removal are not simulated."
    )


def _tank(b, base, radius, wall_height, material="cyan"):
    b.cylinder(base, (0, 0, 0.145), radius - 0.012, 0.035, material, collision=True)
    for a in (0, 2.094, 4.189):
        b.box(
            base,
            (0.72 * radius * math.cos(a), 0.72 * radius * math.sin(a), 0.065),
            (0.10, 0.10, 0.065),
            "shell",
            collision=True,
        )
    segments = 56
    for i in range(segments):
        a = 2 * math.pi * i / segments
        b.box(
            base,
            (radius * math.cos(a), radius * math.sin(a), 0.18 + wall_height / 2),
            (0.025, radius * math.tan(math.pi / segments) + 0.003, wall_height / 2),
            material,
            euler=(0, 0, a),
            collision=True,
        )
    b.ring(base, (0, 0, 0.18 + wall_height), radius, 0.033, material, segments=56)
    # Static display surface only: no fluid dynamics, buoyancy or volume claim.
    b.cylinder(base, (0, 0, 0.18 + wall_height - 0.12), radius - 0.042, 0.004, "glass")


def _aquarium_bridge(b, base, params):
    _tank(b, base, 1.14, 1.0)
    # A bolted metal bridge carries the probe with all posts outside the water.
    for x in (-1.27, 1.27):
        for y in (-0.27, 0.27):
            b.box(base, (x, y, 0.015), (0.12, 0.075, 0.015), "metal", collision=True)
            b.box(base, (x, y, 0.835), (0.023, 0.023, 0.805), "bright", collision=True)
            _bolts(b, base, (x - 0.075, x + 0.075), y, 0.034)
        b.box(base, (x, 0, 1.65), (0.038, 0.32, 0.025), "metal", collision=True)
    for y in (-0.20, 0.20):
        b.box(base, (0, y, 1.665), (1.29, 0.025, 0.025), "metal", collision=True)
        b.rod(base, (-1.16, y, 1.707), (1.16, y, 1.707), 0.012, "bright")
    carriage = b.moving(
        base,
        "probe_position",
        (0, 0, 1.744),
        (1, 0, 0),
        (-0.55, 0.55),
        mass=1.0,
        kp=250,
    )
    b.box(carriage, (0, 0, 0), (0.15, 0.26, 0.026), "blue", collision=True)
    for y in (-0.2, 0.2):
        b.box(carriage, (0, y, -0.025), (0.075, 0.035, 0.012), "dark")
    b.box(carriage, (0, 0, 0.12), (0.08, 0.09, 0.095), "shell")
    b.cylinder(carriage, (0, 0, 0.237), 0.047, 0.035, "dark")
    probe = b.moving(
        carriage, "probe_depth", (0, 0, -0.03), (0, 0, -1), (0, 0.28), mass=0.30
    )
    b.cylinder(probe, (0, 0, -0.235), 0.012, 0.235, "bright")
    b.cylinder(probe, (0, 0, -0.50), 0.022, 0.028, "dark")
    b.cylinder(probe, (0, 0, -0.532), 0.02, 0.004, "cream")
    b.site(probe, "probe_tip", (0, 0, -0.532))
    b.metadata["capabilities"] = [
        "Sampling-probe traverse",
        "Sampling-probe immersion position",
    ]
    b.metadata["limitations"].append(
        "Round tank and yellow rack context derive from UQ references; this motorized sampling bridge is an authored task fixture, not equipment verified at UQ. Static water has no fluid, animal or sensor model."
    )


BUILDERS = {
    "mask_alignment_station": _mask_aligner,
    "chemistry_fume_hood": _hood,
    "membrane_test_skid": _membrane_skid,
    "aquarium_sampling_bridge": _aquarium_bridge,
}
SOURCES = {
    "mask_alignment_station": {
        "reference": "Harvard CNS page-named mask aligner; generic original XY alignment mechanism",
        "url": HARVARD,
        "dimensions_m": [0.96, 0.84, 1.04],
        "dimension_basis": "Estimated envelope and travel, no OEM CAD or dimensional drawing",
    },
    "chemistry_fume_hood": {
        "reference": "Caltech Stoltz photographed glazed fume hoods and supported glassware",
        "url": CALTECH,
        "dimensions_m": [1.7, 0.94, 2.36],
        "dimension_basis": "Estimated from public photographs; sash and supply drawer are original mechanisms",
    },
    "membrane_test_skid": {
        "reference": "NTU SMTC brochure campus industrial project area and membrane test modules",
        "url": NTU,
        "dimensions_m": [1.64, 0.96, 1.96],
        "dimension_basis": "Estimated campus-scale frame, pipe routing and mechanical sample fixture; no Tuas pilot geometry",
    },
    "aquarium_sampling_bridge": {
        "reference": "UQ Moreton Bay aquarium round tank with authored sampling bridge",
        "url": UQ,
        "dimensions_m": [2.78, 2.34, 2.02],
        "dimension_basis": "Tank nominal 4000 L class is documented; model shape and added bridge dimensions are estimated, not surveyed",
    },
}
SAMPLE_INTERFACES = {
    "mask_alignment_station": (
        "wafer",
        (0.15, 0.15, 0.003),
        "clamped",
        "Inert wafer display geometry attached to XY carrier",
    ),
    "chemistry_fume_hood": (
        "sample_carrier",
        (0.18, 0.13, 0.06),
        "clamped",
        "Inert supply cassette mounted in under-deck drawer",
    ),
    "membrane_test_skid": (
        "water_vial",
        (0.064, 0.064, 0.09),
        "clamped",
        "Water-vial geometry retained by carriage",
    ),
    "aquarium_sampling_bridge": (
        "probe_tip",
        (0.04, 0.04, 0.008),
        "clamped",
        "Inert probe sensing cap; no salinity or other scientific measurement",
    ),
}


class _Parts:
    """Namespaced static details, independent of the instrument registry."""

    def __init__(self, world, prefix):
        self.world, self.prefix, self.count = world, prefix, 0

    def geom(self, name, shape, pos, size, color, *, collision=False, **extra):
        self.count += 1
        attrs = dict(
            name=f"facility_{self.prefix}_{name}_{self.count}",
            type=shape,
            size=" ".join(map(str, size)),
            rgba=" ".join(map(str, color)),
            contype="1" if collision else "0",
            conaffinity="1" if collision else "0",
        )
        if "fromto" not in extra:
            attrs["pos"] = " ".join(map(str, pos))
        attrs.update(
            {
                k: " ".join(map(str, v)) if isinstance(v, (tuple, list)) else str(v)
                for k, v in extra.items()
            }
        )
        return ET.SubElement(self.world, "geom", attrs)

    def box(self, name, pos, half, color, **extra):
        return self.geom(name, "box", pos, half, color, **extra)

    def pipe(self, name, start, end, radius=0.018, color=(0.73, 0.77, 0.76, 1)):
        return self.geom(
            name, "capsule", start, (radius,), color, fromto=(*start, *end)
        )

    def cylinder(self, name, pos, radius, half_height, color, **extra):
        return self.geom(name, "cylinder", pos, (radius, half_height), color, **extra)

    def ring(self, name, pos, radius, tube, color, segments=40):
        x, y, z = pos
        for i in range(segments):
            a, c = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
            self.pipe(
                name,
                (x + radius * math.cos(a), y + radius * math.sin(a), z),
                (x + radius * math.cos(c), y + radius * math.sin(c), z),
                tube,
                color,
            )


WHITE = (0.86, 0.87, 0.84, 1)
METAL = (0.53, 0.60, 0.62, 1)
DARK = (0.10, 0.14, 0.15, 1)
YELLOW = (0.91, 0.66, 0.055, 1)
GLASS = (0.48, 0.72, 0.77, 0.20)
BLUE = (0.065, 0.32, 0.51, 1)


def _stool(p, x, y, z=0.66):
    p.cylinder("stool_seat", (x, y, z), 0.21, 0.035, DARK)
    for a in (0, 2.094, 4.189):
        p.pipe(
            "stool_leg",
            (x + 0.17 * math.cos(a), y + 0.17 * math.sin(a), z - 0.035),
            (x + 0.27 * math.cos(a), y + 0.27 * math.sin(a), 0.035),
            0.018,
            METAL,
        )
    p.ring("stool_foot_ring", (x, y, 0.22), 0.245, 0.012, METAL, 20)


def _harvard(world, definition):
    p = _Parts(world, "harvard")
    # Selected bays, not the full building in the dollhouse image.
    for x in (-0.8, 2.75):
        p.box(
            "bay_partition", (x, 1.30, 1.51), (0.045, 2.45, 1.51), WHITE, collision=True
        )
        for y in (-0.35, 1.15, 2.65):
            p.box("bay_pass_window", (x - 0.048, y, 1.65), (0.003, 0.55, 0.32), GLASS)
    # Glazed front partition leaves a 1.2 m access opening into each bay.
    for x, width in ((-5.66, 1.32), (-2.38, 1.40), (0.20, 1.20), (4.52, 2.05)):
        p.box(
            "corridor_spandrel",
            (x, -1.10, 0.47),
            (width / 2, 0.035, 0.47),
            WHITE,
            collision=True,
        )
        p.box(
            "corridor_glazing",
            (x, -1.10, 1.93),
            (width / 2, 0.024, 0.98),
            (0.82, 0.66, 0.23, 0.25) if x < -0.8 else GLASS,
        )
        p.box("corridor_header", (x, -1.10, 3.0), (width / 2, 0.046, 0.11), WHITE)
        for edge in (x - width / 2, x + width / 2):
            p.box(
                "corridor_mullion",
                (edge, -1.10, 1.47),
                (0.019, 0.044, 1.47),
                METAL,
                collision=True,
            )
    p.box(
        "yellow_lithography_floor",
        (-3.59, 1.26, 0.001),
        (2.76, 2.39, 0.0007),
        (0.75, 0.59, 0.21, 1),
    )
    for x in (-5.25, -3.1, -1.7):
        p.box(
            "yellow_task_light",
            (x, 1.40, 2.90),
            (0.45, 0.16, 0.025),
            (1.0, 0.72, 0.19, 1),
        )
    for x in (0.75, 4.35):
        for dx in (-0.52, 0.52):
            for y in (2.43, 3.27):
                p.cylinder(
                    "process_leveling_foot",
                    (x + dx, y, 0.022),
                    0.055,
                    0.022,
                    DARK,
                    collision=True,
                )
        p.box(
            "process_cabinet",
            (x, 2.85, 0.84),
            (0.67, 0.55, 0.80),
            WHITE,
            collision=True,
        )
        p.box("process_chamber_front", (x, 2.292, 1.12), (0.43, 0.009, 0.25), METAL)
        p.cylinder(
            "load_port", (x, 2.27, 1.12), 0.18, 0.024, DARK, euler=(math.pi / 2, 0, 0)
        )
        p.cylinder(
            "load_port_center",
            (x, 2.241, 1.12),
            0.144,
            0.004,
            METAL,
            euler=(math.pi / 2, 0, 0),
        )
        p.box("controller", (x + 0.48, 2.27, 1.40), (0.10, 0.03, 0.15), DARK)
        p.box("controller_display", (x + 0.48, 2.237, 1.40), (0.085, 0.004, 0.13), BLUE)
        p.pipe("exhaust_duct", (x, 3.15, 1.65), (x, 3.15, 2.9), 0.11)
    # Observed corridor character; gowning storage is explicitly an estimate.
    for x in (-5.55, -4.82):
        p.box(
            "gowning_locker", (x, -3.29, 1.0), (0.33, 0.40, 0.98), WHITE, collision=True
        )
        for z in (0.45, 1.15, 1.70):
            p.box("locker_seam", (x, -2.881, z), (0.30, 0.006, 0.008), METAL)
    for x in (-3.8, 0.5, 4.5):
        p.box(
            "corridor_guide",
            (x, -2.24, 0.001),
            (0.5, 0.01, 0.0005),
            (0.34, 0.56, 0.62, 1),
        )


def _caltech(world, definition):
    p = _Parts(world, "caltech")
    # A central service spine with shelves distinguishes a chemistry bay.
    for x in (-1.54, 0, 1.54):
        p.box("bench_service_post", (x, 0.05, 1.38), (0.025, 0.045, 0.46), METAL)
    for z in (1.3, 1.79):
        p.box("reagent_shelf", (0, 0.05, z), (1.68, 0.23, 0.018), WHITE)
        for i in range(11):
            x = -1.5 + 0.29 * i
            p.cylinder(
                "reagent_bottle",
                (x, 0.06, z + 0.076),
                0.035,
                0.055,
                (0.36, 0.20, 0.06, 1),
            )
            p.cylinder("reagent_cap", (x, 0.06, z + 0.139), 0.03, 0.008, WHITE)
            p.box("reagent_label", (x, 0.023, z + 0.07), (0.027, 0.002, 0.035), WHITE)
    for x, y in ((-3.6, 1.02), (-1.2, 1.02), (1.2, 1.02), (1.2, -1.35), (-1.2, -1.35)):
        _stool(p, x, y)
    # Two-port glovebox context at left, on a supported dedicated cabinet.
    for x in (-4.72, -3.64):
        for y in (-0.51, 0.31):
            p.cylinder(
                "glovebox_foot", (x, y, 0.017), 0.045, 0.017, DARK, collision=True
            )
    p.box(
        "glovebox_support",
        (-4.18, -0.10, 0.48),
        (0.67, 0.53, 0.45),
        WHITE,
        collision=True,
    )
    p.box(
        "glovebox_back", (-4.18, 0.28, 1.43), (0.67, 0.08, 0.45), METAL, collision=True
    )
    p.box("glovebox_roof", (-4.18, -0.10, 1.89), (0.69, 0.52, 0.03), METAL)
    p.box("glovebox_floor", (-4.18, -0.10, 0.98), (0.69, 0.52, 0.025), METAL)
    for x in (-4.83, -3.53):
        p.box("glovebox_side", (x, -0.10, 1.44), (0.025, 0.52, 0.45), METAL)
    p.box("glovebox_window", (-4.18, -0.58, 1.44), (0.64, 0.018, 0.43), GLASS)
    for x in (-4.48, -3.91):
        p.cylinder(
            "glove_port",
            (x, -0.618, 1.29),
            0.114,
            0.036,
            DARK,
            euler=(math.pi / 2, 0, 0),
        )
        p.pipe("glove_sleeve", (x, -0.65, 1.29), (x, -0.91, 1.19), 0.079, DARK)
        p.geom("glove_tip", "ellipsoid", (x, -0.95, 1.16), (0.064, 0.08, 0.041), DARK)
    p.cylinder(
        "glovebox_antechamber",
        (-5.02, -0.10, 1.32),
        0.21,
        0.17,
        METAL,
        euler=(0, math.pi / 2, 0),
    )
    for x in (-3.5, -1.25, 1.0):
        p.pipe("hood_exhaust", (x, 2.8, 2.30), (x, 2.8, 3.05), 0.13)
    for x in (-0.85, 0.85):
        p.box("hotplate", (x, -0.39, 0.98), (0.14, 0.14, 0.06), BLUE)
        p.cylinder("hotplate_top", (x, -0.39, 1.05), 0.13, 0.013, METAL)
        p.cylinder("empty_beaker", (x, -0.39, 1.12), 0.065, 0.055, GLASS)


def _ntu(world, definition):
    p = _Parts(world, "ntu")
    # Industrial research aisle with frame rigs and overhead utility carriers.
    for y in (-1.4, 0.9, 2.8):
        p.box("ceiling_service_carrier", (0, y, 2.78), (4.15, 0.055, 0.055), METAL)
        for x in (-3.6, 0, 3.6):
            p.pipe("service_hanger", (x, y, 2.83), (x, y, 3.12), 0.014)
    for x in (-1.5, 1.5):
        p.box("industrial_aisle_marking", (x, 0, 0.001), (0.018, 2.8, 0.0005), YELLOW)
    # Membrane preparation enclosure, visually supported but not a process model.
    x, y = -3.28, 0.4
    for dx in (-0.56, 0.56):
        for dy in (-0.37, 0.37):
            p.cylinder(
                "casting_foot",
                (x + dx, y + dy, 0.017),
                0.042,
                0.017,
                DARK,
                collision=True,
            )
    p.box("casting_cabinet", (x, y, 0.50), (0.69, 0.48, 0.47), METAL, collision=True)
    p.box("casting_deck", (x, y, 1.0), (0.74, 0.53, 0.03), METAL, collision=True)
    for dx in (-0.69, 0.69):
        p.box("casting_guard_post", (x + dx, y + 0.3, 1.4), (0.035, 0.1, 0.38), WHITE)
    p.box("casting_top", (x, y + 0.3, 1.81), (0.74, 0.23, 0.04), WHITE)
    for dx in (-0.3, 0, 0.3):
        p.cylinder(
            "casting_roller",
            (x + dx, y, 1.14),
            0.047,
            0.34,
            METAL,
            euler=(math.pi / 2, 0, 0),
        )
    p.box("casting_control", (x - 0.44, y - 0.492, 0.74), (0.18, 0.012, 0.075), DARK)
    p.box("casting_glazing", (x, y - 0.34, 1.42), (0.66, 0.012, 0.31), GLASS)
    _stool(p, -2.83, -0.85)
    for x in (-3.55, -2.97):
        p.cylinder("feed_vessel", (x, 2.20, 0.52), 0.22, 0.46, WHITE, collision=True)
        p.cylinder("feed_vessel_lid", (x, 2.20, 0.991), 0.232, 0.013, BLUE)
        p.pipe("feed_pipe", (x, 2.20, 1.0), (x, 2.2, 1.8), 0.02)
        p.pipe("feed_pipe_upper", (x, 2.2, 1.8), (x + 0.4, 2.2, 1.8), 0.02)
    for y in (-1.7, 0.1, 2.15):
        p.pipe("utility_drop", (4.23, y, 2.70), (4.23, y, 1.25), 0.018)


def _static_tank(p, name, x, y, radius, height, color):
    p.cylinder(name + "_bottom", (x, y, 0.15), radius, 0.045, color, collision=True)
    for i in range(56):
        a = 2 * math.pi * i / 56
        p.box(
            name + "_wall",
            (x + radius * math.cos(a), y + radius * math.sin(a), 0.19 + height / 2),
            (0.03, radius * math.tan(math.pi / 56) + 0.003, height / 2),
            color,
            euler=(0, 0, a),
            collision=True,
        )
    p.ring(name + "_rim", (x, y, 0.19 + height), radius, 0.038, color, 56)
    p.cylinder(
        name + "_water",
        (x, y, height + 0.075),
        radius - 0.04,
        0.004,
        (0.08, 0.38, 0.40, 0.7),
    )
    for a in (0, 2.094, 4.189):
        p.box(
            name + "_foot",
            (x + radius * 0.74 * math.cos(a), y + radius * 0.74 * math.sin(a), 0.06),
            (0.14, 0.14, 0.06),
            WHITE,
            collision=True,
        )


def _uq(world, definition):
    p = _Parts(world, "uq")
    _static_tank(p, "large_tank", -3.35, -1.6, 1.66, 1.15, (0.60, 0.65, 0.61, 1))
    _static_tank(p, "round_tank", 3.15, -1.6, 1.14, 1.0, (0.02, 0.36, 0.36, 1))
    # Three yellow racks, each with four tanks on three levels.
    for rack_x in (-4.1, 0, 4.1):
        for x in (rack_x - 1.3, rack_x + 1.3):
            for y in (2.76, 3.53):
                p.box(
                    "aquarium_rack_post",
                    (x, y, 1.21),
                    (0.026, 0.026, 1.18),
                    YELLOW,
                    collision=True,
                )
                p.box(
                    "aquarium_rack_foot",
                    (x, y, 0.025),
                    (0.07, 0.07, 0.025),
                    METAL,
                    collision=True,
                )
        for z in (0.35, 1.02, 1.69):
            p.box(
                "aquarium_shelf",
                (rack_x, 3.15, z),
                (1.33, 0.43, 0.025),
                YELLOW,
                collision=True,
            )
            for col in range(4):
                x = rack_x - 0.945 + col * 0.63
                p.box(
                    "glass_tank_base",
                    (x, 3.15, z + 0.033),
                    (0.277, 0.338, 0.009),
                    (0.14, 0.41, 0.39, 1),
                )
                for xx in (x - 0.274, x + 0.274):
                    p.box(
                        "glass_tank_side",
                        (xx, 3.15, z + 0.25),
                        (0.006, 0.338, 0.21),
                        GLASS,
                    )
                for yy in (2.812, 3.488):
                    p.box(
                        "glass_tank_front",
                        (x, yy, z + 0.25),
                        (0.278, 0.006, 0.21),
                        GLASS,
                    )
                p.box(
                    "rack_water",
                    (x, 3.15, z + 0.355),
                    (0.269, 0.328, 0.003),
                    (0.09, 0.41, 0.39, 0.68),
                )
                p.pipe(
                    "rack_supply",
                    (x + 0.20, 3.50, 2.50),
                    (x + 0.20, 3.50, z + 0.31),
                    0.007,
                )
                p.cylinder("standpipe", (x + 0.20, 3.40, z + 0.24), 0.01, 0.17, WHITE)
        p.pipe(
            "rack_feed_header",
            (rack_x - 1.25, 3.56, 2.52),
            (rack_x + 1.25, 3.56, 2.52),
            0.023,
        )
    # Overhead distribution and a wet-floor drainage channel.
    for y in (-1.6, 1.28, 3.60):
        p.pipe("water_main", (-6.7, y, 2.92), (6.7, y, 2.92), 0.027)
        for x in (-5.3, 0, 5.3):
            p.pipe("utility_hanger", (x, y, 2.94), (x, y, 3.24), 0.009)
    for x in (-3.35, 0, 3.15):
        p.pipe(
            "tank_feed_drop", (x + 0.78, -0.82, 2.90), (x + 0.78, -0.82, 1.54), 0.018
        )
        p.pipe(
            "tank_feed_elbow", (x + 0.78, -0.82, 1.54), (x + 0.55, -0.82, 1.54), 0.018
        )
    p.box("drainage_trench", (0, 1.10, 0.002), (6.5, 0.075, 0.002), DARK)
    for x in range(-25, 26):
        p.box("drainage_grate", (x * 0.25, 1.10, 0.006), (0.012, 0.073, 0.004), METAL)
    # An isolated sump beside the wall; pump and valves are visual context.
    p.box(
        "filtration_sump", (5.70, -0.30, 0.5), (0.56, 0.64, 0.43), WHITE, collision=True
    )
    p.cylinder("filter_column", (5.77, -0.30, 1.23), 0.17, 0.30, GLASS)
    p.pipe("filter_return", (5.77, -0.30, 1.55), (5.77, -0.30, 2.91), 0.025)


FEATURES = {
    "harvard_cns_cambridge": _harvard,
    "caltech_stoltz_schlinger": _caltech,
    "ntu_smtc_membranes": _ntu,
    "uq_moreton_aquarium": _uq,
}
