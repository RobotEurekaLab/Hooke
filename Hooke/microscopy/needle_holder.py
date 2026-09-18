"""Independent HI-7/HIR appearance reference with explicitly estimated details."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import body, cylinder, geom, knurled_knob, numbers

HOLDER_LENGTH_M = .140
SHAFT_DIAMETER_M = .004
CAPILLARY_EXPOSED_M = .012


def mount_holder(parent, tool, back):
    """Attach a nominal 140 mm holder behind the unchanged 12 mm glass tip."""
    back = np.asarray(back, dtype=float)
    tangent = np.cross([0., 0., 1.], back)
    tangent /= np.linalg.norm(tangent)
    rotation = np.column_stack((tangent, np.cross(back, tangent), back))
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    node = body(parent, tool+"_needle_holder", back*CAPILLARY_EXPOSED_M,
                quat=numbers(quaternion), gravcomp="1")
    ET.SubElement(node, "inertial", pos="0 0 .070", mass=".000001", diaginertia="1e-9 1e-9 1e-9")
    # Total axial holder envelope is 140 mm; fine nose/endcap details are estimates.
    cylinder(node, tool+"_holder_nose", (0, 0, 0), (0, 0, .010), .003, "graphite")
    knurled_knob(node, tool+"_holder_cap", (0, 0, .006), (0, 0, 1), .0036, .008, ribs=24)
    cylinder(node, tool+"_holder_shaft", (0, 0, .010), (0, 0, .134), SHAFT_DIAMETER_M/2)
    cylinder(node, tool+"_holder_tail", (0, 0, .134), (0, 0, HOLDER_LENGTH_M), .0025, "graphite")
    # HIR official 7 x 7 x 18 mm envelope; its bore/collar details are unverified.
    geom(node, tool+"_rotation_clamp", "box", (.0035, .0035, .009), (0, 0, .118), "graphite")
    knurled_knob(node, tool+"_clamp_screw", (.005, 0, .118), (1, 0, 0), .003, .004, ribs=16)
    # Keep appearance independent of the declared slide inertia and gravity
    # compensation. Vendor mass properties have not been measured.
    for element in node.iter("geom"):
        element.set("mass", "0")
    return dict(name="Independent HI-7/HIR dimensional and photo reference", official_cad=False,
        source_urls=["https://products.narishige-group.com/group1/HI-7/injection/english.html",
                     "https://products.narishige-group.com/group1/HIR/injection/english.html"],
        holder_length_m=HOLDER_LENGTH_M, shaft_diameter_m=SHAFT_DIAMETER_M,
        clamp_envelope_m=[.007, .007, .018], exposed_capillary_length_m=CAPILLARY_EXPOSED_M,
        mass_properties="Massless appearance meshes; gravity-compensated 1 mg numerical body placeholder, not vendor metrology",
        estimates=["Nose, knurling, pressure seal, connector and clamp bore details",
                   "Independent Zaber-to-holder adapter; no verified commercial mounting fit"],
        one_to_one_verified=False)
