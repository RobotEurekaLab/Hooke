"""Independent Sutter BR-AW/BR-MH appearance reference, with estimated details.

The 50 mm rod clamp and 4 mm holder rod are published nominal dimensions.
The holder length, chuck sections and clamp cross-section are estimates from
the installation illustration, not manufacturer CAD or a validated fit.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import annular_tube, body, cylinder, geom, numbers, rounded_box
from microscopy.stand_details import turned_part

ROD_DIAMETER_M = .004
HOLDER_LENGTH_M = .110
CLAMP_LENGTH_M = .050
PIVOT_FROM_BASE_M = .125
SOURCE_URL = "https://www.sutter.com/microinjection/xenoworksmm"
HOLDER_SOURCE_URL = "https://www.sutter.com/hubfs/WEB%20-%20Manuals/XW_BRE_Digital_OpMan.pdf#page=15"


def mount_holder(asset, parent, tool, back, capillary_base, *, terminal_back=None):
    """Mount the holder at the supplied glass rear, independently of its tip tangent."""
    back = np.asarray(back, dtype=float)
    base = np.asarray(capillary_base, dtype=float)
    tangent = np.cross([0., 0., 1.], back)
    tangent /= np.linalg.norm(tangent)
    rotation = np.column_stack((tangent, np.cross(back, tangent), back))
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    node = body(parent, tool+"_needle_holder", base, quat=numbers(quaternion), gravcomp="1")
    ET.SubElement(node, "inertial", pos="0 0 .05", mass=".000001", diaginertia="1e-9 1e-9 1e-9")
    # The open chuck admits the glass; no opaque sphere caps its bore.
    annular_tube(asset, node, tool+"_sutter_chuck_nose", 0., .005, .0032, .00055, "metal")
    turned_part(asset, node, tool+"_sutter_chuck_grip",
                [(.005, .003), (.006, .0035), (.013, .0035), (.014, .0028)],
                material="metal", flutes=32, flute_depth=.00016)
    cylinder(node, tool+"_sutter_seal", (0, 0, .014), (0, 0, .015), .0025, "rubber")
    cylinder(node, tool+"_sutter_holder_rod", (0, 0, .015), (0, 0, .103), ROD_DIAMETER_M/2, "metal")
    turned_part(asset, node, tool+"_sutter_pressure_tail",
                [(.103, .002), (.104, .0027), (.108, .0027), (HOLDER_LENGTH_M, .002)],
                material="metal", flutes=24, flute_depth=.00014)
    # A short, broad adjustable rod carrier runs alongside the actual rod.
    # Its nominal axial extent is documented; cross-sections and bore details
    # remain estimated. The rear bridge terminates at the rotation pivot.
    for side in (-1, 1):
        rounded_box(asset, node, f"{tool}_sutter_carrier_rail_{side}",
                    (.0025, .006, CLAMP_LENGTH_M/2),
                    (side*.005, 0., PIVOT_FROM_BASE_M-CLAMP_LENGTH_M/2), .0007, "metal")
    for z in (PIVOT_FROM_BASE_M-CLAMP_LENGTH_M+.004, PIVOT_FROM_BASE_M-.004):
        rounded_box(asset, node, f"{tool}_sutter_carrier_bridge_{z}",
                    (.0075, .002, .004), (0., .006, z), .0005, "metal")
        cylinder(node, f"{tool}_sutter_carrier_screw_{z}",
                 (0., .005, z), (0., .010, z), .0022, "graphite")
    rounded_box(asset, node, tool+"_sutter_carrier_tail", (.0075, .006, .003),
                (0., 0., PIVOT_FROM_BASE_M-.003), .0007, "metal")
    ET.SubElement(node, "site", name=tool+"_pneumatic_port", pos=numbers((0., 0., HOLDER_LENGTH_M)),
                  size=".0001", group="5")
    for element in node.iter("geom"):
        element.set("mass", "0")
        element.set("contype", "0")
        element.set("conaffinity", "0")
    return dict(name="Independent BR-AW/BR-MH dimensional and assembly reference",
        official_cad=False, one_to_one_verified=False,
        source_urls=[SOURCE_URL, HOLDER_SOURCE_URL],
        rod_diameter_m=ROD_DIAMETER_M, clamp_length_m=CLAMP_LENGTH_M,
        holder_length_m=HOLDER_LENGTH_M, estimated_holder_length=True,
        capillary_base_m=base.tolist(), shaft_back_direction=back.tolist(),
        terminal_back_direction=np.asarray(back if terminal_back is None else terminal_back).tolist(),
        pivot_m=(base+back*PIVOT_FROM_BASE_M).tolist(),
        pneumatic_port_site=tool+"_pneumatic_port",
        estimates=["110 mm holder length and 125 mm glass-rear to rotary-pivot distance",
                   "Chuck, seals, pressure connector and clamp cross-sections; no verified threads or fit"],
        mass_properties="Massless appearance; 1 mg numerical fixed child, not measured holder inertia")
