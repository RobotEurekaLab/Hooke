"""Articulated MP-285 dimensional reference with a XenoWorks-style installation.

Published overall dimensions constrain an independently modelled exterior.
Unmarked internal geometry, link masses, drive gains, holder and TE2000 bracket
details remain estimates. This is neither OEM CAD nor a validated assembly.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import body, cylinder, geom, numbers, rounded_box, slide
from microscopy.sutter_holder import PIVOT_FROM_BASE_M, mount_holder

PROFILE = "sutter-mp285-reference"
DRAWING_URL = "https://www.sutter.com/hubfs/WEB%20-%20PDF%20Files/SutterCatalog.pdf#page=89"
MOUNT_URL = "https://www.sutter.com/manuals/MP-285_RefMan.pdf#page=64"
INSTALL_URL = "https://www.sutter.com/hubfs/WEB%20-%20Manuals/XW_XWM-100_OpMan_r103h.pdf#page=25"
DRAWING_ENVELOPE_M = np.array([4.83, 6.83, 5.78])*.0254
BASE_SIDE_M = 2.75*.0254
MOUNT_SPACING_M = 2.5*.0254
TRAVEL_M = .025
PIVOT_HEIGHT_M = .069  # Digitized exterior proportion, not a dimensioned callout.


class _Exterior:
    """Pose appearance in an assembly frame while leaving slide frames unrotated."""
    def __init__(self, asset, tool, rotation, translation):
        self.asset, self.tool = asset, tool
        self.rotation, self.translation = rotation, translation
        self.quat = numbers(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]])

    def point(self, point):
        return self.translation+self.rotation@np.asarray(point)

    def box(self, node, name, centre, halfsize, material="graphite", bevel=.0007):
        rounded_box(self.asset, node, self.tool+"_sutter_"+name, halfsize, self.point(centre),
                    bevel, material, quat=self.quat, mass="0", contype="0", conaffinity="0")

    def rod(self, node, name, start, end, radius, material="metal"):
        cylinder(node, self.tool+"_sutter_"+name, self.point(start), self.point(end), radius,
                 material, mass="0", contype="0", conaffinity="0")

    def screw(self, node, name, centre, normal=(0., 0., 1.), radius=.0025):
        centre, normal = np.asarray(centre), np.asarray(normal)
        self.rod(node, name, centre-normal*.001, centre+normal*.001, radius)
        self.rod(node, name+"_socket", centre+normal*.001, centre+normal*.0011,
                 radius*.43, "graphite")


def _mounting(exterior, root, origin, home, bench_top):
    """Broad dovetail plate and a declared estimate of the microscope-side bracket."""
    e = exterior
    outward = np.sign(e.rotation[:, 0] @ e.translation) or 1.
    plate_u, support_u = outward*.032, outward*.057
    # Use the catalog's nominal metric adapter envelope; imperial values
    # disagree slightly and are preserved in metadata rather than averaged.
    e.box(root, "adapter_plate", (plate_u, .059, -.005), (.0725, .0775, .004), "metal")
    for u in (-.052, .052):
        e.box(root, f"adapter_slot_{u}", (plate_u+u, .057, -.00095), (.0025, .057, .0001), "graphite", .00008)
        for v in (.011, .097):
            e.screw(root, f"adapter_lock_{u}_{v}", (plate_u+u, v, .0005), radius=.003)
    e.box(root, "adapter_dovetail", (support_u, .057, -.014), (.022, .060, .005), "graphite")
    e.box(root, "bracket_deck", (support_u, .057, -.023), (.031, .065, .004), "graphite")
    # Two compact webs lead to the stand-side saddle; their placement and
    # fit are explicit estimates until the BR-N27 mounting sheet is available.
    # Carry the outboard deck down to the casting below the moving stage and
    # its existing legs; no support crosses the objective turret.
    base_height = origin[2]+home[2]+e.translation[2]
    saddle_z = bench_top+.163-base_height
    web_bottom, web_top = saddle_z+.004, -.027
    for u in (support_u-.021, support_u+.021):
        e.box(root, f"bracket_web_{u}", (u, .057, (web_bottom+web_top)/2),
              (.005, .038, (web_top-web_bottom)/2), "graphite")
    e.box(root, "bracket_saddle", (support_u, .057, saddle_z), (.034, .044, .004), "graphite")
    # A wide transverse bridge meets the side of the stand instead of a
    # slender, freestanding post. The pictured BR-N27 interface is unverified.
    saddle_world = origin+home+e.point((support_u, .057, saddle_z))
    side = np.sign(saddle_world[0]-origin[0]) or 1.
    anchor = np.array([origin[0]+side*.086, saddle_world[1], saddle_world[2]])
    start = saddle_world.copy()
    centre = (start+anchor)/2-(origin+home)
    size = np.array([abs(start[0]-anchor[0])/2+.008, .035, .006])
    rounded_box(e.asset, root, e.tool+"_sutter_stand_bridge", size, centre, .001,
                "graphite", mass="0", contype="0", conaffinity="0")
    return dict(adapter_nominal_metric_m=[.145, .155, .008],
        adapter_catalog_inches=[5.75, 6., .3],
        interface="Estimated BR-N27-style side bracket; mating holes and dovetail unverified",
        support="Broad plate, lower dovetail, boxed webs and transverse stand-side bridge",
        adapter_outward_offset_m=plate_u,
        stand_fit_verified=False, stiffness_verified=False,
        bridge_anchor_world_m=anchor.tolist(), bench_top_m=float(bench_top))


def mount_sutter(world, asset, actuators, tool, home, origin, bench_top, back, *,
                 terminal_back=None, capillary_base=None):
    """Return the real XYZ tool body and provenance, preserving the shared SI TCP."""
    home, origin, back = (np.asarray(value, dtype=float) for value in (home, origin, back))
    if back.shape != (3,) or not np.isclose(np.linalg.norm(back), 1.) or back[2] < 0:
        raise ValueError("Sutter holder direction must be an upward unit vector")
    base = back*.012 if capillary_base is None else np.asarray(capillary_base, dtype=float)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("Capillary base must be a finite TCP-relative position")
    radial = back.copy()
    radial[2] = 0
    radial /= np.linalg.norm(radial)
    depth = np.cross([0., 0., 1.], radial)
    if radial[0] < -.25 or radial[1] > .25:
        depth *= -1
    width = np.cross(depth, [0., 0., 1.])
    rotation = np.column_stack((width, depth, [0., 0., 1.]))
    pivot = base+back*PIVOT_FROM_BASE_M
    # The carrier sits in front of the protractor, not inside the Z carriage.
    translation = pivot+depth*.026-np.array([0., 0., PIVOT_HEIGHT_M])
    e = _Exterior(asset, tool, rotation, translation)
    root = body(world, tool+"_mount", origin+home)
    ET.SubElement(root, "inertial", pos="0 0 0", mass=".000001", diaginertia="1e-8 1e-8 1e-8")
    support = _mounting(e, root, origin, home, bench_top)
    # Fixed base: published 69.85 mm square and four 63.5 mm mounting centres.
    e.box(root, "fixed_base", (0., .081, .003), (BASE_SIDE_M/2, BASE_SIDE_M/2, .003))
    e.box(root, "base_side_lug", (.048419, .081, .003), (.013494, .028575, .003))
    for u in (-MOUNT_SPACING_M/2, MOUNT_SPACING_M/2):
        for v in (-MOUNT_SPACING_M/2, MOUNT_SPACING_M/2):
            p = (u, .081+v, .0065)
            e.screw(root, f"base_mount_{u}_{v}", p, radius=.0026)
            e.rod(root, f"base_washer_{u}_{v}", (u, .081+v, -.001),
                  (u, .081+v, 0.), .003, "metal")
    for v in (.057, .105):
        e.rod(root, f"x_rail_{v}", (-.038, v, .010), (.038, v, .010), .0026)
    e.box(root, "x_motor", (-.0418, .081, .011), (.0068, .025, .009), "metal")
    x = slide(root, actuators, tool+"_x", rotation[:, 0], (-TRAVEL_M/2, TRAVEL_M/2), .08, 8000, 40)
    e.box(x, "x_carriage", (0., .081, .016), (.0345, .0345, .005))
    e.box(x, "y_track", (0., .078, .024), (.028575, .0675, .0035), "metal")
    for u in (-.022, .022):
        e.rod(x, f"y_rail_{u}", (u, .015, .029), (u, .140, .029), .002)
    e.box(x, "y_motor", (.054, .072, .024), (.020, .022, .011), "metal")
    e.rod(x, "y_motor_cap", (.054, .072, .035), (.054, .072, .037), .018, "graphite")
    e.screw(x, "y_motor_centre", (.054, .072, .038), radius=.004)
    y = slide(x, actuators, tool+"_y", rotation[:, 1], (-TRAVEL_M/2, TRAVEL_M/2), .08, 8000, 40)
    e.box(y, "y_carriage", (0., .061, .035), (.028575, .041, .006))
    e.box(y, "z_tower", (0., .024, (.030+DRAWING_ENVELOPE_M[2])/2),
          (.028575, .008, (DRAWING_ENVELOPE_M[2]-.030)/2))
    for u in (-.0215, .0215):
        e.rod(y, f"z_rail_{u}", (u, .014, .034), (u, .014, .141), .0025, "metal")
    e.box(y, "z_motor", (0., .042, .124), (.023, .010, .019))
    for u in (-.020, .020):
        e.screw(y, f"z_top_screw_{u}", (u, .015, .139), (0., -1., 0.))
    # A rear cable connector and strain relief identify the powered stages.
    e.box(y, "motor_connector", (.030, .034, .097), (.004, .009, .006), "metal")
    e.rod(y, "motor_strain_relief", (.034, .034, .097), (.047, .034, .097), .003, "rubber")
    z = slide(y, actuators, tool+"_z", (0., 0., 1.), (-TRAVEL_M/2, TRAVEL_M/2), .05, 8000, 40)
    e.box(z, "z_carriage", (0., .005, .078), (.028575, .008, .041))
    # Swing gate / protractor assembly follows Z; its angle is locked during
    # an experiment, while the three motorized slides remain articulated.
    e.box(z, "swing_gate", (0., -.0075, PIVOT_HEIGHT_M), (.028, .0045, .031), "graphite")
    for u in (-.024, .024):
        e.rod(z, f"gate_pin_{u}", (u, -.006, .098), (u, -.006, .123), .002)
        e.screw(z, f"gate_lock_{u}", (u, -.006, .124), radius=.0035)
    e.rod(z, "angle_dial", (0., -.012, PIVOT_HEIGHT_M), (0., -.015, PIVOT_HEIGHT_M), .024, "metal")
    e.rod(z, "angle_dial_face", (0., -.015, PIVOT_HEIGHT_M), (0., -.0153, PIVOT_HEIGHT_M), .0225, "graphite")
    for angle in range(0, 360, 5):
        a = np.deg2rad(angle)
        inner = .019 if angle % 15 == 0 else .0204
        p = (inner*np.cos(a), -.0155, PIVOT_HEIGHT_M+inner*np.sin(a))
        q = (.022*np.cos(a), -.0155, PIVOT_HEIGHT_M+.022*np.sin(a))
        e.rod(z, f"angle_tick_{angle}", p, q, .00016, "metal")
    e.rod(z, "angle_hub", (0., -.015, PIVOT_HEIGHT_M), (0., -.024, PIVOT_HEIGHT_M), .011, "metal")
    e.screw(z, "angle_lock", (0., -.025, PIVOT_HEIGHT_M), (0., -1., 0.), radius=.004)
    # Attach the tool frame at the shared world-aligned TCP, without placing
    # the instrument itself at that microscopic point.
    tcp = body(z, tool+"_tcp_frame", gravcomp="1")
    ET.SubElement(tcp, "inertial", pos="0 0 0", mass=".000001", diaginertia="1e-9 1e-9 1e-9")
    ET.SubElement(tcp, "site", name=tool+"_tcp", size=".00005", group="5")
    holder = None
    if tool in ("holder", "injector"):
        holder = mount_holder(asset, tcp, tool, back, base, terminal_back=terminal_back)
    else:
        cylinder(tcp, tool+"_sutter_tool_shank", pivot, back*.023, .003, "metal", mass="0")
        cylinder(tcp, tool+"_sutter_tool_collet", back*.023, back*.009, .0035, "graphite", mass="0")
    # Short locked interface joins the rotating carrier to the protractor.
    cylinder(tcp, tool+"_sutter_pivot_spigot", pivot,
             pivot+rotation[:, 1]*.006, .006, "metal", mass="0", contype="0", conaffinity="0")
    evidence = dict(tool=tool, profile=PROFILE, official_cad=False, one_to_one_verified=False,
        source_urls=[DRAWING_URL, MOUNT_URL, INSTALL_URL],
        drawing_envelope_m=DRAWING_ENVELOPE_M.tolist(),
        base_square_m=BASE_SIDE_M, base_mount_square_m=MOUNT_SPACING_M,
        travel_m=TRAVEL_M, chain=[tool+"_"+axis for axis in "xyz"],
        axis_directions_world=rotation.T.tolist(),
        tool_back_direction=back.tolist(),
        terminal_back_direction=np.asarray(back if terminal_back is None else terminal_back).tolist(),
        tool_elevation_deg=float(np.rad2deg(np.arcsin(back[2]))), capillary_base_m=base.tolist(),
        rotary_pivot_m=pivot.tolist(), needle_holder=holder, bench_support=support,
        articulation="Three motorized serial rigid slides; locked swing gate and angle clamp",
        dynamics="Estimated 0.08/0.08/0.05 kg slide masses and position gains; fixed tool frames use 1 mg numerical inertias; no OEM accuracy or rigidity claim",
        dimension_scope="MP-285 catalog assembly pose; XenoWorks mounting architecture, not mixed vendor envelopes",
        estimates=["Unmarked carriage, motor, rail and fastener dimensions digitized from orthographic outlines",
                   "69 mm base-to-rotary-pivot height and 1 mm precision-washer thickness",
                   "BR-N27 bracket and instrument installation fit"],
        rights="Independently authored geometry; manufacturer drawings and photos remain local references")
    return tcp, evidence
