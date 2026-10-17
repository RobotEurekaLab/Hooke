"""Photograph-bounded exterior details for selected laboratory prototypes.

Positions and dimensions are authored estimates. Static context has no process
response; equipment joints, controls and specimen interfaces remain separate.
Reference attribution and uncertainty live in evidence/source_context_details.json.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers


STEEL = (0.57, 0.61, 0.62, 1)
DARK = (0.075, 0.085, 0.09, 1)
WHITE = (0.83, 0.84, 0.80, 1)
BLUE = (0.10, 0.30, 0.47, 1)
WOOD = (0.62, 0.47, 0.28, 1)


class _Parts:
    def __init__(self, parent, prefix):
        self.parent, self.prefix = parent, prefix

    def box(self, key, position, half, color=STEEL, *, collision=True, **attributes):
        return box(self.parent, self.prefix + key, half, position, color,
                   collision=collision, **attributes)

    def rod(self, key, start, end, radius=.006, color=STEEL, *, collision=False):
        return ET.SubElement(
            self.parent, "geom", name=self.prefix + key, type="capsule",
            size=str(radius), fromto=numbers((*start, *end)), rgba=numbers(color),
            contype=str(int(collision)), conaffinity=str(int(collision)),
        )

    def cylinder(self, key, position, radius, half, color=STEEL, *, collision=True,
                 **attributes):
        return ET.SubElement(
            self.parent, "geom", name=self.prefix + key, type="cylinder",
            pos=numbers(position), size=numbers((radius, half)), rgba=numbers(color),
            contype=str(int(collision)), conaffinity=str(int(collision)), **attributes,
        )

    def lead(self, key, points, color=DARK, radius=.004):
        for index, (start, end) in enumerate(zip(points, points[1:])):
            self.rod(f"{key}_{index}", start, end, radius, color)


def _desk(p, key, center, half, height=.82):
    x, y = center
    p.box(key + "_top", (x, y, height - .025), (*half, .025), DARK)
    for dx in (-half[0] + .05, half[0] - .05):
        for dy in (-half[1] + .05, half[1] - .05):
            p.box(f"{key}_leg_{dx}_{dy}", (x + dx, y + dy, (height - .05) / 2),
                  (.025, .025, (height - .05) / 2))
    p.box(key + "_rear_tie", (x, y + half[1] - .05, .25),
          (half[0] - .05, .025, .025))


def _monitor(p, key, position, width=.40):
    x, y, z = position
    p.box(key + "_foot", (x, y, z + .012), (.12, .10, .012), DARK)
    p.box(key + "_stem", (x, y + .025, z + .10), (.024, .024, .076), DARK)
    p.box(key + "_case", (x, y + .025, z + .265), (width / 2, .025, .135), DARK)
    p.box(key + "_blank_display", (x, y - .001, z + .265),
          (width / 2 - .016, .001, .118), (.035, .045, .05, 1), collision=False)


def _brown(world):
    p = _Parts(world.find("./body[@name='afam_test_section']"), "afam_context_")
    # The photograph shows a heterogeneous open rack, not four large displays.
    for index, z in enumerate((.36, .73, 1.10, 1.47)):
        p.box(f"rack_face_{index}", (2.875, .690, z), (.226, .006, .092),
              STEEL if index in (0, 3) else DARK, collision=False)
        p.box(f"rack_small_display_{index}", (2.79, .683, z + .025),
              (.055, .001, .019), DARK, collision=False)
        for j in range(4 if index % 2 else 2):
            p.cylinder(f"rack_socket_{index}_{j}", (2.92 + j * .033, .679, z - .025),
                       .009, .004, STEEL, collision=False,
                       euler=numbers((math.pi / 2, 0, 0)))
        for side in (-1, 1):
            p.rod(f"rack_handle_{index}_{side}", (2.875 + side * .203, .675, z - .05),
                  (2.875 + side * .203, .675, z + .05), .005)
        p.lead(f"rack_harness_{index}", [(3.14, .975, z), (3.22, .975, z),
               (3.22, .975, .14), (2.98, 1.23, .14)], radius=.005)
    # Rear-rail cable clamps terminate the dressing on the rack itself.
    for z in (.20, .62, 1.04, 1.46):
        p.box(f"rack_cable_clip_{z}", (3.19, .975, z), (.055, .018, .012), DARK)


def _caltech(world):
    p = _Parts(world, "stoltz_context_")
    # Narrow service cupboards fill the photographed continuous hood run.
    for index, x in enumerate((-2.375, -.125)):
        p.box(f"service_plinth_{index}", (x, 2.75, .06), (.25, .38, .06), DARK)
        p.box(f"service_cabinet_{index}", (x, 2.75, .51), (.25, .38, .39), WHITE)
        p.box(f"service_worktop_{index}", (x, 2.75, .924), (.26, .40, .024), DARK)
        p.box(f"service_pier_{index}", (x, 3.10, 1.53), (.25, .035, .58), WHITE)
        for z in (1.12, 1.40):
            p.box(f"outlet_plate_{index}_{z}", (x, 3.058, z), (.083, .007, .045), WHITE)
            for dx in (-.035, .035):
                p.box(f"outlet_{index}_{z}_{dx}", (x + dx, 3.050, z),
                      (.018, .001, .022), DARK, collision=False)
        p.rod(f"cabinet_handle_{index}", (x + .17, 2.353, .60),
              (x + .17, 2.353, .78), .007)
    for index, name in enumerate(("hood_primary", "hood_middle", "hood_right")):
        q = _Parts(world.find(f"./body[@name='{name}']"), f"{name}__photo_services_")
        # Photo-visible suspended funnels/stopcocks are retained on the rack.
        x = (.25, -.22, .23)[index]
        q.cylinder("glass_column", (x, .245, 1.57), .028, .16,
                   (.65, .78, .78, .3), collision=False)
        q.rod("clamp", (x, .245, 1.59), (x, .30, 1.59), .011)
        q.rod("clamp_riser", (x, .30, 1.36), (x, .30, 1.59), .006)
        q.cylinder("stopcock", (x, .245, 1.385), .012, .025, WHITE)
        q.rod("stopcock_handle", (x - .025, .245, 1.39),
              (x + .025, .245, 1.39), .006, BLUE)
        q.lead("utility_lead", [(x, .245, 1.73), (x + .06, .245, 1.82),
               (x + .06, .34, 1.93)], (.35, .40, .30, 1))
        # The close photograph shows two horizontal glass manifolds with
        # individual taps, rather than a single unbranched utility rail.
        for row, z in enumerate((1.72, 1.84)):
            q.rod(f"manifold_{row}", (-.62, .22, z), (.62, .22, z),
                  .014, (.59, .72, .64, .70))
            for side in (-1, 1):
                q.rod(f"manifold_support_{row}_{side}", (side * .60, .22, z),
                      (side * .60, .30, z), .009)
            for tap, tx in enumerate((-.52, -.31, -.10, .11, .32, .53)):
                q.rod(f"tap_body_{row}_{tap}", (tx, .22, z),
                      (tx, .14, z - .025), .008, WHITE)
                q.rod(f"tap_handle_{row}_{tap}", (tx - .017, .14, z - .025),
                      (tx + .017, .14, z - .025), .005,
                      (.58, .08, .06, 1) if row else BLUE)
        # Retained bottle/receiver groups stand on the real deck, behind the
        # sash travel plane. They have no reagent or fluid-response model.
        for bottle, (bx, radius, height) in enumerate(
            ((-.66, .034, .16), (-.56, .028, .11), (.29, .027, .10),
             (.41, .040, .15), (.56, .032, .21))
        ):
            by = -.27 + .02 * (index % 2)
            color = WHITE if bottle % 2 else (.55, .31, .09, .85)
            q.cylinder(f"bottle_{bottle}", (bx, by, .935 + height / 2),
                       radius, height / 2, color)
            q.cylinder(f"bottle_cap_{bottle}", (bx, by, .941 + height),
                       radius * .72, .006, BLUE if bottle % 2 else WHITE)


def _chicago(world):
    p = _Parts(world, "chicago_source_")
    for index, x in enumerate((-2.50, -1.78)):
        p.box(f"rack_handle_{index}", (x + .225, 1.124, 1.26), (.009, .012, .11), STEEL)
        for z in (.30, 2.18):
            p.box(f"rack_hinge_{index}_{z}", (x - .284, 1.132, z), (.012, .012, .045), STEEL)
        for row in range(10):
            p.box(f"rack_vent_{index}_{row}", (x, 1.128, .55 + row * .033),
                  (.21, .001, .004), (.22, .24, .24, 1), collision=False)
    # Source-visible wall service boards and the duct above the black cabinets.
    p.box("wall_distribution_box", (-2.75, 2.71, 2.10), (.22, .075, .25), WHITE)
    for x in (-2.88, -2.66):
        p.rod(f"wall_conduit_{x}", (x, 2.67, 2.35), (x, 2.67, 3.06), .017)
    p.box("upper_duct", (-2.24, 2.16, 2.88), (.57, .54, .12), STEEL)
    for y in (1.68, 2.62):
        p.box(f"duct_seam_{y}", (-2.24, y, 2.88), (.58, .012, .13), STEEL)
        p.rod(f"duct_hanger_{y}", (-2.24, y, 3.00), (-2.24, y, 3.20), .008)


def _cmu(world):
    p = _Parts(world, "cmu_photo_")
    # Two rows of varied housings on the photographed utility shelving.
    for x in (-1.20, 1.20):
        p.box(f"shelf_extension_{x}", (x, 1.55, 2.18), (.025, .028, .24), WHITE)
    p.box("upper_shelf", (0, 1.40, 2.39), (1.36, .20, .025), WHITE)
    for index, (x, w, h, color) in enumerate(((-.96, .13, .18, WHITE),
            (-.57, .16, .12, BLUE), (-.10, .20, .22, WOOD), (.47, .16, .14, WHITE),
            (.94, .13, .17, DARK))):
        p.box(f"stored_housing_{index}", (x, 1.41, 2.414 + h / 2), (w, .15, h / 2), color)
        p.box(f"housing_label_{index}", (x, 1.259, 2.414 + h / 2),
              (w * .57, .001, h * .22), WHITE, collision=False)
    for index, (x, w, h) in enumerate(((-.94, .15, .12), (-.46, .12, .19), (.08, .22, .10), (.77, .19, .13))):
        p.box(f"shelf_instrument_{index}", (x, 1.40, 1.99 + h / 2), (w, .15, h / 2), WHITE)
        p.box(f"shelf_front_{index}", (x, 1.248, 1.99 + h / 2), (w * .82, .002, h * .30), DARK, collision=False)
    # The corner photograph shows a second open work run behind the sink island.
    _desk(p, "side_workbench", (-2.56, 1.72), (.43, .85), .90)
    p.box("side_instrument", (-2.56, 2.02, 1.06), (.28, .25, .16), WHITE)
    p.box("instrument_front", (-2.56, 1.764, 1.08), (.22, .006, .10), DARK)
    for x in (-2.76, -2.43):
        p.cylinder(f"small_vessel_{x}", (x, 1.20, .972), .043, .072, WHITE)
        p.cylinder(f"vessel_cap_{x}", (x, 1.20, 1.05), .037, .007, BLUE)
    # Two blue secondary instruments sit behind the photographed 850BP.
    # The lower case contacts the bench; the upper module contacts its lid.
    for index, (z, half_height, width) in enumerate(((1.145, .245, .28), (1.48, .09, .25))):
        p.box(f"blue_instrument_{index}", (-.84, 1.56, z),
              (width, .18, half_height), BLUE)
        p.box(f"blue_instrument_face_{index}", (-.84, 1.374, z),
              (width - .025, .006, half_height - .025), (.08, .20, .31, 1))
        p.box(f"blue_display_{index}", (-.94, 1.367, z + .035),
              (.075, .001, .022), DARK, collision=False)
        for x in (-.75, -.67):
            p.cylinder(f"blue_knob_{index}_{x}", (x, 1.362, z - .025),
                       .016, .006, STEEL, euler=numbers((math.pi / 2, 0, 0)))
    p.box("under_shelf_power_strip", (0, 1.33, 1.915), (.60, .05, .025), WHITE)
    for x in (-.45, -.15, .15, .45):
        p.box(f"power_socket_{x}", (x, 1.279, 1.915),
              (.023, .001, .014), DARK, collision=False)
    for key, points in (
        ("blue_power", [(-.45, 1.28, 1.90), (-1.18, 1.25, 1.80),
                        (-1.18, 1.32, 1.55), (-1.09, 1.38, 1.55)]),
        ("tester_power", [(.45, 1.28, 1.90), (.73, 1.29, 1.82),
                          (.73, 1.36, 1.70), (.63, 1.34, 1.70)]),
    ):
        p.lead(key, points, radius=.006)


def _jhu(world):
    p = _Parts(world, "jhu_photo_")
    island = world.find("./body[@name='central_prep_island']")
    for geom in island.findall("geom"):
        if geom.get("name", "").endswith("_top"):
            geom.set("rgba", numbers(WOOD))
    # Keep the photographed secondary workbench in its peripheral bay while
    # leaving the full metre of approach inside the left entry unobstructed.
    x, y = -2.52, -1.90
    _desk(p, "rear_workbench", (x, y), (.82, .36), .82)
    _monitor(p, "rear_computer", (x - .24, y + .10, .82))
    p.box("rear_instrument", (x + .52, y, .93), (.20, .23, .11), WHITE)
    p.box("rear_instrument_front", (x + .52, y - .234, .93), (.16, .004, .07), DARK)
    for index, y in enumerate((-.90, -.1, .60)):
        p.box(f"island_tray_{index}", (-.57, y, .913), (.20, .18, .013), WHITE)
        for j in range(3):
            x = -.71 + j * .13
            p.cylinder(f"inert_jar_{index}_{j}", (x, y, .969), .043, .043, WHITE)
            p.cylinder(f"jar_cap_{index}_{j}", (x, y, 1.017), .046, .005, BLUE)
    for x in (-.63, .24):
        p.box(f"underbench_case_{x}", (x, -.55, .27), (.18, .27, .27), WOOD)
    # Replace the old three generic blocks with the identifiable computer,
    # balance and bottle groups in the source panorama. These are fixed props.
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith(("jhu_inert_prep_case_", "jhu_prep_calibration_block_")):
            world.remove(geom)
    _monitor(p, "island_computer", (-.12, -1.65, .90), width=.46)
    p.box("island_keyboard", (-.12, -1.99, .913), (.21, .075, .013), DARK)
    p.box("balance_base", (.10, -.60, .94), (.15, .18, .04), WHITE)
    p.cylinder("balance_pan", (.10, -.57, .988), .085, .008, STEEL)
    p.box("balance_blank_display", (.10, -.783, .943),
          (.063, .003, .016), DARK, collision=False)
    for x in (-.04, .24):
        p.box(f"balance_glazing_{x}", (x, -.57, 1.105),
              (.003, .13, .125), (.75, .86, .87, .20), collision=False)
    p.box("balance_rear_glazing", (.10, -.44, 1.105),
          (.14, .003, .125), (.75, .86, .87, .20), collision=False)
    for index, y in enumerate((.22, .46)):
        p.cylinder(f"wash_bottle_{index}", (.20, y, 1.00), .052, .10, WHITE)
        p.cylinder(f"wash_neck_{index}", (.20, y, 1.12), .020, .020, BLUE)
        p.lead(f"wash_spout_{index}", [(.20, y, 1.14), (.20, y - .03, 1.19),
               (.20, y - .10, 1.19)], WHITE, radius=.006)
    # Larger overhead rectangular services visible in the panorama. These share
    # the roof cutaway policy, so a hidden ceiling never leaves dangling hangers.
    q = _Parts(world, "arch_roof_jhu_photo_")
    q.box("duct", (1.25, .20, 3.48), (.28, 3.55, .16), STEEL)
    for y in (-3.0, -.7, 1.6, 3.55):
        q.box(f"duct_seam_{y}", (1.25, y, 3.48), (.29, .014, .17), STEEL)
        q.rod(f"hanger_{y}", (1.25, y, 3.65), (1.25, y, 3.80), .009)


def _ntu(world):
    p = _Parts(world, "smtc_photo_")
    for geom in world.findall("geom"):
        name = geom.get("name", "")
        if "ntu_service_hanger" in name and not name.startswith("arch_roof_"):
            geom.set("name", "arch_roof_" + name)
    # Floor-connected side posts explain utility-carrier support in every view.
    for y in (-1.4, .9, 2.8):
        for x in (-4.15, 4.15):
            p.box(f"carrier_foot_{x}_{y}", (x, y, .025), (.10, .10, .025), DARK)
            p.box(f"carrier_column_{x}_{y}", (x, y, 1.415), (.035, .035, 1.365), DARK)
            p.rod(f"carrier_knee_{x}_{y}", (x, y, 2.22),
                  (x - math.copysign(.50, x), y, 2.78), .023, DARK, collision=True)
    # The complete stand and instrument clear the industrial entry approach;
    # moving only the tabletop would leave a leg blocking the doorway.
    x, y = -3.32, -1.52
    _desk(p, "aisle_service_stand", (x, y), (.53, .42), .90)
    p.box("instrument_housing", (x, y, 1.11), (.36, .30, .21), WHITE)
    p.box("instrument_front", (x, y - .306, 1.13), (.29, .006, .13), DARK)
    p.box("instrument_side_panel", (x + .365, y, 1.11), (.005, .24, .15), STEEL)
    for z in (1.02, 1.08, 1.14, 1.20):
        p.box(f"instrument_vent_{z}", (x + .372, y, z), (.001, .20, .007), DARK, collision=False)
    # Compact auxiliary piping, visibly fastened to the casting cabinet side.
    for z in (.55, .78, 1.01):
        p.rod(f"casting_service_{z}", (-4.02, -.02, z), (-4.02, .83, z), .013)
        for y in (.0, .78):
            p.box(f"pipe_mount_{z}_{y}", (-4.0, y, z), (.04, .02, .024))
    # The brochure shows parallel white pressure housings on an open frame.
    # This compact static auxiliary bay uses that visible arrangement, with no
    # inferred flow, pressure response or claim about hidden plumbing.
    # The frame sits beside the casting cabinet at the aisle boundary. The
    # full marked 3 m corridor stays clear; exact facility spacing is unknown.
    for x in (-2.45, -1.72):
        for y in (.10, .80):
            p.box(f"auxiliary_foot_{x}_{y}", (x, y, .025), (.06, .06, .025), DARK)
            p.box(f"auxiliary_post_{x}_{y}", (x, y, .85), (.025, .025, .80))
    for z in (1.02, 1.31):
        for y in (.18, .72):
            p.box(f"auxiliary_crossbar_{z}_{y}", (-2.085, y, z - .100),
                  (.365, .025, .015))
        for x in (-2.26, -1.91):
            p.cylinder(f"pressure_housing_{x}_{z}", (x, .45, z), .085, .325,
                       WHITE, euler=numbers((math.pi / 2, 0, 0)))
            p.cylinder(f"housing_end_{x}_{z}", (x, .119, z), .090, .006,
                       STEEL, euler=numbers((math.pi / 2, 0, 0)))
            p.rod(f"housing_port_{x}_{z}", (x, .113, z), (x, .02, z), .014, WHITE)
            p.lead(f"housing_route_{x}_{z}", [(x, .02, z), (x, 0, .74),
                   (-2.45, 0, .74), (-2.45, .10, .74)], radius=.009)


def _texas(world):
    p = _Parts(world, "fsel_photo_")
    # Parked mobile work platforms occupy peripheral apparatus bays, leaving
    # the existing central and right-hand circulation lanes unchanged.
    for index, (x, y, height) in enumerate(((6.4, 7.5, 1.20), (-5.4, 7.7, .88))):
        for dx in (-.52, .52):
            for dy in (-.38, .38):
                p.cylinder(f"platform_wheel_{index}_{dx}_{dy}", (x + dx, y + dy, .075),
                           .075, .035, DARK, euler=numbers((math.pi / 2, 0, 0)))
                p.box(f"platform_leg_{index}_{dx}_{dy}", (x + dx, y + dy, (height + .15) / 2),
                      (.025, .025, (height - .15) / 2), BLUE)
        p.box(f"platform_deck_{index}", (x, y, height), (.60, .46, .025), WOOD)
        for dx in (-.57, .57):
            p.rod(f"platform_rail_post_{index}_{dx}", (x + dx, y + .42, height),
                  (x + dx, y + .42, height + .8), .020, BLUE, collision=True)
        p.rod(f"platform_handrail_{index}", (x - .57, y + .42, height + .8),
              (x + .57, y + .42, height + .8), .020, BLUE, collision=True)
    for index, y in enumerate((-7.6, 5.5, 9.3)):
        p.box(f"service_cabinet_{index}", (7.30, y, .53), (.30, .37, .53), STEEL)
        p.box(f"service_front_{index}", (6.994, y, .57), (.006, .31, .39), WHITE)
        p.box(f"service_handle_{index}", (6.976, y - .22, .57), (.012, .012, .10), DARK)
        p.lead(f"service_drop_{index}", [(7.3, y, 1.06), (7.72, y, 1.22),
               (8.74, y, 1.22), (8.74, y, 5.72)], radius=.013)
    from real_labs.structural_hall import add_static_reaction_frame

    add_static_reaction_frame(world, "fsel_photo_rear_left", -2.2, 5.3, 4.30, 1.15)
    add_static_reaction_frame(world, "fsel_photo_rear_right", 4.5, 5.5, 4.65, 1.30)
    # A retained secondary slab and timber falsework are visible in the source;
    # these fixed supports have no load-deformation or material-test model.
    for y in (5.5, 7.5):
        for x in (-2.82, -1.58):
            p.box(f"rear_falsework_leg_{x}_{y}", (x, y, .51), (.065, .09, .51), WOOD)
        p.box(f"rear_falsework_cap_{y}", (-2.2, y, 1.06), (.78, .18, .04), WOOD)
        p.rod(f"rear_falsework_brace_{y}", (-2.82, y - .095, .10),
              (-1.58, y - .095, 1.02), .04, WOOD, collision=True)
    p.box("rear_inert_slab", (-2.2, 6.5, 1.22), (.80, 2.1, .12), WHITE)


_BUILDERS = {
    "brown_breuer_afam_tunnel": _brown,
    "caltech_stoltz_schlinger": _caltech,
    "chicago_mineralphysics_laser_bay": _chicago,
    "cmu_edl_electrochemical_testing": _cmu,
    "jhu_paradim_bulk_crystal_growth": _jhu,
    "ntu_smtc_membranes": _ntu,
    "texas_ferguson_structural_test_hall": _texas,
}


def add_details(root, definition):
    """Extend selected photographed exteriors; all other scenes are untouched."""
    builder = _BUILDERS.get(definition["id"])
    if builder is not None:
        builder(root.find("worldbody"))
