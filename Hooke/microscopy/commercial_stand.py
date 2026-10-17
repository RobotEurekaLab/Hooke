"""Independent TE2000-S appearance reference, not manufacturer CAD.

The cited dimensional diagram supplies stand width/depth, height and eyepoint.
Unmarked surfaces, accessory details and the retained sample-plane height are
estimates. No proprietary mesh, photo texture or vendor logo is embedded.
"""

import numpy as np

from microscopy.geometry import body, cylinder, geom, housing, mesh, rounded_box
from microscopy.stand_details import binocular_head, focus_knob, panel_screw, side_shell, turned_part

SOURCE_URL = "https://www.nikonusa.com/fileuploads/pdfs/TE2000_brochure.pdf"
BASE_WIDTH_M = .195
BASE_DEPTH_M = .4764
HEIGHT_M = .611
EYEPOINT_M = .4494
EYEPIECE_INCLINATION_DEG = 25.
EYEPIECE_SOURCE_URL = ("https://www.microscope.healthcare.nikon.com/about/news/"
                      "nikons-eclipse-te2000-inverted-research-microscope-sets-a-new-standard-for-live-cell-microscopy")
# The drawing's 135.4 mm dimension ends at the stage front, not the optical
# axis. Axis placement is digitised from the side drawing, not an OEM spec.
STAGE_FRONT_FROM_BASE_FRONT_M = .1354
OPTICAL_AXIS_FROM_FRONT_M = .2382
OBSERVER_FRAME_SHIFT_M = .1354-OPTICAL_AXIS_FROM_FRONT_M


def build_stand(asset, world, origin, bench_top, *, stage_supports=True, collection_reference=False):
    scope = body(world, "microscope", [*origin[:2], bench_top])
    centre_y = BASE_DEPTH_M/2-OPTICAL_AXIS_FROM_FRONT_M
    # The long side panel remains planar around the reference collection bore.
    # Shorter rings roll the lower edge and cast upper shoulder into that panel.
    shell_profiles = [
        (.008, .0925, .2332, centre_y, .014),
        (.012, .096, .2367, centre_y, .017),
        (.018, BASE_WIDTH_M/2, BASE_DEPTH_M/2, centre_y, .019),
        (.030, BASE_WIDTH_M/2, BASE_DEPTH_M/2, centre_y, .019),
        (.136, .091, .232, centre_y+.003, .019),
        (.148, .089, .230, centre_y+.004, .020),
        (.160, .0855, .226, centre_y+.007, .022),
        (.169, .0815, .221, centre_y+.009, .023),
        (.173, .077, .216, centre_y+.010, .023),
    ]
    if collection_reference:
        from microscopy.detection_shell import hollow_shell_mesh
        from microscopy.reference_detection import SIDE_PORT_RADIUS_M
        vertices, faces = hollow_shell_mesh(shell_profiles, side_bore_radius_m=SIDE_PORT_RADIUS_M)
        mesh(asset, "te_lower_shell_mesh", vertices, faces)
        geom(scope, "te_lower_shell", "mesh", (), mesh="te_lower_shell_mesh")
    else:
        housing(asset, scope, "te_lower_shell", shell_profiles)
    for x in (-.075, .075):
        for y in (-.104, .302):
            geom(scope, f"te_foot_{x}_{y}", "cylinder", (.011, .004),
                 (x, y+OBSERVER_FRAME_SHIFT_M, .004), "rubber")
    # Separate cast intermediate tube and observation elbow follow the visible
    # assembly seams. Their unmarked surface shapes remain original estimates.
    observer = body(scope, "te_observer", (0, OBSERVER_FRAME_SHIFT_M, 0))
    # The brochure's side outline shows a distinct forward elbow above a
    # vertical intermediate tube, rather than a uniformly tapered pedestal.
    housing(asset, observer, "te_intermediate_tube", [
        (.171, .040, .049, -.178-OBSERVER_FRAME_SHIFT_M, .004),
        (.175, .044, .051, -.178-OBSERVER_FRAME_SHIFT_M, .006),
        (.263, .044, .051, -.178-OBSERVER_FRAME_SHIFT_M, .006),
        (.267, .042, .050, -.178-OBSERVER_FRAME_SHIFT_M, .006)])
    housing(asset, observer, "te_tower_joint", [
        (.267, .0415, .0495, -.178-OBSERVER_FRAME_SHIFT_M, .006),
        (.2685, .0415, .0495, -.178-OBSERVER_FRAME_SHIFT_M, .006)], "graphite")
    observer_outline = np.array([
        (-.229, .269), (-.127, .269), (-.127, .289), (-.152, .289),
        (-.152, .304), (-.193, .366), (-.251, .394), (-.283, .329),
        (-.229, .302)])
    observer_outline[:, 0] -= OBSERVER_FRAME_SHIFT_M
    side_shell(asset, observer, "te_observation_tower", observer_outline, .044, bevel=.003)
    binocular_head(asset, observer, (0, -.379-OBSERVER_FRAME_SHIFT_M, EYEPOINT_M),
                   EYEPIECE_INCLINATION_DEG)
    rounded_box(asset, observer, "te_blank_nameplate", (.017, .0007, .0065),
                (0, -.2297-OBSERVER_FRAME_SHIFT_M, .222), .0005, "graphite")
    for x in (-.020, .020):
        panel_screw(observer, f"te_tube_face_screw_{x}",
                    (x, -.2299-OBSERVER_FRAME_SHIFT_M, .184), (0, -1, 0), .0012)
    # TE2000-S has a left output port; the camera accessory is an estimate.
    if not collection_reference:
        cylinder(scope, "te_side_port", (-.086, .007, .095), (-.126, .007, .095), .020, "graphite")
        cylinder(scope, "te_camera_adapter", (-.125, .007, .095), (-.150, .007, .095), .014)
        rounded_box(asset, scope, "te_camera", (.025, .028, .026), (-.178, .007, .095), .002, "graphite")
        for y in np.linspace(-.020, .030, 7):
            geom(scope, f"te_camera_heat_rib_{y}", "box", (.024, .0006, .0007), (-.178, y, .122), "metal")
    for side in (-1, 1):
        centre = np.array([side*.098, -.023, .082])
        turned_part(asset, observer, f"te_focus_bezel_{side}",
                    [(-.009, .026), (0, .030), (.003, .030), (.004, .027)],
                    centre, (side, 0, 0), "metal")
        focus_knob(asset, observer, f"te_coarse_focus_{side}", (side*.110, -.023, .082),
                   (side, 0, 0), .027, .021)
        focus_knob(asset, observer, f"te_fine_focus_{side}", (side*.127, -.023, .082),
                   (side, 0, 0), .015, .013, flutes=32)
        cylinder(observer, f"te_fine_focus_cap_{side}", (side*.1334, -.023, .082),
                 (side*.1340, -.023, .082), .008, "graphite")
    # The S stand has a plain front casting. The conspicuous front output port
    # belongs to the U/E variants, so it must not be copied into this model.
    turned_part(asset, scope, "te_side_cover_seam", [(-.001, .036), (.001, .036)],
                (.093, .007, .095), (1, 0, 0), "graphite")
    turned_part(asset, scope, "te_side_cover", [(0, .0357), (.0015, .035), (.002, .033)],
                (.093, .007, .095), (1, 0, 0), "ivory")
    for side in (-1, 1):
        for y, z in ((-.071, .133), (.077, .132), (.167, .125)):
            panel_screw(scope, f"te_casting_screw_{side}_{y}", (side*.0914, y, z),
                        (side, 0, 0), .0013)
    rounded_box(asset, scope, "te_intensity_panel", (.0007, .015, .027),
                (.096, -.183, .050), .0005, "graphite")
    focus_knob(asset, scope, "te_intensity_control", (.101, -.183, .051),
               (1, 0, 0), .012, .010, flutes=32)
    # Rear transmitted-light pillar, compact 30 W head and condenser linkage.
    housing(asset, scope, "te_illumination_pillar", [
        (.169, .036, .032, .1806, .008), (.204, .036, .032, .1806, .008),
        (.242, .029, .026, .1806, .006), (.558, .029, .026, .1806, .006),
        (.587, .033, .027, .1806, .008)])
    # Rounded shoulder and sloping front reproduce the characteristic compact
    # transmitted-light head; its top retains the documented overall height.
    housing(asset, scope, "te_lamp_arm", [
        (.535, .035, .119, .086, .010), (.539, .042, .122, .086, .011),
        (.556, .044, .121, .086, .012), (.590, .044, .099, .108, .013),
        (.606, .039, .090, .114, .014), (HEIGHT_M, .032, .081, .112, .014)])
    rounded_box(asset, scope, "te_lamp_rear", (.035, .003, .019),
                (0, .2075, .559), .002, "graphite")
    for x in np.linspace(-.027, .027, 12):
        geom(scope, f"te_lamp_vent_{x}", "box", (.0007, .0004, .013),
             (x, .2105, .559), "rubber")
    for side in (-1, 1):
        for y in (.031, .052):
            rounded_box(asset, scope, f"te_filter_window_{side}_{y}",
                        (.0006, .003, .011), (side*.044, y, .565), .0004, "graphite")
        # Filter slider control emerges from a shallow side recess, as on a
        # compact transmitted illuminator; accessory dimensions are estimates.
        cylinder(scope, f"te_filter_control_bezel_{side}", (side*.043, .075, .565),
                 (side*.045, .075, .565), .0075, "metal")
        focus_knob(asset, scope, f"te_filter_control_{side}", (side*.047, .075, .565),
                   (side, 0, 0), .0065, .005, flutes=24)
    # Removable filter slide and locating ring sit flush against the head.
    cylinder(scope, "te_field_mount", (0, 0, .523), (0, 0, .540), .028, "metal")
    focus_knob(asset, scope, "te_field_iris", (0, 0, .519), (0, 0, 1), .034, .013, flutes=40)
    turned_part(asset, scope, "te_condenser_upper_barrel", [
        (.465, .029), (.469, .034), (.506, .034), (.513, .032), (.514, .030)])
    housing(asset, scope, "te_condenser_carriage", [
        (.428, .031, .027, .1426, .006), (.433, .037, .032, .1426, .007),
        (.460, .037, .032, .1426, .007), (.468, .030, .027, .1426, .006)])
    rounded_box(asset, scope, "te_condenser_arm", (.018, .074, .005),
                (0, .080, .444), .003, "graphite")
    turned_part(asset, scope, "te_condenser_turret", [
        (.429, .029), (.432, .039), (.435, .043), (.450, .043), (.454, .036), (.465, .028)])
    turned_part(asset, scope, "te_condenser_barrel", [
        (.390, .023), (.393, .025), (.422, .025), (.426, .026), (.432, .026)])
    for z, radius in ((.393, .0253), (.424, .0263), (.451, .039)):
        cylinder(scope, f"te_condenser_ring_{z}", (0, 0, z), (0, 0, z+.001), radius, "metal")
    cylinder(scope, "te_condenser_lens", (0, 0, .389), (0, 0, .390), .019, "lens")
    focus_knob(asset, scope, "te_condenser_focus", (.050, .1416, .448), (1, 0, 0), .018, .027, flutes=32)
    # Exposed condenser rack and clamp connect the optical head to its upright
    # carriage; the teeth and hardware are appearance estimates, not a drive.
    rounded_box(asset, scope, "te_condenser_guide", (.010, .001, .104),
                (0, .1538, .426), .0006, "metal")
    for index, z in enumerate(np.linspace(.335, .517, 33)):
        geom(scope, f"te_condenser_rack_tooth_{index}", "box", (.008, .00035, .0003),
             (0, .1525, z), "graphite", mass="0")
    cylinder(scope, "te_condenser_stop_shaft", (-.023, .142, .474),
             (-.053, .142, .474), .0035, "metal", mass="0")
    focus_knob(asset, scope, "te_condenser_stop", (-.054, .142, .474),
               (-1, 0, 0), .007, .008, flutes=24)
    cylinder(scope, "te_condenser_aperture_lever", (.015, -.034, .442),
             (.023, -.057, .442), .0012, "metal", mass="0")
    geom(scope, "te_condenser_aperture_grip", "sphere", (.003,),
         (.023, -.057, .442), "graphite", mass="0")
    for side in (-1, 1):
        for z in (.207, .547):
            panel_screw(scope, f"te_pillar_screw_{side}_{z}",
                        (side*.0293, .179, z), (side, 0, 0), .0014)
    for x in (-.030, .030):
        cylinder(scope, f"te_condenser_centre_screw_{x}", (x, -.025, .443),
                 (x*1.44, -.033, .443), .004, "metal")
    if stage_supports:
        for x in (-.078, .078):
            cylinder(scope, f"te_stage_leg_{x}", (x, .024, .173), (x, .024, .276), .012, "graphite")
    sample_height = float(origin[2]-bench_top)
    evidence = dict(name="TE2000-S independent dimensional/photo reference", stand_profile="te2000-s-reference",
        source_url=SOURCE_URL, source_page=24, official_cad=False, one_to_one_verified=False,
        verified_reference_dimensions_m=dict(stand_base_width=BASE_WIDTH_M, stand_base_depth=BASE_DEPTH_M,
                                            configured_height=HEIGHT_M, eyepoint=EYEPOINT_M),
        eyepiece_inclination_degrees=EYEPIECE_INCLINATION_DEG,
        eyepiece_inclination_source=EYEPIECE_SOURCE_URL,
        sample_plane_height_m=sample_height, optical_profile="estimated",
        axis_from_base_front_estimate_m=OPTICAL_AXIS_FROM_FRONT_M,
        stage_front_from_base_front_reference_m=STAGE_FRONT_FROM_BASE_FRONT_M,
        axis_estimate_basis="Digitised side drawing; 135.4 mm labels the stage front, not the optical axis",
        generic_stage_supports=stage_supports,
        collection_reference=collection_reference,
        appearance_revision="oem-outline-reference-v4",
        visual_detail_scope="Original cast assembly sections, split-prism binocular housing, diopter controls and condenser carriage",
        shell_profiles_m=shell_profiles,
        estimates=["Unmarked shell curves, accessory shapes and all mounting interfaces",
                   "Retained sample plane and generic electric XY stage, not Nikon/ProScan CAD",
                   "Camera, condenser working distance and electric focus mechanism"],
        scope="Independent appearance reference; no calibrated optical path, vendor mass properties or fabrication fit.")
    if collection_reference:
        evidence["collection_housing"] = dict(authoring="Original cavity and bores; not Nikon internal CAD",
            wall_coordinate_inset_m=.003, roof_bore_radius_m=.012, side_bore_radius_m=SIDE_PORT_RADIUS_M,
            manufacturing_fit_verified=False, mass_properties_calibrated=False)
    return scope, evidence
