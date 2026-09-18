"""Independent TE2000-S appearance reference, not manufacturer CAD.

The cited dimensional diagram supplies stand width/depth, height and eyepoint.
Unmarked surfaces, accessory details and the retained sample-plane height are
estimates. No proprietary mesh, photo texture or vendor logo is embedded.
"""

import numpy as np

from microscopy.geometry import body, cylinder, geom, housing, knurled_knob, mesh, rounded_box

SOURCE_URL = "https://www.nikonusa.com/fileuploads/pdfs/TE2000_brochure.pdf"
BASE_WIDTH_M = .195
BASE_DEPTH_M = .4764
HEIGHT_M = .611
EYEPOINT_M = .4494
# The drawing's 135.4 mm dimension ends at the stage front, not the optical
# axis. Axis placement is digitised from the side drawing, not an OEM spec.
STAGE_FRONT_FROM_BASE_FRONT_M = .1354
OPTICAL_AXIS_FROM_FRONT_M = .2382
OBSERVER_FRAME_SHIFT_M = .1354-OPTICAL_AXIS_FROM_FRONT_M


def build_stand(asset, world, origin, bench_top, *, stage_supports=True, collection_reference=False):
    scope = body(world, "microscope", [*origin[:2], bench_top])
    centre_y = BASE_DEPTH_M/2-OPTICAL_AXIS_FROM_FRONT_M
    shell_profiles = [
        (.008, BASE_WIDTH_M/2, BASE_DEPTH_M/2, centre_y, .009),
        (.025, BASE_WIDTH_M/2, BASE_DEPTH_M/2, centre_y, .010),
        (.145, .089, .230, centre_y+.003, .011),
        (.173, .083, .221, centre_y+.011, .012),
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
    # Front viewing tower and sloping binocular prism make the stand's silhouette.
    observer = body(scope, "te_observer", (0, OBSERVER_FRAME_SHIFT_M, 0))
    housing(asset, observer, "te_observation_tower", [
        (.145, .041, .035, -.098, .009),
        (.265, .040, .035, -.103, .009),
        (.338, .042, .034, -.119, .009),
        (.362, .045, .037, -.126, .010),
    ])
    rounded_box(asset, observer, "te_binocular_prism", (.048, .041, .025),
                (0, -.148, .370), .008, "graphite", euler=".55 0 0")
    axis = np.array([0., -2**-.5, 2**-.5])
    for x in (-.032, .032):
        end = np.array([x, -.201, EYEPOINT_M])
        cylinder(observer, f"te_eyepiece_barrel_{x}", end-axis*.070, end-axis*.014, .014, "graphite")
        knurled_knob(observer, f"te_diopter_{x}", end-axis*.020, axis, .016, .014, ribs=24)
        cylinder(observer, f"te_eye_cup_{x}", end-axis*.014, end-axis*.0012, .017, "rubber")
        cylinder(observer, f"te_eyepiece_lens_{x}", end-axis*.001, end, .012, "lens")
    geom(observer, "te_tower_joint", "box", (.039, .0005, .0015), (0, -.139, .265), "graphite")
    geom(observer, "te_blank_nameplate", "box", (.021, .0005, .007), (0, -.140, .223), "blue")
    # TE2000-S has a left output port; the camera accessory is an estimate.
    if not collection_reference:
        cylinder(scope, "te_side_port", (-.086, .007, .095), (-.126, .007, .095), .020, "graphite")
        cylinder(scope, "te_camera_adapter", (-.125, .007, .095), (-.150, .007, .095), .014)
        rounded_box(asset, scope, "te_camera", (.025, .028, .026), (-.178, .007, .095), .002, "graphite")
        for y in np.linspace(-.020, .030, 7):
            geom(scope, f"te_camera_heat_rib_{y}", "box", (.024, .0006, .0007), (-.178, y, .122), "metal")
    for side in (-1, 1):
        knurled_knob(observer, f"te_coarse_focus_{side}", (side*.108, -.023, .082),
                     (1, 0, 0), .027, .021, ribs=36)
        knurled_knob(observer, f"te_fine_focus_{side}", (side*.125, -.023, .082),
                     (1, 0, 0), .015, .010, ribs=24)
    cylinder(observer, "te_front_selector", (0, -.130, .126), (0, -.143, .126), .026, "graphite")
    cylinder(observer, "te_front_selector_rim", (0, -.142, .126), (0, -.144, .126), .021, "rubber")
    cylinder(observer, "te_port_selector", (0, -.132, .051), (0, -.145, .051), .009, "graphite")
    # Rear transmitted-light pillar, compact 30 W head and condenser linkage.
    rounded_box(asset, scope, "te_illumination_pillar", (.034, .026, .209),
                (0, .1806, .378), .006)
    rounded_box(asset, scope, "te_lamp_arm", (.044, .136, .026),
                (0, .1028, HEIGHT_M-.026), .008)
    rounded_box(asset, scope, "te_lamp_rear", (.044, .029, .025),
                (0, .2146, HEIGHT_M-.025), .004, "graphite")
    for x in np.linspace(-.037, .037, 18):
        geom(scope, f"te_lamp_vent_{x}", "box", (.0009, .0004, .018),
             (x, .2446, HEIGHT_M-.024), "rubber")
    cylinder(scope, "te_field_iris", (0, 0, .545), (0, 0, .565), .025, "graphite")
    rounded_box(asset, scope, "te_condenser_carriage", (.038, .033, .020),
                (0, .1426, .448), .004)
    rounded_box(asset, scope, "te_condenser_arm", (.030, .070, .008),
                (0, .080, .423), .004)
    cylinder(scope, "te_condenser_turret", (0, 0, .395), (0, 0, .423), .035, "graphite")
    cylinder(scope, "te_condenser_barrel", (0, 0, .338), (0, 0, .395), .020, "graphite")
    for z in (.345, .363, .382):
        cylinder(scope, f"te_condenser_ring_{z}", (0, 0, z), (0, 0, z+.003), .0215)
    cylinder(scope, "te_condenser_lens", (0, 0, .337), (0, 0, .338), .016, "lens")
    knurled_knob(scope, "te_condenser_focus", (.050, .1416, .448), (1, 0, 0), .018, .027, ribs=24)
    if stage_supports:
        for x in (-.078, .078):
            cylinder(scope, f"te_stage_leg_{x}", (x, .024, .173), (x, .024, .276), .012, "graphite")
    sample_height = float(origin[2]-bench_top)
    evidence = dict(name="TE2000-S independent dimensional/photo reference", stand_profile="te2000-s-reference",
        source_url=SOURCE_URL, source_page=24, official_cad=False, one_to_one_verified=False,
        verified_reference_dimensions_m=dict(stand_base_width=BASE_WIDTH_M, stand_base_depth=BASE_DEPTH_M,
                                            configured_height=HEIGHT_M, eyepoint=EYEPOINT_M),
        sample_plane_height_m=sample_height, optical_profile="estimated",
        axis_from_base_front_estimate_m=OPTICAL_AXIS_FROM_FRONT_M,
        stage_front_from_base_front_reference_m=STAGE_FRONT_FROM_BASE_FRONT_M,
        axis_estimate_basis="Digitised side drawing; 135.4 mm labels the stage front, not the optical axis",
        generic_stage_supports=stage_supports,
        collection_reference=collection_reference,
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
