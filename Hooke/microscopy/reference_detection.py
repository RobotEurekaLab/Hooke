"""Original CFI collection relay with distinct mechanical and optical datums.

MXA22018 dimensions inform the envelope. The internal prescription, original
fold layout, enclosure and virtual sensor are not manufacturer reproductions.
"""

import xml.etree.ElementTree as ET

import numpy as np

from microscopy.geometry import annular_tube, body, camera, cylinder, geom, mesh, numbers
from microscopy.reference_camera import C_MOUNT_FLANGE_DISTANCE_M, build_reference_camera

SOURCE_URL = "https://downloads.microscope.healthcare.nikon.com/phase7/literature/Brochures/OEM_2CE-MUZH-9_12602T_82P.pdf"
TUBE_FOCAL_LENGTH_M = .200
IMAGE_DATUM_DISTANCE_M = .148
UNIT_BODY_LENGTH_M = .028
VIRTUAL_SENSOR_WIDTH_M = .0064
FOLD_HEIGHT_M = .095
CAMERA_FLANGE_X_M = -.170
SIDE_PORT_RADIUS_M = .016


def _site(parent, name, position):
    ET.SubElement(parent, "site", name=name, pos=numbers(position), size=".00001", rgba="0 0 0 0")


def _port_seat(asset, scope, rings):
    """Match a hollow original collar to the actual sloping left wall."""
    rings = np.asarray(rings)
    angles = np.arange(64)*2*np.pi/64
    vertices, faces = [], []
    for radius, seat in ((.020, True), (.020, False), (.014, True), (.014, False)):
        yz = np.c_[radius*np.cos(angles), FOLD_HEIGHT_M+radius*np.sin(angles)]
        x = -np.interp(yz[:, 1], rings[:, 0], rings[:, 1]) if seat else np.full(64, -.126)
        vertices.extend(np.c_[x, yz].tolist())
    for start, end, reverse in ((0, 64, False), (128, 192, True), (64, 192, False), (128, 0, False)):
        for i in range(64):
            j = (i+1) % 64
            triangles = ((start+i, start+j, end+j), (start+i, end+j, end+i))
            faces.extend(triangle[::-1] if reverse else triangle for triangle in triangles)
    # Axial travel is towards X−, opposite to the YZ outline's +X normal.
    faces = [triangle[::-1] for triangle in faces]
    mesh(asset, "collection_port_seat_mesh", vertices, faces)
    ET.SubElement(scope, "geom", name="collection_port_seat", type="mesh", mesh="collection_port_seat_mesh",
                  material="graphite", mass="0", contype="0", conaffinity="0")


def build_detection_path(asset, scope, focus, origin, bench_top, shoulder_offset_m, rings):
    """Fit the fixed relay and a moving collection tube to the focus assembly."""
    fold = np.array([0., 0., FOLD_HEIGHT_M])
    sensor_x = CAMERA_FLANGE_X_M-C_MOUNT_FLANGE_DISTANCE_M
    datum_x = sensor_x+IMAGE_DATUM_DISTANCE_M
    unit = body(scope, "collection_tube_unit", [datum_x, 0, FOLD_HEIGHT_M],
                quat=".707106781 0 .707106781 0")
    ET.SubElement(unit, "inertial", pos="0 0 0", mass=".001", diaginertia=".0000001 .0000001 .0000001")
    annular_tube(asset, unit, "collection_unit_body", 0, UNIT_BODY_LENGTH_M-.005, .018, .0145)
    annular_tube(asset, unit, "collection_unit_thread_envelope", UNIT_BODY_LENGTH_M-.005, UNIT_BODY_LENGTH_M, .019, .0145, "metal")
    annular_tube(asset, unit, "collection_unit_output_barrel", -.012, 0, .0165, .0127, "metal")
    # This display envelope is intentionally not a guessed single-lens prescription.
    geom(unit, "collection_tube_lens_envelope", "cylinder", (.01425, .004),
         (0, 0, UNIT_BODY_LENGTH_M/2), "lens", mass="0")
    annular_tube(asset, unit, "collection_camera_tube", CAMERA_FLANGE_X_M-datum_x, -.012, .014, .0127)
    _site(unit, "collection_tube_image_datum", [0, 0, 0])
    _site(unit, "collection_tube_body_entry", [0, 0, UNIT_BODY_LENGTH_M])
    _port_seat(asset, scope, rings)
    camera_evidence = build_reference_camera(asset, scope, [CAMERA_FLANGE_X_M, 0, FOLD_HEIGHT_M],
                                             face_positive_x=True, nominal_sensor_width_m=VIRTUAL_SENSOR_WIDTH_M)
    normal = np.array([1., 0., -1.])/2**.5
    cylinder(scope, "collection_fold_mirror", fold-normal*.00025, fold+normal*.00025, .0127, "metal", mass="0")
    mirror_rim = body(scope, "collection_mirror_rim", fold, quat=".382683432 0 .923879533 0")
    ET.SubElement(mirror_rim, "inertial", pos="0 0 0", mass=".001", diaginertia=".0000001 .0000001 .0000001")
    annular_tube(asset, mirror_rim, "collection_mirror_retainer", -.0005, .0005, .014, .0127, "graphite")
    # Original side support stays outside the declared 14 mm collection bundle.
    cylinder(scope, "collection_mirror_post", (0, .016, .011), (0, .016, FOLD_HEIGHT_M), .002, "metal", mass="0")
    geom(scope, "collection_mirror_support", "box", (.002, .002, .001),
         (0, .015, FOLD_HEIGHT_M), "graphite", mass="0")
    ET.SubElement(scope, "light", name="collection_inspection_fill", pos=".045 -.040 .145",
                  directional="false", diffuse=".004 .004 .005", specular="0 0 0", castshadow="false", bulbradius=".002")
    shoulder = np.asarray(origin)+[0, 0, shoulder_offset_m]
    annular_tube(asset, focus, "collection_vertical_tube", bench_top+FOLD_HEIGHT_M+.023, shoulder[2], .011, .009,
                 pos=numbers([*origin[:2], 0]))
    _site(focus, "collection_objective_exit", shoulder)
    _site(scope, "collection_fold_axis", fold)
    return dict(profile="original-cfi-collection-reference", official_cad=False, one_to_one_verified=False,
        source_url=SOURCE_URL, source_pdf_page=15, source_printed_pages="28–29", tube_unit_reference="MXA22018; not MXA20696",
        reference_dimensions_m=dict(tube_focal_length=TUBE_FOCAL_LENGTH_M, image_datum_to_image_plane=IMAGE_DATUM_DISTANCE_M,
                                    unit_body_length=UNIT_BODY_LENGTH_M, body_diameter=.036, output_diameter=.033),
        nominal_objective_focal_length_m=TUBE_FOCAL_LENGTH_M/40, nominal_magnification=40,
        virtual_sensor_width_m=VIRTUAL_SENSOR_WIDTH_M, camera=camera_evidence,
        design_estimates=["Internal compound-lens display envelope and clear apertures; no lens prescription or principal planes",
                          "Original 45 degree fold, mirror mount, collection tubes and sloping port seat",
                          "12 mm output extension and original 5 mm thread-envelope split within the 28 mm body; omitted thread teeth",
                          "Virtual square sensor/crop at nominal C-mount image plane; not measured vendor hardware"],
        physical_optics_calibrated=False, fabrication_fit_verified=False,
        fixed_body_inertia="Uncalibrated compilation placeholders on the fixed unit and mirror rim; not vendor mass properties",
        inspection_lighting="Illustrative cavity point light for world-camera inspection; no radiometric or microscope-image calibration",
        scope="Nominal collection geometry and dimensional reference; synthetic microscope image formation remains a separate uncalibrated model")


def add_detection_view(world, origin, bench_top):
    """Look into the real cavity without hiding or modifying its walls."""
    eye = np.asarray(origin)+[.060, -.050, 0.]
    target = np.asarray(origin)+[-.035, 0., 0.]
    eye[2] = bench_top+FOLD_HEIGHT_M+.045
    target[2] = bench_top+FOLD_HEIGHT_M
    camera(world, "detection_path_detail", eye, target, 42)


def installed_dimensions(model, data):
    """Measure datums in the observed joint pose; not optical calibration."""
    point = lambda name: data.site_xpos[model.site(name).id].copy()
    exit, fold = point("collection_objective_exit"), point("collection_fold_axis")
    entry, datum = point("collection_tube_body_entry"), point("collection_tube_image_datum")
    sensor = point("collection_nominal_sensor_plane")
    return dict(profile="original-cfi-collection-reference",
        tube_image_datum_to_nominal_sensor_m=float(np.linalg.norm(datum-sensor)),
        objective_shoulder_to_unit_entry_air_path_m=float(np.linalg.norm(exit-fold)+np.linalg.norm(fold-entry)),
        virtual_sensor_width_m=VIRTUAL_SENSOR_WIDTH_M, nominal_magnification=40,
        nominal_object_field_width_m=VIRTUAL_SENSOR_WIDTH_M/40,
        physical_optics_calibrated=False, actual_vendor_sensor_plane_verified=False,
        measurement="Compiled nominal sites in observed pose; compound optics, image quality, pupil locations and fabrication not calibrated")
