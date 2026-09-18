"""Independent camera enclosure reference; no proprietary vendor CAD."""

import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation

from microscopy.geometry import annular_tube, body, cylinder, geom, mesh, numbers, rounded_box

C_MOUNT_FLANGE_DISTANCE_M = .017526
C_MOUNT_SOURCE_URL = "https://www.edmundoptics.co.uk/knowledge-center/application-notes/imaging/lens-mounts"


def build_reference_camera(asset, scope, front, *, face_positive_x=False, nominal_sensor_width_m=None):
    """Place local Z=0 at the flange, with local +Z pointing towards the lens."""
    quaternion = Rotation.from_euler("y", 90 if face_positive_x else -90, degrees=True).as_quat()[[3, 0, 1, 2]]
    node = body(scope, "scope_camera_reference", front, quat=numbers(quaternion))
    ET.SubElement(node, "inertial", pos="0 0 0", mass=".05", diaginertia=".00001 .00001 .00001")
    barrel_length = .01176
    if nominal_sensor_width_m is None:
        rounded_box(asset, node, "scope_camera", (.0145, .0145, .015),
                    (0, 0, -barrel_length-.015), .0007, "graphite", mass="0")
    else:
        from microscopy.detection_shell import hollow_shell_mesh
        rings = [(z, .0145, .0145, 0., .0007)
                 for z in (-barrel_length-.030, -barrel_length-.015, -barrel_length)]
        vertices, faces = hollow_shell_mesh(rings, wall_m=.0005,
                                           roof_bore_radius_m=.0127, side_bore_radius_m=None)
        mesh(asset, "scope_camera_mesh", vertices, faces)
        geom(node, "scope_camera", "mesh", (), mesh="scope_camera_mesh", material="graphite", mass="0")
    annular_tube(asset, node, "scope_camera_c_mount", -barrel_length, 0, .01395, .0127)
    annular_tube(asset, node, "scope_camera_mount_rim", -.0004, 0, .01395, .0127, "metal")
    rear = -barrel_length-.030
    geom(node, "scope_camera_back_plate", "box", (.014, .014, .0003),
         (0, 0, rear-.0003), "graphite", mass="0")
    geom(node, "scope_camera_usb_shell", "box", (.005, .002, .001),
         (0, -.0075, rear-.0013), "metal", mass="0")
    geom(node, "scope_camera_usb_socket", "box", (.0044, .00145, .0001),
         (0, -.0075, rear-.0024), "rubber", mass="0")
    cylinder(node, "scope_camera_gpio", (.006, .007, rear-.0006),
             (.006, .007, rear-.003), .0025, "metal", mass="0")
    geom(node, "scope_camera_status_led", "box", (.0006, .0006, .0003),
         (-.010, .010, rear-.0007), "led_green", mass="0")
    evidence = dict(name="Independent 29 mm USB3 C-mount camera reference",
        source="Public FLIR FL2-020-R0 dimensional drawing, sheet 1; not vendor CAD",
        verified_body_dimensions_m=[.029, .029, .030], barrel_length_m=barrel_length,
        barrel_reference="11.76 mm is the drawing's IMX183 C-mount variant, not BFS-U3-51S5-C",
        front_installation_plane_m=list(front),
        estimates=["Corner radius", "Outer barrel diameter", "Connectors, LED and rear plate", "Thread geometry omitted"],
        rights="Independently authored reference geometry; vendor STEP and drawing are not embedded",
        dimensional_reproduction_verified=False, sensor_plane_verified=False,
        inertia="Uncalibrated compilation placeholder on a fixed body")
    if nominal_sensor_width_m is not None:
        geom(node, "scope_camera_nominal_sensor", "box", (nominal_sensor_width_m/2, nominal_sensor_width_m/2, .000025),
             (0, 0, -C_MOUNT_FLANGE_DISTANCE_M), "cyan", mass="0")
        ET.SubElement(node, "site", name="collection_nominal_sensor_plane",
                      pos=numbers([0, 0, -C_MOUNT_FLANGE_DISTANCE_M]), size=".00001", rgba="0 0 0 0")
        evidence.update(nominal_flange_distance_m=C_MOUNT_FLANGE_DISTANCE_M,
                        flange_distance_source_url=C_MOUNT_SOURCE_URL,
                        nominal_sensor_width_m=nominal_sensor_width_m,
                        original_enclosure_design=dict(cavity_coordinate_inset_m=.0005, front_aperture_radius_m=.0127),
                        sensor_basis="Original virtual square sensor/crop at the standard C-mount nominal image plane; not a measured vendor sensor")
    return evidence
