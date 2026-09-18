"""Metrology-based openFrame optical mounts and independent peripherals.

The open hardware mounts use unscaled source CAD. Camera dimensions are an
independent mechanical reference, not vendor CAD or an optical calibration.
No proprietary camera geometry or product drawing is embedded here.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import body, cylinder, geom, mesh, numbers, rounded_box
from microscopy.geometry import annular_tube as _ring
from microscopy.reference_camera import build_reference_camera as _reference_camera

OPENFRAME_COMMIT = "19c312931f4fdcbfaf3725fbef3db075a32413c6"
SOURCES = {
    "OF-LL-CORE": "6688e1f0dde5761c3649a2894219f4feae0b25e8691b7fdbc3404a164a38529d",
    "OF-CL-SM2": "a16132df75f3b0ee511053537ccc6ef158bfa96294fc832a11918d1c24debb08",
    "OF-AD-SBSM2-CM": "33f22f3b0cad1d46f2b2442c8860ec8369bb7e21e7c016b5a7a7f2af8d533051",
    "OF-LL-FL-MOT": "7b1a1978f9ec1864a2955ceea650aa53ba23766234f1658017e42c9986839a44",
    "OF-AD-FL-PZ-PI-Q545": "c7fb177a5cb1e33de6047e2ac23e36abd2f9cc9006eb00c2822aea6d4183321c",
    "OF-AD-PZ-PI-Q545-OBH": "7214fe964cea0695e17f9954d91c21562d46e8456c299c5f8c3fab3e16fe5b5b",
    "OF-AD-TI-PILLAR-TI-ARM": "182596a78e040fef6dbad6265afb0d3696edfc4adc130460b115495369257460",
    "OF-TI-ARM": "bc11493e6b880bd3d8596f6ecb3e00a2a01791dc903d26a21787a8cc8c9a9bce",
    "OF-AD-TI-ARM-LED-CAIRN": "690fa3061fb3b9cdb2dfbbc8c217f7e859c3200b86553620cf3330d7a0809cfd",
}


def camera_profile():
    value = os.environ.get("HOOKE_MICROSCOPY_OPTICS", "estimated")
    if value not in ("estimated", "mechanical", "assembled"):
        raise ValueError("HOOKE_MICROSCOPY_OPTICS must be estimated, mechanical or assembled")
    return value


def _metrology(cad, component):
    name = "openframe/"+component
    report = cad.read(name, SOURCES[component])
    if report.get("upstream_commit") != OPENFRAME_COMMIT:
        raise ValueError("Optical mount differs from the inspected openFrame revision")
    return cad.interfaces(name)


def _cylinder(interfaces, radius_mm, axis, centre_mm, *, extent_mm=None):
    axis, centre = np.asarray(axis, dtype=float), np.asarray(centre_mm, dtype=float)
    for item in interfaces["cylinders"]:
        offset = np.asarray(item["point_mm"])-centre
        if (abs(item["radius_mm"]-radius_mm) < 1e-5
                and abs(np.dot(item["axis"], axis)) > .999999
                and np.linalg.norm(offset-axis*np.dot(offset, axis)) < 1e-5):
            bounds = np.asarray(item.get("bounds_mm", []))
            if bounds.shape != (6,) or not np.isfinite(bounds).all():
                raise ValueError("Optical metrology lacks finite mounting face bounds")
            if extent_mm is not None:
                coordinate = int(np.argmax(np.abs(axis)))
                if not np.allclose(bounds[[coordinate, coordinate+3]], extent_mm, atol=1e-5, rtol=0):
                    continue
            return item
    raise ValueError("Expected optical mounting cylinder is absent from the CAD")


def _plane(interfaces, axis, coordinate_mm, min_area_mm2):
    for item in interfaces.get("planes", []):
        bounds = np.asarray(item.get("bounds_mm", []))
        if (abs(item["normal"][axis]) > .999999
                and abs(item["point_mm"][axis]-coordinate_mm) < 1e-5
                and item["area_mm2"] >= min_area_mm2
                and bounds.shape == (6,) and np.isfinite(bounds).all()
                and abs(bounds[axis+3]-bounds[axis]) < 1e-5):
            return float(item["point_mm"][axis])*1e-3
    raise ValueError("Expected finite optical installation plane is absent from the CAD")


def mount_camera(cad, asset, scope):
    """Fit a clamped SM2 tube to CORE's real X+ port, with a C-mount detector."""
    core = _metrology(cad, "OF-LL-CORE")
    clamp = _metrology(cad, "OF-CL-SM2")
    adapter = _metrology(cad, "OF-AD-SBSM2-CM")
    port = _cylinder(core, 27.95, [1, 0, 0], [0, 0, 35], extent_mm=(53.5, 68.))
    # The two ports share an analytic axis. The positive installation plane
    # identifies X+ without mistaking an arbitrary axis origin for a shoulder.
    port_inner = _plane(core, 0, 53.5, 300)
    core_face = _plane(core, 0, 68.5, 1000)
    _plane(clamp, 1, 0, 2000)
    clamp_bore = _cylinder(clamp, 27.95, [0, 1, 0], [0, 0, 0])
    tube = _cylinder(adapter, 27.95, [1, 0, 0], [0, 0, 0])
    c_mount = _cylinder(adapter, 12.7, [1, 0, 0], [0, 0, 0])
    adapter_back = _plane(adapter, 0, 49.65, 300)
    camera_shoulder = _plane(adapter, 0, 4, 2800)
    # CORE begins 82 mm above the microscope's base; its optical axis is Z=35.
    port_z = .082+.035
    # The recessed screw face is not the outside of the cylindrical frame.
    # A clamp mounted directly on it intersects CORE and adjacent layers.
    # Original hollow standoffs bridge that recess, preserving all source CAD.
    outside = round(float(cad.read('openframe/OF-LL-CORE')['cad_bounds_mm'][3]), 6)*1e-3
    spacing = outside-core_face
    if spacing <= 0:
        raise ValueError("Camera standoff cannot bridge the inspected recessed port")
    clamp_position = np.array([outside, 0, port_z])
    clamp_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    adapter_position = np.array([port_inner+adapter_back+spacing, 0, port_z])
    adapter_rotation = np.diag([-1., -1., 1.])
    hole_errors, standoffs = [], []
    for x, z in ((-25, -25), (-25, 25), (25, -25)):
        _cylinder(clamp, 2., [0, 1, 0], [x, 0, z])
        mounted = clamp_position+clamp_rotation@np.array([x, 0, z])*1e-3
        tapped = _cylinder(core, 1.621, [1, 0, 0], [0, mounted[1]*1e3, z+35])
        point = np.asarray(tapped["point_mm"])*1e-3+[0, 0, .082]
        # Axial spacing is intentional; screw-hole alignment is transverse.
        hole_errors.append(float(np.linalg.norm((mounted-point)[1:])))
        name = f'scope_camera_standoff_{x}_{z}'
        _ring(asset, scope, name, 0, spacing, .003, .0021, 'metal')
        node = scope.find(f"geom[@name='{name}']")
        front = np.array([core_face, point[1], point[2]])
        quaternion = Rotation.from_euler('y', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
        node.set('pos', numbers(front)); node.set('quat', numbers(quaternion))
        standoffs.append(dict(position_m=front.tolist(), rotation=Rotation.from_euler('y', 90, degrees=True).as_matrix().tolist(),
                              length_m=spacing, outer_radius_m=.003, inner_radius_m=.0021))
        # Original M4 screw appearance follows the actual clearance/tap axes.
        cylinder(scope, f"scope_camera_mount_screw_{x}_{z}", point-[.006, 0, 0],
                 point+[.0235+spacing, 0, 0], .0019, "metal", mass="0")
    if max(hole_errors) > 1e-6:
        raise ValueError("Camera clamp holes do not align with CORE's mounting pattern")
    transforms = []
    for name, position, rotation in (("OF-CL-SM2", clamp_position, clamp_rotation),
                                      ("OF-AD-SBSM2-CM", adapter_position, adapter_rotation)):
        cad.attach(asset, scope, "openframe/"+name, 0, position, rotation, "graphite")
        transforms.append(dict(component=name, source_sha256=SOURCES[name],
                               position_m=position.tolist(), rotation=rotation.tolist()))
    camera_front = adapter_position+adapter_rotation@[camera_shoulder, 0, 0]
    camera = _reference_camera(asset, scope, camera_front)
    clamp_span = outside-np.array(clamp_bore["bounds_mm"])[[4, 1]]*1e-3
    tube_span = adapter_position[0]-np.array(tube["bounds_mm"])[[3, 0]]*1e-3
    engagement = float(min(clamp_span[1], tube_span[1])-max(clamp_span[0], tube_span[0]))
    if engagement <= 0:
        raise ValueError("Camera adapter does not engage the clamp's finite bore")
    port_span = np.asarray(port['bounds_mm'])[[0, 3]]*1e-3
    adapter_rear = adapter_position[0]-adapter_back
    port_engagement = float(port_span[1]-max(port_span[0], adapter_rear))
    if port_engagement <= 0:
        raise ValueError("Camera adapter does not enter the finite positive frame port")
    bore_diameter_error = 2*abs(clamp_bore["radius_mm"]-tube["radius_mm"])*1e-3
    port_line = np.array([0., 0., port["point_mm"][2]*1e-3+.082])
    port_alignment_error = float(np.linalg.norm((adapter_position-port_line)[1:]))
    if bore_diameter_error > 1e-6 or port_alignment_error > 1e-6:
        raise ValueError("Camera adapter does not align with its nominal mounting bores")
    return dict(profile="mechanical", components=transforms, camera=camera,
        core_port_axis=[1, 0, 0], core_port_height_m=port_z,
        clamp_mount_face_m=core_face, mounting_hole_alignment_error_m=max(hole_errors),
        clamp_mounting_plane_m=outside, original_standoffs=standoffs,
        standoff_length_m=spacing, adapter_frame_engagement_m=port_engagement,
        clamp_tube_engagement_m=engagement,
        c_mount_nominal_diameter_m=2*c_mount["radius_mm"]*1e-3,
        camera_installation_shoulder_m=float(camera_front[0]),
        adapter_port_alignment_error_m=port_alignment_error,
        clamp_bore_nominal_diameter_error_m=bore_diameter_error,
        scope="Nominal axes, planes and three mounting holes verified; no helical threads, fit tolerances or optical calibration")
