"""Finite openFrame illumination support and independently designed condenser.

Mounting axes and engagement are checked against source CAD. The ARM/clamp
thread drawing discrepancies remain explicit; the connector is an original
thread-free visual reference, not proof of a fabricated threaded assembly.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.cad_optics import _cylinder, _metrology, _plane
from microscopy.geometry import cylinder
from microscopy.illumination_design import layout


def support_pose(cad, pillar):
    clamp = _metrology(cad, "OF-AD-TI-PILLAR-TI-ARM")
    arm = _metrology(cad, "OF-TI-ARM")
    led_clamp = _metrology(cad, "OF-AD-TI-ARM-LED-CAIRN")
    pole_bore = _cylinder(clamp, 12.525, [0, 0, 1], [10, 0, 0])
    arm_bore = _cylinder(clamp, 8.025, [0, 1, 0], [-15.0614290239508, 0, 10])
    shaft = _cylinder(arm, 8., [0, 0, 1], [0, 0, 0])
    sleeve_bore = _cylinder(led_clamp, 15.025, [0, 0, 1], [0, 0, 0])
    _cylinder(arm, 1.65, [0, 0, 1], [0, 0, 0])
    _cylinder(led_clamp, 2.067, [1, 0, 0], [0, 0, 10])
    _plane(arm, 2, 0, 100)
    _plane(arm, 2, 200, 100)
    _plane(led_clamp, 0, 23, 100)
    delta = (np.asarray(arm_bore['point_mm'])-np.array([10., 0., 0.]))*1e-3
    pole = np.asarray(pillar, dtype=float)
    # Both tangents are possible around the real pillar. This one approaches
    # from southwest rather than sharing the holding needle's X-Z plane.
    theta = np.arctan2(pole[1], pole[0])-np.arccos(-delta[0]/np.linalg.norm(pole[:2]))
    rotation = Rotation.from_euler('z', theta).as_matrix()
    position = pole+[0, 0, .210]-rotation@[.010, 0, 0]
    bore_point = position+rotation@np.asarray(arm_bore['point_mm'])*1e-3
    direction = rotation@np.asarray(arm_bore['axis'])
    direction *= np.sign(np.dot(direction[:2], bore_point[:2]))
    error = np.linalg.norm(bore_point[:2]-direction[:2]*np.dot(direction[:2], bore_point[:2]))
    if error > 1e-6:
        raise ValueError("Illumination support axis misses the optical centre")
    vertical = np.array([0., 0., 1.])
    led_rotation = np.column_stack((direction, np.cross(vertical, direction), vertical))
    led_position = np.array([0., 0., bore_point[2]-.010])
    side_face = led_position+led_rotation@[.023, 0, .010]
    arm_front = side_face+direction*.008
    arm_rotation = np.column_stack((vertical, np.cross(direction, vertical), direction))
    endpoints = [position+rotation@np.array([arm_bore['point_mm'][0], y, 10.])*1e-3
                 for y in np.array(arm_bore['bounds_mm'])[[1, 4]]]
    distances = sorted(float(np.dot(point-arm_front, direction)) for point in endpoints)
    length = (shaft['bounds_mm'][5]-shaft['bounds_mm'][2])*1e-3
    if distances[0] < 0 or distances[1] > length:
        raise ValueError("Illumination rod does not span its finite clamping bore")
    pole_span = np.array(pole_bore['bounds_mm'])[[2, 5]]*1e-3+.210
    if pole_span[0] < 0 or pole_span[1] > .250:
        raise ValueError("Illumination clamp is outside its 250 mm pillar")
    return dict(clamp_position=position, clamp_rotation=rotation,
        arm_position=arm_front, arm_rotation=arm_rotation,
        led_position=led_position, led_rotation=led_rotation,
        side_face=side_face, direction=direction,
        evidence=dict(axis_alignment_error_m=float(error),
            rod_bore_engagement_m=distances[1]-distances[0],
            rod_front_to_clamp_span_m=distances, rod_length_m=float(length),
            rod_bore_radial_clearance_m=(arm_bore['radius_mm']-shaft['radius_mm'])*1e-3,
            pillar_bore_radial_clearance_m=(pole_bore['radius_mm']-12.5)*1e-3,
            led_sleeve_radial_clearance_m=(sleeve_bore['radius_mm']-15)*1e-3,
            original_connector_length_m=.008,
            threaded_fit_verified=False,
            discrepancies=["ARM STEP has 3.3 mm bore but fabrication PDF labels M5",
                "LED clamp STEP has 4.134 mm nominal minor bore but PDF labels M4"],
            scope="Finite shaft and mounting-bore engagement verified; connector threads/torque and fabrication unverified"))


def mount_illuminator(cad, asset, scope, support, glass_height_m):
    name = "independent-critical-illuminator-v2"
    report = cad.read(name)
    reference = report.get('dimensional_reference', {})
    separation = (support['led_position'][2]+.010-glass_height_m)*1e3
    expected = layout(float(separation))
    design = reference.get('design', {})
    if (reference.get('manufacturer_cad') is not False or design.keys() != expected.keys()
            or any(not np.isclose(design[key], value, atol=1e-8, rtol=0)
                   if isinstance(value, (int, float)) else design[key] != value
                   for key, value in expected.items())):
        raise ValueError("Illuminator CAD does not match its original optical/mechanical design")
    metrology = cad.interfaces(name)
    _cylinder(metrology, 15., [0, 0, 1], [0, 0, 0])
    _plane(metrology, 2, expected['lens_flat_plane_z'], 300)
    position = support['led_position']+[0, 0, .010]
    parts = []
    for envelope in reference['part_envelopes']:
        matches = [i for i, p in enumerate(report['parts']) if p['watertight'] and np.allclose(
            np.asarray(p['cad_bounds_mm']).ravel(), envelope['cad_bounds_mm'], atol=1e-5, rtol=0)]
        if len(matches) != 1:
            raise ValueError("Illuminator parts differ from their declared CAD envelopes")
        index = matches[0]
        cad.attach(asset, scope, name, index, position, support['led_rotation'], envelope['material'])
        parts.append(dict(role=envelope['role'], part=index, position_m=position.tolist(),
                          rotation=support['led_rotation'].tolist()))
    if len(parts) != len(report['parts']) or len({p['part'] for p in parts}) != len(parts):
        raise ValueError("Illuminator contains missing or undeclared solids")
    # Visible coupling sleeve; hidden threads deliberately not inferred from
    # the contradictory source drawing/CAD drill diameters.
    cylinder(scope, 'illumination_arm_connector', support['side_face'],
             support['arm_position'], .004, 'metal', mass='0')
    evidence = dict(support['evidence'], profile='assembled',
        illuminator_component=name, source_sha256=report['sha256'],
        position_m=position.tolist(), rotation=support['led_rotation'].tolist(),
        original_design=reference, parts=parts,
        glass_to_arm_m=float(separation)*1e-3,
        condenser_front_to_glass_m=expected['lens_to_sample']*1e-3,
        microscope_image_illumination_calibrated=False,
        scope="Original illuminator CAD and real mounting chain; synthetic image illumination, optical aberrations and radiometry not calibrated")
    return evidence
