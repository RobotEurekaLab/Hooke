"""Dimensioned inverted objective on a real openFrame focus mounting chain.

The electric slide and objective are independently authored references; the
fixed and moving mounting plates are source CAD. Servo dynamics, threads and
internal optics remain uncalibrated. Optional CAD generation stays offline.
"""

import xml.etree.ElementTree as ET

import numpy as np

from microscopy.cad_optics import _cylinder, _metrology, _plane, _ring
from microscopy.geometry import cylinder, numbers
from microscopy.reference_cad import FOCUS_DRAWING, M25_MOUNT_SOURCE, OBJECTIVE_DRAWING


def _part(cad, name, bounds):
    """Select a STEP solid by its finite envelope, independently of part order."""
    report = cad.read(name)
    matches = [index for index, part in enumerate(report["parts"])
               if part["watertight"] and np.allclose(
                   np.asarray(part["cad_bounds_mm"]).ravel(), bounds, atol=1e-5, rtol=0)]
    if len(matches) != 1:
        raise ValueError(f"Reference CAD {name} differs from its mounting envelope")
    return matches[0]


def _reference(cad, name, drawing):
    report = cad.read(name)
    reference = report.get("dimensional_reference", {})
    if reference.get("manufacturer_cad") is not False or reference.get("drawing") != drawing:
        raise ValueError(f"Reference CAD {name} lacks its dimensional provenance")
    return cad.interfaces(name)


def mount_focus(cad, asset, scope, focus, origin, bench_top, focal_reference_m):
    """Bind objective travel to the focus encoder and its calibrated zero plane."""
    if not np.isfinite(focal_reference_m) or abs(focal_reference_m) > .0055:
        raise ValueError("Optical focus zero is outside the 13 mm hardware travel")
    frame = _metrology(cad, "OF-LL-FL-MOT")
    fixed = _metrology(cad, "OF-AD-FL-PZ-PI-Q545")
    holder = _metrology(cad, "OF-AD-PZ-PI-Q545-OBH")
    stage_name = "independent-q545-mounting"
    objective_name = "independent-mrl00102-outline"
    disk_name = "openframe-m25-nominal"
    stage = _reference(cad, stage_name, FOCUS_DRAWING)
    objective = _reference(cad, objective_name, OBJECTIVE_DRAWING)
    disk_report = cad.read(disk_name)
    derivative = disk_report.get("dimensional_reference", {})
    if (derivative.get("original_sha256") != M25_MOUNT_SOURCE
            or derivative.get("thread_fit_verified") is not False
            or derivative.get("fabrication_validated") is not False):
        raise ValueError("Nominal M25 derivative lacks its source and modification record")
    disk = cad.interfaces(disk_name)
    base_index = _part(cad, stage_name, [-22.5, -24, -5, 22.5, 24, 0])
    moving_index = _part(cad, stage_name, [-22.5, -24, -15, 22.5, 24, -5])
    objective_index = _part(cad, objective_name, [-13.5, -13.5, -5, 13.5, 13.5, 49.4])
    disk_index = _part(cad, disk_name, [-20, -20, 0, 20, 20, 6])
    _plane(frame, 0, -56.5, 100)
    _plane(fixed, 2, 11.5, 500)
    _plane(fixed, 2, 0, 500)
    _plane(stage, 2, 0, 500)
    _plane(stage, 2, -15, 500)
    _plane(holder, 0, -27.5, 100)
    _plane(holder, 2, 2.3, 200)
    _plane(disk, 2, 1.5, 200)
    _plane(disk, 2, 6, 200)
    _plane(objective, 2, 0, 20)
    _cylinder(holder, 15, [0, 0, 1], [2.5, 0, 0])
    _cylinder(holder, 21, [0, 0, 1], [2.5, 0, 0])
    _cylinder(disk, 12.5, [0, 0, 1], [0, 0, 0])
    _cylinder(objective, 12.5, [0, 0, 1], [0, 0, 0])
    rotation = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    fixed_position = np.array([-.045, 0., .2105])
    zero = np.array([0., 0., focal_reference_m])
    holder_position = np.array([-.0025, 0., .215])+zero
    disk_position = np.array([0., 0., .2158])+zero
    # Three explicit interfaces constrain the mounting chain; analytic axis
    # origins alone do not establish a finite mating plane.
    errors = []
    for y, frame_z in ((-15., 41.5), (10., 66.5)):
        _cylinder(fixed, 1.621, [0, 0, 1], [0, y, 11.5])
        _cylinder(frame, 2.4, [1, 0, 0], [0, 0, frame_z])
        mounted = fixed_position+rotation@np.array([0, y, 11.5])*1e-3
        target = np.array([-.0565, 0, .154+frame_z*1e-3])
        errors.append(float(np.linalg.norm(mounted-target)))
        cylinder(scope, f"focus_fixed_screw_{y}", target+[-.0125, 0, 0],
                 target+[.008, 0, 0], .0019, "metal", mass="0")
    for x in (-18.5, 18.5):
        for y in (-18.5, 18.5):
            _cylinder(fixed, 1.0065, [0, 0, 1], [x, y, 0])
            _cylinder(stage, 1.45, [0, 0, 1], [x, y, 0])
    for x in (-12.5, 12.5):
        _cylinder(stage, 1., [0, 0, 1], [x, 12.5, 0])
        _cylinder(holder, 1., [1, 0, 0], [-27.5, -x, 8])
        mounted = fixed_position+zero+rotation@np.array([x, 12.5, -15])*1e-3
        target = holder_position+np.array([-27.5, -x, 8])*1e-3
        errors.append(float(np.linalg.norm(mounted-target)))
    for x, y in ((0., -17.5), (0., 17.5), (-17.5, 0.), (17.5, 0.)):
        _cylinder(disk, 1.4, [0, 0, 1], [x, y, 0])
        _cylinder(holder, 1.0065, [0, 0, 1], [x+2.5, y, 0])
    if max(errors) > 1e-6:
        raise ValueError("Objective focus mounting holes are misaligned")
    microscope_origin = np.r_[origin[:2], bench_top]
    transforms = []

    def attach(parent, name, index, position, orientation=np.eye(3)):
        cad.attach(asset, parent, name, index, position, orientation, "metal")
        transforms.append(dict(component=name, part=index,
            position_m=position.tolist(), rotation=orientation.tolist(),
            parent=parent.get("name"), source_sha256=cad.read(name)["sha256"]))

    attach(scope, "openframe/OF-AD-FL-PZ-PI-Q545", 0, fixed_position, rotation)
    attach(scope, stage_name, base_index, fixed_position, rotation)
    attach(focus, stage_name, moving_index, microscope_origin+fixed_position+zero, rotation)
    attach(focus, "openframe/OF-AD-PZ-PI-Q545-OBH", 0, microscope_origin+holder_position)
    attach(focus, disk_name, disk_index, microscope_origin+disk_position)
    # Original extension: keep published 60.06 mm parfocal height consistent
    # with this workstation's raised sample platform; no factory part claim.
    parfocal = .06006
    shoulder = origin+zero-np.array([0., 0., parfocal])
    disk_top = microscope_origin+disk_position+[0, 0, .006]
    extension = float(shoulder[2]-disk_top[2])
    if extension <= .005:
        raise ValueError("Objective extension cannot accommodate the nominal thread envelope")
    # The lower male sleeve enters the derived disk; the upper female sleeve
    # accepts the objective's 5 mm thread region without mesh penetration.
    centre = disk_top.copy()
    _ring(asset, focus, "objective_spacer_male", centre[2]-.005, centre[2], .0125, .004)
    _ring(asset, focus, "objective_spacer_female", centre[2], shoulder[2], .0135, .0125)
    # Ring meshes are authored about XY=0, then positioned on the optical axis.
    for name in ("objective_spacer_male", "objective_spacer_female"):
        focus.find(f"geom[@name='{name}']").set("pos", numbers([origin[0], origin[1], 0]))
    attach(focus, objective_name, objective_index, shoulder)
    for name, point in (("objective_shoulder", shoulder),
                        ("objective_front", shoulder+[0, 0, .0494]),
                        ("objective_nominal_focal_plane", origin+zero)):
        ET.SubElement(focus, "site", name=name, pos=numbers(point), size=".0001", rgba="0 0 0 0")
    evidence = dict(profile="assembled", components=transforms,
        fixed_to_moving_hole_alignment_error_m=max(errors),
        encoder_axis=[0, 0, 1], hardware_total_travel_m=.013,
        software_focus_range_m=[-.001, .001], calibrated_zero_offset_m=focal_reference_m,
        objective_parfocal_reference_m=parfocal, objective_outline_length_m=.0494,
        front_to_nominal_focal_plane_m=parfocal-.0494,
        supplier_working_distance_m=.0105, working_distance_discrepancy_m=.00016,
        original_spacer_length_m=extension,
        estimates=["Slide 5/10 mm thickness split and outside contour", "Servo mass/dynamics", "Unmarked objective surfaces"],
        derived_adapter=derivative,
        scope="Finite mounting planes and nominal holes verified; no thread fit, optical ray calibration, mass calibration or nanometre precision claim")
    cad.evidence["microscope"]["focus_assembly"] = evidence
    cad.evidence["microscope"]["original_peripherals"] = [
        "Independent dimensioned camera, objective and electric focus reference",
        "Custom parfocal spacer and fine XY stage/risers",
        "Independent critical illuminator and condenser CAD; screws and remaining optical connections estimated",
    ]
    return evidence
