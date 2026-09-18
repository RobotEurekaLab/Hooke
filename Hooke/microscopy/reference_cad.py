"""Offline, independently authored mechanical outlines from published dimensions.

OpenCascade is optional and is imported only when producing a STEP. These
outlines do not include product drawings, trademarks, vendor meshes or claims
of complete commercial device reproduction.
"""

import hashlib
import json
from pathlib import Path


OBJECTIVE_DRAWING = "https://www.microscope.healthcare.nikon.com/images/diagrams/Optics/CFI-Plan-Achromat-Series/cfi_plan_achromat_10x_dl_10x.svg"
FOCUS_DRAWING = "https://www.physikinstrumente.com/en/?downloadFileUid=377&downloadUid=374&type=5600"
M25_MOUNT_SOURCE = "c12e93b70ffcc420bc3fcbbc8e13c743c19e74bddb636b54a3095f2b617790f0"


def write_nominal_m25_mount(original, output):
    """Derive an explicitly thread-free M25 envelope from pinned open hardware.

    The original drawing specifies M25x0.75 but its STEP bore is 22.835 mm.
    Removing that ambiguous thread region prevents intersecting the nominal
    25 mm male barrel. This is a simulation simplification, not a fabrication
    fix or a verified threaded fit. The derivative retains CERN-OHL-P-2.0.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    original, output = Path(original).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(".dimensions.json").exists():
        raise FileExistsError("Derived CAD output must be new")
    if hashlib.sha256(original.read_bytes()).hexdigest() != M25_MOUNT_SOURCE:
        raise ValueError("M25 mount source differs from the inspected open hardware")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(original)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError("Could not read the pinned M25 mount")
    cutter = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 12.5, 8.).Shape()
    shape = BRepAlgoAPI_Cut(reader.OneShape(), cutter).Shape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("Derived M25 nominal mount is not valid CAD")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(output)) != IFSelect_RetDone:
        raise ValueError("Could not export the derived M25 mount")
    evidence = dict(name="openFrame M25 nominal thread-envelope derivative", units="mm",
        original_sha256=M25_MOUNT_SOURCE, design_version=1,
        modification="Removed material inside radius 12.5 mm; original bore radius 11.4175 mm",
        modification_purpose="Nonintersecting nominal threaded envelope for simulation only",
        thread_fit_verified=False, fabrication_validated=False,
        rights="CERN-OHL-P-2.0; retain original source, license, notices and this modification record")
    output.with_suffix(".dimensions.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def write_focus_stage(source):
    """Write separate fixed/moving references with published mounting holes.

    The rear mounting plane is Z=0; the front carriage plane is Z=-15 mm.
    Their 5/10 mm thickness split is an estimate, not a factory section.
    Travel along CAD Y is represented by the runtime joint, not this STEP.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    source = Path(source).resolve()
    if source.exists() or source.with_suffix(".dimensions.json").exists():
        raise FileExistsError("Reference CAD output must be new")
    source.parent.mkdir(parents=True, exist_ok=True)

    def hole(shape, x, y, z, radius, depth):
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1)),
                                       radius, depth).Shape()
        return BRepAlgoAPI_Cut(shape, tool).Shape()

    base = BRepPrimAPI_MakeBox(gp_Pnt(-22.5, -24, -5), 45., 48., 5.).Shape()
    for x in (-18.5, 18.5):
        for y in (-18.5, 18.5):
            base = hole(base, x, y, -5.1, 1.45, 5.2)
            base = hole(base, x, y, -5.1, 2.75, 2.7)
    for x in (-20., 20.):
        base = hole(base, x, 0, -5.1, 1.25, 3.9)
    carriage = BRepPrimAPI_MakeBox(gp_Pnt(-22.5, -24, -15), 45., 48., 10.).Shape()
    for spacing, radius, depth in ((25., 1., 4.), (20., 1., 4.),
                                    (37., 1.25, 5.), (30., 1.25, 4.)):
        for x in (-spacing/2, spacing/2):
            for y in (-spacing/2, spacing/2):
                carriage = hole(carriage, x, y, -15.1, radius, depth+.1)
    for y in (-20., 20.):
        carriage = hole(carriage, 0, y, -15.1, 1.25, 2.6)
    writer = STEPControl_Writer()
    for shape in (base, carriage):
        if not BRepCheck_Analyzer(shape).IsValid():
            raise ValueError("Independent focus reference is not valid CAD")
        writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(source)) != IFSelect_RetDone:
        raise ValueError("Could not export the independent focus STEP")
    evidence = dict(name="Independent Q-545.140 mounting reference", units="mm",
        drawing=FOCUS_DRAWING, manufacturer_cad=False, design_version=1,
        verified_dimensions=dict(body_width=45., body_length=48., body_thickness=15.,
            base_hole_square=37., base_clearance_diameter=2.9,
            base_counterbore_diameter=5.5, base_counterbore_depth=2.6,
            carriage_M2_hole_squares=[25., 20.], carriage_M2_depth=4.,
            carriage_M2_5_hole_squares=[37., 30.], travel=13.),
        estimates=dict(base_thickness=5., carriage_thickness=10.,
            carriage_outline="Same outside envelope as base; internal contour omitted"),
        omitted=["Internal bearing and piezo mechanism", "Helical threads", "Connector and cable", "Logo"],
        part_roles=["fixed_base", "moving_carriage"], travel_axis=[0, 1, 0],
        scope="Published mounting envelope and hole patterns; no precision, mass or whole-device reproduction",
        rights="Independently authored geometry from dimensional facts; source drawing not included")
    source.with_suffix(".dimensions.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def write_objective(source):
    """Write the MRL00102 dimensioned outside outline, with an estimated bore.

    Coordinates are millimetres. Z=0 is the mounting shoulder; the front of
    this objective points towards +Z, as installed on an inverted microscope.
    Helical threads, internal glass and unmarked profile details are omitted.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    source = Path(source).resolve()
    if source.exists() or source.with_suffix(".dimensions.json").exists():
        raise FileExistsError("Reference CAD output must be new")
    source.parent.mkdir(parents=True, exist_ok=True)
    # Published diameters/lengths constrain these sections. Unmarked chamfer
    # positions and the continuous internal bore are recorded estimates.
    sections = ((-5., 0., 12.5, 12.5), (0., 9., 13.5, 13.5),
                (9., 10., 13.5, 12.5), (10., 44., 12.5, 12.5),
                (44., 47., 12.5, 9.5), (47., 49.4, 9.5, 6.))
    shape = None
    for lower, upper, first, last in sections:
        axis = gp_Ax2(gp_Pnt(0, 0, lower), gp_Dir(0, 0, 1))
        maker = (BRepPrimAPI_MakeCylinder(axis, first, upper-lower) if first == last
                 else BRepPrimAPI_MakeCone(axis, first, last, upper-lower))
        shape = maker.Shape() if shape is None else BRepAlgoAPI_Fuse(shape, maker.Shape()).Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -6), gp_Dir(0, 0, 1)), 4., 57.).Shape()
    shape = BRepAlgoAPI_Cut(shape, bore).Shape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("Independent objective outline is not a valid CAD shape")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(source)) != IFSelect_RetDone:
        raise ValueError("Could not export the independent objective STEP")
    evidence = dict(name="Independent MRL00102 mechanical outline reference", units="mm",
        drawing=OBJECTIVE_DRAWING, manufacturer_cad=False,
        verified_dimensions=dict(maximum_diameter=27., lower_barrel_diameter=25.,
            nose_diameter=19., front_diameter=12., upper_barrel_length=10.,
            lower_barrel_length=37., nose_length=2.4, thread_length=5.,
            thread_nominal_diameter=25., thread_pitch=.75, length_excluding_threads=49.4),
        estimates=dict(internal_bore_diameter=8., chamfer_positions="Axial Z=9..10 and 44..47",
                       nose_profile="Linear frustum between published end diameters"),
        omitted=["Helical threads", "Glass elements", "Unmarked rim/groove details", "Logo and engraving"],
        optical_references=dict(NA=.25, magnification=10., working_distance_mm=10.5,
            manufacturer_parfocal_distance_mm=60.06, supplier_parfocal_distance_mm=59.9),
        scope="Published outside dimensions; estimated unmarked surfaces; no exact optical or whole-device reproduction",
        rights="Independently authored geometry from dimensional facts; source drawing not included")
    source.with_suffix(".dimensions.json").write_text(json.dumps(evidence, indent=2))
    return evidence
