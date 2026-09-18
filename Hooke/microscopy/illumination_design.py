"""Independent critical-illumination design; no commercial illuminator replica.

Optical layout is a first-order spherical-lens calculation with an assumed
index. It does not establish aberrations, radiometry or fabricated performance.
OpenCascade is optional and imported only by the offline STEP producer.
"""

import json
import math
from pathlib import Path


def layout(glass_to_arm_mm=155.):
    if not math.isfinite(glass_to_arm_mm) or glass_to_arm_mm <= 0:
        raise ValueError("Glass-to-arm separation must be finite and positive")
    radius, clear_radius, edge, index, source_z = 15., 12.7, 2., 1.5, 25.
    thickness = edge+radius-math.sqrt(radius**2-clear_radius**2)
    focal = radius/(index-1)
    reduced = thickness/index
    source_to_sample = glass_to_arm_mm+source_z
    fixed = source_to_sample-thickness
    discriminant = (fixed-reduced)**2-4*(focal*(fixed+reduced)-fixed*reduced)
    if discriminant <= 0:
        raise ValueError("Illuminator has no real first-order sample conjugate")
    working = (fixed-reduced-math.sqrt(discriminant))/2
    flat_z = working-glass_to_arm_mm
    return dict(design_version=1, units="mm", glass_to_arm=glass_to_arm_mm,
        barrel_diameter=30., lens_diameter=25.4, lens_radius=radius,
        lens_edge_thickness=edge, lens_centre_thickness=thickness,
        assumed_lens_index=index, equivalent_focal_length=focal,
        led_plane_z=source_z, lens_flat_plane_z=flat_z,
        lens_to_sample=working, aperture_radius=working*.25/math.sqrt(1-.25**2),
        target_illumination_NA=.25,
        scope="Original nominal critical illuminator, not Koehler or manufacturer CAD; first-order conjugate only")


def write_illuminator(source, glass_to_arm_mm=155.):
    """Export the hollow mounting sleeve, lens, stop, heatsink and LED parts."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    design = layout(glass_to_arm_mm)
    source = Path(source).resolve()
    if source.exists() or source.with_suffix(".dimensions.json").exists():
        raise FileExistsError("Independent illuminator output must be new")
    source.parent.mkdir(parents=True, exist_ok=True)

    def cylinder(lower, upper, radius):
        return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, lower), gp_Dir(0, 0, 1)),
                                       radius, upper-lower).Shape()

    def ring(lower, upper, outer, inner):
        return BRepAlgoAPI_Cut(cylinder(lower, upper, outer), cylinder(lower-.1, upper+.1, inner)).Shape()

    z, thickness = design['lens_flat_plane_z'], design['lens_centre_thickness']
    barrel = ring(z+1, 40., 15., 13.)
    collar = ring(z+1, z+12, 18., 15.)
    stop = ring(z-2, z, 18., design['aperture_radius'])
    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, z+thickness-15), 15.).Shape()
    lens = BRepAlgoAPI_Common(sphere, cylinder(z, z+thickness+.1, 12.7)).Shape()
    sink = ring(15., 40., 16., 15.)
    for lower in (18., 21., 24., 27., 30., 33., 36.):
        sink = BRepAlgoAPI_Fuse(sink, ring(lower, lower+1, 22., 16.)).Shape()
    cap = BRepAlgoAPI_Fuse(ring(40., 42., 15., 13.), cylinder(42., 44., 15.)).Shape()
    board = cylinder(25., 26.6, 12.5)
    led = BRepPrimAPI_MakeBox(gp_Pnt(-.5, -.5, 24.6), 1., 1., .4).Shape()
    parts = (("barrel", barrel, "ivory"), ("condenser_collar", collar, "graphite"),
             ("aperture_stop", stop, "graphite"), ("lens", lens, "lens"),
             ("heatsink", sink, "metal"), ("cap", cap, "graphite"),
             ("led_board", board, "cyan"), ("led", led, "ivory"))
    writer, envelopes = STEPControl_Writer(), []
    for role, shape, material in parts:
        if not BRepCheck_Analyzer(shape).IsValid():
            raise ValueError(f"Independent illuminator {role} is invalid CAD")
        bounds = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, bounds, False, False)
        envelopes.append(dict(role=role, cad_bounds_mm=list(bounds.Get()), material=material))
        writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(source)) != IFSelect_RetDone:
        raise ValueError("Could not export the independent illuminator STEP")
    evidence = dict(name="Hooke independent critical illuminator", manufacturer_cad=False,
        design=design, part_envelopes=envelopes,
        dimensions="Original design values; not verified against a commercial device",
        constraints="30 mm sleeve for inspected 30.05 mm openFrame clamping bore",
        estimates=["Nominal lens index", "Paraxial layout; aberrations and LED radiometry unresolved",
                   "Thermal/electrical implementation and mass", "Lens seating and machining tolerances"],
        omitted=["Electrical wiring and driver internals", "Helical threads", "Optical coatings", "Brand marks"],
        rights="Independently authored Hooke design; no vendor geometry or drawings embedded")
    source.with_suffix(".dimensions.json").write_text(json.dumps(evidence, indent=2))
    return evidence
