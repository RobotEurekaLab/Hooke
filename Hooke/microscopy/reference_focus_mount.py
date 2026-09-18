"""Original motorized nosepiece mount; dimension facts are not OEM fit claims.

The existing focus servo owns the moving carriage. A Q-545 mounting outline
is used as a geometric reference, not as proof it can carry this nosepiece.
Piezo internals, load capacity and the Nikon focus drivetrain are unmodeled.
"""

import numpy as np
from scipy.spatial import Delaunay

from microscopy.geometry import cylinder, geom, mesh
from microscopy.reference_cad import FOCUS_DRAWING

ROTATION = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def _plate(asset, parent, name, outline, holes, depth, position):
    """Extrude a planar outline and open bores, along the mounting normal."""
    theta = np.arange(48)*2*np.pi/48
    circles = [np.asarray(centre)+radius*np.column_stack((np.cos(theta), np.sin(theta)))
               for centre, radius in holes]
    points = np.vstack((outline, *circles))
    triangles = Delaunay(points).simplices
    centres = points[triangles].mean(axis=1)
    for centre, radius in holes:
        triangles = triangles[np.linalg.norm(centres-centre, axis=1) > radius]
        centres = points[triangles].mean(axis=1)
    count = len(points)
    vertices = np.vstack((np.column_stack((points, np.zeros(count))),
                          np.column_stack((points, np.full(count, depth)))))
    faces = [tuple(face[::-1]) for face in triangles]
    faces.extend(tuple(face+count) for face in triangles)
    edges = {}
    for face in triangles:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key in edges:
                del edges[key]
            else:
                edges[key] = (int(a), int(b))
    for a, b in edges.values():
        faces.extend(((a, b, b+count), (a, b+count, a+count)))
    mesh(asset, name+"_mesh", vertices@ROTATION.T, faces)
    geom(parent, name, "mesh", (), position, "graphite", mesh=name+"_mesh", mass="0")


def build_focus_mount(asset, scope, focus, origin, bench_top, shoulder_m, cad=None):
    """Connect the fixed stand top to the joint-owned, seated nosepiece."""
    stand_origin = np.r_[origin[:2], bench_top]
    plate_bottom = np.asarray(origin)+[0., 0., shoulder_m-.005]
    fixed = plate_bottom+[-.0475, 0., -.026]
    moving = fixed.copy()
    stand_top = bench_top+.173
    foot_height = float(fixed[2]-.024-stand_top)
    if foot_height <= 0:
        raise ValueError("Focus mount does not fit above the stand's top face")
    base_holes = [(np.array([x, y]), .00145) for x in (-.0185, .0185)
                  for y in (-.0185, .0185)]
    bottom = stand_top-fixed[2]
    _plate(asset, scope, "focus_mount_backplate",
           np.array([[-.026, bottom], [.026, bottom], [.026, .024], [-.026, .024]]),
           base_holes, .0075, fixed-stand_origin)
    geom(scope, "focus_mount_foot", "box", (.0025, .026, foot_height/2),
         fixed-stand_origin+[.0025, 0., -.024-foot_height/2], "graphite", mass="0")
    source = "Independent unbored mounting-envelope geometry"
    source_sha256 = None
    if cad is not None:
        from microscopy.focus_assembly import _part, _reference
        name = "independent-q545-mounting"
        _reference(cad, name, FOCUS_DRAWING)
        for parent, role, bounds, position in (
                (scope, "fixed", [-22.5, -24, -5, 22.5, 24, 0], fixed-stand_origin),
                (focus, "moving", [-22.5, -24, -15, 22.5, 24, -5], moving)):
            cad.attach(asset, parent, name, _part(cad, name, bounds), position, ROTATION,
                       prefix="focus_mount_"+role)
        source = "Independent Q-545 dimension-reference STEP; original manufacturer CAD unavailable"
        source_sha256 = cad.read(name)["sha256"]
    else:
        geom(scope, "focus_mount_fixed_outline", "box", (.0025, .0225, .024),
             fixed-stand_origin+[.0025, 0., 0.], "metal", mass="0")
        geom(focus, "focus_mount_moving_outline", "box", (.005, .0225, .024),
             moving+[.01, 0., 0.], "metal", mass="0")
    # A bored original angle joins the carriage's 25 mm M2 hole square to
    # the underside of the nosepiece hub. No source STEP is modified.
    carriage_holes = [(np.array([x, y]), .0011) for x in (-.0125, .0125)
                      for y in (-.0125, .0125)]
    front = moving+[.015, 0., 0.]
    _plate(asset, focus, "focus_mount_carrier_web",
           np.array([[-.016, -.016], [.016, -.016], [.016, .024], [-.016, .024]]),
           carriage_holes, .003, front+[.003, 0., 0.])
    geom(focus, "focus_mount_carrier_cap", "box", (.0065, .016, .001),
         plate_bottom+[-.036, 0., -.001], "graphite", mass="0")
    for index, (point, _) in enumerate(base_holes):
        hole = fixed+ROTATION@[*point, 0.]
        cylinder(scope, f"focus_mount_fixed_screw_{index}",
                 hole-stand_origin+[-.009, 0., 0.],
                 hole-stand_origin+[.003, 0., 0.], .00125, "metal", mass="0")
        cylinder(scope, f"focus_mount_fixed_head_{index}",
                 hole-stand_origin+[.003, 0., 0.],
                 hole-stand_origin+[.0049, 0., 0.], .0025, "metal", mass="0")
    for index, (point, _) in enumerate(carriage_holes):
        hole = front+ROTATION@[*point, 0.]
        cylinder(focus, f"focus_mount_carrier_screw_{index}",
                 hole+[-.0035, 0., 0.], hole+[.003, 0., 0.], .001, "metal", mass="0")
        cylinder(focus, f"focus_mount_carrier_head_{index}",
                 hole+[.003, 0., 0.], hole+[.0045, 0., 0.], .0018, "metal", mass="0")
    return dict(profile="original-nosepiece-linear-focus-mount", source=source,
        dimensional_reference_url=FOCUS_DRAWING, source_sha256=source_sha256,
        fixed_mount_world_m=fixed.tolist(), moving_mount_zero_world_m=moving.tolist(),
        carriage_zero_offset_m=0., encoder_axis=[0., 0., 1.],
        software_focus_range_m=[-.001, .001], base_hole_square_m=.037,
        carriage_hole_square_m=.025, foot_height_m=foot_height,
        stand_attachment_plane_world_z_m=stand_top,
        nosepiece_attachment_plane_world_z_m=float(plate_bottom[2]),
        official_cad=False, one_to_one_verified=False, load_capacity_validated=False,
        estimates=["Original stand backplate, foot, nosepiece carrier and fasteners",
                   "Q-545 5/10 mm shell split and carriage contour are independently estimated",
                   "No Nikon coarse/fine drivetrain, piezo internals, load or manufacturing validation"],
        scope="Visible nominal mechanical chain with fixed/moving joint ownership; geometric and runtime verification required")
