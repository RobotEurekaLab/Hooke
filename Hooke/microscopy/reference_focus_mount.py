"""Original motorized support for the estimated inclined objective carousel.

A rear guide carries a low fork and central spindle. All moving parts belong
to the existing focus joint; the selected objective and collection datums are
unchanged. This assembly is not Nikon or Q-545 CAD and has no rated load model.
"""

import numpy as np
from scipy.spatial import Delaunay

from microscopy.geometry import cylinder, geom, mesh
from microscopy.stand_details import turned_part
from microscopy.turret_geometry import CENTRE_M

ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def _plate(asset, parent, name, outline, holes, depth, position):
    """Closed X/Z mounting plate with real clearance bores, extruded toward Y−."""
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
    """Seat an original rear guide on the stand and carry the turret from below."""
    stand_origin = np.r_[origin[:2], bench_top]
    shoulder = np.asarray(origin)+[0., 0., shoulder_m]
    stand_top = bench_top+.173
    fixed = np.array([shoulder[0]+CENTRE_M[0], shoulder[1]+.099, stand_top+.034])
    moving = shoulder+[CENTRE_M[0], .082, -.026]
    foot_height = .008
    geom(scope, "focus_mount_foot", "box", (.025, .012, foot_height/2),
         [fixed[0]-stand_origin[0], .091, .173+foot_height/2], "graphite", mass="0")
    holes = [(np.array([x, z]), .0017) for x in (-.018, .018) for z in (-.018, .018)]
    _plate(asset, scope, "focus_mount_backplate",
           np.array([[-.025, -.026], [.025, -.026], [.025, .026], [-.025, .026]]),
           holes, .008, fixed-stand_origin)
    geom(scope, "focus_mount_fixed_outline", "box", (.012, .002, .024),
         fixed-stand_origin+[0., -.010, 0.], "metal", mass="0")
    geom(focus, "focus_mount_moving_outline", "box", (.016, .005, .021),
         moving, "metal", mass="0")
    carriage_holes = [(np.array([x, z]), .0011) for x in (-.010, .010) for z in (-.013, .013)]
    _plate(asset, focus, "focus_mount_carrier_web",
           np.array([[-.016, -.021], [.016, -.021], [.016, .021], [-.016, .021]]),
           carriage_holes, .002, moving+[0., -.005, 0.])
    # The fork stays below the lowest inactive socket. A central through bore
    # in the carousel clears the spindle, keeping the active collection tube
    # at X=0 entirely separate from this mechanical support at X=-35 mm.
    geom(focus, "focus_mount_carrier_cap", "box", (.009, .041, .003),
         shoulder+[CENTRE_M[0], .041, -.050], "graphite", mass="0")
    cylinder(focus, "focus_mount_spindle", shoulder+[CENTRE_M[0], 0., -.047],
             shoulder+[CENTRE_M[0], 0., -.002], .006, "metal", mass="0")
    turned_part(asset, focus, "focus_mount_hub", [(-.005, .008), (-.001, .012),
                 (.003, .012), (.005, .009)], shoulder+CENTRE_M, material="graphite")
    for index, (point, _) in enumerate(holes):
        centre = fixed+ROTATION@[*point, 0.]
        cylinder(scope, f"focus_mount_fixed_screw_{index}", centre-stand_origin+[0., -.008, 0.],
                 centre-stand_origin+[0., .001, 0.], .0015, "metal", mass="0")
    for index, (point, _) in enumerate(carriage_holes):
        centre = moving+ROTATION@[*point, 0.]
        cylinder(focus, f"focus_mount_carrier_screw_{index}", centre+[0., -.0078, 0.],
                 centre+[0., -.0048, 0.], .001, "metal", mass="0")
    return dict(profile="original-inclined-carousel-focus-support",
        source="Independent rear guide, low fork and spindle; no manufacturer CAD",
        source_sha256=None, fixed_mount_world_m=fixed.tolist(), moving_mount_zero_world_m=moving.tolist(),
        carriage_zero_offset_m=0., encoder_axis=[0., 0., 1.], software_focus_range_m=[-.001, .001],
        foot_height_m=foot_height, stand_attachment_plane_world_z_m=stand_top,
        nosepiece_attachment_plane_world_z_m=float(shoulder[2]),
        official_cad=False, one_to_one_verified=False, load_capacity_validated=False,
        estimates=["All guide, fork, spindle, bearing and fastener dimensions",
                   "No OEM focus drivetrain, piezo internals or manufacturing fit",
                   "Existing ideal focus servo retained; no measured load or vibration model"],
        scope="Original visible support and verified joint ownership; the optical datum remains unchanged")
