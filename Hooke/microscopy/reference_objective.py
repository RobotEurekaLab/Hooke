"""Independent exterior reconstruction from Nikon's MRH08430 dimension drawing.

The drawing uses 1.2 mm glass and 3.1 mm working distance. Internal bores,
front lens and motorized mounting are original estimates, not OEM CAD or a
calibrated optical prescription. All builders use metres.
"""

import math

import numpy as np
from scipy.spatial import Delaunay

from microscopy.geometry import geom, mesh

COVER_GLASS_M = .0012
WORKING_DISTANCE_M = .0031
SHOULDER_TO_SPECIMEN_M = .06041
BODY_LENGTH_M = SHOULDER_TO_SPECIMEN_M-COVER_GLASS_M-WORKING_DISTANCE_M
DRAWING_URL = "https://www.microscope.healthcare.nikon.com/images/diagrams/Optics/Super-Plan-Fluor-Series/CFI-S-Plan-Fluor-ELWD-40XC_2.svg"
DRAWING_SHA256 = "110dd353cc43f21858cfd4f5df0a43f5cd131bdb5c5b3ea09963d284f48027d8"


def installed_dimensions(model, data):
    """Measure modeled nose-to-glass separation in the actual joint pose."""
    nose = model.geom("reference_40x_nose")
    shape = model.mesh(int(nose.dataid[0]))
    start, count = int(shape.vertadr[0]), int(shape.vertnum[0])
    vertices = (model.mesh_vert[start:start+count]
                @ data.geom_xmat[nose.id].reshape(3, 3).T+data.geom_xpos[nose.id])
    glass = model.geom("sample_glass")
    normal = data.geom_xmat[glass.id].reshape(3, 3)[:, 2]
    bottom = data.geom_xpos[glass.id]-normal*glass.size[2]
    return dict(profile="mrh08430-drawing-reference", nominal_magnification=40,
        nominal_numerical_aperture=.6, glass_thickness_m=2*float(glass.size[2]),
        reference_working_distance_m=WORKING_DISTANCE_M,
        geometric_working_distance_m=float(np.min((bottom-vertices)@normal)),
        measurement="Compiled nose mesh to glass underside plane in observed pose; not optical focus calibration",
        physical_optics_calibrated=False, official_cad=False, one_to_one_verified=False)


def _lathe(asset, parent, name, profile, position, material="metal", *, knurled=False):
    """Revolve a closed meridian; inner profiles describe estimated bores."""
    sides = 144
    theta = np.arange(sides)*2*np.pi/sides
    vertices = []
    for z, radius in profile:
        radii = radius-.00012*(1-np.cos(48*theta)) if knurled and radius > .0169 else radius
        vertices.extend(np.column_stack((radii*np.cos(theta), radii*np.sin(theta),
                                         np.full(sides, z))))
    faces = []
    for ring in range(len(profile)):
        next_ring = (ring+1) % len(profile)
        for i in range(sides):
            j = (i+1) % sides
            a, b, c, d = ring*sides+i, ring*sides+j, next_ring*sides+j, next_ring*sides+i
            faces.extend(((a, b, c), (a, c, d)))
    mesh(asset, name+"_mesh", vertices, faces)
    geom(parent, name, "mesh", (), position, material, mesh=name+"_mesh", mass="0")


def _mount(asset, focus, position):
    """Original flat six-position plate; neither Nikon inclination nor CAD."""
    centre = np.array([-.035, 0.])
    angles = np.arange(6)*np.pi/3
    slots = centre+.035*np.column_stack((np.cos(angles), np.sin(angles)))
    theta = np.arange(192)*2*np.pi/192
    boundary = centre+.055*np.column_stack((np.cos(theta), np.sin(theta)))
    theta = np.arange(96)*2*np.pi/96
    holes = slots[:, None, :]+.0125*np.column_stack((np.cos(theta), np.sin(theta)))
    points = np.vstack((boundary, holes.reshape(-1, 2)))
    triangles = Delaunay(points).simplices
    centroids = points[triangles].mean(axis=1)
    triangles = triangles[np.all(np.linalg.norm(centroids[:, None]-slots, axis=2) > .0125, axis=1)]
    n = len(points)
    vertices = np.vstack((np.column_stack((points, np.full(n, -.005))),
                          np.column_stack((points, np.zeros(n)))))
    faces = [tuple(triangle[::-1]) for triangle in triangles]
    faces.extend(tuple(triangle+n) for triangle in triangles)
    edges = {}
    for triangle in triangles:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key in edges:
                del edges[key]
            else:
                edges[key] = (int(a), int(b))
    for a, b in edges.values():
        faces.extend(((a, b, b+n), (a, b+n, a+n)))
    mesh(asset, "reference_objective_mount_mesh", vertices, faces)
    geom(focus, "nosepiece", "mesh", (), position, "graphite",
         mesh="reference_objective_mount_mesh", mass="0")
    return slots


def build_reference_objective(asset, focus, origin, focal_reference_m):
    position = np.asarray(origin)+[0., 0., focal_reference_m-SHOULDER_TO_SPECIMEN_M]
    slots = _mount(asset, focus, position)
    # Retain the three legacy, unidentified reference objectives in separate
    # slots; their exterior dimensions and optical properties are estimates.
    from microscopy.instrument import detail_objectives
    # The legacy barrel shoulder was at -47 mm. Seat it on this plate's
    # shoulder plane; retaining its previous world height left it floating.
    legacy_origin = np.asarray(origin)+[0., 0., focal_reference_m-SHOULDER_TO_SPECIMEN_M+.047]
    occupied = (1, 2, 5)
    detail_objectives(focus, legacy_origin, placements=[tuple(slots[index]) for index in occupied])
    for index, slot in enumerate(occupied):
        _lathe(asset, focus, f"reference_objective_bushing_{index}",
               [(-.005, .0125), (0., .0125), (0., .0105), (-.005, .0105)],
               position+[*slots[slot], 0.], "bronze")
    # Derived end radii follow the drawing's 7 and 5 degree exterior tapers.
    rear_radius = .016-.0128*math.tan(math.radians(7))
    front_radius = .01625-.0247*math.tan(math.radians(5))
    nose_straight = BODY_LENGTH_M-(.011-.00525)*math.tan(math.radians(30))
    parts = {
        "thread": ([(-.005, .0125), (0., .0125), (0., .008), (-.005, .008)], "bronze"),
        "rear": ([(0., rear_radius), (.0128, .016), (.0195, .01625),
                   (.0195, .008), (0., .008)], "metal"),
        "correction": ([(.0195, .0165), (.020, .017), (.024, .017), (.0245, .0165),
                         (.0245, .008), (.0195, .008)], "metal"),
        "barrel": ([(.0245, .01625), (.0492, front_radius), (.0492, .008), (.0245, .008)], "metal"),
        "lip": ([(.0492, .01375), (.0505, .01375), (.051, .01325),
                  (.051, .008), (.0492, .008)], "metal"),
        "nose": ([(.051, .011), (nose_straight, .011), (BODY_LENGTH_M, .00525),
                   (BODY_LENGTH_M, .0045), (.051, .0045)], "metal"),
        "band": ([(.026, .01613), (.0288, .015885), (.0288, .0157), (.026, .0157)], "graphite"),
    }
    for name, (profile, material) in parts.items():
        _lathe(asset, focus, "reference_40x_"+name, profile, position, material,
               knurled=name == "correction")
    # Display lens only; the internal optical prescription is not available.
    geom(focus, "reference_40x_lens", "cylinder", (.0045, .00005),
         position+[0., 0., BODY_LENGTH_M-.00005], "lens", mass="0")
    return dict(profile="mrh08430-drawing-reference", official_cad=False, one_to_one_verified=False,
                drawing_url=DRAWING_URL, drawing_sha256=DRAWING_SHA256,
                magnification=40, numerical_aperture=.6, geometry_scale=1.,
                cover_glass_m=COVER_GLASS_M, reference_working_distance_m=WORKING_DISTANCE_M,
                shoulder_to_specimen_m=SHOULDER_TO_SPECIMEN_M, body_length_m=BODY_LENGTH_M,
                focal_reference_m=focal_reference_m, maximum_diameter_m=.034,
                mount_slots_m=slots.tolist(), mount_estimated=True,
                legacy_occupied_slots=list(occupied), legacy_shoulder_offset_m=focal_reference_m-SHOULDER_TO_SPECIMEN_M,
                inactive_objectives="Three legacy unidentified reference bodies; dimensions/optics unverified",
                interface="M25 x 0.75; simplified cylindrical thread exterior",
                estimates=["Internal bores and display lens", "Knurl pattern and black band",
                           "Original flat six-hole support; Nikon left inclination and rotary drive not reconstructed",
                           "Original 25-to-21 mm legacy objective sleeves; thread forms omitted",
                           "Focal datum offset into cell phantom; no refractive-index correction"],
                physical_optics_calibrated=False,
                rights="Independent code geometry from dimension facts; source drawing not redistributed")
