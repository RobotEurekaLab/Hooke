"""Original turned instrument parts, expressed in metres in the parent frame."""

import numpy as np

from microscopy.geometry import body, cylinder, geom, housing, mesh, numbers


def binocular_head(asset, parent, eyepoint, inclination_degrees):
    """Original split-prism shell and eyetubes with a common optical axis.

    The eyepoint and inclination are published TE2000 dimensions. The shell
    sections, interpupillary bridge and diopter detailing are photo estimates.
    """
    angle = np.deg2rad(inclination_degrees)
    axis = np.array([0., -np.cos(angle), np.sin(angle)])
    eyepoint = np.asarray(eyepoint)
    quaternion = np.r_[1+axis[2], -axis[1], axis[0], 0.]
    quaternion /= np.linalg.norm(quaternion)
    head = body(parent, "te_binocular_head", eyepoint-axis*.097,
                quat=numbers(quaternion))
    # The prism housing widens towards the two sockets. It is distinct from
    # the two removable eyepieces and has a shallow, closed rear shoulder.
    housing(asset, head, "te_binocular_prism", [
        (-.066, .030, .021, 0., .012), (-.061, .035, .025, 0., .014),
        (-.036, .047, .030, 0., .018), (-.010, .051, .028, 0., .016),
        (.002, .049, .025, 0., .013), (.008, .044, .023, 0., .011)], "graphite")
    cylinder(head, "te_interpupillary_bridge", (-.032, 0, .009),
             (.032, 0, .009), .009, "graphite", mass="0")
    for x in (-.032, .032):
        end = eyepoint+[x, 0., 0.]
        turned_part(asset, parent, f"te_eyepiece_barrel_{x}", [
            (-.105, .0185), (-.101, .021), (-.087, .021),
            (-.078, .019), (-.072, .015), (-.032, .015),
            (-.029, .0157), (-.013, .0157)], end, axis)
        focus_knob(asset, parent, f"te_diopter_{x}", end-axis*.024,
                   axis, .0162, .016, flutes=40)
        turned_part(asset, parent, f"te_eye_cup_{x}", [
            (-.015, .0154), (-.012, .017), (-.006, .018),
            (-.002, .018), (-.001, .0155)], end, axis, "rubber")
        cylinder(parent, f"te_eyepiece_lens_{x}", end-axis*.0005,
                 end, .012, "lens", mass="0")
        cylinder(parent, f"te_ocular_witness_{x}", end-axis*.034,
                 end-axis*.0333, .0151, "metal", mass="0")
        # A few index ticks make the diopter ring legible without baking a
        # manufacturer's label or a fictitious numeric calibration into it.
        across = np.array([1., 0., 0.])
        other = np.cross(axis, across)
        for index, theta in enumerate(np.linspace(-.6, .6, 7)):
            offset = .01525*(np.cos(theta)*other+np.sin(theta)*across)
            start = end-axis*.043+offset
            cylinder(parent, f"te_ocular_index_{x}_{index}", start,
                     start+axis*(.003 if index == 3 else .0017), .00016,
                     "ivory", mass="0")


def panel_screw(parent, name, centre, normal, radius=.0015):
    """Restrained cross-head surface detail; it does not imply a mounting hole."""
    centre, normal = np.asarray(centre), np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    cylinder(parent, name, centre-normal*.0003, centre, radius, "metal", mass="0")
    tangent = np.cross(normal, [0, 0, 1] if abs(normal[2]) < .9 else [1, 0, 0])
    tangent /= np.linalg.norm(tangent)
    for index, direction in enumerate((tangent, np.cross(normal, tangent))):
        cylinder(parent, f"{name}_slot_{index}", centre+normal*.00008-direction*radius*.6,
                 centre+normal*.00008+direction*radius*.6, radius*.10, "graphite", mass="0")


def side_shell(asset, parent, name, outline, half_width, *, bevel=.003, material="ivory"):
    """Extrude a rounded, possibly concave side outline with closed bevelled caps.

    The outline contains Y/Z coordinates. Small quadratic corner transitions
    retain the recognizable elbow of a cast observation tube. Cap triangles
    use ear clipping so a concave elbow is not filled by a convex fan.
    """
    outline = np.asarray(outline, dtype=float)
    area = np.sum(outline[:, 0]*np.roll(outline[:, 1], -1)
                  - outline[:, 1]*np.roll(outline[:, 0], -1))
    if area < 0:
        outline = outline[::-1]
    points = []
    for previous, point, following in zip(np.roll(outline, 1, axis=0), outline,
                                           np.roll(outline, -1, axis=0)):
        before, after = previous-point, following-point
        trim = min(bevel*2, np.linalg.norm(before)*.2, np.linalg.norm(after)*.2)
        start, end = point+before*trim/np.linalg.norm(before), point+after*trim/np.linalg.norm(after)
        for t in np.linspace(0, 1, 5):
            points.append((1-t)**2*start+2*t*(1-t)*point+t*t*end)
    points = np.asarray(points)
    edges = np.roll(points, -1, axis=0)-points
    normal = np.c_[-edges[:, 1], edges[:, 0]]/np.linalg.norm(edges, axis=1)[:, None]
    inward = normal+np.roll(normal, 1, axis=0)
    inward /= np.sum(inward*normal, axis=1)[:, None]
    vertices, faces = [], []
    rings = [(-half_width, bevel), (-half_width+bevel*.293, bevel*.293),
             (-half_width+bevel, 0), (half_width-bevel, 0),
             (half_width-bevel*.293, bevel*.293), (half_width, bevel)]
    for x, inset in rings:
        vertices.extend(np.c_[np.full(len(points), x), points+inward*inset])
    count = len(points)
    for ring in range(len(rings)-1):
        for i in range(count):
            a, b = ring*count+i, ring*count+(i+1) % count
            faces.extend(((a, b, b+count), (a, b+count, a+count)))

    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]

    cap = points+inward*bevel
    remaining = list(range(count))
    while len(remaining) > 2:
        for index, b in enumerate(remaining):
            a, c = remaining[index-1], remaining[(index+1) % len(remaining)]
            triangle = cap[[a, b, c]]
            if cross(triangle[1]-triangle[0], triangle[2]-triangle[1]) <= 1e-14:
                continue
            inside = [p for p in remaining if p not in (a, b, c) and
                      all(cross(triangle[(j+1) % 3]-triangle[j], cap[p]-triangle[j]) >= -1e-14
                          for j in range(3))]
            if inside:
                continue
            offset = (len(rings)-1)*count
            faces.extend(((a, c, b), (offset+a, offset+b, offset+c)))
            remaining.pop(index)
            break
        else:
            raise ValueError("Observation-shell outline could not be triangulated")
    mesh(asset, name+"_mesh", vertices, faces)
    return geom(parent, name, "mesh", (), mesh=name+"_mesh", material=material,
                mass="0", contype="0", conaffinity="0")


def turned_part(asset, parent, name, profile, centre=(0, 0, 0), axis=(0, 0, 1),
                material="graphite", *, flutes=0, flute_depth=0.):
    """Loft axial position/radius pairs into one closed, non-contact visual mesh.

    The lightly fluted middle section replaces separate knurl cylinders. The
    end rings stay circular, giving the machined chamfer an uninterrupted rim.
    This geometry describes an estimated finish, not a manufactured thread.
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    # Author close to the local origin, then pose the geom. Baking a metre-scale
    # translation into millimetre-scale vertices amplifies XML writer rounding.
    # Near-circular inertia axes may rotate on recompilation; 256 radial samples
    # bound the resulting circular polygon support error below 0.008 percent.
    sides = max(256, flutes*8)
    angle = np.arange(sides)*2*np.pi/sides
    vertices, faces = [], []
    for index, (position, radius) in enumerate(profile):
        relief = (flute_depth*(.5+.5*np.cos(flutes*angle))
                  if flutes and 0 < index < len(profile)-1 else 0.)
        radii = np.broadcast_to(radius-relief, (sides,))
        vertices.extend(np.c_[radii*np.cos(angle), radii*np.sin(angle), np.full(sides, position)])
    for ring in range(len(profile)-1):
        for i in range(sides):
            a, b = ring*sides+i, ring*sides+(i+1) % sides
            faces.extend(((a, b, b+sides), (a, b+sides, a+sides)))
    bottom, top = len(vertices), len(vertices)+1
    vertices.extend(((0, 0, profile[0][0]), (0, 0, profile[-1][0])))
    offset = (len(profile)-1)*sides
    for i in range(sides):
        j = (i+1) % sides
        faces.extend(((bottom, j, i), (top, offset+i, offset+j)))
    mesh(asset, name+"_mesh", vertices, faces)
    quat = np.r_[1+axis[2], -axis[1], axis[0], 0.]
    quat = quat/np.linalg.norm(quat) if np.linalg.norm(quat) > 1e-12 else np.array([0., 1., 0., 0.])
    return geom(parent, name, "mesh", (), centre, mesh=name+"_mesh", material=material,
                quat=numbers(quat), mass="0", contype="0", conaffinity="0")


def focus_knob(asset, parent, name, centre, axis, radius, length, *, flutes=36):
    """Bevelled rotary control with a recessed end face and restrained grip."""
    half = length/2
    edge = min(.0015, length*.15)
    profile = [(-half, radius-edge), (-half+edge, radius),
               (half-edge, radius), (half, radius-edge)]
    turned_part(asset, parent, name, profile, centre, axis,
                flutes=flutes, flute_depth=.00035)
