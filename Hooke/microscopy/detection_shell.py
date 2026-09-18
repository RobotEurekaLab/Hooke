"""Original hollow collection housing; internal features are design estimates.

The rounded exterior profiles can come from a dimensional reference. Wall
insets, roof bore and side bore do not reproduce Nikon's internal mechanics.
"""

import numpy as np
from scipy.spatial import Delaunay


def hollow_shell_mesh(rings, *, wall_m=.003, roof_bore_radius_m=.012,
                      side_bore_radius_m=.014, side_bore_centre_yz_m=(0., .095)):
    """Return a closed loft with a roof opening and an optional X− port.

    Profiles use (z, halfwidth, halfdepth, centre_y, corner_radius), as in
    geometry.housing. Insets are nominal coordinate offsets, rather than
    a calibrated constant normal wall thickness. Bores use 48-sided outlines.
    """
    rings = np.asarray(rings, dtype=float)
    has_side_port = side_bore_radius_m is not None
    dimensions = np.asarray([wall_m, roof_bore_radius_m]
                            + ([side_bore_radius_m] if has_side_port else []))
    centre = np.asarray(side_bore_centre_yz_m, dtype=float)
    if (rings.ndim != 2 or rings.shape[1] != 5 or len(rings) < 3
            or not np.isfinite(rings).all() or not np.isfinite(dimensions).all()
            or centre.shape != (2,) or not np.isfinite(centre).all()
            or np.any(dimensions <= 0) or np.any(np.diff(rings[:, 0]) <= 0)
            or np.any(rings[:, 4] <= wall_m)
            or np.any(rings[:, 1:3] <= rings[:, 4, None])
            or rings[-1, 0]-rings[0, 0] <= 2*wall_m):
        raise ValueError("Invalid hollow collection-shell dimensions")
    inner = rings.copy()
    inner[:, [1, 2, 4]] -= wall_m
    inner[0, 0] += wall_m
    inner[-1, 0] -= wall_m
    if np.any(np.diff(inner[:, 0]) <= 0):
        raise ValueError("Wall inset leaves no cavity")
    # The bore must stay entirely in one flat left wall segment in both lofts.
    candidates = []
    for index in range(len(rings)-1) if has_side_port else ():
        bounds = [profile[index:index+2] for profile in (rings, inner)]
        if all(centre[1]-side_bore_radius_m > pair[0, 0]
               and centre[1]+side_bore_radius_m < pair[1, 0]
               and abs(centre[0]-pair[:, 3]).max()+side_bore_radius_m
               < (pair[:, 2]-pair[:, 4]).min() for pair in bounds):
            candidates.append(index)
    top = inner[-1]
    if ((has_side_port and len(candidates) != 1)
            or roof_bore_radius_m >= min(top[1]-top[4], top[2]-top[4]-abs(top[3]))):
        raise ValueError("Collection bore does not fit within a flat shell panel")
    side_ring = candidates[0] if has_side_port else None
    vertices, faces = [], []
    count = 36

    def add_points(points):
        start = len(vertices)
        vertices.extend(np.asarray(points, dtype=float).tolist())
        return np.arange(start, len(vertices))

    def profile_points(profile):
        points = []
        for z, hx, hy, cy, radius in profile:
            for x, y, angle in ((hx-radius, hy-radius, 0), (-hx+radius, hy-radius, 90),
                                (-hx+radius, -hy+radius, 180), (hx-radius, -hy+radius, 270)):
                for a in np.deg2rad(np.linspace(angle, angle+90, 9)):
                    points.append((x+radius*np.cos(a), cy+y+radius*np.sin(a), z))
        return np.asarray(points)

    outer_ids = add_points(profile_points(rings)).reshape(-1, count)
    inner_ids = add_points(profile_points(inner)).reshape(-1, count)
    angles = np.linspace(0, 2*np.pi, 48, endpoint=False)

    def panel(boundary, hole, axes, normal):
        ids = np.r_[boundary, hole]
        points = np.asarray(vertices)[ids]
        planar = points[:, axes]
        for triangle in Delaunay(planar).simplices:
            # With one convex polygonal hole, all inner vertices are on its
            # empty circumcircle; discard triangles entirely within that hole.
            if np.all(triangle >= len(boundary)):
                continue
            corners = points[triangle]
            if np.dot(np.cross(corners[1]-corners[0], corners[2]-corners[0]), normal) < 0:
                triangle = triangle[::-1]
            faces.append(ids[triangle].tolist())

    def bridge(lower, upper, reverse=False):
        for i in range(len(lower)):
            j = (i+1) % len(lower)
            for triangle in ((lower[i], lower[j], upper[j]), (lower[i], upper[j], upper[i])):
                faces.append(triangle[::-1] if reverse else triangle)

    for ids, reverse in ((outer_ids, False), (inner_ids, True)):
        for level in range(len(rings)-1):
            for i in range(count):
                if level == side_ring and i == 17:
                    continue
                j = (i+1) % count
                triangles = ((ids[level, i], ids[level, j], ids[level+1, j]),
                             (ids[level, i], ids[level+1, j], ids[level+1, i]))
                faces.extend(triangle[::-1] if reverse else triangle for triangle in triangles)

    roof_holes = []
    side_holes = []
    for profile, ids, inward in ((rings, outer_ids, False), (inner, inner_ids, True)):
        bottom = ids[0]
        for i in range(1, count-1):
            triangle = (bottom[0], bottom[i+1], bottom[i])
            faces.append(triangle[::-1] if inward else triangle)
        roof = add_points(np.c_[roof_bore_radius_m*np.cos(angles),
                                roof_bore_radius_m*np.sin(angles),
                                np.full(len(angles), profile[-1, 0])])
        panel(ids[-1], roof, [0, 1], [0, 0, -1 if inward else 1])
        roof_holes.append(roof)
        if not has_side_port:
            continue
        yz = centre+side_bore_radius_m*np.c_[np.cos(angles), np.sin(angles)]
        x = -np.interp(yz[:, 1], profile[:, 0], profile[:, 1])
        side = add_points(np.c_[x, yz])
        boundary = ids[[side_ring, side_ring, side_ring+1, side_ring+1], [17, 18, 18, 17]]
        panel(boundary, side, [1, 2], [1 if inward else -1, 0, 0])
        side_holes.append(side)
    bridge(roof_holes[1], roof_holes[0], reverse=True)
    if has_side_port:
        bridge(side_holes[0], side_holes[1], reverse=True)
    return np.asarray(vertices), np.asarray(faces, dtype=int)
