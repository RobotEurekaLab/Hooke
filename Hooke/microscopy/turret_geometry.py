"""Original conical six-position turret; its inclination is a photo estimate.

The selected objective stays vertical. Each inactive socket follows the same
cone, instead of tilting a flat disc while leaving all barrels upright. Bores
are computed along their socket axes and remain actual through openings.
"""

import numpy as np
from scipy.spatial import Delaunay

INCLINATION_DEGREES = 18.
CENTRE_M = np.array([-.035, 0., 0.])
THICKNESS_M = .005


def build_geometry():
    angle = np.deg2rad(INCLINATION_DEGREES)
    sine, cosine, slope = np.sin(angle), np.cos(angle), np.tan(angle)
    rotation = np.array([[cosine, 0., -sine], [0., 1., 0.], [sine, 0., cosine]])
    ring = .035*cosine
    azimuths = np.arange(6)*np.pi/3
    local_centres = np.c_[ring*np.cos(azimuths), ring*np.sin(azimuths),
                          np.full(6, -ring*slope)]
    local_axes = np.c_[sine*np.cos(azimuths), sine*np.sin(azimuths), np.full(6, cosine)]
    sockets = local_centres@rotation.T+CENTRE_M
    axes = local_axes@rotation.T
    # The small central bore clears the vertical focus spindle. It is not
    # another objective socket and uses the world-vertical drilling axis.
    bores = list(zip(local_centres, local_axes, [.0127]*6))
    bores.append((np.zeros(3), rotation.T@np.array([0., 0., 1.]), .009))

    def surface(points, axis, depth):
        result = np.array(points, copy=True)
        for _ in range(12):
            radius = np.maximum(np.linalg.norm(result[:, :2], axis=1), 1e-12)
            residual = result[:, 2]+slope*radius+depth
            derivative = axis[2]+slope*(result[:, :2]@axis[:2])/radius
            result -= (residual/derivative)[:, None]*axis
        return result

    theta = np.arange(192)*2*np.pi/192
    outer = np.c_[.055*np.cos(theta), .055*np.sin(theta), np.full(192, -.055*slope)]
    upper, lower = [outer], [outer-[0., 0., THICKNESS_M]]
    theta = np.arange(96)*2*np.pi/96
    for centre, axis, radius in bores:
        across = np.cross(axis, [0., 0., 1.] if abs(axis[2]) < .95 else [1., 0., 0.])
        across /= np.linalg.norm(across)
        other = np.cross(axis, across)
        outline = centre+radius*(np.cos(theta)[:, None]*across+np.sin(theta)[:, None]*other)
        upper.append(surface(outline, axis, 0.))
        lower.append(surface(outline, axis, THICKNESS_M))
    top, bottom = np.vstack(upper), np.vstack(lower)
    triangles = Delaunay(top[:, :2]).simplices
    centroids = top[triangles].mean(axis=1)
    centroids[:, 2] = -slope*np.linalg.norm(centroids[:, :2], axis=1)
    keep = np.ones(len(triangles), dtype=bool)
    for centre, axis, radius in bores:
        delta = centroids-centre
        perpendicular = delta-(delta@axis)[:, None]*axis
        keep &= np.linalg.norm(perpendicular, axis=1) > radius
    triangles = triangles[keep]
    count = len(top)
    faces = [tuple(triangle) for triangle in triangles]
    faces.extend(tuple(triangle[::-1]+count) for triangle in triangles)
    edges = {}
    for triangle in triangles:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key in edges:
                del edges[key]
            else:
                edges[key] = (int(a), int(b))
    for a, b in edges.values():
        faces.extend(((a, b+count, b), (a, a+count, b+count)))
    vertices = np.vstack((top, bottom))@rotation.T+CENTRE_M
    return vertices, np.asarray(faces), sockets, axes
