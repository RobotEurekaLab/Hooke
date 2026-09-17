"""Closed static terrain mesh from the compiled source height samples."""

import numpy as np


def heightfield_mesh(model, index):
    rows, cols = int(model["hfield_nrow"][index]), int(model["hfield_ncol"][index])
    sx, sy, sz, base = model["hfield_size"][index]
    if rows < 2 or cols < 2 or min(sx, sy, sz, base) <= 0:
        raise ValueError(
            "Heightfield requires a positive 3D extent and at least 2×2 samples"
        )
    start = int(model["hfield_adr"][index])
    heights = model["hfield_data"][start : start + rows * cols].reshape(rows, cols)
    if not np.isfinite(heights).all():
        raise ValueError("Heightfield samples must be finite")
    xx, yy = np.meshgrid(np.linspace(-sx, sx, cols), np.linspace(-sy, sy, rows))
    points = np.column_stack((xx.ravel(), yy.ravel(), (heights * sz).ravel()))
    grid = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = (
        grid[:-1, :-1].ravel(),
        grid[:-1, 1:].ravel(),
        grid[1:, :-1].ravel(),
        grid[1:, 1:].ravel(),
    )
    faces = np.concatenate((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
    border = np.r_[grid[0, :], grid[1:, -1], grid[-1, -2::-1], grid[-2:0:-1, 0]]
    lower = np.arange(len(border)) + len(points)
    bottom = points[border].copy()
    bottom[:, 2] = -base
    next_border, next_lower = np.roll(border, -1), np.roll(lower, -1)
    walls = np.concatenate(
        (
            np.column_stack((border, lower, next_lower)),
            np.column_stack((border, next_lower, next_border)),
        )
    )
    # Convex rectangular bottom, triangulated with downward-facing normals.
    center = len(points) + len(bottom)
    base_faces = np.column_stack((np.full(len(lower), center), next_lower, lower))
    return np.concatenate(
        (points, bottom, np.array([[0.0, 0.0, -base]]))
    ), np.concatenate((faces, walls, base_faces)).astype(np.int32)


def heightfield_normals(points, faces, rows, columns):
    """Smooth the visible top without blending its normals with base walls."""
    top = points[: rows * columns].reshape(rows, columns, 3)
    dy, dx = np.gradient(
        top[:, :, 2], top[1, 0, 1] - top[0, 0, 1], top[0, 1, 0] - top[0, 0, 0]
    )
    vertex = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1).reshape(-1, 3)
    vertex /= np.linalg.norm(vertex, axis=1, keepdims=True)
    triangles = points[faces]
    flat = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    normals = np.repeat(flat[:, None, :], 3, axis=1)
    count = 2 * (rows - 1) * (columns - 1)
    normals[:count] = vertex[faces[:count]]
    return normals.reshape(-1, 3)
