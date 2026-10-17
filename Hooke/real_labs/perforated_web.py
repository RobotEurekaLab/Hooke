"""Conforming visual surfaces for rectangular cells with polygonal bores."""

from collections import Counter, defaultdict


def web_surface(sectors, half_thickness):
    """Extrude adjacent convex sectors without leaving internal prism walls.

    Sectors are counterclockwise polygons in X/Z. Adjacent cells meet along
    horizontal/vertical outer edges; radial edges within each cell already
    match. Subdivide every axis-aligned edge at all neighboring vertices so
    different cell widths or bore subdivisions cannot leave T-junctions.
    """
    if half_thickness <= 0:
        raise ValueError("Web thickness must be positive")
    sectors = [tuple((round(x, 12), round(z, 12)) for x, z in q) for q in sectors]
    columns, rows = defaultdict(set), defaultdict(set)
    for sector in sectors:
        for x, z in sector:
            columns[x].add(z)
            rows[z].add(x)

    vertices, indices, triangles = [], {}, []

    def vertex(point):
        if point not in indices:
            indices[point] = len(vertices)
            vertices.append(point)
        return indices[point]

    for sector in sectors:
        ring = []
        for first, last in zip(sector, sector[1:] + sector[:1]):
            ring.append(first)
            if first[0] == last[0]:
                coordinates = columns[first[0]]
                axis = 1
            elif first[1] == last[1]:
                coordinates = rows[first[1]]
                axis = 0
            else:
                continue
            low, high = sorted((first[axis], last[axis]))
            interior = sorted(
                (value for value in coordinates if low < value < high),
                reverse=first[axis] > last[axis],
            )
            ring.extend(
                (first[0], value) if axis == 1 else (value, first[1])
                for value in interior
            )
        center = tuple(sum(p[axis] for p in sector) / len(sector) for axis in (0, 1))
        middle = vertex(center)
        for first, last in zip(ring, ring[1:] + ring[:1]):
            area_twice = (first[0] - center[0]) * (last[1] - center[1]) - (
                first[1] - center[1]
            ) * (last[0] - center[0])
            if area_twice <= 1e-15:
                raise ValueError(
                    "Web sectors must be nondegenerate and counterclockwise"
                )
            triangles.append((middle, vertex(first), vertex(last)))

    edges = Counter(
        tuple(sorted(edge))
        for face in triangles
        for edge in zip(face, face[1:] + face[:1])
    )
    if any(count > 2 for count in edges.values()):
        raise ValueError("Web sectors must not overlap")
    boundary = [
        (a, b)
        for face in triangles
        for a, b in zip(face, face[1:] + face[:1])
        if edges[tuple(sorted((a, b)))] == 1
    ]
    count = len(vertices)
    mesh_vertices = [
        (x, y, z) for y in (-half_thickness, half_thickness) for x, z in vertices
    ]
    # In X/Z, a counterclockwise front triangle points toward negative Y.
    faces = triangles + [(a + count, c + count, b + count) for a, b, c in triangles]
    for a, b in boundary:
        faces.extend(((a, a + count, b + count), (a, b + count, b)))
    return mesh_vertices, faces
