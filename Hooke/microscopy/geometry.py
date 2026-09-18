"""Reusable MJCF geometry and drive builders for original instrument assets."""

import math
import xml.etree.ElementTree as ET

import numpy as np


def numbers(values):
    return " ".join(f"{value:.9g}" for value in values)


def body(parent, name, pos=(0, 0, 0), **attributes):
    return ET.SubElement(parent, "body", name=name, pos=numbers(pos), **attributes)


def geom(parent, name, kind, size, pos=(0, 0, 0), material="ivory", **attributes):
    return ET.SubElement(
        parent, "geom", name=name, type=kind, size=numbers(size),
        pos=numbers(pos), material=material, **attributes,
    )


def cylinder(parent, name, start, end, radius, material="metal", **attributes):
    return geom(parent, name, "cylinder", (radius,), material=material,
                fromto=numbers([*start, *end]), **attributes)


def annular_tube(asset, parent, name, lower, upper, outer, inner, material="graphite", **attributes):
    """Hollow visual tube along local Z; no contact or mass by default."""
    sides = 64
    angles = np.arange(sides)*2*np.pi/sides
    vertices = [(radius*np.cos(a), radius*np.sin(a), z)
                for z, radius in ((lower, outer), (upper, outer),
                                  (lower, inner), (upper, inner)) for a in angles]
    faces = []
    for start, end, reverse in ((0, sides, False), (2*sides, 3*sides, True),
                                (sides, 3*sides, False), (2*sides, 0, False)):
        for i in range(sides):
            j = (i+1) % sides
            triangles = ((start+i, start+j, end+j), (start+i, end+j, end+i))
            faces.extend(tuple(t[::-1] for t in triangles) if reverse else triangles)
    mesh(asset, name+"_mesh", vertices, faces)
    attributes = dict(material=material, mass="0", contype="0", conaffinity="0") | attributes
    return ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name+"_mesh", **attributes)


def slide(parent, actuators, name, axis, bounds, mass=0.05, kp=4000, kv=30):
    node = body(parent, name, gravcomp="1")
    ET.SubElement(node, "inertial", pos="0 0 0", mass=str(mass),
                  diaginertia=".00001 .00001 .00001")
    ET.SubElement(node, "joint", name=name, type="slide", axis=numbers(axis),
                  range=numbers(bounds), damping="0.05")
    ET.SubElement(actuators, "position", name=name + "_drive", joint=name,
                  kp=str(kp), kv=str(kv), ctrlrange=numbers(bounds))
    return node


def camera(parent, name, position, target, fovy=40):
    z = np.asarray(position) - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(parent, "camera", name=name, pos=numbers(position),
                  xyaxes=numbers([*x, *y]), fovy=str(fovy))


def frustum(asset, name, length, radius, tip_radius):
    vertices = []
    sides = 24
    for z, r in ((-length, radius), (0, tip_radius)):
        vertices.extend((r * math.cos(i * 2 * math.pi / sides),
                         r * math.sin(i * 2 * math.pi / sides), z)
                        for i in range(sides))
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.extend(((i, j, sides + j), (i, sides + j, sides + i)))
    for i in range(1, sides - 1):
        faces.extend(((0, i + 1, i), (sides, sides + i, sides + i + 1)))
    ET.SubElement(asset, "mesh", name=name,
                  vertex=numbers(np.asarray(vertices).ravel()),
                  face=" ".join(map(str, np.asarray(faces).ravel())))


def mesh(asset, name, vertices, faces):
    ET.SubElement(asset, "mesh", name=name, vertex=numbers(np.asarray(vertices).ravel()),
                  face=" ".join(map(str, np.asarray(faces).ravel())))


def rounded_box(asset, parent, name, halfsize, pos, radius=.006, material="ivory", **attributes):
    """Original filleted shell, including edge and corner curvature."""
    halfsize = np.asarray(halfsize)
    radius = min(radius, float(halfsize.min()) * .8)
    vertices, faces = [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        coordinates = [np.unique(np.r_[np.linspace(-h, -h+radius, 5),
                                      -h+radius, h-radius, np.linspace(h-radius, h, 5)]) for h in halfsize]
        for sign in (-1, 1):
            start = len(vertices)
            for a in coordinates[u]:
                for b in coordinates[v]:
                    point = np.zeros(3)
                    point[axis], point[u], point[v] = sign * halfsize[axis], a, b
                    core = np.clip(point, -halfsize+radius, halfsize-radius)
                    delta = point-core
                    vertices.append(core + radius * delta / np.linalg.norm(delta))
            rows, columns = len(coordinates[u]), len(coordinates[v])
            for i in range(rows - 1):
                for j in range(columns - 1):
                    a = start + i * columns + j
                    triangles = ((a, a+columns, a+columns+1), (a, a+columns+1, a+1))
                    faces.extend(triangles if sign == 1 else tuple(t[::-1] for t in triangles))
    mesh(asset, name + "_mesh", vertices, faces)
    ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name + "_mesh",
                  pos=numbers(pos), material=material, **attributes)


def housing(asset, parent, name, rings, material="ivory"):
    """Loft rounded rectangular profiles: (z, halfwidth, halfdepth, centre_y, radius)."""
    vertices, faces = [], []
    for z, hx, hy, cy, radius in rings:
        for x, y, angle in ((hx-radius, hy-radius, 0), (-hx+radius, hy-radius, 90),
                            (-hx+radius, -hy+radius, 180), (hx-radius, -hy+radius, 270)):
            for a in np.linspace(angle, angle+90, 9):
                vertices.append((x+radius*np.cos(np.deg2rad(a)),
                                 cy+y+radius*np.sin(np.deg2rad(a)), z))
    count = 36
    for ring in range(len(rings)-1):
        for i in range(count):
            a, b = ring*count+i, ring*count+(i+1) % count
            faces.extend(((a, b, b+count), (a, b+count, a+count)))
    top = (len(rings)-1) * count
    for i in range(1, count-1):
        faces.extend(((0, i+1, i), (top, top+i, top+i+1)))
    mesh(asset, name + "_mesh", vertices, faces)
    ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name + "_mesh", material=material)


def knurled_knob(parent, name, centre, axis, radius, length, material="graphite", ribs=32):
    centre, axis = np.asarray(centre), np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    start, end = centre-axis*length/2, centre+axis*length/2
    cylinder(parent, name, start, end, radius, material)
    tangent = np.cross(axis, [0, 0, 1] if abs(axis[2]) < .9 else [1, 0, 0])
    tangent /= np.linalg.norm(tangent)
    other = np.cross(axis, tangent)
    for i in range(ribs):
        angle = i*2*np.pi/ribs
        offset = radius * (np.cos(angle)*tangent + np.sin(angle)*other)
        cylinder(parent, f"{name}_rib_{i}", start+offset, end+offset, radius*.018, "rubber")
