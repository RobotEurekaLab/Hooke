"""Independent mounting cradle for the upright open parallel mechanism.

The tower/deck layout follows laboratory manipulator stands. The cradle and
its fasteners are original estimated parts, not a verified commercial fit.
They remain fixed, non-colliding appearance geometry; stiffness is not solved.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import body, cylinder, geom, mesh, numbers, rounded_box


SOURCE_URL = "https://www.sutter.com/stages/mt70"
BASE_FACE_LOCAL_M = np.array([-.03025, -.03025, -.0475])
DECK_TOP_LOCAL_M = -.110


def _extrusion(asset, parent, name, centre, bottom, top):
    """Square section with four recessed channels, authored in metres."""
    profile = []
    side = np.array([[.030, -.030], [.030, -.004], [.026, -.004],
                     [.026, .004], [.030, .004], [.030, .030]])
    for index in range(4):
        angle = index*np.pi/2
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        profile.extend((side[:-1] @ rotation.T).tolist())
    count = len(profile)
    vertices = [[x+centre[0], y+centre[1], z]
                for z in (bottom, top) for x, y in profile]
    # The non-convex outline has only shallow slots. Centre fans triangulate
    # the caps without filling the channels along the vertical walls.
    vertices.extend([[*centre, bottom], [*centre, top]])
    faces = []
    for i in range(count):
        j = (i+1) % count
        faces.extend(((i, j, j+count), (i, j+count, i+count),
                      (2*count, j, i), (2*count+1, i+count, j+count)))
    mesh(asset, name+"_mesh", vertices, faces)
    geom(parent, name, "mesh", (), mesh=name+"_mesh", material="metal", mass="0")


def _beam(asset, parent, name, start, end, width, height, material="graphite"):
    start, end = np.asarray(start), np.asarray(end)
    direction = end-start
    length = float(np.linalg.norm(direction[:2]))
    yaw = float(np.arctan2(direction[1], direction[0]))
    rounded_box(asset, parent, name, (length/2, width/2, height/2),
                (start+end)/2, .001, material, euler=f"0 0 {yaw}", mass="0")


def _gusset(asset, parent, name, points, width):
    """Closed triangular web in the radial vertical plane."""
    tangent = np.array([1., -1., 0.])/np.sqrt(2)
    vertices = [np.asarray(point)+side*tangent*width/2
                for side in (-1, 1) for point in points]
    faces = [(0, 2, 1), (3, 4, 5), (0, 1, 4), (0, 4, 3),
             (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)]
    mesh(asset, name+"_mesh", vertices, faces)
    geom(parent, name, "mesh", (), mesh=name+"_mesh", material="graphite", mass="0")


def _bench_fasteners(plate, rotation, column, bench_top):
    """Select four existing breadboard holes beneath the mounting flange."""
    candidates = []
    for x in np.arange(-.55, .56, .05):
        for y in np.arange(-.32, .33, .05):
            local = rotation.T @ (np.array([x, y, bench_top+.016])-plate)
            offset = local[:2]-column[:2]
            if np.max(abs(offset)) < .070 and np.max(abs(offset)) > .040:
                candidates.append(local)
    selected = []
    for x, y in ((-.06, -.06), (-.06, .06), (.06, -.06), (.06, .06)):
        target = column[:2]+[x, y]
        if not candidates:
            raise ValueError("Manipulator foot does not cover four breadboard holes")
        index = min(range(len(candidates)), key=lambda i: np.linalg.norm(candidates[i][:2]-target))
        selected.append(candidates.pop(index))
    return selected


def add_support(asset, world, tool, plate, rotation, bench_top):
    """Seat a fixed deck below the motors and clamp the base from above.

    ``plate`` is the CAD BaseBlock's bottom-face centre, not the bottom of
    the complete mechanism. ``rotation`` is its upright world orientation;
    the local negative-X/negative-Y diagonal points away from the specimen.
    A solid plate at this face would pass through the lower motor, so the
    deck is 62.5 mm lower and an open rear cradle reaches the base instead.
    """
    plate, rotation = np.asarray(plate), np.asarray(rotation)
    if not np.allclose(rotation[:, 2], [0., 0., 1.], atol=1e-9):
        raise ValueError("The manipulator deck requires an upright base frame")
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    support = body(world, tool+"_support", plate, quat=numbers(quaternion))
    # Retain a small numerical mass for the fixed body without assigning
    # unrealistic material densities to individual appearance meshes.
    ET.SubElement(support, "inertial", pos="0 0 0", mass=".004",
                  diaginertia="1e-7 1e-7 1e-7")
    radial = np.array([-1., -1., 0.])/np.sqrt(2)
    table_z = float(bench_top-plate[2])
    deck_z = DECK_TOP_LOCAL_M-BASE_FACE_LOCAL_M[2]
    column = radial*.090
    deck = radial*.050
    rounded_box(asset, support, tool+"_foot", (.0765, .0765, .008),
                [*column[:2], table_z+.008], .002, "graphite", mass="0")
    fasteners = _bench_fasteners(plate, rotation, column, bench_top)
    for index, point in enumerate(fasteners):
        cylinder(support, f"{tool}_foot_washer_{index}", point,
                 point+[0, 0, .001], .0055, "metal", mass="0")
        cylinder(support, f"{tool}_foot_screw_{index}", point+[0, 0, .001],
                 point+[0, 0, .004], .003, "graphite", mass="0")
    _extrusion(asset, support, tool+"_support_column", column[:2],
               table_z+.016, deck_z-.008)
    for z in (table_z+.019, deck_z-.011):
        rounded_box(asset, support, f"{tool}_column_flange_{z:.6f}", (.034, .034, .003),
                    [*column[:2], z], .001, "graphite", mass="0")
    rounded_box(asset, support, tool+"_mount_deck", (.064, .062, .004),
                [*deck[:2], deck_z-.004], .001, "graphite", mass="0")
    _beam(asset, support, tool+"_deck_beam", column+[0, 0, deck_z-.020],
          [0, 0, deck_z-.020], .044, .024)
    tangent = np.array([1., -1., 0.])/np.sqrt(2)
    for side in (-1, 1):
        shift = tangent*side*.018
        points = [column+shift+[0, 0, deck_z-.070],
                  column+shift+[0, 0, deck_z-.032],
                  radial*.005+shift+[0, 0, deck_z-.032]]
        _gusset(asset, support, f"{tool}_deck_gusset_{side}", points, .004)
    # The motor cases reach 98 mm below the platform datum. Keep this
    # original C-shaped cradle behind their negative-X/negative-Y corner.
    rear = np.array([-.075, -.075, .021])-BASE_FACE_LOCAL_M
    front = np.array([-.039, -.039, .021])-BASE_FACE_LOCAL_M
    rounded_box(asset, support, tool+"_cradle_back", (.009, .009, (rear[2]-deck_z)/2),
                [*rear[:2], (rear[2]+deck_z)/2], .001, "graphite", mass="0")
    _beam(asset, support, tool+"_cradle_bridge", rear, front, .012, .008)
    # The fixed MotorMount plate covers the BaseBlock's upper face. Seat on
    # its exposed Z=-6 mm surface, never pass through it to the cube below.
    contact_z = -.006-BASE_FACE_LOCAL_M[2]
    rounded_box(asset, support, tool+"_cradle_clamp", (.006, .006, (front[2]-contact_z)/2),
                [*front[:2], (front[2]+contact_z)/2], .0008, "graphite", mass="0")
    for offset in (-.003, .003):
        point = front+[offset, -offset, .004]
        cylinder(support, f"{tool}_clamp_bolt_{offset}", point,
                 point+[0, 0, .003], .002, "metal", mass="0")
    return dict(profile="original-upright-cradle", reference_url=SOURCE_URL,
                base_face_world_m=plate.tolist(), deck_top_world_z_m=float(plate[2]+deck_z),
                column_section_m=[.060, .060], deck_size_m=[.128, .124, .008],
                column_radial_offset_m=.090, mounting_fit_verified=False,
                breadboard_bolt_world_xy_m=[(plate+rotation@point)[:2].tolist()
                                            for point in fasteners],
                stiffness_validated=False, support_physics="fixed rigid body",
                note="Original estimated rear cradle; commercial mounting principle only")
