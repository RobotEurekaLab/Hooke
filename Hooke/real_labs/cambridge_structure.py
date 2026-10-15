"""Source-shaped Turner exterior around a separately authored inspection fixture.

The designer documents four 400 by 200 mm hollow members, end bridges and a
two-arm swinging carriage. Wall thickness, local spacing and parked elevations
below are estimates. Swing motion, torsion compliance and high-g loads are not
implemented: these parts are welded to the inspection rotor in its parked pose.
"""

import xml.etree.ElementTree as ET


def _box(b, body, key, pos, half, material="metal"):
    return b.box(body, pos, half, material, collision=True,
                 name=f"{b.name}__turner_{key}")


def principal_members(b, rotor):
    """Four open rectangular tubes; each contact wall has its real thickness."""
    for side, y in (("front", -0.48), ("rear", 0.48)):
        for level, z in (("lower", -0.155), ("upper", 0.095)):
            key = f"member_{side}_{level}"
            for face, dz in (("bottom", -0.09), ("top", 0.09)):
                _box(b, rotor, key + "_" + face, (0, y, z + dz),
                     (4.72, 0.20, 0.01))
            for face, dy in (("front", -0.19), ("rear", 0.19)):
                _box(b, rotor, key + "_" + face, (0, y + dy, z),
                     (4.72, 0.01, 0.08))
    for x in (-4.62, 4.62):
        for level, z in (("lower", -0.155), ("upper", 0.095)):
            _box(b, rotor, f"bridge_{x}_{level}", (x, 0, z),
                 (0.06, 0.28, 0.08))
        for y in (-0.68, 0.68):
            _box(b, rotor, f"bridge_cheek_{x}_{y}", (x, y, -0.03),
                 (0.09, 0.012, 0.235), "bright")
    # Gussets connect the four members to the central shaft collar.
    for y in (-0.26, 0.26):
        for z in (-0.155, 0.095):
            _box(b, rotor, f"shaft_gusset_{y}_{z}", (0, y, z),
                 (0.80, 0.04, 0.07), "bright")


def upper_bearing_mount(b, base):
    """Four floor-supported columns carry a plate with a genuine shaft opening."""
    for x in (-0.92, 0.92):
        for y in (-1.10, 1.10):
            _box(b, base, f"upper_mount_foot_{x}_{y}", (x, y, 0.035),
                 (0.13, 0.13, 0.035))
            _box(b, base, f"upper_mount_column_{x}_{y}", (x, y, 0.87),
                 (0.045, 0.045, 0.80))
    for side, y in (("front", -0.715), ("rear", 0.715)):
        _box(b, base, "upper_mount_plate_" + side, (0, y, 1.695),
             (1.02, 0.455, 0.025))
    for side, x in (("left", -0.64), ("right", 0.64)):
        _box(b, base, "upper_mount_plate_" + side, (x, 0, 1.695),
             (0.38, 0.26, 0.025))


def parked_swing_carriages(b, rotor):
    """Source-visible pivot arms/platforms; rigidly parked, not a swing actuator."""
    for end in (-1, 1):
        x = end * 4.15
        _box(b, rotor, f"parked_swing_platform_{end}", (x, 0, -0.49),
             (0.40, 0.27, 0.025), "bright")
        for y in (-0.23, 0.23):
            key = f"parked_swing_{end}_{y}"
            # Tapered plate cheeks, rather than an invented slender linkage.
            cross = [(end * 3.805, 0.08), (end * 3.895, 0.08),
                     (end * 4.28, -0.465), (end * 4.02, -0.465)]
            if end < 0:
                cross.reverse()
            vertices = [(px, py, pz) for py in (y - 0.025, y + 0.025)
                        for px, pz in cross]
            faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
            for i in range(4):
                j = (i + 1) % 4
                faces.extend(((i, i + 4, j + 4), (i, j + 4, j)))
            mesh = f"{b.name}__turner_{key}_arm_mesh"
            ET.SubElement(b.root.find("asset"), "mesh", name=mesh,
                          vertex=" ".join(str(v) for p in vertices for v in p),
                          face=" ".join(str(v) for f in faces for v in f))
            b.geom(rotor, "mesh", (), mesh=mesh, material="bright", collision=True,
                   name=f"{b.name}__turner_{key}_arm")
            b.rod(rotor, (end * 3.85, y - 0.06, 0.08),
                  (end * 3.85, y + 0.06, 0.08), 0.05, "metal",
                  collision=True, name=f"{b.name}__turner_{key}_pivot")
            b.rod(rotor, (end * 3.15, y, 0.03),
                  (end * 4.50, y, 0.03), 0.018, "bright",
                  collision=True, name=f"{b.name}__turner_{key}_torsion_bar")
            for px in (3.15, 4.50):
                _box(b, rotor, f"{key}_torsion_mount_{px}",
                     (end * px, y, 0.03), (0.035, 0.075, 0.045))
        _box(b, rotor, f"parked_swing_stop_{end}", (end * 4.56, 0, -0.29),
             (0.035, 0.27, 0.20))
