"""EPFL fixed apparatus and moving thermal-stage geometry.

The source supports the arrangement and visible categories. Every dimension is
authored, including the 500 mL-like bulb, clamp contact, condenser and bath.
There is no claim of vessel capacity, chemical connection or OEM identification.
The separate scissor mechanism supplies the moving deck; this file adds no DOF.
"""

from __future__ import annotations

from collections import defaultdict
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

from real_labs.mounting import annular_mount


BULB_RADIUS = 0.052
NECK_RADIUS = 0.016
BULB_HEIGHT = 0.381  # Fixed sphere centre above the bench.
HOTPLATE_TOP = 0.074  # Heights below are relative to the jack deck top.
BATH_FLOOR = 0.078
BATH_RIM = 0.140
BATH_INNER = 0.072
BATH_OUTER = 0.075
GLASS_WALL = 0.0015
RACK_RADIUS = 0.008
RACK_POST_BOTTOM = 0.970
RACK_POST_TOP = 2.300


def _name(b, key):
    return f"{b.name}__assembly_{key}"


def _body(b, parent, key, pos):
    return ET.SubElement(parent, "body", name=_name(b, key),
                         pos=" ".join(str(v) for v in pos))


def _annulus(b, parent, key, pos, inner, outer, half, material="metal"):
    annular_mount(b, parent, key, pos, inner, outer, half, axis="z",
                  proxies=True, material=material, namespace="assembly")


def _post_collar(b, parent, key, x, y, z):
    """Actual 16 mm bore: a post is not hidden inside a solid clamp block."""
    _annulus(b, parent, key + "_rack_collar", (x, y, z),
             RACK_RADIUS, 0.017, 0.012)
    # A visible captive knob touches the collar; its hidden thread is omitted.
    b.cylinder(parent, (x + 0.020, y, z), 0.005, 0.003, "dark",
               euler=(0, math.pi / 2, 0),
               name=_name(b, key + "_boss_screw"))


def add_grounded_rack(b, base, *, bench_z=0.934, rack_y=0.30,
                     post_xs=(-0.28,), existing_posts=False):
    """Bench-anchored sockets for exact production 16 mm capsule posts.

    With existing_posts=True, preserve supplied posts running from z=.970 to
    2.300, and add their missing grounding only. Otherwise create those same
    posts. Default is a minimal independent post for the main apparatus.
    No coincident coplanar crossbars are created. Existing production rods at
    equal Y intersect at their crossings; this independent post helper adds
    none. The complete hood helper supplies offset crossbars and bored couplers.
    """
    if bench_z + .012 >= RACK_POST_BOTTOM - RACK_RADIUS:
        raise ValueError("Bench is too high for the specified original rack endpoints")
    result = {}
    for x in post_xs:
        key = f"rack_{x}"
        # The 12 mm plate meets the bench, with a socket floor above it.
        foot = b.box(base, (x, rack_y, bench_z + .006), (.034, .026, .006),
                     "metal", collision=True, name=_name(b, key + "_foot"))
        floor_bottom, floor_top = bench_z + .012, RACK_POST_BOTTOM - RACK_RADIUS
        floor = b.cylinder(base, (x, rack_y, (floor_bottom + floor_top) / 2),
                           .018, (floor_top - floor_bottom) / 2, "metal",
                           collision=True, name=_name(b, key + "_socket_floor"))
        sleeve_top = RACK_POST_BOTTOM + .038
        _annulus(b, base, key + "_socket", (x, rack_y, (floor_top + sleeve_top) / 2),
                 RACK_RADIUS, .018, (sleeve_top - floor_top) / 2)
        for side in (-1, 1):
            b.cylinder(base, (x + side * .026, rack_y, bench_z + .014),
                       .004, .002, "bright",
                       name=_name(b, f"{key}_anchor_head_{side}"))
        post = None
        if not existing_posts:
            post = b.rod(base, (x, rack_y, RACK_POST_BOTTOM),
                         (x, rack_y, RACK_POST_TOP), RACK_RADIUS, "bright",
                         collision=True, name=_name(b, key + "_post"))
        result[x] = dict(foot=foot.get("name"), floor=floor.get("name"),
                         post=post.get("name") if post is not None else None)
    return result


def _flask_shell(b, base, key, center):
    """Original hollow lathe shell: spherical bowl, smooth shoulder, open neck.

    Outer profile never exceeds the 52 mm envelope. Inner vertices are offset
    along the profile normal by 1.5 mm. Its presentation mesh is not used as a
    convex collider; a separate helper derives local hollow-wall collision cells.
    """
    radius = BULB_RADIUS
    profile = []
    theta_end = math.asin(.036 / radius)
    for i in range(33):
        theta = -math.pi / 2 + (theta_end + math.pi / 2) * i / 32
        r, z = radius * math.cos(theta), radius * math.sin(theta)
        profile.append((r, z, math.cos(theta), math.sin(theta)))
    r0 = math.sqrt(radius * radius - .036 * .036)
    points = ((r0, .036), (r0 - .036 / r0 * .010, .046),
              (NECK_RADIUS, .053), (NECK_RADIUS, .063))
    for i in range(1, 13):
        t = i / 12
        u = 1 - t
        r, z = (sum(weight * p[j] for weight, p in zip(
            (u**3, 3*u*u*t, 3*u*t*t, t**3), points)) for j in (0, 1))
        dr, dz = (3 * sum(weight * (points[k+1][j] - points[k][j])
                         for k, weight in enumerate((u*u, 2*u*t, t*t)))
                  for j in (0, 1))
        length = math.hypot(dr, dz)
        profile.append((r, z, dz / length, -dr / length))
    profile.append((NECK_RADIUS, .109, 1, 0))
    # Include the exact equator so the visual and conservative envelope agree.
    profile.append((radius, 0, 1, 0))
    profile.sort(key=lambda point: point[1])
    segments, vertices, faces = 64, [], []
    layers = []
    for inner in (False, True):
        rings = []
        for i, (r, z, nr, nz) in enumerate(profile):
            if inner:
                r, z = r - GLASS_WALL * nr, z - GLASS_WALL * nz
            if i == 0:
                rings.append([len(vertices)])
                vertices.append((center[0], center[1], center[2] + z))
                continue
            ring = []
            for n in range(segments):
                angle = math.tau * n / segments
                ring.append(len(vertices))
                vertices.append((center[0] + r * math.cos(angle),
                                 center[1] + r * math.sin(angle), center[2] + z))
            rings.append(ring)
        for lower, upper in zip(rings, rings[1:]):
            for n in range(segments):
                nxt = (n + 1) % segments
                triangles = ((lower[0], upper[nxt], upper[n]),) if len(lower) == 1 else (
                    (lower[n], lower[nxt], upper[nxt]),
                    (lower[n], upper[nxt], upper[n]))
                faces.extend(tuple(reversed(face)) if inner else face for face in triangles)
        layers.append(rings)
    outer, inner = layers[0][-1], layers[1][-1]
    for n in range(segments):
        nxt = (n + 1) % segments
        faces.extend(((outer[n], outer[nxt], inner[nxt]),
                      (outer[n], inner[nxt], inner[n])))
    name = _name(b, key + "_flask_shell_mesh")
    ET.SubElement(b.root.find("asset"), "mesh", name=name,
                  vertex=" ".join(f"{v:.10g}" for p in vertices for v in p),
                  face=" ".join(str(v) for p in faces for v in p))
    b.geom(base, "mesh", (), (0, 0, 0), "glass", mesh=name,
           name=_name(b, key + "_flask_shell"))
    return {"vertices": len(vertices), "faces": len(faces),
            "wall_m": GLASS_WALL, "max_outer_radius_m": BULB_RADIUS,
            "open_neck_inner_radius_m": NECK_RADIUS - GLASS_WALL}


def add_hotplate_open_bath(b, deck, *, surface=(0.075, 0, 0.018), key="main"):
    """Attach only hotplate and bath to a supplied scissor deck.

    surface is the deck-top centre in the deck body's local frame. It matches
    the paired scissor-jack module by default. The outer body has small feet with actual
    deck contact. Hollow bath sectors retain the full circular entry opening.
    """
    payload = _body(b, deck, key + "_thermal_stage", surface)
    ET.SubElement(payload, "inertial", mass="1.8", pos="0 .005 .052",
                  diaginertia=".008 .008 .010")
    for x in (-0.068, 0.068):
        for y in (-0.074, 0.074):
            b.cylinder(payload, (x, y, 0.003), 0.011, 0.003, "rubber",
                       collision=True, name=_name(b, f"{key}_foot_{x}_{y}"))
    b.housing(payload, ((0.006, 0.087, 0.093, 0),
                        (0.060, 0.082, 0.084, 0.006)), material="cream")
    b.box(payload, (0, 0.006, 0.033), (0.080, 0.081, 0.027), "cream",
          collision=True, name=_name(b, key + "_hotplate_body"))
    b.box(payload, (0, -0.085, 0.030), (0.084, 0.006, 0.024), "blue",
          name=_name(b, key + "_blue_fascia"))
    for x in (-0.037, 0.036):
        b.cylinder(payload, (x, -0.095, 0.030), 0.012, 0.005, "cream",
                   euler=(math.pi / 2, 0, 0),
                   name=_name(b, f"{key}_control_{x}"))
    b.cylinder(payload, (0, 0, 0.067), 0.076, 0.007, "bright",
               collision=True, name=_name(b, key + "_plate"))
    b.cylinder(payload, (0, 0, 0.076), BATH_OUTER, 0.002, "bright",
               collision=True, name=_name(b, key + "_bath_floor"))
    annular_mount(b, payload, key + "_bath_wall", (0, 0, 0.109),
                  BATH_INNER, BATH_OUTER, 0.031, axis="z", proxies=True,
                  material="bright", namespace="assembly")
    b.ring(payload, (0, 0, BATH_RIM - 0.0015), 0.0735, 0.0015, "bright")
    b.site(payload, key + "_bath_inner_floor", (0, 0, BATH_FLOOR))
    b.site(payload, key + "_bath_rim", (0, 0, BATH_RIM))
    return payload


def _neck_clamp(b, parent, key, *, axis=(0, 0), z=0, radius=NECK_RADIUS,
                rack_y=0.30):
    """Static two-jaw clamp, anchored to a rack post at the same X position.

    Pads touch the neck tangentially; the fixed body is the retention model.
    This is not a frictional grasp or a simulated adjustable clamp.
    """
    x, y = axis
    _post_collar(b, parent, key, x, rack_y, z)
    b.rod(parent, (x, rack_y - 0.022, z), (x, y + 0.045, z), 0.005, "bright",
          name=_name(b, key + "_stem"))
    b.rod(parent, (x - radius - 0.019, y + 0.045, z),
          (x + radius + 0.019, y + 0.045, z), 0.004, "metal",
          name=_name(b, key + "_crossbar"))
    for side in (-1, 1):
        for j, (a, c) in enumerate((
            ((radius + 0.019, 0.045), (radius + 0.012, 0.018)),
            ((radius + 0.012, 0.018), (radius + 0.008, 0)),
        )):
            b.rod(parent, (x + side * a[0], y + a[1], z),
                  (x + side * c[0], y + c[1], z), 0.004, "metal",
                  name=_name(b, f"{key}_jaw_{side}_{j}"))
        b.box(parent, (x + side * (radius + 0.002), y, z),
              (0.002, 0.008, 0.010), "rubber", collision=True,
              name=_name(b, f"{key}_pad_{side}"))


def add_fixed_glassware(b, base, *, axis=(-0.28, -0.005), bench_z=0.934,
                        rack_y=0.30, key="main", condenser=True):
    """Separately retain dry, empty-looking glassware on the static hood rack.

    The hollow glass shell has an authored 1.5 mm wall and conservative convex
    wall cells derived from its exact presentation mesh. Its fixed attachment
    does not model clamp forces, glass fracture or contents. No coloured fill
    is inserted. Condenser-like jacket and sidearms are evidence-category
    detail with inferred dimensions, not an identified source apparatus.
    """
    x, y = axis
    z = bench_z + BULB_HEIGHT
    shell = _flask_shell(b, base, key, (x, y, z))
    b.geom(base, "sphere", (BULB_RADIUS,), (x, y, z), "glass",
           collision=True, rgba=(0, 0, 0, 0), group="3",
           name=_name(b, key + "_bulb"))
    b.cylinder(base, (x, y, z + 0.076), NECK_RADIUS, 0.033, "glass",
               collision=True, rgba=(0, 0, 0, 0), group="3",
               name=_name(b, key + "_neck"))
    b.cylinder(base, (x, y, z + 0.0495), .038, .0135, "glass",
               collision=True, rgba=(0, 0, 0, 0), group="3",
               name=_name(b, key + "_shoulder_proxy"))
    _neck_clamp(b, base, key + "_flask_clamp", axis=axis, z=z + 0.076,
                rack_y=rack_y)
    b.site(base, "inert_flask", (x, y, z))
    from real_labs.epfl_flask_collision import install_flask_collision
    collision = install_flask_collision(b, base, key=key, center=(x, y, z))
    result = {"bulb_center": (x, y, z), "bulb_bottom": z - BULB_RADIUS,
              "glassware_top": z + .109, "flask_shell": shell,
              "required_post_x": x, "required_post_y": rack_y,
              "flask_collision": collision}
    if not condenser:
        return result
    # Narrow core plus transparent annular outer jacket preserve a visible bore.
    bottom, top = z + 0.109, z + 0.365
    _annulus(b, base, key + "_condenser_core", (x, y, (bottom + top) / 2),
             .0045, .006, (top - bottom) / 2 - .0015, "glass")
    annular_mount(b, base, key + "_condenser_jacket",
                  (x, y, (bottom + top) / 2), 0.012, 0.015,
                  (top - bottom) / 2 - .0015, axis="z", proxies=True,
                  material="glass", namespace="assembly")
    for index, joint_z in enumerate((bottom + 0.008, top - 0.008)):
        b.rod(base, (x + 0.012, y, joint_z),
              (x + 0.043, y, joint_z + 0.010), 0.004, "glass",
              name=_name(b, f"{key}_condenser_sidearm_{index}"))
    for end, cap_z in (("bottom", bottom + .00075), ("top", top - .00075)):
        _annulus(b, base, key + "_condenser_" + end + "_bridge", (x, y, cap_z),
                 .0045, .015, .00075, "glass")
    _neck_clamp(b, base, key + "_condenser_clamp", axis=axis,
                z=z + 0.29, radius=0.015, rack_y=rack_y)
    result["glassware_top"] = top
    return result


def _wash_bottle(b, base, key, x, y, bench_z, cap="cream"):
    b.cylinder(base, (x, y, bench_z + 0.047), 0.029, 0.047, "cream",
               name=_name(b, key + "_body"))
    b.cylinder(base, (x, y, bench_z + 0.103), 0.017, 0.009, cap,
               name=_name(b, key + "_cap"))
    b.rod(base, (x, y, bench_z + 0.108), (x, y, bench_z + 0.151),
          0.004, "cream", name=_name(b, key + "_spout_up"))
    b.rod(base, (x, y, bench_z + 0.151),
          (x - 0.033, y, bench_z + 0.168), 0.004, "cream",
          name=_name(b, key + "_spout_tip"))


def _secondary_hotplate(b, base, side, bench_z, back_y):
    """Source-visible second hotplate on a retained, unpowered laboratory jack."""
    x, y = (-.61 if side == "left" else -.60), .06
    key = side + "_secondary"
    for z in (bench_z + .010, bench_z + .080):
        b.box(base, (x, y, z), (.105, .095, .010), "metal", collision=True,
              name=_name(b, f"{key}_jack_plate_{z}"))
    for dy in (-.070, .070):
        for sign in (-1, 1):
            b.rod(base, (x - sign * .080, y + dy, bench_z + .020),
                  (x + sign * .080, y + dy, bench_z + .070), .006, "bright",
                  name=_name(b, f"{key}_jack_link_{dy}_{sign}"))
    b.box(base, (x, y, bench_z + .121), (.110, .107, .031),
          "blue" if side == "left" else "dark", collision=True,
          name=_name(b, key + "_hotplate"))
    b.cylinder(base, (x, y, bench_z + .159), .099, .007, "bright", collision=True,
               name=_name(b, key + "_platen"))
    for dx in (-.062, .062):
        b.cylinder(base, (x + dx, y - .113, bench_z + .119), .013, .006,
                   "cream", euler=(math.pi / 2, 0, 0),
                   name=_name(b, f"{key}_dial_{dx}"))
    # Fixed power dressing ends at a supported rear-wall junction, away from
    # the independently actuated main bath and clamped glassware.
    points = ((x, y + .107, bench_z + .116), (x, .235, bench_z + .025),
              (x - .095, .235, bench_z + .025), (x - .095, back_y - .032, bench_z + .19))
    for index, (start, end) in enumerate(zip(points, points[1:])):
        b.rod(base, start, end, .004, "dark", name=_name(b, f"{key}_lead_{index}"))
    b.box(base, (x - .095, back_y - .016, bench_z + .19), (.027, .016, .033), "cream",
          collision=True, name=_name(b, key + "_junction"))


def add_asymmetric_props(b, base, *, side="left", bench_z=0.934, rack_y=.30,
                         back_y=.40):
    """Source-visible secondary apparatus; all additional objects remain fixed."""
    if side not in {"left", "right"}:
        raise ValueError("Expected hood side left or right")
    _secondary_hotplate(b, base, side, bench_z, back_y)
    bottles = ((0.34, -0.12, "cream"), (0.56, -0.10, "warning"),
               (-0.71, -0.17, "cream")) if side == "left" else (
                   (0.48, 0.18, "cream"),)
    for i, (x, y, cap) in enumerate(bottles):
        _wash_bottle(b, base, f"{side}_wash_{i}", x, y,
                     bench_z + (0.024 if side == "right" else 0), cap)
    count = 6 if side == "left" else 2
    for i in range(count):
        x = 0.22 + (i % 3) * 0.066
        y = -0.28 + (i // 3) * 0.060
        b.cylinder(base, (x, y, bench_z + 0.028), 0.017, 0.028, "glass",
                   name=_name(b, f"{side}_vial_{i}"))
        b.cylinder(base, (x, y, bench_z + 0.061), 0.018, 0.005, "cream",
                   name=_name(b, f"{side}_vial_cap_{i}"))
    if side == "left":
        # The source's red rack is visually distinctive. Keep all tubes inert.
        b.box(base, (0.045, -0.16, bench_z + 0.007), (0.10, 0.052, 0.007),
              "guard_red", name=_name(b, "left_tube_rack_floor"))
        for x in (-0.047, 0.137):
            b.box(base, (x, -0.16, bench_z + 0.038), (0.008, 0.052, 0.024),
                  "guard_red", name=_name(b, f"left_tube_rack_end_{x}"))
        for i in range(7):
            x = -0.028 + i * 0.024
            for j, y in enumerate((-0.181, -0.141)):
                b.cylinder(base, (x, y, bench_z + 0.046), 0.007, 0.032,
                           "glass", name=_name(b, f"left_rack_tube_{i}_{j}"))
        # Rails have 0.3 mm tube clearance; every tube rests on the rack floor.
        # Their ends touch the end supports without slicing tube interiors.
        for y, width in ((-.1903, .002), (-.161, .0127), (-.1317, .002)):
            b.box(base, (.045, y, bench_z + .058), (.084, width, .004),
                  "guard_red", name=_name(b, f"left_tube_rack_rail_{y}"))
        # Shelves stand behind the existing post plane on attached cantilevers.
        # They need posts x=.16 and .59; no invisible back wall is assumed.
        for z in (1.66, 1.99):
            b.box(base, (.375, rack_y + .063, z), (.240, .037, .005), "bright",
                  name=_name(b, f"left_rear_ledge_{z}"))
            for x in (.16, .59):
                key = f"left_ledge_mount_{x}_{z}"
                _post_collar(b, base, key, x, rack_y, z - .011)
                b.rod(base, (x, rack_y + .023, z - .011),
                      (x, rack_y + .09, z - .011), .006, "metal",
                      name=_name(b, key + "_arm"))
    else:
        b.box(base, (0.56, 0.20, bench_z + 0.012), (0.17, 0.11, 0.012),
              "cream", name=_name(b, "right_supply_tray"))


def add_supported_fixed_assembly(b, base, *, axis=(-.28, -.005), bench_z=.934,
                                 rack_y=.30, key="main", side=None,
                                 existing_posts=False, condenser=True):
    """Convenience API: grounded rack, fixed glassware, optional supported props.

    side=None adds the minimal main post and glassware only. side='left' also
    grounds the .16/.59 shelf posts; side='right' adds bench-supported supplies.
    existing_posts=True grounds matching supplied production posts instead of
    duplicating them. All added posts use the documented production endpoints.
    """
    posts = {axis[0]}
    if side == "left":
        posts.update((.16, .59))
    rack = add_grounded_rack(b, base, bench_z=bench_z, rack_y=rack_y,
                             post_xs=tuple(sorted(posts)), existing_posts=existing_posts)
    apparatus = add_fixed_glassware(b, base, axis=axis, bench_z=bench_z,
                                    rack_y=rack_y, key=key, condenser=condenser)
    if side is not None:
        add_asymmetric_props(b, base, side=side, bench_z=bench_z, rack_y=rack_y)
    return {"rack": rack, "apparatus": apparatus}


def clearances(min_height=0.105, max_height=0.235):
    """Analytic authored clearances; integrated dynamics must verify them."""
    bulb_bottom = BULB_HEIGHT - BULB_RADIUS
    return {
        "minimum_floor_to_bulb_m": bulb_bottom - (max_height + BATH_FLOOR),
        "lowered_rim_to_bulb_bottom_m": bulb_bottom - (min_height + BATH_RIM),
        "minimum_radial_bath_to_bulb_m": BATH_INNER - BULB_RADIUS,
        "upper_rim_relative_to_bulb_center_m": max_height + BATH_RIM - BULB_HEIGHT,
        "dimensions_basis": "Authored conservative working clearance, not measured or certified",
    }


def refine_bearings(b, mount):
    meshes = {mesh.get("name"): mesh for mesh in b.root.findall("asset/mesh")}
    groups = defaultdict(list)
    parents = {child: parent for parent in mount.iter() for child in parent}
    for geom in mount.iter("geom"):
        match = re.fullmatch(r"(.+)_([0-9]+)", geom.get("name", ""))
        if geom.get("type") == "mesh" and match:
            groups[match[1]].append((int(match[2]), geom))
    refined = []
    for name, pieces in groups.items():
        if {number for number, _ in pieces} != set(range(12)):
            continue
        geoms = [geom for _, geom in sorted(pieces)]
        if len({id(parents[g]) for g in geoms}) != 1:
            raise ValueError("A bearing's sectors must share their rigid body")
        vertices = np.concatenate([
            np.fromstring(meshes[g.get("mesh")].get("vertex"), sep=" ").reshape(-1, 3)
            for g in geoms
        ])
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        spans = np.ptp(vertices, axis=0)
        first_sector = np.fromstring(meshes[geoms[0].get("mesh")].get("vertex"),
                                     sep=" ").reshape(-1, 3)
        # A long nut can be longer than its diameter. Its extrusion axis is
        # defined by its two planar four-vertex ends, never its shortest span.
        axes = np.flatnonzero((np.ptp(first_sector[:4], axis=0) < 1e-12)
                              & (np.ptp(first_sector[4:], axis=0) < 1e-12))
        if len(axes) != 1 or int(axes[0]) not in (0, 1):
            raise ValueError("Expected one X- or Y-axis bearing extrusion")
        axis = int(axes[0])
        radii = np.linalg.norm(np.delete(vertices - center, axis, axis=1), axis=1)
        parent = parents[geoms[0]]
        old_children = set(parent)
        annular_mount(b, parent, name, (0, 0, 0), float(radii.min()),
                      float(radii.max()), float(spans[axis] / 2), axis="x",
                      material="metal", namespace="jack_presentation")
        visual = next(child for child in parent if child not in old_children)
        visual.set("mass", "0")
        visual.set("pos", " ".join(f"{value:.15g}" for value in center))
        if axis == 1:
            visual.set("euler", f"0 0 {math.pi / 2:.15g}")
        for geom in geoms:
            geom.set("group", "3")
        refined.append(name)
    return refined


def separate_housing_caps(root, payload):
    """Give planar caps their own vertices so their normals remain planar."""
    assets = {item.get("name"): item for item in root.findall("asset/mesh")}
    changed = []
    for geom in payload.findall("geom"):
        name = geom.get("mesh", "")
        if "housing_mesh" not in name:
            continue
        mesh = assets[name]
        vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3).tolist()
        original = np.asarray(vertices)
        faces = np.fromstring(mesh.get("face"), dtype=int, sep=" ").reshape(-1, 3)
        maps = {}
        for face in faces:
            zs = original[face, 2]
            if np.ptp(zs) > 1e-12:
                continue
            z = float(zs[0])
            if z not in (float(original[:, 2].min()), float(original[:, 2].max())):
                continue
            for index, old in enumerate(face.copy()):
                key = (z, int(old))
                if key not in maps:
                    maps[key] = len(vertices)
                    vertices.append(original[old].tolist())
                face[index] = maps[key]
        mesh.set("vertex", " ".join(f"{v:.15g}" for point in vertices for v in point))
        mesh.set("face", " ".join(str(v) for face in faces for v in face))
        changed.append(name)
    return changed


def refine_assembly_presentation(b, mount, payload, *, key="main"):
    proxy = payload.find(f"geom[@name='{_name(b, key + '_hotplate_body')}']")
    if proxy is None or proxy.get("contype") != "1":
        raise ValueError("Expected the enabled conservative hotplate body collider")
    # This coplanar core is needed for physics, but the rounded housing already
    # supplies the actual presentation surface at the same top elevation.
    proxy.set("group", "3")
    return dict(
        seam_free_bearings=refine_bearings(b, mount),
        split_planar_cap_meshes=separate_housing_caps(b.root, payload),
        hidden_duplicate_collider=proxy.get("name"),
        collision_surfaces_preserved=True,
    )
