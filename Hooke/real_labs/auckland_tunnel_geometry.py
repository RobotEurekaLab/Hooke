"""Open bearings for an authored dry calibration fixture in Auckland's tunnel.

The official small photograph supports a clear vertical article and end plate;
these hidden guide and bearing dimensions are explicit engineering estimates.
Original slide/yaw frames, masses, drives, foil dimensions and sites are retained.
Ideal joints carry the load across declared running gaps; no aerodynamic model.
"""
import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

PREFIX = "primary_apparatus__tunnel_fixture_"


def _box(b, body, key, pos, half, material="metal"):
    return b.box(body, pos, half, material, collision=True, density=0,
                 name=PREFIX + key)


def _tube(b, body, key, center, inner, outer, half, axis="z", material="metal"):
    """Watertight open sleeve; invisible enabled convex sectors retain the bore."""
    count = 48
    def point(u, v, w):
        p = (w, u, v) if axis == "x" else (u, v, w)
        return tuple(a + c for a, c in zip(p, center))
    vertices = [point(r * math.cos(math.tau * i / count),
                      r * math.sin(math.tau * i / count), z)
                for z in (-half, half) for r in (inner, outer)
                for i in range(count)]
    faces = []
    for i in range(count):
        j = (i + 1) % count
        for a, c, d, e in (
            (i, j, count+j, count+i),
            (2*count+i, 3*count+i, 3*count+j, 2*count+j),
            (i, 2*count+i, 2*count+j, j),
            (count+i, count+j, 3*count+j, 3*count+i),
        ):
            faces.extend(((a, c, d), (a, d, e)))
    mesh = PREFIX + key + "_mesh"
    ET.SubElement(b.root.find("asset"), "mesh", name=mesh,
                  vertex=numbers([v for p in vertices for v in p]),
                  face=" ".join(str(v) for f in faces for v in f))
    b.geom(body, "mesh", (), material=material, mesh=mesh, density=0,
           name=PREFIX + key + "_visual")
    radius, thick = (inner + outer) / 2, (outer - inner) / 2
    for i in range(count):
        angle = math.tau * (i + .5) / count
        pos = point(radius * math.cos(angle), radius * math.sin(angle), 0)
        size = ((half, thick, outer * math.tan(math.pi/count)) if axis == "x"
                else (thick, outer * math.tan(math.pi/count), half))
        euler = (angle, 0, 0) if axis == "x" else (0, 0, angle)
        b.box(body, pos, size, material, collision=True, density=0,
              euler=euler, group=3, rgba=(.4, .4, .4, 0),
              name=PREFIX + key + f"_contact_{i}")


def add_fixture(b, base):
    # Neutral clear acrylic separates the specimen and glazing from the shared
    # blue glass role. Synthetic fill emitters are hidden in reflections;
    # physical panels, contact shapes and actual fixture lights remain intact.
    material = b.name + "__mat_clear_acrylic"
    ET.SubElement(b.root.find("asset"), "material", name=material,
                  rgba=".92 .96 .98 .08", specular=".5", shininess=".7")
    b.metadata["inspection_softbox_visible_in_glass"] = False
    for panel in base.findall("geom"):
        if panel.get("name", "").startswith(b.name + "__test_side_"):
            panel.set("material", material)
    # Fill the former 10 mm floor/base gap with explicit visible feet. Lower
    # the estimated base top by 10 mm for the new open sleeve undersides;
    # all moving frames remain at their original authored heights.
    for i, x in enumerate((-.73, .73)):
        for j, y in enumerate((-.40, .40)):
            _box(b, base, f"foot_{i}_{j}", (x, y, .405), (.065, .05, .005), "dark")
    _box(b, base, "base", (0, 0, .440), (.85, .47, .030), "dark")
    for i, y in enumerate((-.39, .39)):
        b.rod(base, (-.75, y, .50), (.75, y, .50), .016, "bright",
              collision=True, density=0, name=PREFIX+f"rail_{i}")
        for j, x in enumerate((-.70, .70)):
            _box(b, base, f"rail_mount_{i}_{j}", (x, y, .487),
                 (.026, .030, .017))
    carriage = b.moving(base, "specimen_x", (0, 0, .53), (1, 0, 0),
                        (-.22, .22), mass=4, kp=1200, force=1600)
    # The old full-width solid plate intersected both rails. A central plate
    # and four crown brackets connect to open sleeves without filling the holes.
    _box(b, carriage, "carriage_plate", (0, 0, .010), (.46, .34, .025))
    for i, y in enumerate((-.39, .39)):
        for j, x in enumerate((-.30, .30)):
            _tube(b, carriage, f"bearing_{i}_{j}", (x, y, -.030),
                  .0166, .024, .060, axis="x", material="dark")
            _box(b, carriage, f"bearing_crown_{i}_{j}",
                 (x, math.copysign(.373, y), .008), (.062, .039, .017))
    # A fixed shaft is mounted to the carriage; the original rotary plate now
    # has a real open bore. 0.6 mm radial and 1 mm thrust clearances are authored,
    # not bearing metrology. The original ideal hinge/inertia remain unchanged.
    b.cylinder(carriage, (0, 0, .070), .050, .035, "bright", collision=True,
               density=0, name=PREFIX+"fixed_rotary_shaft")
    b.cylinder(carriage, (0, 0, .037), .110, .002, "metal", collision=True,
               density=0, name=PREFIX+"thrust_support")
    turn = b.moving(carriage, "specimen_yaw", (0, 0, .075), (0, 0, 1),
                    (-.30, .30), kind="hinge", mass=2, kp=400, force=500)
    _tube(b, turn, "rotary_plate", (0, 0, 0), .0506, .420, .035)
    b.cylinder(turn, (0, 0, .12), .085, .085, "dark", collision=True,
               density=0, name=PREFIX+"foil_mount")
    b.box(turn, (0, 0, .92), (.045, .28, .74), "clear_acrylic", collision=True,
          density=0, name="primary_apparatus__clear_foil_30")
    b.box(turn, (.047, 0, .92), (.005, .02, .74), "bright",
          name=PREFIX+"foil_edge")
    # The previous end plate was 8 mm above the foil. This clamp joins both
    # while preserving the end plate and foil's original poses/dimensions.
    _box(b, turn, "upper_foil_clamp", (0, 0, 1.664), (.05, .29, .004))
    b.cylinder(turn, (0, 0, 1.68), .48, .012, "clear_acrylic", collision=True,
               density=0, name=PREFIX+"upper_plate")
    b.ring(turn, (0, 0, 1.68), .48, .005, "bright")
    b.site(turn, "clear_specimen", (0, 0, .92))
    b.metadata["fixture_support"] = {
        "revision": 1,
        "source_status": "Clear article/end plate observed; bearings/mounts are authored estimates",
        "linear_running_clearance_m": .0006,
        "rotary_running_clearance_m": .0006,
        "thrust_running_clearance_m": .001,
        "full_slide_travel_m": .44,
        "bearing_length_m": .12,
        "ideal_joint_load_support": True,
        "scope": "Retained inert fixture positioning, not aerodynamic forces or real bearing performance",
        "appearance": "Neutral clear-acrylic inspection finish estimated from the local photograph; optical calibration and coating identity are not established",
    }
