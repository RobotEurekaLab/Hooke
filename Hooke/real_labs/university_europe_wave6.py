"""Historical reference-informed materials, food-process and NMR workcells.

Original mechanical access fixtures accompany photographed apparatus exteriors.
No vacuum, synthesis, food process, magnetic field or scientific measurement is
implemented. Dimensions are estimates unless explicitly cited otherwise.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

IMPERIAL = "https://blogs.imperial.ac.uk/photography/2018/02/06/featured-lab-the-thin-film-technology-lab/"
NOTTINGHAM = "https://www.nottingham.ac.uk/biosciences/documents/business/food-processing-equipment-and-facilities.pdf"
WARWICK = "https://warwick.ac.uk/fac/sci/physics/research/condensedmatt/nmr/850/1ghzinstallation/"
WHITE = (0.85, 0.85, 0.81, 1)
STEEL = (0.49, 0.53, 0.56, 1)
DARK = (0.045, 0.062, 0.076, 1)


class _Room:
    def __init__(self, world, name):
        self.world, self.name, self.index = world, name, 0

    def box(self, key, pos, half, color, collision=False, **kwargs):
        self.index += 1
        return box(
            self.world,
            f"europe6_{self.name}_{key}_{self.index}",
            half,
            pos,
            color,
            collision=collision,
            **{
                k: numbers(v) if isinstance(v, (tuple, list)) else str(v)
                for k, v in kwargs.items()
            },
        )

    def rod(self, key, start, end, radius, color):
        self.index += 1
        return ET.SubElement(
            self.world,
            "geom",
            name=f"europe6_{self.name}_{key}_{self.index}",
            type="capsule",
            size=str(radius),
            fromto=numbers((*start, *end)),
            rgba=numbers(color),
            contype="0",
            conaffinity="0",
        )


def _hollow_wall(b, parent, center, outer, inner, half, material, segments=48):
    vertices, faces = [], []
    for z in (-half, half):
        for radius in (outer, inner):
            for i in range(segments):
                a = 2 * math.pi * i / segments
                vertices.append(
                    (
                        center[0] + radius * math.cos(a),
                        center[1] + radius * math.sin(a),
                        center[2] + z,
                    )
                )
    for i in range(segments):
        j = (i + 1) % segments
        for a, c, d, e in (
            (i, j, 2 * segments + j, 2 * segments + i),
            (segments + j, segments + i, 3 * segments + i, 3 * segments + j),
            (j, i, segments + i, segments + j),
            (2 * segments + i, 2 * segments + j, 3 * segments + j, 3 * segments + i),
        ):
            faces.extend(((a, c, d), (a, d, e)))
    mesh = b.unique("hollow_wall")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([x for v in vertices for x in v]),
        face=" ".join(str(x) for f in faces for x in f),
    )
    b.geom(parent, "mesh", (), material=material, mesh=mesh)
    for i in range(segments):
        a = 2 * math.pi * i / segments
        r = (outer + inner) / 2
        b.box(
            parent,
            (center[0] + r * math.cos(a), center[1] + r * math.sin(a), center[2]),
            ((outer - inner) / 2, outer * math.tan(math.pi / segments), half),
            material,
            collision=True,
            euler=(0, 0, a),
            rgba=(0.5, 0.5, 0.5, 0),
        )


def _flange(b, parent, center, radius, plane="xy"):
    b.ring(parent, center, radius, 0.015, "bright", plane, 28)
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    for i in range(12):
        p = list(center)
        angle = math.tau * i / 12
        p[axes[0]] += radius * math.cos(angle)
        p[axes[1]] += radius * math.sin(angle)
        b.geom(parent, "sphere", (0.009,), p, "dark")


def _imperial_thinfilm_cluster(b, base, params):
    # Two installed multiport apparatus silhouettes follow the historical photo.
    for x, y in ((-0.95, -0.32), (0.94, 0.38)):
        for dx in (-0.46, 0.46):
            for dy in (-0.37, 0.37):
                b.cylinder(
                    base,
                    (x + dx, y + dy, 0.025),
                    0.045,
                    0.025,
                    "rubber",
                    collision=True,
                )
                b.box(
                    base,
                    (x + dx, y + dy, 0.50),
                    (0.021, 0.021, 0.45),
                    "bright",
                    collision=True,
                )
        b.box(base, (x, y, 0.54), (0.46, 0.37, 0.37), "cream", collision=True)
        b.box(base, (x, y, 0.93), (0.49, 0.40, 0.020), "metal", collision=True)
        for dx in (-0.29, 0.29):
            b.box(
                base, (x + dx, y, 1.055), (0.038, 0.27, 0.105), "metal", collision=True
            )
        b.cylinder(base, (x, y, 1.47), 0.29, 0.30, "metal", collision=True)
        b.geom(base, "ellipsoid", (0.29, 0.29, 0.11), (x, y, 1.775), "metal")
        for z in (1.18, 1.74):
            _flange(b, base, (x, y, z), 0.294)
        for angle in (-math.pi / 2, -math.pi / 6, math.pi / 3, math.pi):
            dx, dy = math.cos(angle), math.sin(angle)
            start = (x + 0.25 * dx, y + 0.25 * dy, 1.48)
            end = (x + 0.47 * dx, y + 0.47 * dy, 1.48)
            b.rod(base, start, end, 0.085, "bright", collision=True)
            b.geom(
                base,
                "cylinder",
                (0.112,),
                material="bright",
                fromto=(*end, end[0] + 0.035 * dx, end[1] + 0.035 * dy, end[2]),
            )
            b.geom(
                base,
                "cylinder",
                (0.075,),
                material="lens",
                fromto=(
                    end[0] + 0.037 * dx,
                    end[1] + 0.037 * dy,
                    end[2],
                    end[0] + 0.040 * dx,
                    end[1] + 0.040 * dy,
                    end[2],
                ),
            )
        for i, dx in enumerate((-0.15, 0.08)):
            b.cylinder(
                base, (x + dx, y, 1.985 + i * 0.10), 0.043, 0.17 + i * 0.10, "bright"
            )
            for z in (1.86, 2.06 + i * 0.18):
                _flange(b, base, (x + dx, y, z), 0.061)
            b.box(base, (x + dx, y, 2.19 + i * 0.12), (0.058, 0.044, 0.04), "dark")
        for dx, dy in ((-0.24, -0.23), (0.23, -0.21), (0.20, 0.22)):
            b.rod(
                base,
                (x + dx, y + dy, 1.20),
                (x + dx * 2.2, y + dy * 2.2, 0.78),
                0.066,
                "bright",
            )
            p = (x + dx * 2.2, y + dy * 2.2, 0.76)
            b.cylinder(base, p, 0.084, 0.027, "bright")
            b.rod(base, (p[0], p[1], 0.73), (p[0], p[1] - 0.1, 0.40), 0.007, "blue")
            b.rod(
                base,
                (p[0], p[1] - 0.1, 0.40),
                (x + 0.41, y - 0.38, 0.16),
                0.007,
                "blue",
            )
        # Visible cable bundles remain passive, fixed routing geometry.
        for i in range(4):
            b.rod(
                base,
                (x + 0.31 + i * 0.013, y + 0.23, 1.78),
                (x + 0.55 + i * 0.016, y + 0.26, 1.01),
                0.004,
                "warning" if i % 2 else "blue",
            )
            b.rod(
                base,
                (x + 0.55 + i * 0.016, y + 0.26, 1.01),
                (x + 0.40, y + 0.30, 0.15 + i * 0.012),
                0.004,
                "warning" if i % 2 else "blue",
            )
    # An explicitly authored dry access station extends the right machine.
    for x in (0.59, 1.30):
        for y in (-0.96, -0.51):
            b.box(base, (x, y, 0.42), (0.025, 0.025, 0.42), "metal", collision=True)
    b.box(base, (0.945, -0.735, 0.865), (0.41, 0.29, 0.025), "metal", collision=True)
    # Open-sided hollow chamber and a real free lid sweep.
    for x in (0.615, 1.275):
        b.box(base, (x, -0.73, 1.08), (0.025, 0.225, 0.18), "metal", collision=True)
    b.box(base, (0.945, -0.485, 1.08), (0.355, 0.02, 0.18), "metal", collision=True)
    b.box(base, (0.945, -0.73, 0.921), (0.355, 0.245, 0.02), "metal", collision=True)
    b.box(base, (0.945, -0.73, 1.261), (0.355, 0.245, 0.020), "metal", collision=True)
    door = b.moving(
        base,
        "access_door",
        (0.583, -0.993, 1.085),
        (0, 0, -1),
        (0, 1.2),
        kind="hinge",
        mass=0.48,
    )
    door.set("euler", numbers((0, 0, -0.72)))
    door.find("joint").set("ref", ".72")
    b.box(door, (0.362, 0, 0), (0.348, 0.015, 0.166), "bright", collision=True)
    b.box(door, (0.362, -0.018, 0), (0.238, 0.006, 0.112), "lens")
    b.rod(door, (0.60, -0.06, -0.065), (0.60, -0.06, 0.065), 0.011, "dark")
    for y in (-0.80, -0.65):
        b.rod(base, (0.68, y, 0.97), (1.19, y, 0.97), 0.007, "bright")
    carrier = b.moving(
        base, "coupon_x", (0.94, -0.73, 1.015), (1, 0, 0), (-0.12, 0.12), mass=0.25
    )
    b.box(carrier, (0, 0, 0), (0.09, 0.09, 0.024), "metal", collision=True)
    b.box(carrier, (0, 0, 0.032), (0.049, 0.045, 0.007), "copper", collision=True)
    b.box(carrier, (0, 0, 0.042), (0.036, 0.034, 0.003), "lens")
    for x in (-0.063, 0.063):
        b.box(carrier, (x, 0, 0.039), (0.010, 0.043, 0.012), "dark", collision=True)
    b.site(carrier, "inert_thinfilm_coupon", (0, 0, 0.042), size=0.004)
    b.metadata["capabilities"] = [
        "Mechanical access-door positioning",
        "Dry retained-coupon translation",
    ]
    b.metadata["limitations"].append(
        "Historical 2018 Bessemer photograph, not a current White City floorplan. Main vessels/ports/cables and paired machine arrangement are source-informed; dry access chamber, coupon mechanism, all dimensions and travel are authored estimates. No vacuum, thin-film growth, plasma, heating, chemical delivery, optical measurement or validated deposition process."
    )


def _imperial_bay(world, definition):
    p = _Room(world, "imperial")
    # Overhead service carriers connect to the building, with no hanging loose fixtures.
    for x in (-2.0, 0.0, 2.0):
        p.box("roof_duct", (x, 0.15, 2.97), (0.14, 2.12, 0.11), STEEL)
    for x in (-1.08, 1.20):
        p.rod(
            "green_service",
            (x, -2.15, 2.87),
            (x, 2.51, 2.87),
            0.030,
            (0.10, 0.43, 0.16, 1),
        )
        for y in (-1.50, 1.6):
            p.rod("service_bracket", (x, y, 2.87), (x, y, 3.18), 0.013, STEEL)
    p.box("rear_desk", (0, 2.13, 0.83), (1.04, 0.35, 0.035), WHITE, True)
    for x in (-0.90, 0.90):
        for y in (1.86, 2.4):
            p.box("desk_leg", (x, y, 0.4), (0.028, 0.028, 0.4), STEEL, True)
    for x in (-0.50, 0.47):
        p.box("monitor_foot", (x, 2.16, 0.887), (0.15, 0.10, 0.018), DARK)
        p.rod("monitor_post", (x, 2.16, 0.90), (x, 2.16, 1.17), 0.018, DARK)
        p.box("monitor", (x, 2.16, 1.29), (0.27, 0.026, 0.17), DARK)
    for x in (-2.40, 2.39):
        p.box("electronics_rack", (x, 1.39, 0.87), (0.24, 0.34, 0.82), DARK, True)
        for dx in (-0.17, 0.17):
            for dy in (-0.25, 0.25):
                p.box(
                    "rack_foot",
                    (x + dx, 1.39 + dy, 0.025),
                    (0.035, 0.035, 0.025),
                    DARK,
                    True,
                )
        for i in range(7):
            z = 0.27 + i * 0.18
            p.box("rack_panel", (x, 1.042, z), (0.214, 0.009, 0.078), STEEL)
            p.box(
                "rack_display",
                (x - 0.07, 1.031, z),
                (0.052, 0.005, 0.02),
                (0.08, 0.23, 0.33, 1),
            )

    for geom in world.findall("geom"):
        name = geom.get("name", "")
        if any(
            name.startswith("europe6_imperial_" + key)
            for key in ("roof_duct", "green_service", "service_bracket")
        ):
            geom.set("name", "arch_roof_" + name)


def _pan_shell(b, parent):
    # Original smooth lathed bowl; the published 25 L value is working capacity,
    # not a surveyed exterior diameter. Conservative ring proxies remain hollow.
    profile = [
        (-0.22, 0.10, 0.075),
        (-0.19, 0.18, 0.153),
        (-0.13, 0.23, 0.203),
        (-0.04, 0.27, 0.243),
        (0.19, 0.27, 0.243),
    ]
    n = 48
    vertices = []
    faces = []
    for z, outer, inner in profile:
        for radius in (outer, inner):
            for i in range(n):
                a = math.tau * i / n
                vertices.append((radius * math.cos(a), radius * math.sin(a), z))
    for k in range(len(profile) - 1):
        for side in (0, 1):
            for i in range(n):
                j = (i + 1) % n
                a = 2 * n * k + side * n + i
                c = 2 * n * k + side * n + j
                d = c + 2 * n
                e = a + 2 * n
                faces.extend(
                    ((a, c, d), (a, d, e)) if side == 0 else ((a, d, c), (a, e, d))
                )
    for k in (0, len(profile) - 1):
        o = 2 * n * k
        for i in range(n):
            j = (i + 1) % n
            faces.extend(((o + i, o + n + i, o + n + j), (o + i, o + n + j, o + j)))
    mesh = b.unique("smooth_jacketed_bowl")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for p in vertices for v in p]),
        face=" ".join(str(v) for f in faces for v in f),
    )
    b.geom(parent, "mesh", (), material="bright", mesh=mesh)
    for (z0, ro0, ri0), (z1, ro1, ri1) in zip(profile, profile[1:]):
        outer = max(ro0, ro1)
        inner = min(ri0, ri1)
        r = (outer + inner) / 2
        for i in range(32):
            a = math.tau * i / 32
            b.box(
                parent,
                (r * math.cos(a), r * math.sin(a), (z0 + z1) / 2),
                ((outer - inner) / 2, outer * math.tan(math.pi / 32), (z1 - z0) / 2),
                "metal",
                euler=(0, 0, a),
                rgba=(0.5, 0.5, 0.5, 0),
                collision=True,
            )
    b.cylinder(parent, (0, 0, -0.23), 0.098, 0.014, "metal", collision=True)
    b.ring(parent, (0, 0, 0.194), 0.271, 0.009, "bright")


def _nottingham_pan(b, base, params):
    # White braced mobile trunnion frame from the equipment photograph.
    for x in (-0.355, 0.355):
        b.box(base, (x, 0, 0.118), (0.052, 0.42, 0.042), "cream", collision=True)
        for y in (-0.35, 0.35):
            b.cylinder(
                base,
                (x, y, 0.044),
                0.044,
                0.021,
                "rubber",
                euler=(0, math.pi / 2, 0),
                collision=True,
            )
            b.box(base, (x, y, 0.09), (0.023, 0.025, 0.023), "metal")
        b.box(base, (x, 0.07, 0.515), (0.046, 0.09, 0.355), "cream", collision=True)
        b.rod(base, (x, -0.31, 0.15), (x, 0.04, 0.81), 0.036, "cream", collision=True)
        b.cylinder(
            base,
            (x, 0, 0.93),
            0.085,
            0.047,
            "cream",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
    b.box(base, (0, 0.24, 0.22), (0.35, 0.033, 0.026), "cream", collision=True)
    b.box(base, (0, -0.30, 0.535), (0.30, 0.23, 0.017), "metal", collision=True)
    for i in range(17):
        b.rod(
            base,
            (-0.28 + i * 0.035, -0.51, 0.555),
            (-0.28 + i * 0.035, -0.09, 0.555),
            0.004,
            "bright",
        )
    pan = b.moving(
        base,
        "pan_tilt",
        (0, 0, 0.93),
        (1, 0, 0),
        (0, 0.40),
        kind="hinge",
        mass=1.8,
        kp=420,
    )
    _pan_shell(b, pan)
    b.rod(pan, (-0.35, 0, 0), (-0.27, 0, 0), 0.026, "metal")
    b.rod(pan, (0.27, 0, 0), (0.35, 0, 0), 0.026, "metal")
    # Source-like long manual tilt handles mounted outside the bowl.
    for x in (-0.43, 0.43):
        b.rod(pan, (x, 0, -0.03), (x, 0.08, 0.41), 0.016, "dark")
        b.rod(pan, ((-0.35 if x < 0 else 0.35), 0, 0), (x, 0, 0), 0.025, "copper")
    b.box(pan, (0, 0, -0.10), (0.063, 0.053, 0.022), "dark", collision=True)
    b.box(pan, (0, 0, -0.070), (0.047, 0.037, 0.008), "cream", collision=True)
    b.rod(pan, (0, 0, -0.215), (0, 0, -0.122), 0.025, "metal", collision=True)
    b.site(pan, "inert_pan_coupon", (0, 0, -0.070), size=0.006)
    lid = b.moving(
        pan,
        "inspection_cover",
        (0, 0.294, 0.225),
        (-1, 0, 0),
        (0, 1.20),
        kind="hinge",
        mass=0.4,
    )
    lid.set("euler", numbers((-0.82, 0, 0)))
    lid.find("joint").set("ref", ".82")
    b.cylinder(lid, (0, -0.294, 0), 0.279, 0.012, "metal", collision=True)
    for x in (-0.06, 0.06):
        b.rod(lid, (x, -0.294, 0.014), (x, -0.294, 0.065), 0.007, "dark")
    b.rod(lid, (-0.06, -0.294, 0.065), (0.06, -0.294, 0.065), 0.008, "dark")
    b.metadata["capabilities"] = [
        "Unloaded pan trunnion tilt",
        "Dry mechanical cover access",
    ]
    b.metadata["limitations"].append(
        "Historical official facilities brochure circa 2013 supplies hall and jacketed-pan photographs and approximately 25 L working capacity. Pan exterior dimensions, internal retained calibration coupon, cover kinematics and placement in the selected hall are estimates. No steam, heating, food production, mixing, fermentation, contamination control, liquid spill or food-quality process."
    )


def _nottingham_hall(world, definition):
    p = _Room(world, "nottingham")
    # Long parallel stainless work banks and wheeled foreground tables preserve
    # the hall photo, without merging the separate quality-control room below it.
    for y in (0.45, 1.82):
        p.box("steel_bank", (0, y, 0.96), (2.45, 0.30, 0.025), STEEL, True)
        for x in (-2.27, -0.76, 0.76, 2.27):
            for dy in (-0.24, 0.24):
                p.box(
                    "bank_leg", (x, y + dy, 0.468), (0.026, 0.026, 0.468), STEEL, True
                )
        for x in (-2.43, -1.62, -0.81, 0, 0.81, 1.62, 2.43):
            p.rod(
                "mesh_guard_upright",
                (x, y + 0.30, 0.98),
                (x, y + 0.30, 1.17),
                0.008,
                STEEL,
            )
        for z in (1.005, 1.05, 1.095, 1.14):
            p.rod("guard_wire", (-2.43, y + 0.30, z), (2.43, y + 0.30, z), 0.003, STEEL)
    for x in (-2.25, -0.92):
        p.box("mobile_table", (x, -1.40, 0.84), (0.57, 0.34, 0.026), WHITE, True)
        p.box("lower_shelf", (x, -1.40, 0.22), (0.54, 0.30, 0.018), WHITE, True)
        for dx in (-0.51, 0.51):
            for dy in (-0.28, 0.28):
                p.box(
                    "trolley_leg",
                    (x + dx, -1.40 + dy, 0.43),
                    (0.022, 0.022, 0.36),
                    STEEL,
                    True,
                )
                p.rod(
                    "castor",
                    (x + dx - 0.019, -1.40 + dy, 0.040),
                    (x + dx + 0.019, -1.40 + dy, 0.040),
                    0.040,
                    DARK,
                )
    p.box("left_sink_unit", (-3.39, 0.93, 0.50), (0.33, 0.71, 0.50), WHITE, True)
    p.box("sink_rim", (-3.39, 0.93, 1.035), (0.35, 0.73, 0.020), STEEL)
    p.box("sink_bowl", (-3.39, 0.90, 1.057), (0.23, 0.35, 0.003), DARK)
    p.rod("tap_riser", (-3.60, 1.09, 1.06), (-3.60, 1.09, 1.39), 0.014, STEEL)
    p.rod("tap_spout", (-3.60, 1.09, 1.39), (-3.34, 1.09, 1.39), 0.014, STEEL)
    for x in (-1.5, 0, 1.50):
        p.box("rear_oven", (x, 3.04, 0.69), (0.52, 0.34, 0.59), WHITE, True)
        for dx in (-0.42, 0.42):
            p.box("oven_foot", (x + dx, 3.04, 0.05), (0.033, 0.26, 0.05), DARK, True)
        p.box("oven_front", (x, 2.689, 0.80), (0.40, 0.008, 0.35), STEEL)
        p.box("oven_window", (x, 2.678, 0.80), (0.27, 0.005, 0.25), DARK)
        p.rod(
            "oven_handle",
            (x + 0.34, 2.638, 0.65),
            (x + 0.34, 2.638, 0.93),
            0.009,
            STEEL,
        )
    # Slanted glazing is an observed hall feature. The angle and span are estimated.
    for y in (-1.8, -0.6, 0.6, 1.8, 3.0):
        p.rod("sloped_glazing_mullion", (3.75, y, 1.25), (3.11, y, 3.82), 0.035, WHITE)
    for z in (1.25, 2.54, 3.82):
        x = 3.75 - (z - 1.25) * 0.64 / 2.57
        p.rod("glazing_transom", (x, -1.8, z), (x, 3.0, z), 0.035, WHITE)
    for y in (-1.2, 0, 1.2, 2.4):
        p.box(
            "sloped_glass",
            (3.43, y, 2.535),
            (0.010, 0.565, 1.31),
            (0.54, 0.65, 0.70, 0.21),
            euler=(0, -0.244, 0),
        )
    p.rod(
        "wall_service_line",
        (-3.89, 2.60, 3.26),
        (2.65, 2.60, 3.26),
        0.021,
        (0.68, 0.54, 0.28, 1),
    )

    for geom in world.findall("geom"):
        name = geom.get("name", "")
        if any(
            name.startswith("europe6_nottingham_" + key)
            for key in ("sloped_glazing", "glazing_transom", "sloped_glass")
        ):
            geom.set("name", "arch_cutaway_wall_" + name)


def _warwick_nmr_magnet(b, base, params):
    # Three high isolation columns and a cream cryostat silhouette from the
    # 2021 installation photographs. No cryogenic or magnetic physics is active.
    for a in (math.pi / 6, 5 * math.pi / 6, 3 * math.pi / 2):
        x, y = 0.84 * math.cos(a), 0.84 * math.sin(a)
        b.cylinder(base, (x, y, 0.060), 0.205, 0.035, "cream", collision=True)
        for i in range(4):
            t = math.tau * i / 4
            b.cylinder(
                base,
                (x + 0.14 * math.cos(t), y + 0.14 * math.sin(t), 0.015),
                0.021,
                0.015,
                "metal",
                collision=True,
            )
        b.cylinder(base, (x, y, 0.575), 0.135, 0.485, "cream", collision=True)
        b.cylinder(base, (x, y, 1.095), 0.19, 0.035, "metal", collision=True)
        for z in (1.147, 1.192, 1.237, 1.282, 1.327):
            b.cylinder(base, (x, y, z), 0.174, 0.014, "dark", collision=True)
        b.cylinder(base, (x, y, 1.355), 0.205, 0.018, "metal", collision=True)
        b.rod(
            base,
            (x, y, 1.40),
            (0.53 * math.cos(a), 0.53 * math.sin(a), 1.40),
            0.085,
            "metal",
            collision=True,
        )
    _hollow_wall(b, base, (0, 0, 2.12), 0.666, 0.065, 0.746, "cream")
    for z in (1.375, 2.875):
        _hollow_wall(b, base, (0, 0, z), 0.694, 0.062, 0.022, "cream")
    for i in range(24):
        a = math.tau * i / 24
        b.cylinder(
            base,
            (0.684 * math.cos(a), 0.684 * math.sin(a), 1.407),
            0.012,
            0.011,
            "metal",
        )
    b.cylinder(base, (0, 0, 2.95), 0.59, 0.035, "metal")
    for x, y, height in ((-0.35, 0.10, 0.48), (0.23, 0.17, 0.62), (0.02, -0.26, 0.36)):
        b.cylinder(base, (x, y, 2.98 + height / 2), 0.052, height / 2, "bright")
        for z in (3.00, 2.98 + height):
            _flange(b, base, (x, y, z), 0.074)
        b.cylinder(base, (x, y, 3.02 + height), 0.078, 0.025, "metal")
    b.rod(base, (-0.35, 0.10, 3.45), (-0.35, 0.57, 3.45), 0.031, "bright")
    b.rod(base, (-0.35, 0.57, 3.45), (0.58, 0.57, 3.45), 0.031, "bright")
    for x in (-0.29, 0.23):
        b.rod(base, (x, 0.20, 3.30), (x, 0.55, 3.75), 0.029, "metal")
        for j in range(8):
            b.ring(
                base,
                (x, 0.22 + j * 0.042, 3.33 + j * 0.051),
                0.032,
                0.005,
                "bright",
                "xy",
                12,
            )
    # Supported original dry access fixture below the cryostat. The travel stops
    # below the bore; it is not a real cryoprobe/sample installation procedure.
    b.box(base, (0, -0.20, 0.15), (0.28, 0.27, 0.05), "metal", collision=True)
    for x in (-0.23, 0.23):
        for y in (-0.40, 0):
            b.box(base, (x, y, 0.055), (0.037, 0.037, 0.055), "dark", collision=True)
        b.rod(base, (x, -0.20, 0.20), (x, -0.20, 1.12), 0.019, "bright", collision=True)
    lift = b.moving(
        base, "inspection_lift", (0, -0.20, 0.60), (0, 0, 1), (0, 0.20), mass=0.45
    )
    b.box(lift, (0, 0, 0), (0.205, 0.13, 0.028), "metal", collision=True)
    b.rod(lift, (0, 0, 0.029), (0, 0, 0.099), 0.024, "metal", collision=True)
    holder = b.moving(
        lift,
        "holder_rotation",
        (0, 0, 0.123),
        (0, 0, 1),
        (-0.35, 0.35),
        kind="hinge",
        mass=0.15,
    )
    b.cylinder(holder, (0, 0, 0), 0.078, 0.025, "dark", collision=True)
    b.cylinder(holder, (0, 0, 0.085), 0.026, 0.060, "cream", collision=True)
    b.box(holder, (0.029, 0, 0.079), (0.007, 0.010, 0.027), "blue")
    for x in (-0.043, 0.043):
        b.box(holder, (x, 0, 0.047), (0.010, 0.032, 0.028), "metal", collision=True)
    b.site(holder, "inert_nmr_dummy", (0, 0, 0.085), size=0.004)
    # Source-like pump cart remains a passive exterior fixture.
    for x in (1.11, 1.49):
        for y in (0.13, 0.57):
            b.cylinder(
                base,
                (x, y, 0.04),
                0.04,
                0.022,
                "rubber",
                euler=(0, math.pi / 2, 0),
                collision=True,
            )
    b.box(base, (1.30, 0.35, 0.105), (0.25, 0.28, 0.030), "blue", collision=True)
    b.box(base, (1.30, 0.35, 0.245), (0.19, 0.21, 0.105), "cream", collision=True)
    b.cylinder(
        base, (1.30, 0.10, 0.245), 0.105, 0.045, "metal", euler=(math.pi / 2, 0, 0)
    )
    for x in (1.17, 1.23, 1.29, 1.35, 1.41):
        b.box(base, (x, 0.06, 0.245), (0.008, 0.006, 0.075), "dark")
    b.rod(base, (1.12, 0.36, 0.33), (0.80, 0.36, 0.61), 0.026, "rubber")
    b.rod(base, (0.80, 0.36, 0.61), (0.51, 0.30, 1.38), 0.026, "rubber")
    b.metadata["capabilities"] = [
        "Dry under-cryostat inspection-carrier lift",
        "Retained inert dummy-holder rotation",
    ]
    b.metadata["limitations"].append(
        "Warwick 1 GHz installation photographs establish the tall cream cryostat, three isolation columns, service necks and pump cart. Selected room and dimensions are estimates. Temporary scaffolds, crates and protective covers are not presented as permanent laboratory furniture. The under-cryostat dry carrier is an original mechanical surrogate and stops below the bore. No magnetic field, RF pulse, NMR spectrum, cryogenics, vacuum, ionising radiation or scientific measurement."
    )


def _warwick_room(world, definition):
    p = _Room(world, "warwick")
    for x in (-1.87, 1.66):
        p.box(
            "perimeter_worktop",
            (x, 2.44, 0.83),
            (0.83, 0.34, 0.035),
            (0.63, 0.47, 0.28, 1),
            True,
        )
        for dx in (-0.71, 0.71):
            for y in (2.16, 2.71):
                p.box(
                    "timber_leg",
                    (x + dx, y, 0.40),
                    (0.035, 0.035, 0.40),
                    (0.62, 0.45, 0.25, 1),
                    True,
                )
        p.box(
            "low_shelf",
            (x, 2.44, 0.25),
            (0.78, 0.30, 0.026),
            (0.63, 0.47, 0.28, 1),
            True,
        )
    p.box("side_control_cabinet", (-2.38, 0.91, 0.91), (0.25, 0.33, 0.86), WHITE, True)
    for x in (-2.55, -2.21):
        for y in (0.68, 1.14):
            p.box("cabinet_foot", (x, y, 0.025), (0.036, 0.036, 0.025), DARK, True)
    p.box("cabinet_display", (-2.38, 0.57, 1.27), (0.18, 0.009, 0.14), DARK)
    p.rod(
        "wall_red_service",
        (-2.90, 2.83, 3.86),
        (2.9, 2.83, 3.86),
        0.016,
        (0.6, 0.07, 0.05, 1),
    )
    p.rod(
        "service_drop",
        (2.64, 2.83, 3.86),
        (2.64, 2.83, 0.75),
        0.016,
        (0.6, 0.07, 0.05, 1),
    )
    p.rod("overhead_duct", (-2.44, 1.55, 4.14), (2.58, 1.55, 4.14), 0.10, STEEL).set(
        "name", "arch_roof_warwick_vent"
    )
    for x in (-2.0, 2.0):
        p.rod("vent_bracket", (x, 1.55, 4.14), (x, 1.55, 4.40), 0.016, STEEL).set(
            "name", f"arch_roof_warwick_vent_support_{x}"
        )


BUILDERS = {
    "imperial_historical_thinfilm_cluster": _imperial_thinfilm_cluster,
    "nottingham_jacketed_pan_fixture": _nottingham_pan,
    "warwick_nmr_cryostat_fixture": _warwick_nmr_magnet,
}
SOURCES = {
    "imperial_historical_thinfilm_cluster": dict(
        reference="Imperial College 2018 Thin Film Technology Lab installed photograph",
        url=IMPERIAL,
        dimensions_m=[3.25, 2.33, 2.34],
        dimension_basis="Estimated relative to standing operator and floor cabinets; original dry access fixture, no manufacturer CAD",
    ),
    "nottingham_jacketed_pan_fixture": dict(
        reference="Nottingham Food Processing Facilities brochure, hall p4 and jacketed-pan p9",
        url=NOTTINGHAM,
        dimensions_m=[0.92, 1.12, 1.61],
        dimension_basis="Approximately 25 L published working capacity; exterior dimensions, lid and retained inert coupon mechanics estimated",
    ),
    "warwick_nmr_cryostat_fixture": dict(
        reference="Warwick 1 GHz NMR official 2021 installation gallery",
        url=WARWICK,
        dimensions_m=[2.60, 2.10, 3.81],
        dimension_basis="Human-scale photographic estimate; published 1 GHz designation is not a dimensional or field simulation claim; original dry inspection carriage",
    ),
}
SAMPLE_INTERFACES = {
    "imperial_historical_thinfilm_cluster": (
        "inert_thinfilm_coupon",
        (0.098, 0.090, 0.020),
        "clamped",
        "Visible inert solid coupon retained on the authored dry access carrier",
    ),
    "nottingham_jacketed_pan_fixture": (
        "inert_pan_coupon",
        (0.094, 0.074, 0.016),
        "clamped",
        "Visible inert inspection coupon retained inside a dry, unloaded representative pan",
    ),
    "warwick_nmr_cryostat_fixture": (
        "inert_nmr_dummy",
        (0.052, 0.052, 0.12),
        "clamped",
        "Visible retained original solid dummy on the dry under-cryostat inspection carrier",
    ),
}
FEATURES = {
    "imperial_thinfilm_bessemer": _imperial_bay,
    "nottingham_food_process_hall": _nottingham_hall,
    "warwick_highfield_nmr_1ghz": _warwick_room,
}
