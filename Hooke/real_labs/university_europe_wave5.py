"""Independent reference-informed optics, chamber and pilot-plant workcells.

Mechanical access/positioning is explicitly separated from experimental process
models. Historical photographs are identified; no OEM CAD or textures are copied.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

PSL = "https://lira.observatoiredeparis.psl.eu/Photos-Galerie"
LUND = "https://www.nano.lu.se/labs/lund-nano-characterization-labs/aerosollab"
POLIMI = (
    "https://www.aero.polimi.it/it/laboratori-sperimentali/laboratorio-aerodinamico"
)
SHEFFIELD = "https://www.sheffield.ac.uk/engineering/diamond-engineering/our-facilities/pilot-plant"

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
            f"europe5_{self.name}_{key}_{self.index}",
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
            name=f"europe5_{self.name}_{key}_{self.index}",
            type="capsule",
            size=str(radius),
            fromto=numbers((*start, *end)),
            rgba=numbers(color),
            contype="0",
            conaffinity="0",
        )


def _round_optic(b, base, x, y, z, radius=0.050):
    b.box(base, (x, y, 0.942), (0.059, 0.053, 0.013), "bright")
    b.rod(base, (x, y, 0.955), (x, y, z - radius - 0.01), 0.013, "bright")
    b.box(base, (x, y, z - radius - 0.003), (0.030, 0.016, 0.017), "metal")
    b.ring(base, (x, y, z), radius + 0.007, 0.008, "metal", plane="xz")
    b.cylinder(base, (x, y, z), radius, 0.005, "lens", euler=(math.pi / 2, 0, 0))
    b.box(base, (x + radius + 0.016, y, z), (0.012, 0.016, 0.010), "dark")


def _sesame_relay_bench(b, base, params):
    """Three-metre historical optical bench with two dry alignment mechanisms."""
    for x in (-1.22, 1.22):
        for y in (-0.48, 0.48):
            b.cylinder(base, (x, y, 0.06), 0.12, 0.06, "rubber", collision=True)
            b.cylinder(base, (x, y, 0.405), 0.085, 0.345, "metal", collision=True)
    b.box(base, (0, 0, 0.845), (1.50, 0.75, 0.078), "dark", collision=True)
    b.box(base, (0, 0, 0.927), (1.494, 0.744, 0.004), "bright")
    # Decimated visual hole marks; the documented real grid is 25 mm.
    for i in range(39):
        for j in range(19):
            b.cylinder(
                base,
                (-1.425 + i * 0.075, -0.675 + j * 0.075, 0.9325),
                0.0027,
                0.0006,
                "dark",
            )
    for x, y, r in [
        (-1.12, -0.38, 0.05),
        (-0.69, 0.07, 0.045),
        (-0.23, 0.29, 0.06),
        (0.14, 0.31, 0.042),
        (0.53, 0.31, 0.072),
        (0.98, 0.31, 0.067),
        (1.20, -0.07, 0.05),
        (0.76, -0.27, 0.057),
        (0.36, -0.28, 0.038),
        (-0.08, -0.34, 0.048),
        (-0.79, 0.46, 0.051),
        (-1.21, 0.38, 0.045),
    ]:
        _round_optic(b, base, x, y, 1.081, r)
    # Original mechanical mirror stage; optical transformation is not evaluated.
    b.box(base, (-0.46, -0.37, 0.958), (0.095, 0.085, 0.021), "dark", collision=True)
    b.cylinder(base, (-0.46, -0.37, 1.007), 0.034, 0.028, "bright")
    mirror = b.moving(
        base,
        "mirror_yaw",
        (-0.46, -0.37, 1.081),
        (0, 0, 1),
        (-0.15, 0.15),
        kind="hinge",
        mass=0.12,
    )
    b.ring(mirror, (0, 0, 0), 0.061, 0.008, "metal", plane="xz")
    b.cylinder(
        mirror,
        (0, 0, 0),
        0.053,
        0.008,
        "bright",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    b.box(mirror, (0, 0, -0.065), (0.032, 0.02, 0.015), "dark", collision=True)
    b.cylinder(mirror, (0, 0, -0.080), 0.013, 0.021, "bright")
    # Retained calibration coupon is an explicit authored task fixture.
    b.box(base, (0.80, 0.03, 0.953), (0.20, 0.11, 0.020), "dark", collision=True)
    for y in (-0.05, 0.11):
        b.rod(base, (0.61, y, 0.982), (1.0, y, 0.982), 0.006, "bright")
    target = b.moving(
        base, "coupon_x", (0.80, 0.03, 1.012), (1, 0, 0), (-0.045, 0.045), mass=0.17
    )
    b.box(target, (0, 0, 0), (0.09, 0.093, 0.018), "metal", collision=True)
    b.rod(target, (0, 0, 0.018), (0, 0, 0.069), 0.011, "bright", collision=True)
    b.box(target, (0, 0, 0.103), (0.036, 0.007, 0.034), "cream", collision=True)
    b.box(target, (0, -0.009, 0.103), (0.021, 0.001, 0.0015), "dark")
    b.box(target, (0, -0.009, 0.103), (0.0015, 0.001, 0.021), "dark")
    b.site(target, "inert_optical_coupon", (0, 0, 0.103), size=0.004)
    # Rectangular cameras, small mounts and retained cable routes follow the dense bench.
    for x, y in [(-1.08, 0.08), (-0.20, -0.05), (0.17, -0.51), (1.21, 0.49)]:
        b.box(base, (x, y, 0.966), (0.072, 0.062, 0.027), "metal")
        b.rod(base, (x, y, 0.99), (x, y, 1.055), 0.009, "bright")
        b.box(base, (x, y, 1.081), (0.032, 0.037, 0.031), "dark")
        b.cylinder(
            base, (x, y - 0.043, 1.081), 0.018, 0.009, "lens", euler=(math.pi / 2, 0, 0)
        )
        b.rod(base, (x, y + 0.035, 1.07), (x + 0.10, y + 0.09, 0.947), 0.004, "rubber")
    for y in (-0.61, 0.60):
        b.rod(base, (-1.27, y, 0.947), (1.31, y, 0.947), 0.004, "rubber")
    b.box(base, (-0.84, 0.54, 1.01), (0.18, 0.16, 0.075), "guard_red")
    for z in (0.96, 1.00, 1.04):
        b.box(base, (-0.84, 0.376, z), (0.145, 0.004, 0.005), "dark")
    b.metadata["capabilities"] = [
        "Dry kinematic-mirror yaw",
        "Retained inert optical-coupon translation",
    ]
    b.metadata["limitations"].append(
        "Historical SESAME 2009 gallery; official documentation gives a 3 x 1.5 m "
        "table and optical axes 0.15 m above it. Hole marks are visually decimated "
        "from the documented 25 mm grid. Exact optical prescription, stage design "
        "and travel are not reproduced. No adaptive optics, optical propagation, "
        "wavefront reconstruction, camera image or telescope-performance solver."
    )


def _sesame_room(world, definition):
    p = _Room(world, "sesame")
    for x in (-1.74, -0.13, 1.49):
        p.box("rear_desk", (x, 1.89, 0.77), (0.74, 0.36, 0.032), WHITE, True)
        for dx in (-0.60, 0.60):
            for y in (1.64, 2.14):
                p.box("desk_leg", (x + dx, y, 0.37), (0.025, 0.025, 0.37), STEEL, True)
        p.box("monitor_foot", (x, 1.97, 0.82), (0.13, 0.09, 0.017), DARK)
        p.rod("monitor_post", (x, 1.97, 0.83), (x, 1.97, 1.08), 0.022, DARK)
        p.box("monitor", (x, 1.97, 1.20), (0.25, 0.024, 0.16), DARK)
        p.box(
            "screen", (x, 1.941, 1.20), (0.231, 0.002, 0.14), (0.045, 0.075, 0.095, 1)
        )
        p.box("keyboard", (x, 1.66, 0.82), (0.19, 0.065, 0.015), DARK)
    # Curtain is attached to a wall-side rail, with visible support brackets.
    p.rod("curtain_rail", (2.40, -0.40, 2.50), (2.40, 2.40, 2.50), 0.021, STEEL)
    for y in (-0.40, 2.40):
        p.rod("curtain_bracket", (2.40, y, 2.50), (3.09, y, 2.50), 0.014, STEEL)
    for i in range(24):
        y = -0.34 + i * 0.117
        p.box(
            "black_curtain_fold",
            (2.41 + 0.026 * math.sin(i * 1.6), y, 1.40),
            (0.029, 0.066, 1.075),
            DARK,
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


def _lund_dry_chamber(b, base, params):
    """Red characterization exterior with dry, inert inspection mechanisms."""
    for x in (-0.56, 0.56):
        for y in (-0.34, 0.34):
            b.box(base, (x, y, 0.407), (0.027, 0.027, 0.407), "metal", collision=True)
            b.box(base, (x, y, 0.024), (0.036, 0.036, 0.024), "rubber")
    b.box(
        base,
        (0, 0, 0.844),
        (0.69, 0.44, 0.030),
        "cream",
        collision=True,
        rgba=(0.63, 0.48, 0.29, 1),
    )
    b.housing(
        base,
        ((0.882, 0.40, 0.31, 0), (1.108, 0.39, 0.30, 0)),
        radius=0.027,
        material="guard_red",
    )
    b.box(base, (0, 0, 0.996), (0.38, 0.29, 0.107), "guard_red", collision=True)
    for x in (-0.25, 0, 0.25):
        b.box(base, (x, -0.305, 1.017), (0.041, 0.022, 0.035), "dark")
        b.cylinder(
            base, (x, -0.337, 1.019), 0.028, 0.014, "cyan", euler=(math.pi / 2, 0, 0)
        )
    b.rod(base, (-0.29, -0.335, 0.969), (0.28, -0.335, 0.969), 0.008, "cream")
    b.cylinder(base, (0, 0, 1.125), 0.237, 0.022, "metal", collision=True)
    _hollow_wall(b, base, (0, 0, 1.229), 0.224, 0.198, 0.082, "bright")
    b.ring(base, (0, 0, 1.319), 0.222, 0.015, "bright")
    # The authored default is open for inspection; zero is the closed lid position.
    lid = b.moving(
        base,
        "access_lid",
        (0, 0.234, 1.355),
        (-1, 0, 0),
        (0, 1.30),
        kind="hinge",
        mass=0.45,
    )
    lid.set("euler", numbers((-0.85, 0, 0)))
    lid.find("joint").set("ref", ".85")
    b.cylinder(lid, (0, -0.234, 0), 0.215, 0.027, "dark", collision=True)
    b.cylinder(lid, (0, -0.234, 0.039), 0.183, 0.014, "dark")
    for x in (-0.06, 0.06):
        b.rod(lid, (x, -0.234, 0.051), (x, -0.234, 0.094), 0.007, "metal")
    b.rod(lid, (-0.06, -0.234, 0.094), (0.06, -0.234, 0.094), 0.008, "metal")
    for x in (-0.072, 0.072):
        b.box(base, (x, 0.234, 1.322), (0.020, 0.020, 0.035), "metal")
    tray = b.moving(
        base,
        "coupon_index",
        (0, 0, 1.184),
        (0, 0, 1),
        (-0.40, 0.40),
        kind="hinge",
        mass=0.25,
    )
    b.cylinder(tray, (0, 0, 0), 0.174, 0.025, "cream", collision=True)
    b.cylinder(tray, (0, 0, 0.029), 0.091, 0.006, "bright")
    for i in range(16):
        a = 2 * math.pi * i / 16
        b.cylinder(
            tray,
            (0.139 * math.cos(a), 0.139 * math.sin(a), 0.029),
            0.019,
            0.008,
            "metal",
        )
        b.cylinder(
            tray,
            (0.139 * math.cos(a), 0.139 * math.sin(a), 0.039),
            0.009,
            0.003,
            "sample",
        )
    b.box(tray, (-0.034, -0.036, 0.043), (0.028, 0.023, 0.006), "blue", collision=True)
    b.box(tray, (-0.034, -0.036, 0.052), (0.020, 0.015, 0.003), "cream")
    b.site(tray, "inert_characterization_coupon", (-0.034, -0.036, 0.052), size=0.004)
    for i in range(12):
        a = 2 * math.pi * i / 12
        b.cylinder(
            base,
            (0.223 * math.cos(a), 0.223 * math.sin(a), 1.331),
            0.008,
            0.012,
            "metal",
        )
    for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        x, y = 0.248 * math.cos(a), 0.248 * math.sin(a)
        b.box(base, (x, y, 1.223), (0.036, 0.024, 0.047), "dark", euler=(0, 0, a))
    b.cylinder(
        base, (-0.488, 0, 0.996), 0.086, 0.215, "bright", euler=(math.pi / 2, 0, 0)
    )
    b.rod(base, (-0.488, -0.22, 0.996), (-0.30, -0.334, 0.969), 0.009, "cream")
    b.rod(base, (0.24, 0.18, 1.213), (0.44, 0.27, 1.13), 0.008, "orange")
    b.rod(base, (0.44, 0.27, 1.13), (0.47, 0.20, 0.91), 0.008, "orange")
    b.metadata["capabilities"] = [
        "Mechanical cover access for dry inspection",
        "Retained inert coupon indexing",
    ]
    b.metadata["limitations"].append(
        "Photographs establish a red bench housing and clamped circular chamber; "
        "the opening hinge, internal inert coupon tray and motion ranges are authored "
        "mechanical surrogates. Default lid pose is open for inspection. No particle "
        "generation, aerosols, exposure, dispersion, collection physics, biological "
        "process or scientific characterization result is implemented."
    )


def _lund_workstation(world, definition):
    p = _Room(world, "lund")
    # The extra control frame is based on a separate detail, without claiming adjacency.
    p.box("controller_top", (1.01, 0.78, 0.88), (0.34, 0.27, 0.025), STEEL, True)
    for x in (0.73, 1.29):
        for y in (0.57, 0.99):
            p.box(
                "controller_frame_leg",
                (x, y, 0.426),
                (0.018, 0.018, 0.426),
                STEEL,
                True,
            )
    p.box("controller_base", (1.01, 0.78, 0.14), (0.30, 0.25, 0.025), STEEL)
    p.rod("monitor_stand", (1.01, 0.89, 0.91), (1.01, 0.89, 1.18), 0.017, STEEL)
    p.box("monitor", (1.01, 0.89, 1.29), (0.24, 0.025, 0.16), DARK)
    p.box(
        "monitor_screen",
        (1.01, 0.86, 1.29),
        (0.216, 0.002, 0.137),
        (0.05, 0.16, 0.22, 1),
    )
    p.box("control_keyboard", (1.01, 0.58, 0.932), (0.20, 0.066, 0.018), DARK)
    for z in (0.23, 0.43, 0.63):
        p.box("rack_electronics", (1.01, 0.80, z), (0.27, 0.23, 0.075), WHITE)
    p.box("wall_service_panel", (0, 1.717, 1.12), (0.65, 0.033, 0.32), WHITE)
    for x in (-0.42, 0.02, 0.45):
        p.box("service_fixture", (x, 1.676, 1.12), (0.026, 0.012, 0.040), STEEL)


def _polimi_openjet(b, base, params):
    """Representative inert specimen and microphone-positioning station."""
    # This asset also supplies one reusable wedge mesh for the room's wall panels.
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=f"{b.name}__acoustic_wedge",
        vertex="-.18 -.18 0 .18 -.18 0 0 -.18 .22 -.18 .18 0 .18 .18 0 0 .18 .22",
        face="0 2 1 3 4 5 0 1 4 0 4 3 1 2 5 1 5 4 2 0 3 2 3 5",
    )
    # Net 0.7 x 0.5 m opening; outer enclosure/support dimensions are estimates.
    for x in (-0.86, -0.04):
        b.box(base, (x, 1.64, 1.52), (0.055, 0.30, 0.36), "shell", collision=True)
        b.box(base, (x, 1.64, 0.56), (0.052, 0.18, 0.56), "metal", collision=True)
    for z in (1.215, 1.825):
        b.box(base, (-0.45, 1.64, z), (0.355, 0.30, 0.055), "shell", collision=True)
    b.box(base, (-0.45, 1.946, 1.52), (0.35, 0.012, 0.25), "dark")
    # Independent support stand and retained original rigid calibration body.
    b.cylinder(base, (-0.45, 0.36, 0.065), 0.18, 0.065, "dark", collision=True)
    b.cylinder(base, (-0.45, 0.36, 0.644), 0.024, 0.52, "metal", collision=True)
    b.cylinder(base, (-0.45, 0.36, 1.185), 0.060, 0.035, "bright", collision=True)
    specimen = b.moving(
        base,
        "specimen_yaw",
        (-0.45, 0.36, 1.238),
        (0, 0, 1),
        (-0.35, 0.35),
        kind="hinge",
        mass=0.14,
    )
    b.cylinder(specimen, (0, 0, 0), 0.066, 0.017, "metal", collision=True)
    b.rod(specimen, (0, 0, 0.02), (0, 0, 0.10), 0.010, "bright", collision=True)
    b.geom(
        specimen,
        "ellipsoid",
        (0.095, 0.19, 0.034),
        (0, 0, 0.126),
        "cream",
        collision=True,
    )
    b.box(specimen, (0, 0, 0.160), (0.035, 0.09, 0.002), "blue")
    b.site(specimen, "inert_aeroacoustic_model", (0, 0, 0.126), size=0.006)
    for x in (-0.82, 0.90):
        b.cylinder(base, (x, -0.28, 0.04), 0.13, 0.04, "dark", collision=True)
        b.rod(base, (x, -0.28, 0.08), (x, -0.28, 1.66), 0.019, "metal", collision=True)
    for z in (1.56, 1.65):
        b.rod(base, (-0.82, -0.28, z), (0.90, -0.28, z), 0.011, "bright")
    mic = b.moving(
        base, "microphone_x", (0.10, -0.28, 1.606), (1, 0, 0), (-0.23, 0.23), mass=0.12
    )
    b.box(mic, (0, 0, 0), (0.042, 0.045, 0.055), "dark", collision=True)
    b.rod(mic, (0, 0, -0.054), (0, 0, -0.26), 0.007, "metal", collision=True)
    b.geom(mic, "sphere", (0.014,), (0, 0, -0.27), "dark", collision=True)
    # A signal lead runs down a supported post, without simulating acoustic output.
    b.rod(base, (0.92, -0.28, 1.64), (0.92, -0.28, 0.12), 0.005, "rubber")
    b.metadata["capabilities"] = [
        "Retained inert model yaw",
        "Supported microphone-positioning traverse",
    ]
    b.metadata["limitations"].append(
        "The white chamber is separate from Polimi's blue and green tunnels. "
        "Published room size is 4 x 4 x 4 m and useful jet section 0.5 x 0.7 m; "
        "opening orientation, wedge depth, supports, inert model and travel are "
        "authored estimates. No airflow, acoustic propagation, noise spectrum, "
        "microphone signal, aerodynamic force or measurement calibration."
    )


def _polimi_chamber(world, definition):
    item = next(
        e
        for e in definition["equipment"]
        if e["kind"] == "polimi_white_openjet_station"
    )
    mesh = f"{item['id']}__acoustic_wedge"
    # Genuine triangular wedges; right/front and ceiling follow renderer cutaways.
    for wall in ("left", "back", "right", "front", "roof"):
        for i in range(10):
            a = -1.8 + i * 0.4
            for j in range(10):
                c = 0.2 + j * 0.4
                if wall == "left" and -1.65 < a < -0.45 and c < 2.55:
                    continue
                if wall == "back" and -1.13 < a < 0.20 and 0.93 < c < 2.16:
                    continue
                if wall == "left":
                    pos = (-1.995, a, c)
                    angle = (0, math.pi / 2, 0)
                elif wall == "back":
                    pos = (a, 1.995, c)
                    angle = (math.pi / 2, 0, 0)
                elif wall == "right":
                    pos = (1.995, a, c)
                    angle = (0, -math.pi / 2, 0)
                elif wall == "front":
                    pos = (a, -1.995, c)
                    angle = (-math.pi / 2, 0, 0)
                else:
                    pos = (a, -1.8 + j * 0.4, 3.995)
                    angle = (math.pi, 0, 0)
                prefix = (
                    "arch_roof_"
                    if wall == "roof"
                    else (
                        "arch_cutaway_wall_"
                        if wall in ("right", "front")
                        else "europe5_"
                    )
                )
                ET.SubElement(
                    world,
                    "geom",
                    name=f"{prefix}polimi_wedge_{wall}_{i}_{j}",
                    type="mesh",
                    mesh=mesh,
                    pos=numbers(pos),
                    euler=numbers(angle),
                    rgba=numbers(WHITE),
                    contype="1",
                    conaffinity="1",
                )
    p = _Room(world, "polimi")
    # Black absorber panels are mechanically fixed to a wall-side support.
    p.box("absorber_bracket", (1.78, 0.87, 2.54), (0.20, 0.025, 0.025), STEEL)
    p.box("black_absorber", (1.54, 0.87, 2.54), (0.045, 0.42, 0.38), DARK)
    for y in (0.55, 0.66, 0.77, 0.88, 0.99, 1.10, 1.21):
        p.box("absorber_ridge", (1.474, y, 2.54), (0.026, 0.024, 0.34), DARK)


def _dipp_pilot_skid(b, base, params):
    """Stair-supported process-bay exterior with an inert access task."""
    # Raised stainless platform and continuous diagonal stair stringers.
    for x in (-1.47, 1.54):
        for y in (0.53, 1.62):
            b.box(base, (x, y, 0.58), (0.05, 0.05, 0.58), "metal", collision=True)
            b.box(base, (x, y, 0.035), (0.10, 0.10, 0.035), "bright", collision=True)
    b.box(base, (0.02, 1.075, 1.218), (1.59, 0.65, 0.062), "metal", collision=True)
    for i in range(34):
        b.box(base, (-1.47 + i * 0.09, 1.075, 1.282), (0.007, 0.60, 0.002), "dark")
    for x in (0.85, 1.62):
        b.rod(base, (x, -1.77, 0.02), (x, 0.49, 1.23), 0.047, "metal", collision=True)
    for j in range(6):
        y = -1.52 + j * 0.36
        z = 0.105 + j * 0.20
        b.box(base, (1.235, y, z), (0.42, 0.185, 0.09), "metal", collision=True)
        for dx in (-0.29, -0.14, 0, 0.14, 0.29):
            b.rod(
                base,
                (1.235 + dx - 0.027, y - 0.065, z + 0.091),
                (1.235 + dx + 0.027, y + 0.015, z + 0.091),
                0.0035,
                "bright",
            )
    for x in (0.80, 1.67):
        for y, z in ((-1.65, 0.17), (-0.57, 0.77), (0.49, 1.28)):
            b.rod(base, (x, y, z), (x, y, z + 0.86), 0.018, "bright")
        b.rod(base, (x, -1.65, 1.03), (x, 0.49, 2.14), 0.023, "bright")
        b.rod(base, (x, -1.65, 0.62), (x, 0.49, 1.73), 0.015, "bright")
    for x in (-1.51, -0.46, 0.59, 1.59):
        b.rod(base, (x, 1.70, 1.28), (x, 1.70, 2.18), 0.019, "bright")
    for z in (1.72, 2.18):
        b.rod(base, (-1.51, 1.70, z), (1.59, 1.70, z), 0.019, "bright")
    for y in (0.48, 1.68):
        b.rod(base, (-1.53, y, 1.28), (-1.53, y, 2.18), 0.019, "bright")
    for z in (1.72, 2.18):
        b.rod(base, (-1.53, 0.48, z), (-1.53, 1.68, z), 0.019, "bright")
    # Large process housing behind the operator deck: original, unbranded exterior.
    b.housing(
        base,
        ((1.29, 0.79, 0.36, 1.29), (3.05, 0.77, 0.35, 1.29)),
        radius=0.045,
        material="metal",
    )
    b.box(base, (0, 1.29, 2.17), (0.75, 0.34, 0.88), "metal", collision=True)
    b.box(base, (0, 0.939, 2.17), (0.006, 0.008, 0.80), "dark")
    for x in (-0.13, 0.13):
        b.rod(base, (x, 0.91, 1.91), (x, 0.91, 2.14), 0.009, "bright")
    b.cylinder(base, (0.11, 1.18, 3.22), 0.22, 0.17, "bright", collision=True)
    b.ring(base, (0.11, 1.18, 3.395), 0.23, 0.009, "metal")
    # Front cylindrical vessel, with source-like flange and nonfunctional side fittings.
    for x in (-0.73, -0.10):
        b.rod(base, (x, -0.10, 0.05), (x, -0.10, 0.68), 0.039, "metal", collision=True)
        b.cylinder(base, (x, -0.10, 0.035), 0.075, 0.035, "bright", collision=True)
    b.cylinder(base, (-0.42, -0.10, 0.50), 0.31, 0.045, "metal", collision=True)
    b.geom(base, "ellipsoid", (0.385, 0.385, 0.17), (-0.42, -0.10, 0.665), "metal")
    _hollow_wall(b, base, (-0.42, -0.10, 1.092), 0.393, 0.367, 0.42, "metal")
    b.cylinder(base, (-0.42, -0.10, 0.695), 0.367, 0.019, "metal", collision=True)
    b.ring(base, (-0.42, -0.10, 1.526), 0.407, 0.018, "bright")
    for i in range(12):
        a = 2 * math.pi * i / 12
        b.cylinder(
            base,
            (-0.42 + 0.403 * math.cos(a), -0.10 + 0.403 * math.sin(a), 1.539),
            0.012,
            0.016,
            "metal",
        )
    lid = b.moving(
        base,
        "vessel_cover",
        (-0.42, 0.33, 1.569),
        (-1, 0, 0),
        (0, 1.10),
        kind="hinge",
        mass=0.9,
        kp=240,
    )
    lid.set("euler", numbers((-0.62, 0, 0)))
    lid.find("joint").set("ref", ".62")
    b.cylinder(lid, (0, -0.43, 0), 0.392, 0.021, "metal", collision=True)
    for x in (-0.10, 0.10):
        b.rod(lid, (x, -0.43, 0.024), (x, -0.43, 0.10), 0.011, "bright")
    b.rod(lid, (-0.10, -0.43, 0.10), (0.10, -0.43, 0.10), 0.012, "bright")
    # Fixed blank interior is deliberately dry; it has no product/process content.
    b.cylinder(base, (-0.42, -0.10, 0.799), 0.13, 0.075, "dark")
    for z in (0.83, 1.17):
        b.cylinder(
            base, (-0.42, -0.505, z), 0.068, 0.024, "cream", euler=(math.pi / 2, 0, 0)
        )
        b.cylinder(
            base, (-0.42, -0.535, z), 0.036, 0.009, "lens", euler=(math.pi / 2, 0, 0)
        )
    b.rod(base, (-0.10, -0.06, 1.39), (0.19, -0.06, 1.39), 0.036, "bright")
    b.rod(base, (0.19, -0.06, 1.39), (0.19, 0.85, 2.03), 0.036, "bright")
    # Supported receiver slide holds a visible empty inert cup outside the vessel.
    b.rod(
        base, (-0.23, -0.10, 0.90), (-0.23, -0.71, 0.90), 0.032, "metal", collision=True
    )
    b.box(base, (-0.15, -0.71, 0.923), (0.34, 0.17, 0.022), "metal", collision=True)
    for y in (-0.81, -0.62):
        b.rod(base, (-0.44, y, 0.964), (0.14, y, 0.964), 0.009, "bright")
    drawer = b.moving(
        base, "receiver_slide", (-0.18, -0.71, 0.996), (1, 0, 0), (0, 0.16), mass=0.4
    )
    b.box(drawer, (0, 0, 0), (0.14, 0.14, 0.023), "metal", collision=True)
    b.cylinder(drawer, (0, 0, 0.035), 0.075, 0.012, "dark", collision=True)
    _hollow_wall(b, drawer, (0, 0, 0.103), 0.062, 0.052, 0.052, "cream", segments=24)
    b.cylinder(drawer, (0, 0, 0.053), 0.052, 0.006, "cream", collision=True)
    for x in (-0.075, 0.075):
        b.box(drawer, (x, 0, 0.075), (0.012, 0.066, 0.041), "dark", collision=True)
    b.site(drawer, "inert_sampling_cup", (0, 0, 0.103), size=0.006)
    b.metadata["capabilities"] = [
        "Mechanical vessel-cover access",
        "Retained inert receiver positioning",
    ]
    b.metadata["limitations"].append(
        "The primary Sheffield photo establishes the platform, stairs, handrails, "
        "large process housing and front vessel. Dimensions, lid hinge, empty cup "
        "and receiver slide are authored estimates; no GEA CAD or logo is copied. "
        "No flow, powders, tablet manufacture, reaction, fermentation, sterility, "
        "thermal control, process recipe or validated plant operation."
    )


def _sheffield_bay(world, definition):
    p = _Room(world, "sheffield")
    # Building-scale fittings follow the primary view; exact facility plan is unknown.
    p.rod("upper_service_duct", (-3.18, 2.72, 3.57), (-0.91, 2.72, 3.57), 0.09, STEEL)
    p.box("wall_vent_plenum", (-2.04, 2.82, 3.11), (1.14, 0.16, 0.20), STEEL)
    for index, x in enumerate((-2.91, -2.33, -1.75, -1.17)):
        ET.SubElement(
            world,
            "geom",
            name=f"europe5_sheffield_vent_rim_{index}",
            type="cylinder",
            size=".137 .029",
            pos=numbers((x, 2.637, 3.11)),
            euler=numbers((math.pi / 2, 0, 0)),
            rgba=numbers(WHITE),
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            world,
            "geom",
            name=f"europe5_sheffield_vent_dark_{index}",
            type="cylinder",
            size=".093 .006",
            pos=numbers((x, 2.602, 3.11)),
            euler=numbers((math.pi / 2, 0, 0)),
            rgba=numbers(DARK),
            contype="0",
            conaffinity="0",
        )
    p.box("left_service_cabinet", (-2.65, 1.80, 0.64), (0.33, 0.41, 0.64), WHITE, True)
    p.box("cabinet_display", (-2.65, 1.38, 0.98), (0.20, 0.008, 0.12), DARK)


BUILDERS = {
    "sheffield_dipp_pilot_skid": _dipp_pilot_skid,
    "polimi_white_openjet_station": _polimi_openjet,
    "lund_red_dry_characterization_chamber": _lund_dry_chamber,
    "sesame_historical_relay_bench": _sesame_relay_bench,
}
SOURCES = {
    "sheffield_dipp_pilot_skid": dict(
        reference="Sheffield Diamond Pilot Plant installed stainless platform/vessel photograph",
        url=SHEFFIELD,
        dimensions_m=[3.42, 3.58, 3.42],
        dimension_basis="Estimated person-scale platform, stairs and vessel; representative cover/receiver mechanics are authored, not OEM CAD",
    ),
    "polimi_white_openjet_station": dict(
        reference="Polimi DAER white aeroacoustic chamber photograph and specifications",
        url=POLIMI,
        dimensions_m=[1.98, 2.43, 1.93],
        dimension_basis="Published 4 m cubical chamber and 0.5 x 0.7 m useful jet; outer nozzle, specimen/supports and travel estimated",
    ),
    "lund_red_dry_characterization_chamber": dict(
        reference="NanoLund official red chamber and control-rack photographs",
        url=LUND,
        dimensions_m=[1.38, 0.88, 1.75],
        dimension_basis="Estimated bench/chamber dimensions from installed photographs; lid mechanism, inert coupon tray and travel authored",
    ),
    "sesame_historical_relay_bench": dict(
        reference="PSL/Observatoire de Paris SESAME 2009 gallery; official interfacing dimensions",
        url=PSL,
        dimensions_m=[3.0, 1.5, 1.17],
        dimension_basis="Official 3 x 1.5 m table and 0.15 m optical-axis height; detailed mount positions/travel estimated; hole pattern decimated",
    ),
}
SAMPLE_INTERFACES = {
    "sheffield_dipp_pilot_skid": (
        "inert_sampling_cup",
        (0.124, 0.124, 0.116),
        "clamped",
        "Visible empty rigid inert receiver cup retained in a supported sliding tray",
    ),
    "polimi_white_openjet_station": (
        "inert_aeroacoustic_model",
        (0.19, 0.38, 0.068),
        "clamped",
        "Visible original rigid inert calibration body retained on a yaw support",
    ),
    "lund_red_dry_characterization_chamber": (
        "inert_characterization_coupon",
        (0.056, 0.046, 0.018),
        "clamped",
        "Visible original inert calibration coupon retained on a mechanical indexing tray; no process material",
    ),
    "sesame_historical_relay_bench": (
        "inert_optical_coupon",
        (0.072, 0.014, 0.068),
        "clamped",
        "Visible retained inert crosshair coupon on an authored translation fixture",
    ),
}
FEATURES = {
    "sheffield_diamond_pilot_plant": _sheffield_bay,
    "polimi_aeroacoustic_white": _polimi_chamber,
    "psl_sesame_adaptive_optics": _sesame_room,
    "lund_nanoparticle_characterization": _lund_workstation,
}
