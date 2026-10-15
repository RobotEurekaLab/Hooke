"""Estimated PICE workroom furniture and open dry-carriage guides.

Two official photographs constrain visible furniture relationships. Hidden
bearing details and all room dimensions remain authored estimates. No powered
cutting, refrigeration, melting or ice-property model is supplied.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

PREFIX = "ice_station__pice_"
BLUE = (0.07, 0.25, 0.38, 1)
WOOD = (0.63, 0.46, 0.28, 1)
STEEL = (0.49, 0.53, 0.56, 1)
DARK = (0.055, 0.065, 0.075, 1)
WHITE = (0.84, 0.85, 0.83, 1)


def _bearing_y(b, carrier, key, x):
    """Continuous open visual sleeve, with separate convex contact sectors."""
    count, inner, outer, half = 48, 0.0076, 0.0136, 0.024
    center = (x, 0, -0.029)
    vertices = [
        (center[0] + r * math.cos(math.tau * i / count), y,
         center[2] + r * math.sin(math.tau * i / count))
        for y in (-half, half)
        for r in (inner, outer)
        for i in range(count)
    ]
    faces = []
    for i in range(count):
        j = (i + 1) % count
        for a, c, d, e in (
            (i, count + i, count + j, j),
            (2 * count + i, 2 * count + j, 3 * count + j, 3 * count + i),
            (i, j, 2 * count + j, 2 * count + i),
            (count + j, count + i, 3 * count + i, 3 * count + j),
        ):
            faces.extend(((a, c, d), (a, d, e)))
    name = PREFIX + key
    ET.SubElement(b.root.find("asset"), "mesh", name=name + "_mesh",
                  vertex=numbers([v for point in vertices for v in point]),
                  face=" ".join(str(v) for face in faces for v in face))
    b.geom(carrier, "mesh", (), material="dark", mesh=name + "_mesh",
           name=name + "_shell", density=0)
    radius, thickness = (outer + inner) / 2, (outer - inner) / 2
    for i in range(count):
        angle = math.tau * (i + 0.5) / count
        b.box(carrier,
              (x + radius * math.cos(angle), 0, -0.029 + radius * math.sin(angle)),
              (thickness, half, outer * math.tan(math.pi / count)), "dark",
              collision=True, euler=(0, -angle, 0), rgba=(0.5, 0.5, 0.5, 0),
              group=3, density=0, name=name + f"_contact_{i}")
    # Crown block touches the sleeve and the original plate underside without
    # entering the bore. The ideal slide carries load across the running gap.
    b.box(carrier, (x, 0, -0.014), (0.013, 0.026, 0.003), "metal",
          collision=True, density=0, name=name + "_crown")


def add_carriage_guides(b, base, carrier):
    """Preserve the original +/-75 mm slide, core and moving inertial model."""
    for index, x in enumerate((-0.22, -0.02)):
        b.rod(base, (x, -0.36, 0.891), (x, -0.055, 0.891), 0.007, "bright",
              collision=True, name=PREFIX + f"rail_{index}")
        for end, y in enumerate((-0.36, -0.055)):
            b.box(base, (x, y, 0.8855), (0.019, 0.012, 0.0095), "metal",
                  collision=True, name=PREFIX + f"rail_{index}_support_{end}")
        _bearing_y(b, carrier, f"bearing_{index}", x + 0.115)
    b.metadata["carriage_guides"] = {
        "revision": 1,
        "axis": "Original Y slide; no axis/frame/range/drive change",
        "rail_radius_m": 0.007,
        "bearing_inner_radius_m": 0.0076,
        "running_clearance_m": 0.0006,
        "bearing_length_m": 0.048,
        "rail_centerline_y_m": [-0.36, -0.055],
        "dimension_basis": "Authored open bearing repair for the existing dry carriage; not visible OEM internals",
        "scope": "Ideal joint support with explicit open contact geometry; no real bearing stiffness, powered cutting or ice science",
    }


class _Furniture:
    def __init__(self, world):
        self.world = world

    def box(self, key, pos, half, color, collision=True, **kwargs):
        return box(self.world, "pice_room_" + key, half, pos, color,
                   collision=collision, **kwargs)

    def rod(self, key, start, end, radius, color, collision=True):
        return ET.SubElement(self.world, "geom", name="pice_room_" + key,
                             type="capsule", fromto=numbers((*start, *end)),
                             size=str(radius), rgba=numbers(color),
                             contype="1" if collision else "0",
                             conaffinity="1" if collision else "0")

    def bench(self, key, x, y, hx, hy, height=0.88):
        self.box(key + "_wood_top", (x, y, height), (hx, hy, 0.028), WOOD)
        for side in (-1, 1):
            self.box(key + f"_rail_{side}", (x, y + side * (hy - 0.04), height - 0.068),
                     (hx, 0.04, 0.04), BLUE)
        for i, xx in enumerate((x - hx + 0.055, x + hx - 0.055)):
            for j, yy in enumerate((y - hy + 0.05, y + hy - 0.05)):
                self.box(key + f"_leg_{i}_{j}", (xx, yy, (height - 0.05) / 2),
                         (0.032, 0.032, (height - 0.05) / 2), BLUE)
                for k, z in enumerate((0.30, 0.43, 0.56, 0.69)):
                    if z < height - 0.12:
                        self.box(key + f"_hole_{i}_{j}_{k}", (xx, yy - 0.033, z),
                                 (0.007, 0.001, 0.010), DARK, False)

    def bin(self, key, x, y):
        color = (0.43, 0.45, 0.46, 1)
        self.box(key + "_bottom", (x, y, 0.016), (0.28, 0.235, 0.016), color)
        for side in (-1, 1):
            self.box(key + f"_side_{side}", (x + side * 0.271, y, 0.215),
                     (0.009, 0.235, 0.20), color)
            self.box(key + f"_end_{side}", (x, y + side * 0.226, 0.215),
                     (0.27, 0.009, 0.20), color)
        for i, dx in enumerate((-0.20, -0.07, 0.07, 0.20)):
            self.box(key + f"_rib_{i}", (x + dx, y - 0.24, 0.22),
                     (0.009, 0.01, 0.18), color)


def add_workroom(world, definition):
    """Reconcile adjacent-wall photos without inventing a surveyed floor plan."""
    p = _Furniture(world)
    p.bench("left_bench", -1.34, 1.30, 0.84, 0.43)
    p.bench("right_bench", 1.14, 1.30, 0.64, 0.43)
    p.bin("left_bin", -0.84, 1.28)
    p.bin("right_bin", 0.93, 1.28)
    p.bench("near_table", -1.51, -0.40, 0.72, 0.34, 0.86)
    # The second photograph looks along the workbench wall toward a distinct
    # end wall. Placing this freezer panel behind a worktop blocked that view.
    for i, x in enumerate((-2.35, -1.55, -0.75, 0.05, 0.85, 1.65, 2.35)):
        p.box(f"back_panel_seam_{i}", (x, 2.24, 1.40), (0.004, 0.010, 1.40),
              (0.67, 0.68, 0.67, 1), False)
    # End-wall parts share the established cut-away naming convention so the
    # overview cannot leave an unsupported door frame floating without its wall.
    def wall_box(key, pos, half, color, collision=True):
        g = p.box(key, pos, half, color, collision)
        g.set("name", "arch_cutaway_wall_right_pice_" + key)
        return g
    wall_box("freezer_closed_panel", (2.468, -0.20, 1.075), (0.032, 0.55, 1.025), WHITE)
    for i, y in enumerate((-0.795, 0.395)):
        wall_box(f"freezer_jamb_{i}", (2.453, y, 1.075), (0.047, 0.035, 1.075), STEEL)
    wall_box("freezer_header", (2.453, -0.20, 2.165), (0.047, 0.63, 0.04), STEEL)
    wall_box("freezer_latch", (2.400, -0.65, 1.02), (0.036, 0.036, 0.034), DARK)
    wall_box("freezer_exit_backing", (2.480, -0.20, 2.35), (0.020, 0.16, 0.075), WHITE)
    wall_box("freezer_exit_panel", (2.457, -0.20, 2.35), (0.003, 0.14, 0.05), (0.12, 0.55, 0.34, 1), False)
    # Open galvanised shelving to the left of the doorway in the source view.
    for i, x in enumerate((1.89, 2.40)):
        for j, y in enumerate((0.63, 1.18)):
            p.box(f"shelf_post_{i}_{j}", (x, y, 0.94), (0.018, 0.018, 0.94), STEEL)
    for i, z in enumerate((0.12, 0.57, 1.06, 1.54)):
        p.box(f"shelf_{i}", (2.145, 0.905, z), (0.272, 0.293, 0.015), STEEL)
    p.box("shelf_grey_box", (2.15, 0.91, 1.17), (0.21, 0.21, 0.095), (0.39, 0.41, 0.43, 1))
    # A floor-supported cart and stacked closed storage boxes remain outside
    # the clear door approach. They are passive furniture, not process assets.
    for i, x in enumerate((1.81, 2.24)):
        for j, y in enumerate((-1.29, -0.90)):
            ET.SubElement(world, "geom", name=f"pice_room_cart_wheel_{i}_{j}",
                                  type="cylinder", pos=numbers((x, y, 0.043)),
                                  size=".043 .014", euler=numbers((math.pi / 2, 0, 0)),
                                  rgba=numbers(DARK), contype="1", conaffinity="1")
            p.box(f"cart_fork_{i}_{j}", (x, y, 0.082), (0.018, 0.020, 0.035), STEEL)
    p.box("cart_deck", (2.025, -1.095, 0.124), (0.29, 0.245, 0.014), WOOD)
    p.box("cart_lower_box", (2.025, -1.095, 0.325), (0.24, 0.205, 0.187), WHITE)
    p.box("cart_lower_lid", (2.025, -1.095, 0.526), (0.252, 0.217, 0.014), WHITE)
    p.box("cart_upper_box", (2.04, -1.08, 0.68), (0.19, 0.17, 0.14), WHITE)
    p.bench("end_bench", 2.05, -1.83, 0.36, 0.33)
    p.box("end_bench_tray", (2.05, -1.83, 0.922), (0.29, 0.25, 0.014), STEEL)
    # A visible wall-mounted dustpan and parked leads reproduce modest source
    # density. Service routes terminate at visible brackets; no live feed claim.
    p.box("dustpan", (1.44, 2.215, 1.16), (0.105, 0.022, 0.115), DARK, False)
    p.rod("dustpan_handle", (1.44, 2.215, 1.26), (1.44, 2.215, 1.62), 0.014, DARK, False)
    p.box("dustpan_hook", (1.44, 2.235, 1.60), (0.023, 0.015, 0.024), STEEL, False)
    for i, y in enumerate((0.15, 0.27)):
        g = p.rod(f"endwall_service_{i}", (2.499, y, 2.20), (2.499, y, 2.78), 0.005, STEEL, False)
        g.set("name", f"arch_cutaway_wall_right_pice_service_{i}")
    # All dimensions above are estimates, including the 5 x 4.5 m room.
