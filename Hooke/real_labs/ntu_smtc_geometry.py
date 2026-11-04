"""Supported membrane-skid exterior and cold mechanical access fixtures.

The SMTC brochure informs frame/module/hosing appearance. Dimensions, hidden
connections and the retained sampling carriage are authored estimates. There
is no pressure, fluid transport, membrane response or calibrated gauge output.
"""

import math

from real_labs.mounting import annular_mount


class Parts:
    def __init__(self, builder):
        self.b = builder

    def name(self, key):
        return f"{self.b.name}__smtc_{key}"

    def box(self, parent, key, pos, half, material="metal", **kwargs):
        return self.b.box(
            parent, pos, half, material, name=self.name(key), collision=True, **kwargs
        )

    def cylinder(self, parent, key, pos, radius, half, material="metal", **kwargs):
        return self.b.cylinder(
            parent,
            pos,
            radius,
            half,
            material,
            name=self.name(key),
            collision=True,
            **kwargs,
        )

    def rod(self, parent, key, start, end, radius, material="metal"):
        return self.b.rod(
            parent, start, end, radius, material, name=self.name(key), collision=True
        )

    def collar_z(self, parent, key, pos, bore, outer, half, material="metal"):
        """Smooth open appearance and sixteen conservative contact sectors."""
        annular_mount(
            self.b,
            parent,
            key,
            pos,
            bore,
            outer,
            half,
            axis="z",
            namespace="smtc",
            material=material,
        )
        radius = (bore + outer) / 2
        for i in range(16):
            angle = math.tau * (i + 0.5) / 16
            self.box(
                parent,
                f"{key}_proxy_{i}",
                (
                    pos[0] + radius * math.cos(angle),
                    pos[1] + radius * math.sin(angle),
                    pos[2],
                ),
                ((outer - bore) / 2, outer * math.tan(math.pi / 16), half),
                material,
                euler=(0, 0, angle),
                rgba=(0.5, 0.5, 0.5, 0),
            )

    def bore_y(self, parent, key, pos, bore, outer, half, material="metal"):
        """Rotate a complete open Z-bore to Y in the owning body's frame."""
        first = len(parent)
        self.collar_z(parent, key, (0, 0, 0), bore, outer, half, material)
        for geom in list(parent)[first:]:
            x, y, z = map(float, geom.get("pos", "0 0 0").split())
            angle = float(geom.get("euler", "0 0 0").split()[2])
            geom.set("pos", " ".join(map(str, (pos[0] + x, pos[1] - z, pos[2] + y))))
            geom.set("euler", f"{math.pi/2} 0 {angle}")

    def hose(self, parent, key, points, radius=0.012):
        """Continuous cubic route with connected segment endpoints."""
        a, b, c, d = points
        previous = a
        for i in range(1, 13):
            t = i / 12
            point = tuple(
                (1 - t) ** 3 * a[k]
                + 3 * (1 - t) ** 2 * t * b[k]
                + 3 * (1 - t) * t * t * c[k]
                + t**3 * d[k]
                for k in range(3)
            )
            self.rod(parent, f"{key}_{i}", previous, point, radius, "rubber")
            previous = point


def _frame(p, base):
    for i, x in enumerate((-0.77, 0.77)):
        for j, y in enumerate((-0.43, 0.43)):
            p.cylinder(base, f"foot_{i}_{j}", (x, y, 0.024), 0.048, 0.024, "rubber")
            p.box(base, f"post_{i}_{j}", (x, y, 0.839), (0.022, 0.022, 0.791), "bright")
    for k, z in enumerate((0.16, 0.69, 1.60)):
        for j, y in enumerate((-0.43, 0.43)):
            p.box(base, f"crossbar_{k}_{j}", (0, y, z), (0.79, 0.023, 0.023), "bright")
        for i, x in enumerate((-0.77, 0.77)):
            p.box(base, f"sidebar_{k}_{i}", (x, 0, z), (0.023, 0.44, 0.023), "bright")
        for i, x in enumerate((-0.744, 0.744)):
            p.box(base, f"corner_plate_{k}_{i}", (x, -0.457, z), (0.045, 0.004, 0.042))
            for j, dz in enumerate((-0.022, 0.022)):
                p.cylinder(
                    base,
                    f"corner_bolt_{k}_{i}_{j}",
                    (x, -0.463, z + dz),
                    0.005,
                    0.003,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
    p.box(base, "bottom_deck", (0, 0, 0.185), (0.755, 0.425, 0.012))


def _modules(p, base):
    for i, x in enumerate((-0.44, 0.30)):
        p.cylinder(base, f"feed_foot_{i}", (x, 0.14, 0.200), 0.12, 0.003, "rubber")
        p.cylinder(base, f"feed_vessel_{i}", (x, 0.14, 0.46), 0.16, 0.26, "shell")
        p.cylinder(base, f"feed_lid_{i}", (x, 0.14, 0.73), 0.17, 0.012)
        p.cylinder(base, f"feed_neck_{i}", (x, 0.14, 0.755), 0.025, 0.017, "dark")
        sign = -1 if i == 0 else 1
        p.hose(
            base,
            f"feed_route_{i}",
            (
                (x, 0.14, 0.77),
                (x, 0.12, 0.86),
                (sign * 0.62, -0.16, 0.92),
                (sign * 0.62, -0.26, 0.84),
            ),
        )
    for k, z in enumerate((1.06, 1.36)):
        p.box(
            base,
            f"module_crossbar_{k}",
            (0, 0.43, z - 0.142),
            (0.79, 0.023, 0.025),
            "bright",
        )
        p.cylinder(
            base,
            f"module_{k}",
            (0, 0.12, z),
            0.071,
            0.606,
            "shell",
            euler=(0, math.pi / 2, 0),
        )
        for i, sign in enumerate((-1, 1)):
            x = sign * 0.63
            p.cylinder(
                base,
                f"module_flange_{k}_{i}",
                (x, 0.12, z),
                0.081,
                0.024,
                "cream",
                euler=(0, math.pi / 2, 0),
            )
            p.cylinder(
                base,
                f"module_end_{k}_{i}",
                (sign * 0.657, 0.12, z),
                0.061,
                0.006,
                "dark",
                euler=(0, math.pi / 2, 0),
            )
            p.cylinder(
                base,
                f"module_union_{k}_{i}",
                (sign * 0.674, 0.12, z),
                0.027,
                0.015,
                "bright",
                euler=(0, math.pi / 2, 0),
            )
            for bolt in range(6):
                a = math.tau * bolt / 6
                p.cylinder(
                    base,
                    f"flange_screw_{k}_{i}_{bolt}",
                    (sign * 0.657, 0.12 + 0.070 * math.cos(a), z + 0.070 * math.sin(a)),
                    0.004,
                    0.003,
                    "bright",
                    euler=(0, math.pi / 2, 0),
                )
            p.box(
                base,
                f"module_arm_{k}_{i}",
                (sign * 0.516, 0.275, z - 0.142),
                (0.032, 0.15, 0.025),
                "bright",
            )
            p.box(
                base,
                f"module_saddle_{k}_{i}",
                (sign * 0.516, 0.14, z - 0.09),
                (0.048, 0.055, 0.035),
                "rubber",
            )
            p.hose(
                base,
                f"module_hose_{k}_{i}",
                (
                    (sign * 0.689, 0.12, z),
                    (sign * 0.72, -0.25 - 0.04 * k, z),
                    (sign * 0.69, -0.26, 0.89),
                    (sign * 0.62, -0.26, 0.84),
                ),
            )
    for i, x in enumerate((-0.60, 0.60)):
        p.box(base, f"manifold_support_{i}", (x, -0.343, 0.84), (0.025, 0.087, 0.022))
        p.box(base, f"manifold_post_{i}", (x, -0.43, 0.776), (0.025, 0.024, 0.064))
    p.rod(
        base,
        "manifold_left",
        (-0.62, -0.26, 0.84),
        (0.255, -0.26, 0.84),
        0.02,
        "bright",
    )
    p.rod(
        base,
        "manifold_right",
        (0.345, -0.26, 0.84),
        (0.62, -0.26, 0.84),
        0.02,
        "bright",
    )
    for i, x in enumerate((-0.28, 0.26)):
        p.rod(
            base,
            f"gauge_riser_{i}",
            (x, -0.26, 0.84),
            (x, -0.26, 1.075),
            0.012,
            "bright",
        )
        p.rod(
            base,
            f"gauge_elbow_{i}",
            (x, -0.26, 1.075),
            (x, -0.31, 1.075),
            0.012,
            "bright",
        )
        p.cylinder(
            base,
            f"gauge_case_{i}",
            (x, -0.31, 1.11),
            0.055,
            0.018,
            "bright",
            euler=(math.pi / 2, 0, 0),
        )
        p.cylinder(
            base,
            f"gauge_face_{i}",
            (x, -0.330, 1.11),
            0.047,
            0.002,
            "cream",
            euler=(math.pi / 2, 0, 0),
        )
        p.rod(
            base,
            f"gauge_needle_{i}",
            (x, -0.333, 1.11),
            (x + 0.023, -0.333, 1.134),
            0.0015,
            "dark",
        )
        for j in range(9):
            a = -2.35 + j * 4.7 / 8
            p.rod(
                base,
                f"gauge_tick_{i}_{j}",
                (x + 0.035 * math.sin(a), -0.3327, 1.11 + 0.035 * math.cos(a)),
                (x + 0.041 * math.sin(a), -0.3327, 1.11 + 0.041 * math.cos(a)),
                0.0008,
                "dark",
            )


def _valve(p, base):
    b = p.b
    p.cylinder(
        base,
        "valve_body",
        (0.30, -0.26, 0.84),
        0.043,
        0.045,
        "metal",
        euler=(0, math.pi / 2, 0),
    )
    p.bore_y(base, "valve_bonnet", (0.30, -0.3529, 0.86), 0.0086, 0.024, 0.0395)
    p.box(base, "valve_bonnet_heel", (0.30, -0.311, 0.824), (0.026, 0.033, 0.015))
    valve = b.moving(
        base,
        "isolation_valve",
        (0.30, -0.405, 0.86),
        (0, 1, 0),
        (0, math.pi / 2),
        kind="hinge",
        mass=0.08,
        kp=50,
    )
    p.cylinder(
        valve, "valve_hub", (0, 0, 0), 0.025, 0.012, "metal", euler=(math.pi / 2, 0, 0)
    )
    p.cylinder(
        valve,
        "valve_stem",
        (0, 0.03925, 0),
        0.008,
        0.05125,
        "bright",
        euler=(math.pi / 2, 0, 0),
    )
    p.rod(valve, "valve_lever", (0, -0.015, 0), (0, -0.015, 0.15), 0.019, "blue")
    p.cylinder(
        valve,
        "valve_locknut",
        (0, -0.023, 0),
        0.010,
        0.004,
        "bright",
        euler=(math.pi / 2, 0, 0),
    )
    b.site(valve, "valve_handle_tip", (0, -0.015, 0.15))


def _carriage(p, base):
    b = p.b
    for i, x in enumerate((-0.17, 0.17)):
        p.box(base, f"rail_footplate_{i}", (x, -0.185, 0.207), (0.035, 0.28, 0.010))
        for j, y in enumerate((-0.455, 0.09)):
            p.box(base, f"rail_pedestal_{i}_{j}", (x, y, 0.285), (0.025, 0.025, 0.068))
            p.cylinder(
                base,
                f"rail_boss_{i}_{j}",
                (x, y, 0.36),
                0.020,
                0.018,
                "metal",
                euler=(math.pi / 2, 0, 0),
            )
        p.rod(
            base,
            f"carriage_rail_{i}",
            (x, -0.455, 0.36),
            (x, 0.09, 0.36),
            0.012,
            "bright",
        )
    carriage = b.moving(
        base, "sample_carriage", (0, -0.16, 0.40), (0, -1, 0), (0, 0.15), mass=0.28
    )
    p.box(carriage, "tray", (0, 0, 0), (0.20, 0.115, 0.025), "blue")
    for i, x in enumerate((-0.17, 0.17)):
        for j, y in enumerate((-0.055, 0.055)):
            p.bore_y(
                carriage,
                f"carriage_bearing_{i}_{j}",
                (x, y, -0.04),
                0.0126,
                0.020,
                0.020,
                "dark",
            )
            p.box(
                carriage, f"bearing_cap_{i}_{j}", (x, y, -0.020), (0.016, 0.022, 0.007)
            )
    for i, x in enumerate((-0.12, 0, 0.12)):
        p.cylinder(carriage, f"vial_{i}", (x, 0, 0.065), 0.032, 0.045, "glass")
        p.cylinder(carriage, f"vial_cap_{i}", (x, 0, 0.114), 0.030, 0.006, "dark")
        p.collar_z(
            carriage,
            f"vial_retainer_{i}",
            (x, 0, 0.038),
            0.032,
            0.037,
            0.013,
            material="cream",
        )
    b.site(carriage, "water_vial", (0, 0, 0.065))
    p.rod(
        carriage,
        "tray_handle_left",
        (-0.11, -0.122, 0),
        (-0.11, -0.16, 0),
        0.008,
        "bright",
    )
    p.rod(
        carriage,
        "tray_handle_right",
        (0.11, -0.122, 0),
        (0.11, -0.16, 0),
        0.008,
        "bright",
    )
    p.rod(
        carriage, "tray_handle_grip", (-0.11, -0.16, 0), (0.11, -0.16, 0), 0.009, "dark"
    )


def add_skid(b, base, params):
    p = Parts(b)
    _frame(p, base)
    _modules(p, base)
    _valve(p, base)
    _carriage(p, base)
    p.box(base, "monitor_support", (0.51, 0.35, 1.623), (0.045, 0.103, 0.023))
    p.rod(base, "monitor_stalk", (0.51, 0.27, 1.6), (0.51, 0.27, 1.68), 0.024)
    p.box(base, "monitor_housing", (0.51, 0.27, 1.81), (0.22, 0.10, 0.15), "shell")
    p.box(base, "screen_bezel", (0.51, 0.162, 1.81), (0.167, 0.009, 0.107), "dark")
    p.box(base, "inactive_display", (0.51, 0.152, 1.81), (0.16, 0.001, 0.10), "screen")
    p.hose(
        base,
        "monitor_service",
        (
            (0.65, 0.37, 1.70),
            (0.75, 0.40, 1.63),
            (0.72, 0.41, 1.35),
            (0.748, 0.43, 1.20),
        ),
        0.008,
    )
    b.metadata["capabilities"] = [
        "Supported quarter-turn valve-handle positioning",
        "Guided rigid water-vial carriage access and return",
    ]
    b.metadata["limitations"].extend(
        [
            "Campus frame and module appearance follow the SMTC brochure; exact plumbing, hidden mechanisms and dimensions are authored estimates.",
            "Cold mechanical handle/carriage operation only; pressure, flow, filtration, membrane transport and gauge response are not simulated.",
            "Rigid retained closed vials are not free samples; bearing clearances and original servo masses/gains are uncalibrated surrogates.",
        ]
    )
