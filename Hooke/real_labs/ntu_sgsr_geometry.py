"""Photo-informed NTU apparatus with explicit open mechanical interfaces.

Envelope and exterior cues follow the facility's linked brochures. Bearings,
internal connections and dry inspection fixtures are original engineering
estimates; no OEM CAD, AFM scanning or material-test response is reproduced.
"""

import math

from real_labs.mounting import annular_mount


class Parts:
    """Give meaningful names and enabled contacts to substantial apparatus parts."""

    def __init__(self, builder):
        self.b = builder

    def name(self, key):
        return f"{self.b.name}__ntu_{key}"

    def box(self, parent, key, pos, size, material="metal", **kwargs):
        return self.b.box(
            parent, pos, size, material, name=self.name(key), collision=True, **kwargs
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

    def rod(self, parent, key, first, second, radius, material="metal"):
        return self.b.rod(
            parent, first, second, radius, material, name=self.name(key), collision=True
        )

    def sleeve(self, parent, key, center, axis, half_length, bore, wall=0.004):
        """Four enabled walls around a rectangular, genuinely open guide bore."""
        transverse = [i for i in range(3) if i != axis]
        for offset, dimension in enumerate(transverse):
            other = transverse[1 - offset]
            for sign in (-1, 1):
                p = list(center)
                p[dimension] += sign * (bore[offset] + wall / 2)
                size = [0.0, 0.0, 0.0]
                size[axis] = half_length
                size[dimension] = wall / 2
                size[other] = bore[1 - offset] + (wall if offset == 0 else 0)
                self.box(parent, f"{key}_{dimension}_{sign}", p, size, "dark")

    def collar(self, parent, key, center, inner, outer, half):
        annular_mount(
            self.b,
            parent,
            key,
            center,
            inner,
            outer,
            half,
            axis="z",
            namespace="ntu",
            material="metal",
        )
        count = 16
        radius = (inner + outer) / 2
        tangent = outer * math.tan(math.pi / count)
        for i in range(count):
            angle = math.tau * (i + 0.5) / count
            pos = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                center[2],
            )
            self.box(
                parent,
                f"{key}_sector_{i}",
                pos,
                ((outer - inner) / 2, tangent, half),
                euler=(0, 0, angle),
                rgba=(0.5, 0.5, 0.5, 0),
            )

    def housing(self, parent, key, levels, material, radius=0.012):
        self.b.housing(parent, levels, material=material, radius=radius)
        geom = list(parent)[-1]
        geom.set("name", self.name(key))
        geom.set("contype", "1")
        geom.set("conaffinity", "1")


def build_afm(b, base, params):
    p = Parts(b)
    b.feet(base, 0.65, 0.69)
    p.box(base, "base", (0, 0, 0.0515), (0.315, 0.33, 0.0275), "dark")
    p.housing(
        base,
        "rear_shell",
        [
            (0.05, 0.315, 0.12, 0.20),
            (0.49, 0.305, 0.115, 0.20),
            (0.64, 0.295, 0.10, 0.20),
        ],
        "cream",
        0.035,
    )
    p.box(base, "rear_chassis", (0, 0.21, 0.34), (0.28, 0.09, 0.27), "cream")
    p.box(base, "blue_brow", (0, 0.11, 0.55), (0.273, 0.003, 0.015), "blue")
    b.screen(base, (-0.20, 0.08, 0.49), 0.12, 0.05)
    for i, x in enumerate((-0.25, 0.25)):
        p.box(
            base, f"service_cover_{i}", (x, 0.09, 0.31), (0.016, 0.012, 0.135), "dark"
        )
        for j, z in enumerate((0.2, 0.25, 0.3, 0.35, 0.4)):
            p.cylinder(
                base,
                f"cover_screw_{i}_{j}",
                (x, 0.075, z),
                0.003,
                0.003,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
    for i, y in enumerate((-0.16, -0.06)):
        for j, x in enumerate((-0.225, 0.225)):
            p.box(
                base,
                f"x_rail_pedestal_{i}_{j}",
                (x, y, 0.086),
                (0.018, 0.022, 0.007),
                "dark",
            )
        p.box(base, f"x_rail_{i}", (0, y, 0.102), (0.25, 0.012, 0.009), "bright")
    xstage = b.moving(
        base,
        "coarse_x",
        (0, -0.115, 0.125),
        (1, 0, 0),
        (-0.08, 0.08),
        mass=0.55,
        kp=360,
    )
    p.box(xstage, "x_plate", (0, 0, 0), (0.117, 0.155, 0.013))
    for i, x in enumerate((-0.080, 0.080)):
        for j, y in enumerate((-0.045, 0.055)):
            p.sleeve(
                xstage, f"x_bearing_{i}_{j}", (x, y, -0.023), 0, 0.020, (0.0126, 0.0096)
            )
            p.box(
                xstage, f"x_cap_{i}_{j}", (x, y, 0.012), (0.024, 0.025, 0.001), "bright"
            )
    for i, x in enumerate((-0.07, 0.07)):
        p.box(xstage, f"y_rail_{i}", (x, 0, 0.021), (0.004, 0.145, 0.004), "bright")
        for j, y in enumerate((-0.135, 0.135)):
            p.box(
                xstage,
                f"y_rail_post_{i}_{j}",
                (x, y, 0.016),
                (0.008, 0.007, 0.008),
                "dark",
            )
    ystage = b.moving(
        xstage, "coarse_y", (0, 0, 0.028), (0, 1, 0), (-0.06, 0.06), mass=0.3, kp=300
    )
    p.cylinder(ystage, "chuck", (0, 0, 0.010), 0.105, 0.004, "bright")
    for i, x in enumerate((-0.07, 0.07)):
        for j, y in enumerate((-0.04, 0.04)):
            p.sleeve(
                ystage,
                f"y_bearing_{i}_{j}",
                (x, y, -0.007),
                1,
                0.012,
                (0.0046, 0.0046),
                0.0025,
            )
            p.box(ystage, f"y_cap_{i}_{j}", (x, y, 0.004), (0.007, 0.013, 0.004))
    # Preserve the original site location and put actual material at that datum.
    p.cylinder(ystage, "sample", (0, 0, 0.0175), 0.013, 0.0035, "sample")
    b.site(ystage, "sample_coupon", (0, 0, 0.021), 0.001)
    p.box(base, "head_backplate", (0, 0.108, 0.385), (0.074, 0.011, 0.105))
    for i, x in enumerate((-0.048, 0.048)):
        p.box(base, f"head_rail_{i}", (x, 0.055, 0.385), (0.004, 0.004, 0.10), "bright")
        for j, z in enumerate((0.28, 0.49)):
            p.box(
                base, f"head_rail_mount_{i}_{j}", (x, 0.080, z), (0.010, 0.029, 0.009)
            )
    head = b.moving(
        base,
        "coarse_head",
        (0, 0.014, 0.38),
        (0, 0, 1),
        (-0.025, 0.025),
        mass=0.4,
        kp=320,
    )
    p.box(head, "head_chassis", (0, 0.007, 0), (0.061, 0.016, 0.064), "cream")
    for i, x in enumerate((-0.048, 0.048)):
        for j, z in enumerate((-0.033, 0.033)):
            p.sleeve(
                head,
                f"head_bearing_{i}_{j}",
                (x, 0.041, z),
                2,
                0.012,
                (0.0046, 0.0046),
                0.006,
            )
            p.box(head, f"head_ear_{i}_{j}", (x, 0.027, z), (0.011, 0.006, 0.013))
    p.box(head, "head_top", (0, -0.017, 0.047), (0.062, 0.027, 0.017))
    p.box(head, "head_bottom", (0, -0.017, -0.055), (0.053, 0.027, 0.012))
    for i, x in enumerate((-0.052, 0.052)):
        p.box(head, f"head_side_{i}", (x, -0.012, -0.005), (0.009, 0.027, 0.042))
        p.cylinder(head, f"head_adjuster_{i}", (x, -0.023, 0.073), 0.013, 0.012, "dark")
        p.cylinder(
            head, f"adjuster_cap_{i}", (x, -0.023, 0.086), 0.009, 0.002, "bright"
        )
    p.cylinder(
        head,
        "scanner_face",
        (0, -0.064, -0.026),
        0.062,
        0.023,
        euler=(math.pi / 2, 0, 0),
    )
    p.cylinder(
        head,
        "scanner_bezel",
        (0, -0.088, -0.026),
        0.057,
        0.002,
        "bright",
        euler=(math.pi / 2, 0, 0),
    )
    for i, angle in enumerate((0.35, 2.79, 4.71)):
        x, z = 0.046 * math.cos(angle), -0.026 + 0.046 * math.sin(angle)
        p.cylinder(
            head,
            f"face_screw_{i}",
            (x, -0.091, z),
            0.0035,
            0.002,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    p.box(head, "scanner_shoe", (0, -0.075, -0.106), (0.018, 0.027, 0.024), "bright")
    p.cylinder(head, "probe_stem", (0, -0.095, -0.1526), 0.009, 0.0226)
    p.box(head, "chip_holder", (0, -0.101, -0.1772), (0.005, 0.009, 0.002), "dark")
    p.box(
        head, "probe_chip", (0, -0.110, -0.17935), (0.0008, 0.0016, 0.00015), "copper"
    )
    p.box(
        head,
        "cantilever",
        (0, -0.111689, -0.179501),
        (0.0000175, 0.000091, 0.000001),
        "bright",
    )
    p.cylinder(
        head, "probe_tip", (0, -0.11176, -0.1795045), 0.000001, 0.0000025, "bright"
    )
    b.site(head, "probe_tip", (0, -0.11176, -0.179507), 0.000002)
    # Vertical optical tube and its elbow follow the photograph's visible cue.
    p.box(head, "optic_bracket", (-0.085, -0.015, 0.014), (0.035, 0.012, 0.015), "dark")
    p.cylinder(head, "optic_camera", (-0.135, -0.015, 0.032), 0.027, 0.066, "dark")
    p.rod(
        head,
        "optic_elbow",
        (-0.135, -0.015, -0.031),
        (-0.068, -0.039, -0.063),
        0.019,
        "dark",
    )
    # A rigid local service loop: both ends attach to this moving head. No
    # invented fixed-to-moving deformable lead or cable mechanics is claimed.
    anchors = (
        (0.054, 0.015, 0.050),
        (0.17, 0.015, 0.10),
        (0.21, 0.015, -0.08),
        (0.054, 0.015, -0.040),
    )
    points = []
    for i in range(21):
        t = i / 20
        weights = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3)
        points.append(
            tuple(sum(w * v[k] for w, v in zip(weights, anchors)) for k in range(3))
        )
    for i, (a, c) in enumerate(zip(points, points[1:])):
        p.rod(head, f"head_harness_{i}", a, c, 0.006, "rubber")
    b.metadata["capabilities"] = [
        "coarse_xy_position",
        "coarse_head_clearance",
        "visible_210mm_chuck",
        "dry_probe_axis_inspection",
    ]
    b.metadata["limitations"].extend(
        [
            "Original open guides and ideal gravity-compensated joints; mass, inertia and servo parameters are authored, not OEM calibrated.",
            "The original sample site now lies on a 7 mm thick rigid retained disk. No insertion, clamping or sample deformation is simulated.",
            "The generic 180 micrometre cantilever and 5 micrometre tip are true-scale authored geometry, not an identified Bruker probe. Minimum coarse-Z standoff is about 1.5 mm, not an AFM scan/contact pose.",
            "Photo-informed shell, scanner face and camera tube; rear services, bearings and rigid head-local harness are inferred. No piezo feedback, force, optical focus, isolation or surface metrology.",
        ]
    )


def build_load_frame(b, base, params):
    p = Parts(b)
    b.feet(base, 0.76, 0.715)
    p.housing(
        base,
        "base_shell",
        [(0.024, 0.38, 0.3575, 0), (0.13, 0.375, 0.35, 0), (0.175, 0.355, 0.32, 0)],
        "dark",
    )
    p.box(base, "base", (0, 0, 0.089), (0.35, 0.32, 0.065), "dark")
    p.box(base, "front_trim", (0, -0.351, 0.058), (0.36, 0.007, 0.026))
    for i, x in enumerate((-0.295, 0.295)):
        p.box(
            base, f"column_foot_{i}", (x, 0.14, 0.166), (0.050, 0.075, 0.012), "metal"
        )
        p.box(base, f"column_{i}", (x, 0.14, 0.86), (0.045, 0.071, 0.69), "dark")
        for j, shift in enumerate((-0.03, 0.03)):
            p.box(
                base,
                f"column_strip_{i}_{j}",
                (x + shift, 0.072, 0.86),
                (0.003, 0.003, 0.69),
            )
        rail_x = x * 0.88
        p.cylinder(
            base, f"crosshead_rail_{i}", (rail_x, 0.034, 0.86), 0.014, 0.68, "bright"
        )
        for j, z in enumerate((0.192, 1.528)):
            p.box(
                base, f"rail_mount_{i}_{j}", (rail_x, 0.064, z), (0.026, 0.023, 0.012)
            )
    p.housing(
        base,
        "crown",
        [
            (1.54, 0.34, 0.078, 0.14),
            (1.60, 0.34, 0.078, 0.14),
            (1.64, 0.25, 0.06, 0.14),
        ],
        "dark",
    )
    p.cylinder(base, "lower_mount", (0, 0.015, 0.205), 0.068, 0.031, "bright")
    p.cylinder(base, "lower_stem", (0, 0.015, 0.26), 0.033, 0.033)
    p.box(base, "lower_grip", (0, 0.015, 0.32), (0.064, 0.055, 0.035), "dark")
    p.box(base, "coupon", (0, -0.013, 0.363), (0.012, 0.004, 0.032), "sample")
    b.site(base, "coupon", (0, -0.013, 0.363), 0.003)
    carriage = b.moving(
        base,
        "crosshead",
        (0, 0.14, 0.96),
        (0, 0, 1),
        (-0.25, 0.30),
        mass=8,
        kp=900,
        force=900,
    )
    p.box(carriage, "crosshead", (0, 0, 0), (0.24, 0.075, 0.044), "dark")
    p.box(carriage, "badge", (0, -0.077, 0), (0.035, 0.003, 0.017), "guard_red")
    for i, x in enumerate((-0.2596, 0.2596)):
        p.collar(
            carriage, f"crosshead_bearing_{i}", (x, -0.106, 0), 0.0146, 0.032, 0.04
        )
        p.box(
            carriage,
            f"crosshead_ear_{i}",
            (math.copysign(0.226, x), -0.082, 0),
            (0.019, 0.011, 0.038),
        )
    p.box(
        carriage, "loadcell_mount", (0, -0.105, -0.043), (0.051, 0.057, 0.020), "dark"
    )
    p.cylinder(carriage, "loadcell", (0, -0.125, -0.085), 0.042, 0.04)
    p.cylinder(carriage, "loadcell_ring", (0, -0.125, -0.105), 0.044, 0.01, "guard_red")
    p.cylinder(carriage, "upper_stem", (0, -0.125, -0.155), 0.023, 0.033, "bright")
    p.box(carriage, "upper_grip", (0, -0.1247, -0.207), (0.061, 0.0547, 0.025), "dark")
    for i, x in enumerate((-0.026, 0.026)):
        p.rod(
            carriage,
            f"jaw_rail_{i}",
            (x, -0.222, -0.207),
            (x, -0.155, -0.207),
            0.0025,
            "bright",
        )
        p.cylinder(
            carriage,
            f"jaw_stop_{i}",
            (x, -0.225, -0.207),
            0.004,
            0.002,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    jaw = b.moving(
        carriage,
        "upper_grip",
        (0, -0.19, -0.207),
        (0, 1, 0),
        (-0.012, 0),
        mass=0.15,
        kp=250,
    )
    p.box(jaw, "jaw_web", (0, 0, 0), (0.0229, 0.010, 0.024), "bright")
    for i, x in enumerate((-0.026, 0.026)):
        p.sleeve(jaw, f"jaw_bearing_{i}", (x, 0, 0), 1, 0.010, (0.0031, 0.0031), 0.0069)
        # The sleeve's outer side connects to the central jaw web, while the
        # cylindrical rail remains inside its true 0.6 mm radial-clear bore.
    for i, x in enumerate((-0.035, 0.035)):
        p.rod(base, f"lower_adjuster_{i}", (x, -0.05, 0.32), (x, -0.09, 0.32), 0.011)
    p.rod(base, "dashboard_arm", (0.31, 0.1, 0.64), (0.51, 0.1, 0.64), 0.013)
    p.box(base, "dashboard_back", (0.48, 0.108, 0.75), (0.095, 0.011, 0.12), "dark")
    b.screen(base, (0.48, 0.09, 0.78), 0.21, 0.31)
    p.box(
        base, "emergency_mount", (0.30, -0.17, 0.19), (0.024, 0.027, 0.016), "warning"
    )
    p.cylinder(
        base, "emergency_button", (0.30, -0.17, 0.209), 0.017, 0.014, "guard_red"
    )
    cable = ((0.49, 0.119, 0.66), (0.44, 0.19, 0.4), (0.33, 0.18, 0.15))
    for i, (a, c) in enumerate(zip(cable, cable[1:])):
        p.rod(base, f"dashboard_cable_{i}", a, c, 0.007, "rubber")
    b.metadata["capabilities"] = [
        "crosshead_position",
        "upper_grip_opening",
        "visible_inert_coupon",
        "empty_grip_inspection",
    ]
    b.metadata["limitations"].extend(
        [
            "Only lower-clamped inert coupon is present; the original crosshead range leaves at least 83 mm to the upper grip. This is an empty-grip/lower-coupon inspection, not tensile engagement.",
            "Open column and jaw guides preserve the original axes and ranges; hidden guide reactions, ideal gravity compensation and drive parameters are authored.",
            "Published 681TM envelope and source-visible dark columns, crown, load-cell chain and side dashboard guide the exterior. Exact NTU capacity/accessories, internal hardware and CAD are unconfirmed.",
            "No load measurement, tensile force, strain, deformation, contact stress or OEM interlock model.",
        ]
    )
