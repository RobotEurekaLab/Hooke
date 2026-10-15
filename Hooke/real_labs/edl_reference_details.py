"""Estimated exterior details of CMU's photographed 850e / 850BP assembly.

These original shapes depict installed equipment, not manufacturer CAD. Dial
faces and parked service leads are appearance cues, not measured process state.
"""

import math


REFERENCE_PHOTO = "https://www.cmu.edu/me/tpes/facilities/test-stand-2.jpg"


def _name(b, key):
    return f"{b.name}__edl_{key}"


def _front_cylinder(b, parent, key, position, radius, half_depth, material):
    return b.cylinder(
        parent,
        position,
        radius,
        half_depth,
        material,
        euler=(math.pi / 2, 0, 0),
        collision=True,
        name=_name(b, key),
    )


def _panel(b, parent, key, position, half_size, material="cream"):
    b.box(parent, position, half_size, material, name=_name(b, key))
    x, y, z = position
    width, depth, height = half_size
    for dx in (-width + 0.008, width - 0.008):
        for dz in (-height + 0.008, height - 0.008):
            b.cylinder(
                parent,
                (x + dx, y - depth - 0.001, z + dz),
                0.0025,
                0.001,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )


def _gauge(b, parent, key, x, z, radius, projection):
    # The small regulators project farther forward than the two upper dials.
    _front_cylinder(
        b,
        parent,
        key + "_stem",
        (x, -0.097 - projection / 2, z),
        radius * 0.45,
        projection / 2,
        "metal",
    )
    center_y = -0.097 - projection
    _front_cylinder(b, parent, key + "_rim", (x, center_y, z), radius, 0.007, "bright")
    b.cylinder(
        parent,
        (x, center_y - 0.0075, z),
        radius * 0.87,
        0.001,
        "cream",
        euler=(math.pi / 2, 0, 0),
    )
    face_y = center_y - 0.009
    for i in range(13):
        angle = math.radians(-45 + i * 22.5)
        points = [
            (x + radius * r * math.cos(angle), face_y, z + radius * r * math.sin(angle))
            for r in (0.64, 0.77)
        ]
        b.rod(parent, *points, 0.0007, "dark")
    b.rod(
        parent,
        (x, face_y, z),
        (x - radius * 0.41, face_y, z + radius * 0.39),
        0.0012,
        "dark",
    )


def edl_cabinets(b, parent):
    """Keep the original cabinet footprints and provide continuous supports."""
    for key, x, y, width, depth, height in (
        ("tester", 0.43, 0.10, 0.43, 0.48, 0.79),
        ("pressure", -0.47, 0.10, 0.40, 0.38, 0.40),
    ):
        for i, dx in enumerate((-width * 0.38, width * 0.38)):
            for j, dy in enumerate((-depth * 0.37, depth * 0.37)):
                b.cylinder(
                    parent,
                    (x + dx, y + dy, 0.018),
                    0.018,
                    0.018,
                    "rubber",
                    collision=True,
                    name=_name(b, f"{key}_foot_{i}_{j}"),
                )
        b.box(
            parent,
            (x, y, 0.036 + height / 2),
            (width / 2, depth / 2, height / 2),
            "cream",
            collision=True,
            name=_name(b, key + "_cabinet"),
        )
        _panel(
            b,
            parent,
            key + "_front",
            (x, y - depth / 2 - 0.003, 0.036 + height / 2),
            (width / 2 - 0.007, 0.003, height / 2 - 0.008),
        )
        # Separate rear cover seam and a recessed side grille.
        b.box(
            parent,
            (x + width / 2 + 0.0005, y + 0.12, 0.12 + height / 2),
            (0.0005, 0.035, height * 0.26),
            "dark",
        )
        for i in range(8):
            b.box(
                parent,
                (x + width / 2 + 0.0012, y + 0.12, 0.045 + height / 2 + i * 0.018),
                (0.001, 0.037, 0.0065),
                "cream",
            )

    # Muted, blank LCD and three discrete controllers; no invented readings.
    _panel(b, parent, "controller_mount", (0.525, -0.150, 0.566), (0.079, 0.003, 0.190))
    b.box(parent, (0.525, -0.156, 0.702), (0.063, 0.003, 0.023), "dark")
    b.box(
        parent,
        (0.525, -0.1595, 0.702),
        (0.053, 0.0005, 0.013),
        "plant",
        rgba=(0.25, 0.31, 0.095, 1),
        name=_name(b, "lcd_window"),
    )
    for i, z in enumerate((0.605, 0.522, 0.439)):
        b.box(
            parent,
            (0.525, -0.159, z),
            (0.054, 0.009, 0.029),
            "dark",
            collision=True,
            name=_name(b, f"controller_{i}"),
        )
        b.box(parent, (0.522, -0.169, z + 0.008), (0.042, 0.001, 0.011), "rubber")
        for x in (0.495, 0.518, 0.541, 0.562):
            b.box(parent, (x, -0.171, z - 0.017), (0.005, 0.002, 0.003), "metal")
    # Blank label plates preserve the asymmetric front without fictional text.
    for z, width in ((0.760, 0.155), (0.716, 0.090)):
        b.box(parent, (0.333, -0.1475, z), (width / 2, 0.0008, 0.006), "metal")
    _front_cylinder(
        b, parent, "stop_backplate", (0.529, -0.152, 0.336), 0.033, 0.006, "warning"
    )
    _front_cylinder(
        b, parent, "stop_button", (0.529, -0.168, 0.336), 0.023, 0.012, "guard_red"
    )
    b.box(parent, (0.551, -0.155, 0.260), (0.021, 0.009, 0.023), "dark")
    b.box(parent, (0.551, -0.166, 0.260), (0.007, 0.004, 0.014), "rubber")
    for key, x, z, color in (
        ("yellow", 0.370, 0.275, "warning"),
        ("blue", 0.390, 0.211, "blue"),
        ("red", 0.551, 0.168, "guard_red"),
        ("black", 0.551, 0.105, "rubber"),
    ):
        b.box(parent, (x, -0.148, z), (0.025, 0.003, 0.022), "metal")
        _front_cylinder(
            b, parent, key + "_terminal", (x, -0.165, z), 0.013, 0.017, color
        )
        _front_cylinder(
            b, parent, key + "_socket", (x, -0.184, z), 0.007, 0.003, "bright"
        )

    # A vented silver chassis and a smaller dark unit have separate supports.
    # The photo does not expose their full backs or establish their functions.
    for i, bottom, height, width, depth, material in (
        (0, 0.830, 0.106, 0.228, 0.195, "shell"),
        (1, 0.940, 0.050, 0.175, 0.150, "dark"),
    ):
        front = 0.075 - depth
        center_z = bottom + height / 2
        for j, x in enumerate((0.275, 0.585)):
            b.box(
                parent,
                (x, 0.075, bottom - 0.002),
                (0.018, 0.135, 0.002),
                "rubber",
                collision=True,
                name=_name(b, f"stack_{i}_foot_{j}"),
            )
        b.box(
            parent,
            (0.43, 0.075, center_z),
            (width, depth, height / 2),
            material,
            collision=True,
            name=_name(b, f"stack_{i}_case"),
        )
        _panel(
            b,
            parent,
            f"stack_{i}_panel",
            (0.43, front - 0.004, center_z),
            (width - 0.010, 0.004, height / 2 - 0.008),
            material,
        )
        for row in range(3):
            for col in range(8):
                b.box(
                    parent,
                    (
                        0.43 - width * 0.76 + col * width * 0.14,
                        front - 0.009,
                        center_z + (row - 1) * height * 0.22,
                    ),
                    (width * 0.046, 0.001, 0.0015),
                    "rubber",
                )
        for row in (-1, 0, 1):
            for column in (-2, -1, 0, 1, 2):
                b.box(
                    parent,
                    (
                        0.431 + width,
                        0.075 + column * depth * 0.31,
                        center_z + row * height * 0.25,
                    ),
                    (0.001, depth * 0.10, 0.002),
                    "rubber",
                )
        _front_cylinder(
            b,
            parent,
            f"stack_{i}_knob",
            (0.43 + width * 0.73, front - 0.015, center_z),
            0.009,
            0.006,
            "metal",
        )

    for i, x in enumerate((-0.57, -0.36)):
        _gauge(b, parent, f"dial_{i}", x, 0.31, 0.046, 0.006)
        _gauge(b, parent, f"regulator_{i}", x, 0.17, 0.030, 0.040)
        _front_cylinder(
            b, parent, f"hose_mount_{i}", (x, -0.099, 0.405), 0.018, 0.009, "bright"
        )
        _front_cylinder(
            b, parent, f"hose_port_{i}", (x, -0.127, 0.405), 0.009, 0.020, "metal"
        )
        _front_cylinder(
            b, parent, f"valve_{i}", (x, -0.111, 0.080), 0.014, 0.020, "bright"
        )
        b.box(parent, (x, -0.116, 0.098), (0.027, 0.008, 0.005), "plant")
        b.cylinder(parent, (x, -0.116, 0.093), 0.006, 0.009, "metal")
    # Blue electronics remain within the previous assembly envelope but no
    # longer intersect the left pressure cabinet.
    b.box(
        parent,
        (-0.773, 0.14, 0.015),
        (0.087, 0.12, 0.015),
        "dark",
        collision=True,
        name=_name(b, "blue_base"),
    )
    for i, z in enumerate((0.065, 0.135)):
        b.box(
            parent,
            (-0.773, 0.14, z),
            (0.087, 0.13, 0.035),
            "blue",
            collision=True,
            name=_name(b, f"blue_case_{i}"),
        )
        b.box(parent, (-0.773, 0.008, z), (0.065, 0.002, 0.022), "metal")
        b.box(parent, (-0.778, 0.005, z + 0.004), (0.043, 0.001, 0.010), "dark")


def _curve(b, parent, key, spans, radius, material):
    """Cubic spans retain exact endpoints and stay inside their control hulls."""
    points = [spans[0][0]]
    for controls in spans:
        for step in range(1, 13):
            t = step / 12
            weights = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3)
            points.append(
                tuple(
                    sum(w * p[axis] for w, p in zip(weights, controls))
                    for axis in range(3)
                )
            )
    for i, (first, second) in enumerate(zip(points, points[1:])):
        b.rod(
            parent, first, second, radius, material, name=_name(b, f"{key}_route_{i}")
        )


def edl_service_hardware(b, parent):
    """Supported fixed hardware and visibly parked, disconnected service ends."""
    b.box(
        parent,
        (-0.12, 0.14, 0.012),
        (0.065, 0.070, 0.012),
        "metal",
        collision=True,
        name=_name(b, "service_foot"),
    )
    b.box(
        parent,
        (-0.078, 0.14, 0.16),
        (0.009, 0.030, 0.148),
        "bright",
        collision=True,
        name=_name(b, "service_bracket"),
    )
    b.box(
        parent,
        (-0.114, 0.14, 0.129),
        (0.045, 0.030, 0.006),
        "bright",
        collision=True,
        name=_name(b, "service_shelf"),
    )
    b.cylinder(
        parent,
        (-0.12, 0.14, 0.229),
        0.026,
        0.094,
        "bright",
        collision=True,
        name=_name(b, "service_canister"),
    )
    b.cylinder(
        parent,
        (-0.12, 0.14, 0.329),
        0.010,
        0.010,
        "metal",
        collision=True,
        name=_name(b, "service_top_port"),
    )
    b.cylinder(parent, (-0.12, 0.14, 0.137), 0.028, 0.006, "metal")
    routes = (
        (
            "white_left",
            "hose_port_0",
            (-0.57, -0.147, 0.405),
            (-0.82, -0.27, 0.39),
            (-0.84, -0.52, 0.15),
            (-0.73, -0.59, 0.030),
            (-0.70, -0.61, 0.012),
            (-0.64, -0.61, 0.011),
            (-0.58, -0.61, 0.011),
            0.0035,
            "cream",
        ),
        (
            "white_right",
            "hose_port_1",
            (-0.36, -0.147, 0.405),
            (-0.43, -0.24, 0.37),
            (-0.54, -0.46, 0.15),
            (-0.51, -0.58, 0.030),
            (-0.49, -0.65, 0.012),
            (-0.45, -0.66, 0.011),
            (-0.40, -0.66, 0.011),
            0.0035,
            "cream",
        ),
        (
            "red",
            "red_socket",
            (0.551, -0.187, 0.168),
            (0.76, -0.28, 0.16),
            (0.82, -0.47, 0.032),
            (0.70, -0.56, 0.019),
            (0.58, -0.65, 0.011),
            (0.45, -0.65, 0.011),
            (0.38, -0.65, 0.011),
            0.006,
            "guard_red",
        ),
        (
            "black",
            "black_socket",
            (0.551, -0.187, 0.105),
            (0.66, -0.27, 0.09),
            (0.72, -0.49, 0.021),
            (0.59, -0.60, 0.033),
            (0.51, -0.67, 0.030),
            (0.39, -0.69, 0.023),
            (0.29, -0.69, 0.011),
            0.006,
            "rubber",
        ),
    )
    endpoints = []
    for key, port, start, a, c, middle, d, e, end, radius, material in routes:
        _curve(
            b,
            parent,
            key,
            ((start, a, c, middle), (middle, d, e, end)),
            radius,
            material,
        )
        # A loose metal connector rests on the bench. It is deliberately not
        # attached to the moving coupon: motion cannot stretch a cosmetic lead.
        _front_cylinder(
            b, parent, key + "_parked_connector", end, 0.011, 0.018, "bright"
        )
        b.cylinder(
            parent,
            (end[0], end[1] - 0.0185, end[2]),
            0.006,
            0.001,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
        endpoints.append(
            dict(
                route=key,
                attached_to=_name(b, port),
                parked_connector=_name(b, key + "_parked_connector"),
            )
        )
    # One short fixed hose reaches the supported canister. Its routing is an
    # exterior estimate, not a claimed pressure circuit of the real instrument.
    _front_cylinder(
        b, parent, "service_lower_port", (-0.275, -0.100, 0.16), 0.010, 0.014, "metal"
    )
    _curve(
        b,
        parent,
        "service",
        (
            (
                (-0.275, -0.114, 0.16),
                (-0.20, -0.14, 0.18),
                (-0.13, -0.07, 0.23),
                (-0.10, 0.01, 0.29),
            ),
            (
                (-0.10, 0.01, 0.29),
                (-0.07, 0.09, 0.35),
                (-0.12, 0.12, 0.37),
                (-0.12, 0.14, 0.339),
            ),
        ),
        0.0035,
        "cream",
    )
    b.metadata["visual_reference_details"] = {
        "source": REFERENCE_PHOTO,
        "basis": "Original photo-informed estimates, not OEM CAD or measured dimensions.",
        "observed_features": [
            "Asymmetric LCD / three-controller front and separate lower electrical terminals",
            "A vented silver chassis and smaller dark unit above the tall tester",
            "Two large dial gauges and projecting lower regulators on the shorter cabinet",
            "Green valve levers, blue electronics, a supported metal canister and curved bench leads",
        ],
        "estimated_geometry": "Dimensions, hidden mounts, cable routes and connector internals are estimates.",
        "service_endpoints": endpoints,
        "service_model": "Fixed exterior hoses and parked disconnected leads; no flexible-body, pressure or electrical simulation.",
        "instrument_state": "Blank controller windows and static illustrative gauge needles convey no measured readings.",
        "fixture_scope": "The separate actuated dry coupon fixture remains an authored mechanical demonstrator.",
    }
