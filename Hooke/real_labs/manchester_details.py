"""Source-informed static supports for Manchester's Gen3 substitute.

The reference shows a silver base connector and a curved yellow lead. Its other
end leaves the photograph; the parking bracket below is explicitly authored.
These rigid exterior shapes do not simulate cable deformation or clamp force.
"""


def retaining_pads(b, parent):
    """Bridge the two 2 mm jaw gaps without changing the specimen or its jig."""
    for side, x in (("left", -0.576), ("right", -0.524)):
        b.box(
            parent,
            (x, 0.24, 0.035),
            (0.001, 0.025, 0.009),
            "rubber",
            collision=True,
            name=f"{b.name}__retaining_pad_{side}",
        )


def _bezier(control, segments=12):
    points = []
    for index in range(segments + 1):
        t = index / segments
        weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3)
        points.append(
            tuple(
                sum(w * p[axis] for w, p in zip(weights, control)) for axis in range(3)
            )
        )
    return points


def parked_service_cable(b, parent):
    """Connect the base socket to an attached, explicitly parked connector."""

    def cylinder(key, center, radius, half_length, axis, material):
        b.cylinder(
            parent,
            center,
            radius,
            half_length,
            material,
            zaxis=axis,
            collision=True,
            name=f"{b.name}__service_{key}",
        )

    # The base end overlaps its fixed exterior; the sleeve meets the cable.
    cylinder("base_socket", (-0.051, -0.01, 0.12), 0.012, 0.012, (1, 0, 0), "metal")
    cylinder("base_coupling", (-0.065, -0.01, 0.12), 0.010, 0.006, (1, 0, 0), "bright")
    cylinder(
        "base_strain_relief", (-0.077, -0.01, 0.12), 0.008, 0.008, (1, 0, 0), "rubber"
    )

    # The plate meets the front of the existing left table clamp at y=-0.175.
    b.box(
        parent,
        (-0.09, -0.178, -0.055),
        (0.020, 0.003, 0.026),
        "metal",
        collision=True,
        name=f"{b.name}__service_parking_plate",
    )
    for x in (-0.102, -0.078):
        cylinder(
            f"parking_screw_{x}",
            (x, -0.182, -0.040),
            0.0025,
            0.002,
            (0, 1, 0),
            "bright",
        )
    cylinder(
        "parked_coupling", (-0.09, -0.191, -0.055), 0.011, 0.015, (0, 1, 0), "bright"
    )
    cylinder(
        "parked_strain_relief",
        (-0.09, -0.216, -0.055),
        0.008,
        0.012,
        (0, 1, 0),
        "rubber",
    )
    curves = (
        (
            (-0.085, -0.01, 0.12),
            (-0.235, -0.01, 0.12),
            (-0.29, -0.24, -0.15),
            (-0.27, -0.24, -0.25),
        ),
        (
            (-0.27, -0.24, -0.25),
            (-0.25, -0.24, -0.35),
            (-0.09, -0.30, -0.31),
            (-0.09, -0.30, -0.19),
        ),
        (
            (-0.09, -0.30, -0.19),
            (-0.09, -0.30, -0.07),
            (-0.09, -0.30, -0.055),
            (-0.09, -0.228, -0.055),
        ),
    )
    for curve_index, control in enumerate(curves):
        points = _bezier(control)
        for segment, (start, end) in enumerate(zip(points, points[1:])):
            b.rod(
                parent,
                start,
                end,
                0.006,
                "warning",
                collision=True,
                name=f"{b.name}__service_cable_{curve_index}_{segment}",
            )
    b.metadata["service_geometry"] = {
        "reference_cues": "Silver base connector and curved yellow lead",
        "authored_details": "Rigid lead route and under-table parked connector attached to existing clamp; final connection is outside the source photograph",
        "cable_dynamics": False,
        "clamp_force_model": False,
    }
