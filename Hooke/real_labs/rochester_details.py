"""Original access frames and free-carrier supports for Rochester surrogates.

Dimensions of these internal parts are authored mechanical estimates; they are
not manufacturer CAD. No sequencing, liquid handling or biological process is
implemented by this geometry.
"""


def sequencer_frame(b, base):
    """Leave actual holes for the retained drawer and upper tray envelopes."""
    for key, pos, half in (
        ("front_left", (-0.407, -0.448, 0.70), (0.005, 0.01, 0.61)),
        ("front_right", (0.407, -0.448, 0.70), (0.005, 0.01, 0.61)),
        ("drawer_divider", (0, -0.448, 0.365), (0.008, 0.01, 0.265)),
        ("lower_sill", (0, -0.448, 0.080), (0.402, 0.01, 0.020)),
        ("separator", (0, -0.448, 0.696), (0.402, 0.01, 0.066)),
        ("upper_lintel", (0, -0.448, 1.2025), (0.402, 0.01, 0.1075)),
    ):
        b.box(base, pos, half, "dark", collision=True, name=f"{b.name}__{key}")
    # Continuous guide ledges lie below the actual moving tray floor. Their
    # matching height supplies visible support without crossing the loading bay.
    for side in (-1, 1):
        b.box(
            base,
            (side * 0.337, -0.100, 0.868),
            (0.012, 0.335, 0.009),
            "metal",
            collision=True,
            name=f"{b.name}__flowcell_guide_{side}",
        )
    b.metadata["loading_interface"] = dict(
        design="Original open front frame and tray-mounted covers",
        source_basis="Manufacturer external envelope retained; compartment framing and retaining stops are estimated",
        free_carrier=True,
        loading_control="flowcell_tray",
        authored_tray_force_limit_n=20,
        contact_parameter_basis="20 N tray force cap and 12 ms retaining-wall contact response are numerical mechanical estimates, not manufacturer force/compliance calibration",
        limitations="Open/close and inert carrier retention only; no robot transfer, OEM latch or scientific sequencing model",
    )


def flowcell_retainer(b, tray, x, index):
    """Open-top pocket around a 90 x 140 x 10 mm freely supported carrier."""
    clearance, thickness, height = 0.0003, 0.002, 0.008
    for axis in (0, 1):
        extents = (0.045, 0.070)
        for sign in (-1, 1):
            p = [x, 0.07, -0.025 + height / 2]
            half = [0.0, 0.0, height / 2]
            p[axis] += sign * (extents[axis] + clearance + thickness / 2)
            half[axis] = thickness / 2
            half[1 - axis] = extents[1 - axis] + clearance
            b.box(
                tray,
                p,
                half,
                "metal",
                collision=True,
                name=f"{b.name}__flowcell_stop_{index}_{axis}_{sign}",
                # An authored 12 ms retaining-wall response keeps the free
                # carrier stable under the existing abrupt drawer commands.
                # This is a numerical contact setting, not OEM calibration.
                priority="1",
                solref=".012 1",
                solimp=".95 .99 .001",
            )
