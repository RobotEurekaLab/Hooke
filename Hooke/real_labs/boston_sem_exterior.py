"""Source-visible fittings for the compact BU SEM reconstruction.

The BU facility photograph and cached Supra-family brochure establish an
inclined detector housing, roof fittings, a service cylinder and a cabinet.
Their dimensions, connections and identities are estimated. These fixed
exteriors add no vacuum, electron optics or measurement behavior.
"""

import math

from real_labs.equipment import _Builder


def add_exterior(root, base, namespace):
    """Attach a separately named exterior without changing the seven axes.

    Geometry uses the compact chamber's existing metre-scale coordinates.
    Contact shells remain solid; the fixed cable dressing is visual only.
    Existing inertias are explicit, so all added geoms have zero inferred mass.
    """
    b = _Builder(root, namespace)

    def geom(kind, key, pos, size, material="metal", **kwargs):
        return b.geom(
            base,
            kind,
            size,
            pos,
            material,
            collision=True,
            mass="0",
            name=f"{namespace}__{key}",
            **kwargs,
        )

    def box(key, pos, half, material="metal", **kwargs):
        return geom("box", key, pos, half, material, **kwargs)

    def cylinder(key, pos, radius, half, material="bright", **kwargs):
        return geom("cylinder", key, pos, (radius, half), material, **kwargs)

    # A cabinet face and corner trim are supported by the original case.
    for side in (-1, 1):
        cylinder(
            f"cabinet_trim_{side}",
            (side * 0.298, -0.305, 0.399),
            0.017,
            0.339,
            "dark",
        )
        box(
            f"cabinet_face_{side}",
            (side * 0.139, -0.312, 0.399),
            (0.137, 0.002, 0.331),
            "cream",
        )
    # Service covers stand on the desktop outside the fixed rails. The
    # front remains open over the telescopic guide exit: no disguised runway.
    for side in (-1, 1):
        box(
            f"lower_side_cover_{side}",
            (side * 0.238, 0.025, 0.870),
            (0.008, 0.190, 0.090),
            "cream",
        )
    box("lower_rear_cover", (0, 0.207, 0.870), (0.185, 0.008, 0.090), "cream")

    # The inclined cream housing is visible in the institutional photograph.
    # Its axis and mount are authored, not a recovered detector drawing.
    angle = -math.pi / 4
    direction = (math.sin(angle), 0, math.cos(angle))
    anchor = (-0.135, 0.005, 1.310)

    def along(distance):
        return tuple(a + distance * d for a, d in zip(anchor, direction))

    cylinder("inclined_foot", (-0.135, 0.005, 1.2765), 0.034, 0.0065)
    cylinder("inclined_mount", along(0.008), 0.039, 0.013, euler=(0, angle, 0))
    cylinder("inclined_neck", along(0.047), 0.021, 0.026, euler=(0, angle, 0))
    cylinder("inclined_collar", along(0.078), 0.035, 0.012, euler=(0, angle, 0))
    box(
        "inclined_housing",
        along(0.185),
        (0.048, 0.048, 0.097),
        "cream",
        euler=(0, angle, 0),
    )
    box(
        "inclined_end_cover",
        along(0.284),
        (0.044, 0.044, 0.002),
        "dark",
        euler=(0, angle, 0),
    )
    for index, x in enumerate((-0.142, -0.082)):
        cylinder(f"roof_flange_{index}", (x, -0.100, 1.275), 0.025, 0.005)
        cylinder(f"roof_fitting_{index}", (x, -0.100, 1.302), 0.018, 0.022, "dark")
        cylinder(f"roof_cap_{index}", (x, -0.100, 1.327), 0.021, 0.003)

    # A separate vertical service body is fixed to a side bracket. No
    # connection is allowed to float merely because both parts are static.
    box("service_bracket", (0.235, 0.164, 0.987), (0.050, 0.033, 0.010))
    cylinder("service_body", (0.238, 0.164, 1.127), 0.029, 0.130, "cream")
    cylinder("service_top", (0.238, 0.164, 1.261), 0.032, 0.004)
    cylinder("service_bottom", (0.238, 0.164, 1.001), 0.032, 0.004)
    for z in (1.052, 1.202):
        cylinder(f"service_band_{z}", (0.238, 0.164, z), 0.030, 0.006)

    # Fixed leads terminate at two modeled sockets. Routing is an estimate;
    # it does not pretend to be a flexible hose or a working vacuum line.
    side_axis = (0, math.pi / 2, 0)
    cylinder(
        "case_socket", (-0.316, 0.120, 0.650), 0.017, 0.006, "dark", euler=side_axis
    )
    socket = (-0.30, 0.057, 1.435)
    cylinder("detector_socket", socket, 0.012, 0.007, "dark", euler=(math.pi / 2, 0, 0))
    points = [
        (-0.30, 0.064, 1.435),
        (-0.35, 0.11, 1.410),
        (-0.39, 0.14, 1.320),
        (-0.40, 0.14, 1.100),
        (-0.40, 0.14, 0.870),
        (-0.38, 0.13, 0.710),
        (-0.345, 0.12, 0.665),
        (-0.322, 0.12, 0.650),
    ]
    for index, (start, end) in enumerate(zip(points, points[1:])):
        b.rod(
            base,
            start,
            end,
            0.004,
            "rubber",
            mass="0",
            name=f"{namespace}__fixed_lead_{index}",
        )
    return {
        "reference": "BU PML facility photo and Supra-family brochure",
        "dimension_basis": "Estimated exterior fittings; original compact mechanics unchanged",
        "limitations": [
            "No OEM CAD, confirmed detector identity or dimensional survey",
            "Cabinet panels and covers are fixed; they are not serviceable doors",
            "Fixed cable dressing has no cable forces or electrical behavior",
            "No vacuum, electron optics or calibrated measurement simulation",
        ],
    }
