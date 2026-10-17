"""Photo-informed Kyoto exterior and authored inert-slide inspection mechanics.

The cached official photos establish the installed instrument arrangement, not
hidden guides, optical prescriptions or dimensions. All bearings and the dry
working clearance below are explicit engineering estimates.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount
from real_labs.optical_reference_details import kyoto_microscope_shell
from real_labs.optical_stages import aperture_plate

CAMERA = "tirf_workspace__inspection_camera"
PREFIX = "kyoto_"


def _name(b, key):
    return f"{b.name}__{PREFIX}{key}"


def _box(b, parent, key, position, half, material="metal", **kwargs):
    return b.box(
        parent, position, half, material, collision=True, name=_name(b, key), **kwargs
    )


def _cylinder(b, parent, key, position, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent,
        position,
        radius,
        half,
        material,
        collision=True,
        name=_name(b, key),
        **kwargs,
    )


def _annulus(b, parent, key, position, inner, outer, half, material="metal", axis="z"):
    return annular_mount(
        b,
        parent,
        key,
        position,
        inner,
        outer,
        half,
        axis=axis,
        proxies=True,
        material=material,
        namespace="kyoto",
    )


def _sleeve(b, parent, key, position, half, opening, rotation):
    """Connected rectangular sleeve: local Z is its open translation axis."""
    node = ET.SubElement(
        parent,
        "body",
        name=_name(b, key),
        pos=" ".join(map(str, position)),
        euler=rotation,
    )
    aperture_plate(b, node, PREFIX + key, (0, 0, 0), half, opening, "metal")
    return node


def _stage(b, scope):
    # Two X rails are supported at both ends on the existing microscope foot.
    # 0.5 mm bearing clearance represents an ideal prismatic constraint, not
    # calibrated backlash or a second friction joint imposed on the servo.
    for side in (-1, 1):
        y = -0.09 + side * 0.070
        _box(b, scope, f"x_rail_{side}", (0, y, 0.181), (0.166, 0.007, 0.009), "bright")
        for end in (-1, 1):
            _box(
                b,
                scope,
                f"x_pedestal_{side}_{end}",
                (end * 0.156, y, 0.121),
                (0.010, 0.016, 0.051),
                "dark",
            )
    xstage = b.moving(
        scope,
        "slide_x",
        (0, -0.09, 0.205),
        (1, 0, 0),
        (-0.018, 0.018),
        mass=0.10,
        kp=200,
    )
    aperture_plate(
        b,
        xstage,
        "slide_x_plate",
        (0, 0, 0),
        (0.13, 0.10, 0.007),
        (0.048, 0.044),
        "dark",
    )
    for side in (-1, 1):
        # Rotation maps sleeve local X/Y to world Z/Y; the through axis is X.
        _sleeve(
            b,
            xstage,
            f"x_bearing_{side}",
            (0, side * 0.070, -0.024),
            (0.017, 0.013, 0.034),
            (0.0095, 0.0075),
            f"0 {math.pi/2} 0",
        )
        _box(
            b,
            xstage,
            f"y_rail_{side}",
            (side * 0.079, 0, 0.015),
            (0.003, 0.093, 0.003),
            "bright",
        )
        for end in (-1, 1):
            _box(
                b,
                xstage,
                f"y_rail_mount_{side}_{end}",
                (side * 0.079, end * 0.087, 0.0095),
                (0.006, 0.006, 0.0025),
                "dark",
            )
    ystage = b.moving(
        xstage, "slide_y", (0, 0, 0.019), (0, 1, 0), (-0.014, 0.014), mass=0.055, kp=180
    )
    aperture_plate(
        b,
        ystage,
        "slide_y_plate",
        (0, 0, 0.0015),
        (0.065, 0.055, 0.0055),
        (0.042, 0.024),
        "metal",
    )
    for side in (-1, 1):
        _sleeve(
            b,
            ystage,
            f"y_bearing_{side}",
            (side * 0.079, 0, -0.004),
            (0.007, 0.007, 0.030),
            (0.0035, 0.0035),
            f"{math.pi/2} 0 0",
        )
        _box(
            b,
            ystage,
            f"y_bearing_link_{side}",
            (side * 0.069, 0, 0.0015),
            (0.004, 0.025, 0.0015),
            "dark",
        )
    # This continuous annulus supports all glass edges while retaining the real
    # optical opening. Its upper face remains at the original slide underside.
    aperture_plate(
        b,
        ystage,
        "slide_holder_insert",
        (0, 0, 0.0066),
        (0.047, 0.027, 0.0004),
        (0.029, 0.009),
        "dark",
    )
    _box(b, ystage, "glass_slide", (0, 0, 0.0075), (0.0375, 0.0125, 0.0005), "glass")
    _box(b, ystage, "inert_target", (0, 0, 0.00808), (0.001, 0.001, 0.00008), "sample")
    _box(
        b,
        ystage,
        "registration_target",
        (0.010, -0.006, 0.00808),
        (0.0005, 0.0005, 0.00008),
        "orange",
    )
    b.site(ystage, "inert_target", (0, 0, 0.00816), 0.0003)
    b.site(ystage, "registration_target", (0.010, -0.006, 0.00816), 0.0002)
    for side in (-1, 1):
        x = side * 0.035
        # The post is clear of the glass. A connected bridge/pad retains it.
        _box(
            b,
            ystage,
            f"keeper_post_{side}",
            (x, 0.018, 0.008),
            (0.006, 0.003, 0.001),
            "dark",
        )
        _box(
            b,
            ystage,
            f"keeper_bridge_{side}",
            (x, 0.015, 0.0095),
            (0.008, 0.006, 0.0005),
            "dark",
        )
        _box(
            b,
            ystage,
            f"keeper_pad_{side}",
            (x, 0.011, 0.0085),
            (0.003, 0.0015, 0.0005),
            "rubber",
        )
        _cylinder(
            b,
            ystage,
            f"keeper_screw_{side}",
            (x, 0.018, 0.0105),
            0.0022,
            0.0005,
            "bright",
        )
        for y in (-0.043, 0.043):
            _cylinder(
                b,
                ystage,
                f"insert_screw_{side}_{y}",
                (side * 0.055, y, 0.0075),
                0.003,
                0.0005,
                "bright",
            )
    return xstage, ystage


def _focus(b, scope):
    _cylinder(b, scope, "focus_pedestal", (0, -0.09, 0.077), 0.037, 0.007, "dark")
    _annulus(
        b, scope, "focus_bearing", (0, -0.09, 0.112), 0.0186, 0.030, 0.028, "metal"
    )
    focus = b.moving(
        scope,
        "objective_focus",
        (0, -0.09, 0.132),
        (0, 0, 1),
        (-0.003, 0.003),
        mass=0.04,
        kp=180,
    )
    _cylinder(b, focus, "focus_carriage", (0, 0, 0.010), 0.018, 0.037, "bright")
    _cylinder(b, focus, "objective_barrel", (0, 0, 0.065), 0.014, 0.018, "dark")
    _cylinder(b, focus, "objective_collar", (0, 0, 0.073), 0.0145, 0.003, "warning")
    _cylinder(b, focus, "objective_neck", (0, 0, 0.0885), 0.009, 0.0055, "metal")
    _cylinder(b, focus, "objective_face", (0, 0, 0.0945), 0.006, 0.0005, "lens")
    b.site(focus, "objective_face", (0, 0, 0.095), 0.001)
    # Ideal entrance-pupil view of the inert underside through the glass.
    # This is body-mounted geometry, not calibrated microscope image formation.
    position = (0, 0, 0.0951)
    site = b.site(focus, "inspection_camera", position, 0.0002)
    ET.SubElement(
        focus,
        "camera",
        name=CAMERA,
        pos="0 0 .0951",
        quat="0 1 0 0",
        fovy="65",
        mode="fixed",
    )
    b.metadata["mounted_cameras"] = [
        dict(
            name=CAMERA,
            body=focus.get("name"),
            site=site,
            local_position_m=list(position),
            local_quaternion_wxyz=[0, 1, 0, 0],
            fovy_deg=65,
            purpose="Authored upward entrance-pupil pinhole surrogate attached to the moving objective; geometric dry target inspection, not calibrated microscopy",
        )
    ]


def _lead(b, base, key, points):
    """A fixed parked cable with named, physical end connectors."""
    for i, (start, end) in enumerate(zip(points, points[1:])):
        b.rod(
            base,
            start,
            end,
            0.003,
            "rubber",
            collision=True,
            name=_name(b, f"{key}_segment_{i}"),
        )
    return dict(
        name=key,
        start=list(points[0]),
        end=list(points[-1]),
        state="Fixed instrument-to-controller cable; no pressure/fluid or optical transfer model",
    )


def _table_instruments(b, base):
    # Visible photograph cues: unequal rail stages, mixed lens/mirror mounts,
    # a large end camera/light unit and two differently proportioned controllers.
    # This does not assert the photograph's optical routing or OEM internals.
    for index, (x, y, length, width, height) in enumerate(
        (
            (-0.72, 0.18, 0.22, 0.13, 0.29),
            (-0.37, 0.18, 0.16, 0.14, 0.29),
            (-0.66, -0.20, 0.15, 0.16, 0.23),
            (-0.28, -0.20, 0.11, 0.11, 0.195),
        )
    ):
        _box(
            b,
            base,
            f"optical_base_{index}",
            (x, y, 0.048),
            (length / 2, width / 2, 0.012),
            "dark",
        )
        for sign in (-1, 1):
            _box(
                b,
                base,
                f"translation_rail_{index}_{sign}",
                (x, y + sign * (width / 2 - 0.018), 0.065),
                (length / 2 - 0.008, 0.005, 0.005),
                "bright",
            )
        _box(
            b,
            base,
            f"translation_carriage_{index}",
            (x, y, 0.076),
            (length * 0.30, width * 0.40, 0.006),
            "metal",
        )
        post_top = height - (0.037 if index < 2 else 0.028)
        _cylinder(
            b,
            base,
            f"post_{index}",
            (x, y, (0.082 + post_top) / 2),
            0.012,
            (post_top - 0.082) / 2,
            "bright",
        )
        _cylinder(b, base, f"post_collar_{index}", (x, y, 0.116), 0.019, 0.018, "dark")
        if index < 2:
            _annulus(
                b,
                base,
                f"optic_holder_{index}",
                (x, y, height),
                0.026,
                0.039,
                0.012,
                "dark",
                axis="x",
            )
            _cylinder(
                b,
                base,
                f"optic_face_{index}",
                (x, y, height),
                0.025,
                0.0015,
                "lens",
                euler=(0, math.pi / 2, 0),
            )
            for side in (-1, 1):
                _annulus(
                    b,
                    base,
                    f"optic_retainer_{index}_{side}",
                    (x + side * 0.003, y, height),
                    0.024,
                    0.030,
                    0.0015,
                    "metal",
                    axis="x",
                )
        else:
            angle = 0.3 if index == 2 else -0.4
            _box(
                b,
                base,
                f"mirror_back_{index}",
                (x, y, height),
                (0.028, 0.009, 0.030),
                "dark",
                euler=(0, 0, angle),
            )
            _box(
                b,
                base,
                f"mirror_face_{index}",
                (x + 0.010 * math.sin(angle), y - 0.010 * math.cos(angle), height),
                (0.021, 0.001, 0.023),
                "bright",
                euler=(0, 0, angle),
            )
        _box(
            b,
            base,
            f"micrometer_bracket_{index}",
            (x - length / 2 + 0.005, y, 0.075),
            (0.005, 0.018, 0.015),
            "dark",
        )
        _cylinder(
            b,
            base,
            f"micrometer_{index}",
            (x - length / 2 - 0.020, y, 0.076),
            0.008,
            0.020,
            "metal",
            euler=(0, math.pi / 2, 0),
        )
    _box(b, base, "light_unit", (-0.96, 0.18, 0.21), (0.075, 0.10, 0.174), "cream")
    _cylinder(
        b,
        base,
        "light_adapter",
        (-0.86, 0.18, 0.29),
        0.032,
        0.025,
        "dark",
        euler=(0, math.pi / 2, 0),
    )
    # A connected tube enters the existing photo-informed side imaging port.
    _cylinder(
        b,
        base,
        "camera_tube",
        (0.028, 0.18, 0.29),
        0.026,
        0.118,
        "dark",
        euler=(0, math.pi / 2, 0),
    )
    _box(b, base, "tube_foot", (0.05, 0.18, 0.048), (0.075, 0.05, 0.012), "dark")
    _cylinder(b, base, "tube_post", (0.05, 0.18, 0.162), 0.012, 0.102, "bright")
    for index, (x, y, width, depth, height) in enumerate(
        ((-0.78, -0.42, 0.31, 0.15, 0.135), (-0.28, -0.42, 0.24, 0.15, 0.094))
    ):
        for dx in (-width / 2 + 0.025, width / 2 - 0.025):
            for dy in (-depth / 2 + 0.020, depth / 2 - 0.020):
                _cylinder(
                    b,
                    base,
                    f"controller_foot_{index}_{dx}_{dy}",
                    (x + dx, y + dy, 0.040),
                    0.008,
                    0.004,
                    "rubber",
                )
        _box(
            b,
            base,
            f"controller_{index}",
            (x, y, 0.044 + height / 2),
            (width / 2, depth / 2, height / 2),
            "metal" if index == 0 else "cream",
        )
        b.screen(
            base,
            (x - 0.03, y - depth / 2 - 0.001, 0.044 + height * 0.62),
            width * 0.42,
            height * 0.36,
        )
        for knob in range(2):
            _cylinder(
                b,
                base,
                f"controller_knob_{index}_{knob}",
                (x + width * 0.30, y - depth / 2 - 0.005, 0.068 + knob * 0.027),
                0.008,
                0.004,
                "dark",
                euler=(math.pi / 2, 0, 0),
            )
        for row in range(5):
            b.box(
                base,
                (x + width / 2 + 0.0001, y, 0.065 + row * 0.014),
                (0.0001, depth * 0.32, 0.0014),
                "dark",
                name=_name(b, f"vent_{index}_{row}"),
            )
        _cylinder(
            b,
            base,
            f"service_socket_{index}",
            (x, y + depth / 2 + 0.004, 0.10),
            0.008,
            0.004,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    routes = [
        (
            (0.51, 0.28, 0.79),
            (0.88, 0.38, 0.71),
            (0.92, 0.40, 0.046),
            (0.2, 0.43, 0.046),
            (-0.81, 0.34, 0.046),
            (-0.84, -0.29, 0.046),
            (-0.78, -0.337, 0.10),
        ),
        (
            (-0.96, 0.28, 0.22),
            (-1.015, 0.37, 0.14),
            (-1.015, 0.44, 0.046),
            (-0.12, 0.45, 0.046),
            (-0.08, -0.29, 0.046),
            (-0.28, -0.337, 0.10),
        ),
    ]
    # The first start matches the existing shell's attached camera lead bracket.
    _cylinder(
        b,
        base,
        "light_socket",
        (-0.96, 0.281, 0.22),
        0.008,
        0.006,
        "dark",
        euler=(math.pi / 2, 0, 0),
    )
    return [
        _lead(b, base, f"parked_service_{i}", route) for i, route in enumerate(routes)
    ]


def build_station(b, base, params):
    _box(b, base, "breadboard", (0, 0, 0.018), (1.08, 0.55, 0.018), "metal")
    scope = ET.SubElement(
        base, "body", name=b.name + "__microscope_frame", pos=".51 .1 .036"
    )
    kyoto_microscope_shell(b, scope)
    _stage(b, scope)
    _focus(b, scope)
    services = _table_instruments(b, base)
    required = [g.get("name") for g in base.iter("geom") if g.get("contype") == "1"]
    b.metadata["inspection_interface"] = dict(
        kind="inert_slide_registration",
        camera=CAMERA,
        objective_face=_name(b, "objective_face"),
        objective_site=b.metadata["sites"]["objective_face"],
        glass_geom=_name(b, "glass_slide"),
        target_geom=_name(b, "inert_target"),
        registration_geom=_name(b, "registration_target"),
        registration_site=b.metadata["sites"]["registration_target"],
        home_working_clearance_m=0.004,
        inspection_working_clearance_m=0.002,
        working_clearance_basis="Authored physical lens-face to glass-underside clearance; not an OEM objective working distance or optical focus criterion",
        bearing_clearance_m=0.0005,
        focus_bore_radial_clearance_m=0.0006,
        required_contact_geoms=required,
        parked_services=services,
        slide_attachment="Rigidly retained 75 x 25 x 1 mm inert slide, with visible connected keepers; no free loading or optical material calibration",
    )
    b.metadata["capabilities"] = [
        "xy_slide_position",
        "objective_focus_surrogate",
        "visible_2mm_inert_target",
        "measured_inert_fiducial_registration",
        "body_mounted_geometric_inspection_camera",
    ]
    b.metadata["limitations"].extend(
        [
            "Official exterior photos inform heterogeneous mounts, controllers and microscope shell; hidden guide geometry, dimensions and camera optics are authored estimates.",
            "The target is inert. No TIRF, fluorescence, optical image formation, calibrated focus, fluid or biological process is modeled.",
            "0.5/0.6 mm ideal bearing clearances and original position servos are mechanical surrogates, not nanometric resolution or backlash specifications.",
        ]
    )
