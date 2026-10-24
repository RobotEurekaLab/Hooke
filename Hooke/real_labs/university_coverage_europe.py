"""Six inspected European laboratory references with bounded dry mechanisms.

Original exterior geometry, estimated dimensions and authored travel ranges are
not OEM CAD or calibrated optics, physiology, plasma or aerodynamics models.
The official source photographs are retained locally, never embedded here.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

UCL = "https://www.sainsburywellcome.org/content/advanced-microscopy"
KCL = "https://www.kcl.ac.uk/research/facilities/Microscopy-Innovation-Centre"
LSE = "https://www.lse.ac.uk/behavioural-lab/our-facilities/facilities-list/facilities-photos"
IP = "https://portail.polytechnique.edu/luli/en/facilities/luli2000/experimentals-room"
UVA = "https://www.fon.hum.uva.nl/speechlab/studios.html"
DURHAM = "https://www.durham.ac.uk/departments/academic/engineering/research/nodes/fluid-mechanics-/facilitiescapabilities/"


def _table(b, parent, xy=(0, 0), half=(0.8, 0.4), height=0.75, material="dark"):
    x, y = xy
    for dx in (-half[0] + 0.06, half[0] - 0.06):
        for dy in (-half[1] + 0.06, half[1] - 0.06):
            b.box(
                parent,
                (x + dx, y + dy, (height - 0.04) / 2),
                (0.035, 0.035, (height - 0.04) / 2),
                "metal",
                collision=True,
            )
    b.box(parent, (x, y, height - 0.02), (*half, 0.02), material, collision=True)


def _monitor(b, parent, pos, width=0.45):
    x, y, z = pos
    b.box(parent, (x, y, z), (width / 2, 0.025, width * 0.31), "dark", collision=True)
    b.box(parent, (x, y - 0.027, z), (width * 0.46, 0.002, width * 0.27), "screen")
    b.rod(parent, (x, y, z - width * 0.31), (x, y, z - 0.35), 0.018, "metal")
    b.box(parent, (x, y, z - 0.36), (0.12, 0.09, 0.01), "dark", collision=True)


def _breadboard(b, parent, xy, half, z):
    _table(b, parent, xy, half, z)
    # Sparse procedural hole marks convey the board without image textures.
    for i in range(int(half[0] * 20)):
        for j in range(int(half[1] * 20)):
            b.cylinder(
                parent,
                (
                    xy[0] - half[0] + 0.05 + 0.1 * i,
                    xy[1] - half[1] + 0.05 + 0.1 * j,
                    z + 0.0005,
                ),
                0.004,
                0.0005,
                "rubber",
            )


def _slide(b, parent, key, pos, axis, limits, half=(0.08, 0.06, 0.018), guide_gap=0):
    """Supported carriage with full-travel guides and explicit bearing clearance.

    A 0.5 mm horizontal or 1 mm vertical numerical bearing gap avoids treating
    an ideal joint and a solid bearing surface as competing constraints. These
    are exterior supports around a constrained joint, not measured OEM fits.
    """
    horizontal = axis[2] == 0
    if horizontal:
        a = 0 if axis[0] else 1
        travel = max(abs(limits[0]), abs(limits[1]))
        for side in (-1, 1):
            p = list(pos)
            p[1 - a] += side * (half[1 - a] - 0.018)
            p[2] -= half[2] + 0.0095
            size = [0.01, 0.01, 0.009]
            size[a] = half[a] + travel + 0.015
            b.box(parent, p, size, "bright", collision=True)
    else:
        # Rear guide column and an attached bearing block carry the vertical axis.
        b.box(
            parent,
            (
                pos[0],
                pos[1] + half[1] + 0.026 + guide_gap,
                pos[2] + (limits[0] + limits[1]) / 2,
            ),
            (0.027, 0.025, (limits[1] - limits[0]) / 2 + half[2] + 0.035),
            "bright",
            collision=True,
        )
    carriage = b.moving(parent, key, pos, axis, limits, mass=0.5)
    b.box(carriage, (0, 0, 0), half, "dark", collision=True)
    if not horizontal:
        b.box(
            carriage,
            (0, half[1] - 0.007 + guide_gap / 2, 0),
            (0.035, 0.007 + guide_gap / 2, half[2]),
            "blue",
            collision=True,
        )
    return carriage


def _ucl_optics(b, base, params):
    _breadboard(b, base, (0, 0), (1.05, 0.57), 0.80)
    # Open optical breadboard: separated posts, camera path and electronics.
    for x, y, radius in (
        (-0.66, 0.23, 0.045),
        (-0.43, 0.23, 0.055),
        (-0.18, 0.23, 0.045),
        (0.15, 0.23, 0.055),
    ):
        b.cylinder(base, (x, y, 0.875), 0.018, 0.075, "bright", collision=True)
        b.ring(base, (x, y, 0.99), radius, 0.010, "dark", "yz")
        b.geom(
            base,
            "cylinder",
            (radius - 0.009, 0.003),
            (x, y, 0.99),
            "lens",
            euler=(0, math.pi / 2, 0),
        )
    xaxis = _slide(b, base, "specimen_x", (0.32, -0.16, 0.85), (1, 0, 0), (-0.05, 0.05))
    yaxis = _slide(
        b,
        xaxis,
        "specimen_y",
        (0, 0, 0.055),
        (0, 1, 0),
        (-0.035, 0.035),
        (0.065, 0.045, 0.015),
    )
    b.box(yaxis, (0, 0, 0.02), (0.0375, 0.0125, 0.005), "glass", collision=True)
    b.site(yaxis, "calibration_slide", (0, 0, 0.025), 0.01)
    for dy in (-0.045, 0.045):
        for dz in (-0.045, 0.045):
            b.rod(
                base,
                (-0.70, 0.23 + dy, 0.99 + dz),
                (0.18, 0.23 + dy, 0.99 + dz),
                0.004,
                "bright",
            )
    b.box(base, (0.72, 0.24, 0.87), (0.18, 0.14, 0.07), "cream", collision=True)
    for z in (0.84, 0.89):
        b.box(base, (0.72, 0.095, z), (0.13, 0.004, 0.012), "dark")
    b.rod(base, (0.23, 0.23, 0.99), (0.37, 0.23, 0.99), 0.025, "dark")
    b.box(base, (0.43, 0.23, 0.99), (0.06, 0.07, 0.06), "blue", collision=True)
    _table(b, base, (-1.45, 0.05), (0.32, 0.46), 0.8)
    _monitor(b, base, (-1.45, 0.18, 1.17), 0.49)
    b.metadata["capabilities"] += ["dry_calibration_slide_xy"]


def _kcl_confocal(b, base, params):
    _breadboard(b, base, (0.34, 0), (1.23, 0.54), 0.78)
    # Installed photograph: inverted body in black enclosure, right laser box,
    # dual operator monitors at left. Stage is visible through a front opening.
    b.box(base, (-0.28, 0.04, 0.84), (0.25, 0.27, 0.06), "cream", collision=True)
    b.box(base, (-0.28, 0.27, 1.08), (0.18, 0.065, 0.19), "cream", collision=True)
    b.box(base, (-0.28, 0.22, 1.34), (0.28, 0.13, 0.07), "dark", collision=True)
    for x in (-0.56, 0.0):
        b.box(base, (x, 0.03, 1.19), (0.022, 0.23, 0.18), "dark", collision=True)
    b.box(base, (-0.28, 0.03, 1.385), (0.30, 0.24, 0.025), "dark", collision=True)
    b.cylinder(base, (-0.28, -0.06, 0.99), 0.028, 0.06, "bright", collision=True)
    for y in (-0.142, 0.022):
        b.box(
            base, (-0.28, y, 0.97225), (0.19, 0.018, 0.07225), "cream", collision=True
        )
    xaxis = _slide(
        b,
        base,
        "stage_x",
        (-0.28, -0.06, 1.075),
        (1, 0, 0),
        (-0.035, 0.035),
        (0.14, 0.10, 0.012),
    )
    yaxis = _slide(
        b,
        xaxis,
        "stage_y",
        (0, 0, 0.045),
        (0, 1, 0),
        (-0.025, 0.025),
        (0.11, 0.065, 0.01),
    )
    b.box(yaxis, (0, 0, 0.014), (0.0375, 0.0125, 0.004), "glass", collision=True)
    b.site(yaxis, "retained_slide", (0, 0, 0.018), 0.008)
    for x in (-0.37, -0.19):
        b.rod(base, (x, -0.14, 1.28), (x, -0.33, 1.21), 0.028, "cream")
        b.rod(base, (x, -0.33, 1.21), (x, -0.36, 1.20), 0.032, "dark")
    b.box(base, (0.92, 0.20, 0.96), (0.52, 0.25, 0.18), "dark", collision=True)
    b.box(base, (0.92, -0.055, 0.82), (0.50, 0.006, 0.025), "bright")
    b.box(base, (0.31, -0.23, 0.95), (0.16, 0.21, 0.17), "shell", collision=True)
    _table(b, base, (-1.55, 0.04), (0.55, 0.44), 0.76)
    for x in (-1.82, -1.28):
        _monitor(b, base, (x, 0.18, 1.13), 0.48)
    b.metadata["capabilities"] += ["retained_slide_xy_alignment"]


def _lse_eye_tracking(b, base, params):
    _table(b, base, (0, 0), (0.74, 0.40), 0.74, "cream")
    _monitor(b, base, (0, 0.18, 1.10), 0.49)
    b.box(base, (0, -0.04, 0.755), (0.20, 0.075, 0.015), "dark", collision=True)
    # Adjustable head support has two vertical guides, not a floating chin pad.
    for x in (-0.19, 0.19):
        b.cylinder(base, (x, -0.30, 0.755), 0.05, 0.015, "dark", collision=True)
        b.rod(base, (x, -0.30, 0.77), (x, -0.30, 1.30), 0.012, "dark", collision=True)
    chin = b.moving(base, "chin_height", (0, -0.30, 0.98), (0, 0, 1), (-0.06, 0.10))
    b.box(chin, (0, 0, 0), (0.16, 0.033, 0.012), "dark", collision=True)
    b.geom(chin, "ellipsoid", (0.06, 0.045, 0.016), (0, -0.005, 0.018), "cream")
    for x in (-0.19, 0.19):
        for dx in (-0.023, 0.023):
            b.box(chin, (x + dx, 0, 0), (0.007, 0.03, 0.024), "dark", collision=True)
        for dy in (-0.023, 0.023):
            b.box(chin, (x, dy, 0), (0.016, 0.007, 0.024), "dark", collision=True)
    b.site(chin, "chin_support", (0, -0.005, 0.034), 0.015)
    b.rod(base, (-0.19, -0.30, 1.29), (0.19, -0.30, 1.29), 0.014, "dark")
    # Eye camera tilt is an authored alignment hinge on a table-mounted cradle.
    b.box(base, (0.35, -0.15, 0.78), (0.09, 0.09, 0.04), "dark", collision=True)
    camera = b.moving(
        base,
        "camera_tilt",
        (0.35, -0.15, 0.88),
        (1, 0, 0),
        (-0.2, 0.2),
        kind="hinge",
        mass=0.18,
    )
    camera.set("euler", numbers((0, 0, -1.166)))
    b.box(camera, (0, 0, 0), (0.055, 0.05, 0.035), "dark", collision=True)
    b.geom(
        camera,
        "cylinder",
        (0.022, 0.02),
        (0, -0.07, 0),
        "lens",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    b.site(camera, "eye_camera", (0, -0.09, 0), 0.01)
    _table(b, base, (1.28, -0.10), (0.42, 0.48), 0.74, "cream")
    _monitor(b, base, (1.28, 0.03, 1.10), 0.50)
    b.metadata["capabilities"] += ["head_support_height", "eye_camera_alignment"]


def _milka(b, base, params):
    # Published sphere diameter about 2 m. Exterior-only collider is intentional:
    # this scene operates the external diagnostic mount, not a hidden target.
    b.geom(base, "sphere", (1.0,), (0, 0.45, 1.65), "metal", collision=True)
    for x in (-0.68, 0.68):
        b.rod(
            base, (x, 0.45, 0.03), (x * 0.7, 0.45, 0.88), 0.065, "cream", collision=True
        )
        b.box(base, (x, 0.45, 0.035), (0.18, 0.30, 0.035), "cream", collision=True)
    for az, el in (
        (-1.57, 0),
        (-2.4, 0.25),
        (-0.7, 0.35),
        (0, 0.0),
        (math.pi, 0.0),
        (-1.8, 0.9),
        (-0.9, 0.85),
    ):
        direction = (
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        )
        p = (
            direction[0] * 0.94,
            0.45 + direction[1] * 0.94,
            1.65 + direction[2] * 0.94,
        )
        q = (
            direction[0] * 1.13,
            0.45 + direction[1] * 1.13,
            1.65 + direction[2] * 1.13,
        )
        b.rod(base, p, q, 0.15, "bright")
        b.geom(base, "sphere", (0.14,), q, "dark")
    _breadboard(b, base, (0, -1.27), (0.65, 0.38), 0.86)
    xaxis = _slide(
        b,
        base,
        "diagnostic_x",
        (0, -1.27, 0.92),
        (1, 0, 0),
        (-0.10, 0.10),
        (0.12, 0.10, 0.022),
    )
    zaxis = _slide(
        b,
        xaxis,
        "diagnostic_height",
        (0, 0, 0.25),
        (0, 0, 1),
        (0, 0.12),
        (0.08, 0.06, 0.024),
        guide_gap=0.012,
    )
    b.box(xaxis, (0, 0.12, 0.14), (0.045, 0.035, 0.14), "bright", collision=True)
    b.geom(
        zaxis,
        "cylinder",
        (0.055, 0.065),
        (0, 0, 0.085),
        "dark",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    b.site(zaxis, "alignment_target", (0, 0.065, 0.085), 0.015)
    _table(b, base, (1.65, 0.3), (0.40, 0.85), 0.93, "cream")
    b.box(base, (1.65, 0.65, 1.06), (0.32, 0.22, 0.13), "blue", collision=True)
    b.metadata["capabilities"] += ["external_diagnostic_xy_alignment"]


def _uva_studio(b, base, params):
    # Participant chair is fixed, vacant and supported by a pedestal.
    b.cylinder(base, (0.15, -0.40, 0.035), 0.42, 0.035, "bright", collision=True)
    b.cylinder(base, (0.15, -0.40, 0.26), 0.07, 0.19, "bright", collision=True)
    b.box(base, (0.15, -0.40, 0.48), (0.28, 0.28, 0.045), "rubber", collision=True)
    b.rod(
        base, (-0.06, -0.12, 0.42), (-0.06, 0.24, 0.18), 0.018, "bright", collision=True
    )
    b.rod(
        base, (0.36, -0.12, 0.42), (0.36, 0.24, 0.18), 0.018, "bright", collision=True
    )
    b.box(base, (0.15, 0.26, 0.16), (0.25, 0.16, 0.02), "dark", collision=True)
    b.box(
        base,
        (0.15, -0.68, 0.78),
        (0.28, 0.05, 0.27),
        "rubber",
        collision=True,
        euler=(0.14, 0, 0),
    )
    for x in (-0.2, 0.5):
        b.rod(base, (x, -0.42, 0.44), (x, -0.42, 0.68), 0.018, "bright", collision=True)
        b.box(base, (x, -0.40, 0.70), (0.055, 0.26, 0.035), "rubber", collision=True)
    # Photograph shows a white articulated monitor arm and an under-screen
    # tracker. Authored arm yaw and monitor pitch adjust stimulus presentation.
    b.box(base, (-0.95, 0.50, 1.30), (0.045, 0.10, 0.35), "cream", collision=True)
    b.rod(
        base, (-0.95, 0.50, 0.03), (-0.95, 0.50, 1.62), 0.045, "cream", collision=True
    )
    arm = b.moving(
        base,
        "stimulus_arm_yaw",
        (-0.95, 0.50, 1.72),
        (0, 0, 1),
        (-0.18, 0.18),
        kind="hinge",
        mass=0.6,
    )
    b.box(arm, (0.51, 0, 0), (0.45, 0.035, 0.035), "cream", collision=True)
    b.rod(arm, (0.96, 0.065, 0), (0.96, 0.065, -0.30), 0.025, "cream", collision=True)
    screen = b.moving(
        arm,
        "stimulus_screen_pitch",
        (0.96, 0, -0.30),
        (1, 0, 0),
        (-0.15, 0.15),
        kind="hinge",
        mass=0.4,
    )
    b.box(screen, (0, 0, 0), (0.25, 0.026, 0.15), "dark", collision=True)
    b.box(screen, (0, -0.029, 0), (0.23, 0.002, 0.13), "screen")
    b.box(screen, (0, -0.01, -0.19), (0.13, 0.055, 0.035), "metal", collision=True)
    b.site(screen, "stimulus_center", (0, -0.03, 0), 0.02)
    b.site(screen, "tracker_center", (0, -0.065, -0.19), 0.012)
    # Two real microphone stands and rear keyboard shelf are source cues.
    for x in (0.65, 0.95):
        b.rod(base, (x, 0.52, 0.04), (x, 0.52, 1.15), 0.014, "dark", collision=True)
        b.rod(base, (x, 0.52, 1.15), (x - 0.12, 0.37, 1.45), 0.012, "dark")
        b.geom(base, "sphere", (0.027,), (x - 0.12, 0.37, 1.45), "dark")
        for a in (0, 2.1, 4.2):
            b.rod(
                base,
                (x, 0.52, 0.18),
                (x + 0.18 * math.cos(a), 0.52 + 0.18 * math.sin(a), 0.025),
                0.012,
                "dark",
            )
    _table(b, base, (1.25, 0.20), (0.25, 0.42), 0.90, "cream")
    b.box(base, (1.25, 0.14, 0.913), (0.20, 0.07, 0.013), "cream", collision=True)
    b.metadata["capabilities"] += ["stimulus_arm_yaw", "stimulus_screen_pitch"]


def _durham_tunnel(b, base, params):
    # Fixed-ground mode: published 1.65 m turntable centred 1.8 m downstream
    # from nozzle; published test-section dimensions are 1.75 x 1.15 x 5.5 m.
    b.box(
        base,
        (0, -0.95, 0.22),
        (1.35, 2.75, 0.22),
        "metal",
        collision=True,
        name=b.name + "__test_floor",
    )
    for x in (-0.955, 0.955):
        b.box(
            base,
            (x, 2.20, 1.015),
            (0.08, 0.40, 0.575),
            "cream",
            collision=True,
            name=b.name + ("__nozzle_left" if x < 0 else "__nozzle_right"),
        )
    b.box(
        base,
        (0, 2.20, 1.63),
        (1.035, 0.40, 0.04),
        "cream",
        collision=True,
        name=b.name + "__nozzle_top",
    )
    b.box(base, (0, 2.58, 1.015), (0.875, 0.015, 0.575), "dark", collision=True)
    b.cylinder(base, (0, 0, 0.46), 0.86, 0.02, "dark", collision=True)
    turn = b.moving(
        base,
        "model_yaw",
        (0, 0, 0.5055),
        (0, 0, 1),
        (-0.35, 0.35),
        kind="hinge",
        mass=2,
    )
    b.cylinder(
        turn,
        (0, 0, 0),
        0.825,
        0.025,
        "bright",
        collision=True,
        name=b.name + "__published_1650mm_turntable",
    )
    b.box(turn, (0, 0, 0.17), (0.29, 0.61, 0.12), "blue", collision=True)
    b.box(turn, (0, 0.04, 0.36), (0.255, 0.34, 0.07), "blue", collision=True)
    b.box(turn, (0, -0.31, 0.35), (0.24, 0.035, 0.06), "lens", euler=(0.28, 0, 0))
    for x in (-0.29, 0.29):
        for y in (-0.38, 0.38):
            b.geom(
                turn,
                "cylinder",
                (0.12, 0.035),
                (x, y, 0.13),
                "rubber",
                euler=(0, math.pi / 2, 0),
                collision=True,
            )
    b.site(turn, "vehicle_model", (0, 0, 0.43), 0.03)
    # Separate overhead traverse, tall enough for model rotation at all limits.
    for x in (-1.06, 1.06):
        b.box(base, (x, -0.60, 1.40), (0.045, 0.06, 0.96), "bright", collision=True)
    b.box(base, (0, -0.60, 2.40), (1.11, 0.08, 0.04), "bright", collision=True)
    probe = _slide(
        b,
        base,
        "probe_height",
        (0.75, -0.60, 1.70),
        (0, 0, 1),
        (-0.2, 0.2),
        (0.05, 0.04, 0.04),
        guide_gap=0.12,
    )
    b.rod(
        base,
        (0.75, -0.414, 1.96),
        (0.75, -0.414, 2.40),
        0.022,
        "bright",
        collision=True,
    )
    b.rod(
        base, (0.75, -0.414, 2.40), (0.75, -0.60, 2.40), 0.022, "bright", collision=True
    )
    b.rod(probe, (0, 0, 0), (0, 0, -0.35), 0.008, "bright", collision=True)
    b.rod(probe, (0, 0, -0.35), (0, 0.10, -0.35), 0.004, "bright", collision=True)
    b.site(probe, "probe_tip", (0, 0.10, -0.35), 0.01)
    b.metadata["capabilities"] += ["dry_vehicle_yaw", "probe_height_alignment"]


def _room_detail(world, definition):
    identifier = definition["id"]
    width, depth, _ = definition["room"]["size"]
    y = depth / 2 - 0.09

    def add(key, pos, half, color, collision=False):
        return box(world, identifier + "_" + key, half, pos, color, collision=collision)

    if identifier == "kcl_mic_stellaris":
        # Black light-isolation curtain and high service outlets visible in photo.
        for i in range(13):
            add(
                f"curtain_{i}",
                (-2.48 + i * 0.055, 0.75, 1.35),
                (0.025, 0.65, 1.30),
                (0.025, 0.03, 0.04, 1),
            )
        add("wall_services", (0, y, 1.13), (2.3, 0.035, 0.045), (0.8, 0.8, 0.77, 1))
    elif identifier == "uva_speech_studio2":
        for x in (-1.7, -0.85, 0, 0.85, 1.7):
            add(
                f"acoustic_panel_{x}",
                (x, y, 1.45),
                (0.40, 0.025, 1.20),
                (0.12, 0.31, 0.40, 1),
            )
        add(
            "cable_tray",
            (0, y - 0.035, 2.60),
            (2.1, 0.035, 0.035),
            (0.79, 0.79, 0.73, 1),
        )
    elif identifier == "ipparis_luli2000_milka":
        for x in (-2.3, 2.3):
            add(
                f"service_rack_{x}",
                (x, 1.45, 0.9),
                (0.25, 0.4, 0.9),
                (0.4, 0.45, 0.48, 1),
                True,
            )
            for z in (0.45, 0.85, 1.25, 1.65):
                add(
                    f"rack_panel_{x}_{z}",
                    (x, 1.035, z),
                    (0.22, 0.01, 0.13),
                    (0.07, 0.12, 0.15, 1),
                )
    elif identifier == "durham_2m_wind_tunnel":
        add(
            "observation_glazing",
            (-2.90, -0.1, 1.65),
            (0.025, 1.15, 0.55),
            (0.15, 0.27, 0.34, 1),
        )
        add(
            "yellow_nozzle_beam",
            (-1.52, 2.1, 1.45),
            (0.08, 0.08, 1.45),
            (0.9, 0.71, 0.05, 1),
            True,
        )
    elif identifier == "ucl_swc_optics_training":
        add(
            "rear_cable_tray",
            (0, y - 0.02, 1.38),
            (2.1, 0.055, 0.035),
            (0.70, 0.72, 0.69, 1),
        )


BUILDERS = {
    "ucl_swc_optical_breadboard": _ucl_optics,
    "kcl_mic_confocal_workcell": _kcl_confocal,
    "lse_eyelink_workstation": _lse_eye_tracking,
    "ipparis_milka_diagnostic_bay": _milka,
    "uva_speech_stimulus_station": _uva_studio,
    "durham_fixed_ground_tunnel": _durham_tunnel,
}
SOURCES = {
    kind: dict(
        reference=reference, url=url, dimensions_m=dimensions, dimension_basis=basis
    )
    for kind, reference, url, dimensions, basis in (
        (
            "ucl_swc_optical_breadboard",
            "UCL SWC installed optics-course station",
            UCL,
            [3.3, 1.14, 1.4],
            "Photographic estimates; original dry XY slide, no measured optical alignment",
        ),
        (
            "kcl_mic_confocal_workcell",
            "KCL MIC installed STELLARIS 8 FALCON DIVE room",
            KCL,
            [3.7, 1.08, 1.42],
            "Photographic estimates; inverted-body envelope and room relationships, no OEM CAD",
        ),
        (
            "lse_eyelink_workstation",
            "LSE Behavioural Lab installed EyeLink test setup",
            LSE,
            [2.45, 1.06, 1.32],
            "Photographic estimates; chin support and camera alignment travel are authored",
        ),
        (
            "ipparis_milka_diagnostic_bay",
            "LULI2000 room 1 MILKA, Ecole polytechnique Palaiseau",
            IP,
            [4.0, 3.1, 2.85],
            "Published approximate 2 m sphere diameter; external alignment stage and remaining dimensions estimated",
        ),
        (
            "uva_speech_stimulus_station",
            "UvA Speech Lab studio 2 installed participant station",
            UVA,
            [2.8, 1.7, 1.85],
            "Photographic estimates; articulated monitor-arm and pitch limits are authored",
        ),
        (
            "durham_fixed_ground_tunnel",
            "Durham 2 m wind-tunnel fixed-ground test section",
            DURHAM,
            [2.9, 6.4, 2.5],
            "Published 1.75 x 1.15 x 5.5 m test section, 1.65 m turntable diameter and 1.8 m nozzle distance; remaining dimensions estimated",
        ),
    )
}
SAMPLE_INTERFACES = {
    "ucl_swc_optical_breadboard": (
        "calibration_slide",
        (0.075, 0.025, 0.01),
        "clamped",
        "Inert calibration slide retained on authored XY carriage",
    ),
    "kcl_mic_confocal_workcell": (
        "retained_slide",
        (0.075, 0.025, 0.008),
        "clamped",
        "Inert glass-look slide retained on the dry microscope stage",
    ),
    "lse_eyelink_workstation": (
        "chin_support",
        (0.12, 0.09, 0.032),
        "clamped",
        "Head-support alignment interface; no human or biological specimen",
    ),
    "ipparis_milka_diagnostic_bay": (
        "alignment_target",
        (0.11, 0.13, 0.11),
        "clamped",
        "External dry diagnostic lens mount; no chamber loading",
    ),
    "uva_speech_stimulus_station": (
        "stimulus_center",
        (0.46, 0.004, 0.26),
        "clamped",
        "Stimulus display plane; no participant or gaze-generation model",
    ),
    "durham_fixed_ground_tunnel": (
        "vehicle_model",
        (0.65, 1.22, 0.43),
        "clamped",
        "Generic inert vehicle model rigidly retained on turntable",
    ),
}
FEATURES = dict.fromkeys(
    (
        "ucl_swc_optics_training",
        "kcl_mic_stellaris",
        "lse_behavioural_eyelink",
        "ipparis_luli2000_milka",
        "uva_speech_studio2",
        "durham_2m_wind_tunnel",
    ),
    _room_detail,
)
