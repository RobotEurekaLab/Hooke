"""Version 3 UW autosampler: supported XYZ guides and a contact-retained free vial.

Derived from the independently checked original articulated study, with its
local geometry, inertia, drives and contact assumptions preserved. This is a
new interface: v2 actions are intentionally not mapped to the new frames.
"""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.equipment import _Builder
from real_labs.mounting import annular_mount
from real_labs.uw_isolab_context import add_static_assembly

ID = "uw_cartesian_v3"
HOME_Y = -0.15
TIP_HOME = 0.45
Z_STROKE = 0.18
CONTROLS = ("v3_gantry_x", "v3_gantry_y", "v3_injection_z")


def _rebase_static(parent):
    """Move the estimated beam behind all rack rows; leave cabinet/targets fixed."""
    removed = {
        "y_axis_cover",
        "z_tower",
        "z_guide_recess",
        "z_face_rail_left",
        "z_face_rail_right",
        "tool_carriage",
        "tool_holder",
        "dry_stylus",
        "rail_top_track",
    }
    for owner in parent.iter():
        for geom in list(owner.findall("geom")):
            name = geom.get("name", "").removeprefix("isotope_analyzer__oem_")
            if name in removed:
                owner.remove(geom)
                continue
            move = name.startswith(
                ("stand_column_", "stand_clamp_", "clamp_screw_", "rail_end_")
            ) or name in {
                "rail_housing",
                "rail_front_stripe",
                "tray_clamp",
                "tray_hanger",
                "wash_clamp",
                "wash_hanger",
                "vaporizer_hanger",
            }
            if move:
                pos = list(map(float, geom.get("pos").split()))
                pos[1] += 0.055
                geom.set("pos", " ".join(map(str, pos)))
            if name in {"tray_back", "wash_back"}:
                pos = list(map(float, geom.get("pos").split()))
                pos[1] = 0.1585
                size = list(map(float, geom.get("size").split()))
                size[1] = 0.0415
                geom.set("pos", " ".join(map(str, pos)))
                geom.set("size", " ".join(map(str, size)))
            if name == "tray_shelf":
                # Explicit matched contact response for the actual free vial.
                # This is a numerical contact assumption, not material calibration.
                geom.set("solref", ".002 1")
                geom.set("solimp", ".96 .995 .0005")
            if name == "vaporizer_mount":
                geom.set("pos", ".120 .131 .333")
                geom.set("size", ".050 .099 .005")


class Parts:
    def __init__(self, builder):
        self.b = builder

    def name(self, key):
        return self.b.name + "__" + key

    def box(self, body, key, pos, half, material="metal"):
        return self.b.box(
            body, pos, half, material, collision=True, mass="0", name=self.name(key)
        )

    def rod(self, body, key, a, z, radius, material="bright"):
        return self.b.rod(
            body, a, z, radius, material, collision=True, mass="0", name=self.name(key)
        )

    def cylinder(self, body, key, pos, radius, half, material="bright", **kwargs):
        return self.b.cylinder(
            body,
            pos,
            radius,
            half,
            material,
            collision=True,
            mass="0",
            name=self.name(key),
            **kwargs,
        )

    def bore(self, body, key, center, inner, outer, half, axis):
        if axis == "y":
            body = ET.SubElement(
                body,
                "body",
                name=self.name(key + "_frame"),
                pos=" ".join(map(str, center)),
                quat=f"{math.sqrt(.5)} 0 0 {math.sqrt(.5)}",
                gravcomp="1",
            )
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=".002",
                diaginertia="0.0000001 0.0000001 0.0000001",
            )
            center, axis = (0, 0, 0), "x"
        before = set(body.iter("geom"))
        annular_mount(
            self.b,
            body,
            key,
            center,
            inner,
            outer,
            half,
            axis=axis,
            proxies=True,
            namespace="guide",
            material="dark",
        )
        for geom in body.iter("geom"):
            if geom not in before:
                geom.set("mass", "0")


def _guides(p, base):
    b = p.b
    # The primary cream beam is behind the rear rack row. Open rods sit above it.
    for j, y in enumerate((0.198, 0.244)):
        p.rod(base, f"x_rail_{j}", (-0.267, y, 0.475), (0.267, y, 0.475), 0.004)
        for i, x in enumerate((-0.260, 0.260)):
            p.box(base, f"x_anchor_{i}_{j}", (x, y, 0.470), (0.013, 0.012, 0.010))
    x = b.moving(
        base,
        "gantry_x",
        (0, 0, 0),
        (1, 0, 0),
        (-0.19, 0.19),
        mass=0.7,
        kp=250,
        force=30,
    )
    x.find("inertial").set("pos", "0 .05 .55")
    p.box(x, "x_carriage", (0, 0.221, 0.4905), (0.030, 0.041, 0.006))
    for i, xx in enumerate((-0.014, 0.014)):
        for j, yy in enumerate((0.198, 0.244)):
            p.bore(x, f"x_bearing_{i}_{j}", (xx, yy, 0.475), 0.0043, 0.0095, 0.010, "x")
    for i, xx in enumerate((-0.060, 0.060)):
        p.box(x, f"y_rear_bridge_{i}", (xx / 2, 0.220, 0.503), (0.039, 0.025, 0.007))
        p.box(x, f"y_side_member_{i}", (xx, 0.0205, 0.510), (0.009, 0.2255, 0.009))
    for i, xx in enumerate((-0.035, 0.035)):
        p.rod(x, f"y_rail_{i}", (xx, -0.205, 0.525), (xx, 0.246, 0.525), 0.005)
        for j, yy in enumerate((-0.205, 0.242)):
            p.box(
                x,
                f"y_anchor_{i}_{j}",
                ((-0.050 if i == 0 else 0.050), yy, 0.524),
                (0.025, 0.009, 0.012),
            )
    p.box(x, "y_front_tie", (0, -0.205, 0.510), (0.069, 0.009, 0.009), "cream")
    y = b.moving(
        x, "gantry_y", (0, HOME_Y, 0), (0, 1, 0), (0, 0.285), mass=0.4, kp=250, force=20
    )
    y.find("inertial").set("pos", "0 0 .62")
    for i, xx in enumerate((-0.060, 0.060)):
        p.box(y, f"y_carriage_side_{i}", (xx, 0, 0.550), (0.009, 0.054, 0.008))
        p.box(
            y,
            f"z_column_{i}",
            ((-0.058 if i == 0 else 0.058), 0, 0.666),
            (0.009, 0.014, 0.142),
            "cream",
        )
    for i, yy in enumerate((-0.045, 0.045)):
        p.box(y, f"y_carriage_end_{i}", (0, yy, 0.550), (0.060, 0.009, 0.008))
    for i, xx in enumerate((-0.035, 0.035)):
        for j, yy in enumerate((-0.014, 0.014)):
            p.bore(y, f"y_bearing_{i}_{j}", (xx, yy, 0.525), 0.0053, 0.0125, 0.010, "y")
            p.box(
                y,
                f"y_bearing_lug_{i}_{j}",
                ((-0.048 if i == 0 else 0.048), yy, 0.54275),
                (0.014, 0.010, 0.00525),
            )
    for i, xx in enumerate((-0.021, 0.021)):
        p.rod(y, f"z_rail_{i}", (xx, 0, 0.544), (xx, 0, 0.796), 0.004)
        for j, zz in enumerate((0.538, 0.802)):
            p.box(
                y,
                f"z_anchor_{i}_{j}",
                ((-0.0395 if i == 0 else 0.0395), 0, zz),
                (0.023, 0.013, 0.006),
            )
    p.box(y, "z_top_bridge", (0, 0, 0.802), (0.067, 0.014, 0.006), "cream")
    # The cover is behind the bore; it does not fill the moving stem opening.
    p.box(y, "z_back_cover", (0, 0.027, 0.674), (0.067, 0.004, 0.134), "cream")
    for i, xx in enumerate((-0.058, 0.058)):
        p.box(
            y, f"z_back_spacer_{i}", (xx, 0.017, 0.666), (0.009, 0.010, 0.142), "cream"
        )
    z = b.moving(
        y,
        "injection_z",
        (0, 0, 0.550),
        (0, 0, -1),
        (0, Z_STROKE),
        mass=0.1,
        kp=180,
        force=120,
    )
    p.box(z, "head", (0, 0, 0), (0.014, 0.021, 0.044), "cream")
    p.box(z, "head_face", (0, -0.023, 0), (0.012, 0.002, 0.038), "cyan")
    p.box(z, "z_stem", (0, 0, 0.142), (0.010, 0.010, 0.098))
    for i, xx in enumerate((-0.021, 0.021)):
        for j, zz in enumerate((0.187, 0.227)):
            p.bore(z, f"z_bearing_{i}_{j}", (xx, 0, zz), 0.0043, 0.0085, 0.009, "z")
            p.box(
                z,
                f"z_lug_{i}_{j}",
                ((-0.0115 if i == 0 else 0.0115), 0, zz),
                (0.003, 0.006, 0.008),
            )
    p.rod(z, "stylus_holder", (0, 0, -0.043), (0, 0, -0.061), 0.0015)
    p.rod(z, "dry_stylus", (0, 0, -0.060), (0, 0, -0.0996), 0.0004)
    b.site(z, "physical_tip", (0, 0, -0.100), 0.0002)
    return x, y, z


def _rack_contacts(p, base):
    """Contact sectors preserve every visual bore; one vial remains physically free."""
    b = p.b
    positions = []
    for i in range(6):
        for j in range(16):
            xx, yy = -0.069 + (i - 2.5) * 0.014, 0.012 + (j - 7.5) * 0.014
            positions.append((xx, yy))
            b.site(base, f"rack_slot_{i}_{j}", (xx, yy, 0.281), 0.0006)
            for k in range(16):
                angle = math.tau * (k + 0.5) / 16
                radius, half = (0.00615 + 0.007) / 2, (0.007 - 0.00615) / 2
                geom = b.box(
                    base,
                    (
                        xx + radius * math.cos(angle),
                        yy + radius * math.sin(angle),
                        0.275,
                    ),
                    (half, 0.007 * math.tan(math.pi / 16), 0.006),
                    "bright",
                    euler=(0, 0, angle),
                    collision=True,
                    mass="0",
                    rgba=(0.5, 0.5, 0.5, 0),
                    name=p.name(f"rack_contact_{i}_{j}_{k}"),
                )
    xx, yy = positions[8]  # left column, middle row: an actual supported free object.
    position = np.asarray([xx, yy, 0.269 + 0.016 + 0.00005])
    quaternion = np.fromstring(base.get("quat", "1 0 0 0"), sep=" ")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    position = (
        np.fromstring(base.get("pos", "0 0 0"), sep=" ")
        + rotation.reshape(3, 3) @ position
    )
    body = ET.SubElement(
        b.root.find("worldbody"),
        "body",
        name=p.name("inert_vial"),
        pos=" ".join(map(str, position)),
        quat=" ".join(map(str, quaternion)),
    )
    ET.SubElement(body, "freejoint", name=p.name("inert_vial_free"))
    # Original inert solid surrogate; size follows the published outer vial envelope.
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=".003",
        diaginertia="0.000000283 0.000000283 0.000000054",
    )
    p.cylinder(
        body,
        "vial_collision",
        (0, 0, 0),
        0.006,
        0.016,
        "glass",
        friction=".6 .005 .0001",
        solref=".002 1",
        solimp=".96 .995 .0005",
    )
    b.cylinder(
        body,
        (0, 0, 0.015),
        0.006,
        0.001,
        "blue",
        collision=False,
        mass="0",
        name=p.name("vial_cap"),
    )
    b.site(body, "inert_vial_cap", (0, 0, 0.016), 0.0005)
    b.site(base, "vaporizer_dry_datum", (0.120, 0.060, 0.346), 0.0008)
    b.site(base, "service_clearance_datum", (0, HOME_Y, 0.269), 0.001)
    return {
        "rack_positions_xy_m": [list(point) for point in positions],
        "free_vial_position_index": 8,
        "vial_dimensions_m": [0.012, 0.032],
        "vial_mass_kg": 0.003,
        "vial_mass_basis": "Authored inert surrogate; not published vial mass",
    }


def add_autosampler(builder, base, params):
    """Install the versioned assembly on the existing UW analysis counter."""
    if builder.name != "isotope_analyzer":
        raise ValueError("UW version 3 requires its explicit isotope_analyzer owner")
    if params.get("interface_revision") != 3:
        raise ValueError(
            "Select UW interface revision 3; legacy actions are not mapped"
        )
    root = builder.root
    before = set(root.iter("geom"))
    add_static_assembly(root, base, builder)
    _rebase_static(base)
    moving = _Builder(root, ID)
    parts = Parts(moving)
    _guides(parts, base)
    interface = _rack_contacts(parts, base)
    moving.site(base, "sample_seat", (-0.104, 0.019, 0.285), 0.0005)
    # The isolated reference had MuJoCo's default 20 ms contact time constant
    # except for the explicitly matched vial/shelf pair. Do not inherit the
    # room's different decorative defaults into this instrument.
    for geom in root.iter("geom"):
        if geom not in before:
            geom.attrib.setdefault("solref", ".02 1")
            geom.attrib.setdefault("solimp", ".9 .95 .001 .5 2")
    root.find("option").set("timestep", ".0005")
    for field in ("joints", "actuators", "joint_ranges", "actuator_modes"):
        builder.metadata[field] = {
            "v3_" + k: v for k, v in moving.metadata[field].items()
        }
    builder.metadata["sites"] = moving.metadata["sites"]
    body = ID + "__inert_vial"
    collider = ID + "__vial_collision"
    interface.update(
        revision=3,
        physical_tip_site=ID + "__site_physical_tip",
        carrier_body=body,
        carrier_geom=collider,
        support_geom="isotope_analyzer__oem_tray_shelf",
        sample_standoff_m=0.0045,
        legacy_actions_accepted=False,
        full_joint_box_collision_free=False,
        safe_use="Raised XY corridors with observed vial/port approach and retraction",
    )
    builder.metadata.update(
        interface_revision=3,
        inspection_interface=interface,
        capabilities=[
            "Dry XYZ surface registration",
            "Contact-retained free inert vial",
            "Measured vial and vaporizer-datum inspection",
        ],
        inspection_targets=dict(zip(CONTROLS, (-0.10, 0.03, 0.03))),
        published_analyzer_envelope_m=[0.432, 0.432, 0.191],
        retired_controls={
            key: {
                "reason": "Explicit v3 frame, travel and free-vial migration",
                "automatic_mapping": False,
                "replacement_controls": list(CONTROLS),
            }
            for key in ("carousel", "gantry_x", "gantry_y", "injection_z")
        },
        existing_free_specimen=dict(
            body=body,
            freejoint=ID + "__inert_vial_free",
            collision_geom=collider,
            support_geom="isotope_analyzer__oem_tray_shelf",
            sample_site=ID + "__site_sample_seat",
            dimensions_m=[0.012, 0.012, 0.032],
            mass_kg=0.003,
            description="One inert 12 x 32 mm vial in a 96-position estimated rack",
            mass_basis="Authored solid surrogate mass; not a published vial mass",
            initial_support_clearance_m=0.00005,
        ),
    )
    builder.metadata["limitations"].extend(
        [
            "Only the analyzer cabinet envelope and outer vial dimensions have published bases; guides, drive parameters, tray pitch and hidden mechanics are authored.",
            "The installed analyzer generation and room layout remain unverified.",
            "Version 3 changes all XYZ frames and ranges, including Z 230 to 180 mm; no automatic legacy action mapping.",
            "The full XYZ box contains a wash-support obstruction. Direct unplanned commands can collide; use the checked dry inspection route.",
            "No liquid injection, isotope, pressure or calibrated analytical behavior is modeled.",
            "The loaded 3 g solid vial is a contact-retained free body; no hidden weld or compliant septum model.",
        ]
    )
    from real_labs.uw_source_services import add_analyzer_context

    add_analyzer_context(root, base)
