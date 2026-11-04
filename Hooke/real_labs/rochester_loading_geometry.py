"""Rochester source-informed hinge and authored contact-only plate-loading fixture.

The Flex figure supports a hinged transparent front; installed Rochester model,
hinge sizes and this external manual XYZ fork are estimates. The fork is an
original task tool, not observed OEM hardware. Original pipetting XYZ and the
dry-touch carrier remain separate and unchanged. SI units throughout.
"""

import math
import xml.etree.ElementTree as ET
import numpy as np

P = "rochester_loading__"


def nums(x):
    return " ".join(f"{v:.10g}" for v in x)


def add_loading_fixture(root, base, meta):
    if base.get("name") != "liquid_handler" or base.get("quat") != "1 0 0 0":
        raise ValueError(
            "Rochester fixture requires the named, unrotated liquid handler"
        )

    def geom(parent, key, kind, pos, size, mat="metal", **kw):
        a = dict(
            name=P + key,
            type=kind,
            pos=nums(pos),
            size=nums(size),
            material="liquid_handler__mat_" + mat,
            contype="1",
            conaffinity="1",
            density="0",
            friction="1 .01 .001",
            solref=".008 1",
        )
        a.update(
            {
                k: nums(v) if isinstance(v, (list, tuple)) else str(v)
                for k, v in kw.items()
            }
        )
        return ET.SubElement(parent, "geom", a)

    def box(p, k, xyz, half, mat="metal", **kw):
        return geom(p, k, "box", xyz, half, mat, **kw)

    def rod(p, k, a, b, r=0.006, mat="metal"):
        g = geom(p, k, "capsule", (0, 0, 0), (r,), mat, fromto=(*a, *b))
        g.attrib.pop("pos")
        return g

    def moving(parent, key, pos, axis, span, mass, kp=500, force=80):
        body = ET.SubElement(
            parent, "body", name=P + "body_" + key, pos=nums(pos), gravcomp="1"
        )
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=str(mass),
            diaginertia=nums([mass * 0.008] * 3),
        )
        joint = P + "joint_" + key
        act = P + "actuator_" + key
        ET.SubElement(
            body,
            "joint",
            name=joint,
            type="slide",
            axis=nums(axis),
            range=nums(span),
            damping="5",
            armature=".01",
        )
        ET.SubElement(
            root.find("actuator"),
            "position",
            name=act,
            joint=joint,
            kp=str(kp),
            kv=str(2 * math.sqrt(kp * mass)),
            ctrllimited="true",
            ctrlrange=nums(span),
            forcelimited="true",
            forcerange=nums((-force, force)),
        )
        for field, val in [
            ("joints", joint),
            ("actuators", act),
            ("joint_ranges", list(span)),
            ("actuator_modes", "position"),
        ]:
            meta[field][key] = val
        return body

    # Versioned access. Old slider's actuator and whole moving exterior are removed.
    door = base.find("body[@name='liquid_handler__body_door']")
    old_joint = meta["joints"].pop("door")
    old_act = meta["actuators"].pop("door")
    meta["joint_ranges"].pop("door")
    meta["actuator_modes"].pop("door")
    for a in list(root.find("actuator")):
        if a.get("name") == old_act:
            root.find("actuator").remove(a)
    base.remove(door)
    door = ET.SubElement(
        base,
        "body",
        name="liquid_handler__body_front_door_hinge",
        pos="-.401 -.365 .454",
        gravcomp="1",
    )
    ET.SubElement(
        door, "inertial", pos=".40 0 0", mass="1", diaginertia=".024 .075 .052"
    )
    joint = "liquid_handler__joint_front_door_hinge"
    act = "liquid_handler__actuator_front_door_hinge"
    ET.SubElement(
        door,
        "joint",
        name=joint,
        type="hinge",
        axis="0 0 -1",
        range="0 1.75",
        damping="2",
        armature=".01",
    )
    ET.SubElement(
        root.find("actuator"),
        "position",
        name=act,
        joint=joint,
        kp="70",
        kv="16",
        ctrllimited="true",
        ctrlrange="0 1.75",
        forcelimited="true",
        forcerange="-12 12",
    )
    for field, val in [
        ("joints", joint),
        ("actuators", act),
        ("joint_ranges", [0, 1.75]),
        ("actuator_modes", "position"),
    ]:
        meta[field]["front_door_hinge"] = val
    meta["retired_controls"] = {
        "door": "The former meter-valued vertical slider is retired. Use front_door_hinge in radians; no automatic coordinate conversion."
    }
    box(door, "door_window", (0.402, 0, 0), (0.378, 0.003, 0.263), "glass")
    box(door, "door_left_stile", (0.015, 0, 0), (0.010, 0.007, 0.232), "dark")
    box(door, "door_right_stile", (0.789, 0, 0), (0.009, 0.007, 0.270), "dark")
    for z in [-0.269, 0.270]:
        box(door, "door_rail_" + str(z), (0.407, 0, z), (0.387, 0.008, 0.009), "dark")
    for z in [-0.21, 0.21]:
        for i in range(32):
            a = math.tau * i / 32
            b = math.tau * (i + 1) / 32
            # Open annular knuckle built from vertical capsules, clear of the pin.
            x, y = 0.008 * math.cos(a), 0.008 * math.sin(a)
            rod(door, f"knuckle_{z}_{i}", (x, y, z - 0.015), (x, y, z + 0.015), 0.0025)
        box(door, "knuckle_link_" + str(z), (0.014, 0, z), (0.008, 0.006, 0.012))
    for z in [0.187, 0.720]:
        box(base, "hinge_mount_" + str(z), (-0.421, -0.344, z), (0.023, 0.024, 0.010))
    rod(base, "hinge_pin", (-0.401, -0.365, 0.194), (-0.401, -0.365, 0.714), 0.004)
    for z in [-0.060, 0.060]:
        rod(
            door,
            "handle_standoff_" + str(z),
            (0.752, -0.007, z),
            (0.752, -0.033, z),
            0.007,
            "dark",
        )
    rod(
        door,
        "door_handle",
        (0.752, -0.033, -0.060),
        (0.752, -0.033, 0.060),
        0.009,
        "dark",
    )
    ET.SubElement(
        door,
        "site",
        name="liquid_handler__site_door_handle",
        pos=".752 -.033 0",
        size=".006",
        rgba="0 1 0 0",
    )
    # Move the three screen surfaces to the right of the door's whole swing.
    for g in base.findall("geom"):
        p = np.fromstring(g.get("pos", ""), sep=" ")
        if (
            len(p) == 3
            and 0.23 < p[0] < 0.38
            and -0.41 < p[1] < -0.37
            and 0.44 < p[2] < 0.55
        ):
            p[0] += 0.24
            g.set("pos", nums(p))
    arm = base.find("geom[@name='liquid_handler__display_mount_arm']")
    arm.set("pos", ".46 -.38 .493")
    arm.set("size", ".056 .008 .012")
    # Replacing one old static plate frees front-center deck slot2.
    removed = []
    for g in list(base.findall("geom")):
        p = np.fromstring(g.get("pos", ""), sep=" ")
        s = np.fromstring(g.get("size", ""), sep=" ")
        if len(p) != 3:
            continue
        if (g.get("type") == "box" and np.allclose(p, [0, -0.1635, 0.176])) or (
            g.get("type") == "cylinder"
            and abs(p[0]) < 0.051
            and -0.20 < p[1] < -0.13
            and abs(p[2] - 0.188) < 1e-8
        ):
            base.remove(g)
            removed.append(g.get("name"))
    assert len(removed) == 97, len(removed)
    # Fixture clamps onto the retained counter. Its station parks to the right
    # of the door swing, then X aligns the tool only after the door opens.
    box(base, "mount_platform", (0.80, -0.57, 0.065), (0.32, 0.25, 0.010))
    for x in [0.56, 1.04]:
        box(
            base,
            "bench_pad_" + str(x),
            (x, -0.39, 0.0275),
            (0.050, 0.070, 0.0275),
            "rubber",
        )
        box(base, "frame_post_" + str(x), (x, -0.60, 0.092), (0.020, 0.020, 0.027))
    # Open-top fixed X guides. Runner height extends above their side walls.
    for y in [-0.78, -0.57]:
        box(base, "x_guide_floor_" + str(y), (0.80, y, 0.119), (0.22, 0.019, 0.006))
        for s in [-1, 1]:
            box(
                base,
                f"x_guide_side_{y}_{s}",
                (0.80, y + s * 0.014, 0.132),
                (0.22, 0.004, 0.008),
            )
    # Connect each guide to the common anchored platform.
    for x in [0.62, 0.98]:
        for y in [-0.78, -0.57]:
            box(base, f"x_guide_post_{x}_{y}", (x, y, 0.094), (0.018, 0.018, 0.019))
    xbody = moving(base, "loader_x", (0.70, 0, 0), (1, 0, 0), (-0.70, 0), 1.2)
    for y in [-0.78, -0.57]:
        box(xbody, "x_runner_" + str(y), (0.20, y, 0.139), (0.50, 0.008, 0.013))
    box(xbody, "x_bridge", (0, -0.675, 0.158), (0.116, 0.112, 0.008))
    # Fixed Y channel rides on X; long runners keep a visible support overlap.
    for x in [-0.105, 0.105]:
        box(
            xbody, "y_guide_support_" + str(x), (x, -0.70, 0.194), (0.010, 0.065, 0.029)
        )
        box(xbody, "y_guide_floor_" + str(x), (x, -0.735, 0.229), (0.018, 0.085, 0.006))
        for s in [-1, 1]:
            box(
                xbody,
                f"y_guide_side_{x}_{s}",
                (x + s * 0.013, -0.735, 0.242),
                (0.004, 0.085, 0.008),
            )
    ybody = moving(xbody, "loader_y", (0, 0, 0), (0, 1, 0), (-0.82, 0), 0.7)
    for x in [-0.105, 0.105]:
        box(ybody, "y_runner_" + str(x), (x, -0.34, 0.247), (0.007, 0.40, 0.011))
    # Raised crossmembers connect runners without crossing their fixed U sides.
    for y in [-0.080, -0.120]:
        box(ybody, "y_bridge_" + str(y), (0, y, 0.263), (0.112, 0.008, 0.006))
    # Open vertical fork slides behind the plate; supports are outside its sides.
    for x in [-0.076, 0.076]:
        box(ybody, "z_guide_" + str(x), (x, -0.074, 0.250), (0.006, 0.010, 0.063))
    zbody = moving(
        ybody, "loader_z", (0, 0, 0.173), (0, 0, 1), (0, 0.105), 0.18, kp=650, force=35
    )
    for x in [-0.076, 0.076]:
        box(zbody, "z_pad_" + str(x), (x, -0.054, 0.038), (0.006, 0.009, 0.027), "dark")
    box(zbody, "fork_back", (0, -0.055, 0.008), (0.085, 0.005, 0.014))
    # Two open fork tines fit between both docks' four receiving supports.
    for x in [-0.031, 0.031]:
        box(zbody, "fork_tine_" + str(x), (x, 0, 0), (0.008, 0.050, 0.004))
    for x in [-0.068, 0.068]:
        box(zbody, "fork_side_" + str(x), (x, 0, 0.005), (0.004, 0.051, 0.008))
    for x in [-0.031, 0.031]:
        box(zbody, "fork_front_" + str(x), (x, 0.051, 0.005), (0.008, 0.004, 0.008))
    # Connect the outer retaining rails and tines across the rear crossmember.
    box(zbody, "fork_rear_floor", (0, -0.056, 0), (0.087, 0.007, 0.004))
    # Deck and staging dock share plate size and true separated support points.
    for loc, (cx, cy) in {"stage": (0.70, -0.46), "deck": (0, -0.1635)}.items():
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                zlow = 0.075 if loc == "stage" else 0.167
                box(
                    base,
                    f"{loc}_pad_{sx}_{sy}",
                    (cx + sx * 0.055, cy + sy * 0.032, (zlow + 0.205) / 2),
                    (0.006, 0.009, (0.205 - zlow) / 2),
                    "dark",
                )
    # A genuinely free inert 96-well-sized plate starts on the staging pads.
    world = root.find("worldbody")
    bp = np.fromstring(base.get("pos", "0 0 0"), sep=" ")
    plate = ET.SubElement(
        world, "body", name=P + "plate", pos=nums(bp + np.array([0.70, -0.46, 0.21605]))
    )
    ET.SubElement(plate, "freejoint", name=P + "plate_free")
    geom(
        plate,
        "plate_contact",
        "box",
        (0, 0, 0),
        (0.0639, 0.0428, 0.011),
        "cream",
        mass=".032",
        density="0",
    )
    for row in range(8):
        for col in range(12):
            g = geom(
                plate,
                f"well_{row}_{col}",
                "cylinder",
                ((col - 5.5) * 0.009, (row - 3.5) * 0.009, 0.0115),
                (0.0028, 0.0005),
                "dark",
            )
            g.set("contype", "0")
            g.set("conaffinity", "0")
    meta["plate_loading_interface"] = {
        "revision": 1,
        "body": P + "plate",
        "collision_geom": P + "plate_contact",
        "freejoint": P + "plate_free",
        "home_xyz": [0, 0, 0],
        "dock_world": (bp + np.array([0, -0.1635, 0.216])).tolist(),
        "stage_world": (bp + np.array([0.70, -0.46, 0.216])).tolist(),
        "removed_static_plate_geoms": removed,
        "scope": "Authored bench-mounted manual XYZ fork; free inert plate is lifted, retained by real lips, deposited, released and retrieved by contact only. No OEM gripper, liquid/tip transfer or sequencing process.",
    }

    meta["limitations"] = [
        x for x in meta["limitations"] if "retained sliding access door" not in x
    ]
    meta["limitations"].append(
        "Original authored bench-mounted loading fork; not an observed Rochester or OEM Flex gripper. No liquid transfer, tip pickup, fluid/biological or sequencing process."
    )
    meta["capabilities"].extend(
        [
            "contact-enabled hinged access in radians",
            "free inert plate contact-only pick, place, release and retrieval",
        ]
    )
    return meta
