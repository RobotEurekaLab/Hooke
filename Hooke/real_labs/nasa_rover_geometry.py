"""Authored grounded six-wheel rover and physical sample-staging fixtures.

NASA's cached SLOPE preview establishes the terrestrial high-bay context, not
these rover dimensions. This version retains the original mast/cameras and arm
links, adds supported steering knuckles and opening cradle jaws, and places a
free 25 g inert sample and a grounded lower inspection shelf. Terrain, friction,
masses, contact compliance and dimensions are uncalibrated authored estimates.
"""

import math
import xml.etree.ElementTree as ET
import numpy as np

VERSION = "grounded_staging_v2"
START = (-5.0, 2.5, 0.0)
# The equipment frame uses the wheel contact plane as its vertical datum.
# Geometry below is authored in the earlier frame, 8 mm below that plane.
GROUND_DATUM_OFFSET_M = 0.008


def vals(a):
    return " ".join(map(str, a))


def _use_ground_datum(rover):
    """Translate the root frame while preserving its physical world geometry.

    Only direct children change frames. Their descendants retain all relative
    transforms. The explicit inertial position moves with the geometry, so its
    world centre of mass and principal inertia remain unchanged.
    """
    offset = np.array([0.0, 0.0, GROUND_DATUM_OFFSET_M])
    for child in rover:
        if child.tag == "freejoint":
            continue
        if child.tag not in {"body", "geom", "site", "camera", "inertial"}:
            raise ValueError(f"Unsupported NASA root-frame element: {child.tag}")
        if "fromto" in child.attrib:
            endpoints = np.fromstring(child.get("fromto"), sep=" ").reshape(2, 3)
            child.set("fromto", vals((endpoints - offset).ravel()))
        else:
            position = np.fromstring(child.get("pos", "0 0 0"), sep=" ")
            child.set("pos", vals(position - offset))


def geom(p, name, typ, size, pos, rgba=".5 .55 .6 1", **kw):
    return ET.SubElement(
        p,
        "geom",
        name="nasa_task__" + name,
        type=typ,
        size=vals(size),
        pos=vals(pos),
        rgba=rgba,
        contype="1",
        conaffinity="1",
        solref=".001 1",
        solimp=".999 .9999 .0001",
        priority="1",
        **kw,
    )


def configure_task(root, equipment):
    """Configure only the NASA ground-task scene, once, after its equipment."""
    r = root
    r.find("option").set("timestep", ".0005")
    r.find("option").set("ccd_tolerance", "1e-10")
    r.find("option").set("ccd_iterations", "200")
    w = r.find("worldbody")
    rover = w.find("body[@name='rover']")
    if rover is None:
        raise RuntimeError([x.get("name") for x in w.findall("body")])
    rover.set("pos", vals(START))
    if rover.find("freejoint") is None:
        ET.SubElement(rover, "freejoint", name="rover__free_base")
        ET.SubElement(
            rover, "inertial", pos="0 0 .25", mass="18", diaginertia=".9 1.2 1.5"
        )
    for b in rover.iter("body"):
        b.set("gravcomp", "0")
    for g in rover.iter("geom"):
        if g.get("contype") != "0":
            g.set("solref", ".001 1")
            g.set("solimp", ".999 .9999 .0001")
            g.set("priority", "1")
    for p in w.iter():
        for child in list(p):
            if child.get("name") in ("rover_service_stand",):
                p.remove(child)
    # Supported pickup pedestal is authored handling context, not NASA fixture CAD.
    sx, sy = START[0] + 0.6554, START[1]
    picktop = 0.7143
    geom(w, "pickup_foot", "box", (0.095, 0.095, 0.01), (sx, sy, 0.01))
    geom(
        w,
        "pickup_post",
        "cylinder",
        (0.015, (picktop - 0.03) / 2),
        (sx, sy, (picktop - 0.03) / 2 + 0.02),
    )
    geom(w, "pickup_top", "box", (0.022, 0.008, 0.005), (sx, sy, picktop - 0.005))
    b = ET.SubElement(
        w, "body", name="nasa_task__sample", pos=vals((sx, sy, picktop + 0.012))
    )
    ET.SubElement(b, "freejoint", name="nasa_task__sample_free")
    geom(
        b,
        "sample",
        "box",
        (0.012, 0.018, 0.012),
        (0, 0, 0),
        ".7 .32 .12 1",
        mass=".025",
        friction="1.2 .005 .0001",
    )
    ET.SubElement(
        b, "site", name="nasa_task__sample_center", size=".002", rgba="0 0 0 0"
    )
    # Lower inspection staging shelf joins the front of the existing bench.
    tx, ty, tz = -4.6, 5.95, 0.7143
    geom(w, "stage_shelf", "box", (0.18, 0.235, 0.012), (tx, 6.07, 0.668))
    for x in (tx - 0.145, tx + 0.145):
        geom(w, "stage_foot_" + str(x), "box", (0.06, 0.06, 0.01), (x, 6.16, 0.01))
        geom(
            w,
            "stage_leg_" + str(x),
            "box",
            (0.018, 0.018, (0.68 - 0.044) / 2),
            (x, 6.16, (0.68 - 0.044) / 2 + 0.02),
        )
        geom(
            w,
            "stage_bench_bracket_" + str(x),
            "box",
            (0.022, 0.016, 0.132),
            (x, 6.18, 0.788),
        )
    # Low rim retains sample after physical gripper release. Front left open.
    for sign in (-1, 1):
        geom(
            w,
            "stage_rim_" + str(sign),
            "box",
            (0.003, 0.045, 0.010),
            (tx + sign * 0.045, ty, 0.690),
        )
    geom(w, "stage_backrim", "box", (0.048, 0.003, 0.010), (tx, ty + 0.048, 0.690))
    geom(
        w,
        "stage_seat",
        "box",
        (0.008, 0.022, (tz - 0.68) / 2),
        (tx, ty, (tz + 0.68) / 2),
    )
    ET.SubElement(
        w,
        "site",
        name="nasa_task__stage_target",
        pos=vals((tx, ty, tz + 0.012)),
        size=".003",
        rgba="0 .8 .2 0",
    )
    # Remove old visual-only struts, replacing their endpoints with supported steering bearings.
    for g in list(rover.findall("geom")):
        if g.get("type") == "capsule" and g.get("size") == "0.025":
            rover.remove(g)
    # Authored six steering knuckles: axes pass through existing wheel centres.
    for side, y in (("left", -0.416), ("right", 0.416)):
        for i, x in enumerate((-0.4, 0, 0.4), 1):
            key = f"wheel_{side}_{i}"
            wheel = rover.find(f"body[@name='rover__body_{key}']")
            rover.remove(wheel)
            steer = ET.SubElement(
                rover, "body", name="nasa_task__steer_" + key, pos=vals((x, y, 0.148))
            )
            ET.SubElement(
                steer,
                "joint",
                name="nasa_task__steer_" + key,
                axis="0 0 1",
                range="-1.57 1.57",
                damping=".6",
                armature=".003",
            )
            ET.SubElement(
                steer,
                "inertial",
                mass=".15",
                pos="0 0 .07",
                diaginertia=".002 .002 .002",
            )
            # Open square bearing around an 8 mm radius fixed kingpin; min 1 mm radial gap.
            for sign in (-1, 1):
                geom(
                    steer,
                    "steer_bridge_long_" + key + str(sign),
                    "box",
                    (0.007, 0.09, 0.008),
                    (sign * 0.016, 0, 0.17),
                )
                geom(
                    steer,
                    "steer_bridge_short_" + key + str(sign),
                    "box",
                    (0.009, 0.0405, 0.008),
                    (0, sign * 0.0495, 0.17),
                )
                geom(
                    rover,
                    "kingpin_washer_" + key + str(sign),
                    "cylinder",
                    (0.014, 0.002),
                    (x, y, 0.318 + sign * 0.0105),
                )
                geom(
                    steer,
                    "wheel_stub_" + key + str(sign),
                    "cylinder",
                    (0.025, 0.006),
                    (0, sign * 0.061, 0),
                    quat=".7071067811865476 .7071067811865476 0 0",
                )
            geom(rover, "kingpin_" + key, "cylinder", (0.008, 0.025), (x, y, 0.326))
            g = geom(rover, "suspension_" + key, "capsule", (0.013,), (0, 0, 0))
            g.attrib.pop("pos")
            g.set(
                "fromto", vals((x * 0.65, math.copysign(0.235, y), 0.36, x, y, 0.355))
            )
            for sign in (-1, 1):
                geom(
                    steer,
                    "steer_fork_" + key + str(sign),
                    "box",
                    (0.018, 0.008, 0.089),
                    (0, sign * 0.075, 0.081),
                )
            wheel.set("pos", "0 0 0")
            steer.append(wheel)
            ET.SubElement(
                r.find("actuator"),
                "position",
                name="nasa_task__steer_" + key,
                joint="nasa_task__steer_" + key,
                kp="100",
                kv="8",
                ctrllimited="true",
                ctrlrange="-1.57 1.57",
                forcelimited="true",
                forcerange="-20 20",
            )
    # Captive jaw tool: short front/back lips stop fore/aft creep while closed.
    # Both jaws physically slide clear for release. No object weld or attachment.
    for side, sign in (("left", -1), ("right", 1)):
        finger = rover.find(f".//body[@name='rover__body_arm_gripper_{side}']")
        pos = np.fromstring(finger.get("pos"), sep=" ")
        pos[1] += sign * 0.015
        finger.set("pos", vals(pos))
        finger.find("joint").set("ref", ".015")
        geom(
            finger,
            "jaw_extension_" + side,
            "box",
            (0.011, 0.0031, 0.01178),
            (0.035, 0, 0),
        )
        geom(
            finger,
            "jaw_floor_" + side,
            "box",
            (0.015, 0.010, 0.003),
            (0.029, -sign * 0.001, -0.015),
        )
        for end, x in (("back", 0.014), ("front", 0.044)):
            geom(
                finger,
                "jaw_lip_" + side + "_" + end,
                "box",
                (0.002, 0.009, 0.01178),
                (x, 0, 0),
            )
    # Gentle finite-force clamp; the connected cradle carries the inert sample's weight.
    for a in r.find("actuator"):
        if "arm_gripper" in a.get("name", ""):
            a.set("kp", "100")
            a.set("kv", "4")

    item = next(row for row in equipment if row["id"] == "rover")
    item["pos"] = list(START)
    item["freejoint"] = "rover__free_base"
    item["dimensions_m"] = [1.4, 1.14, 1.2]
    item["dimension_basis"] = (
        "Conservative authored bound including steering sweep and sample tool; no NASA hardware dimensions"
    )
    item["limitations"] = [
        s for s in item["limitations"] if "Base is anchored" not in s
    ]
    item["capabilities"] += [
        "six finite-torque steering knuckles",
        "wheel-contact bay traversal",
        "opening sample cradle and physical lower-shelf staging",
    ]
    for side in ("left", "right"):
        for i in (1, 2, 3):
            wheel = f"wheel_{side}_{i}"
            key = "steer_" + wheel
            name = "nasa_task__" + key
            item["joints"][key] = name
            item["actuators"][key] = name
            item["joint_ranges"][key] = [-1.57, 1.57]
            item["actuator_modes"][key] = "position"
            item["control_contracts"][wheel][
                "scope"
            ] = "Authored wheel-contact ground drive, 12 Nm torque limit; uncalibrated traction"
    item["sample_transfer_interface"] = {
        "kind": "rigid_sample_staging_v1",
        "sample_geom": "nasa_task__sample",
        "stage_seat_geom": "nasa_task__stage_seat",
        "native_contact_offset_m": 0.00095,
    }
    item["nasa_rover_task"] = {
        "version": VERSION,
        "entry_point": "real_labs.nasa_rover_task.traverse_and_stage_nasa",
        "sample_body": "nasa_task__sample",
        "sample_joint": "nasa_task__sample_free",
        "sample_mass_kg": 0.025,
        "sample_half_sizes_m": [0.012, 0.018, 0.012],
        "pickup_top": "nasa_task__pickup_top",
        "stage_seat": "nasa_task__stage_seat",
        "stage_target": "nasa_task__stage_target",
        "tcp": "rover__site_arm_tcp",
        "start_base_xyz_m": list(START),
        "base_frame_datum": "Wheel contact plane at the authored reset pose",
        "geometry_frame_offset_m": [0.0, 0.0, -GROUND_DATUM_OFFSET_M],
        "wheel_radius_m": 0.14,
        "wheel_track_m": 0.832,
        "waypoints_xy_m": [[-5.25, 2.5], [-5.25, 3.6], [-4.6, 4.6], [-4.6, 5.38]],
        "controller_limits": {
            "linear_m_s": 0.13,
            "yaw_rad_s": 0.35,
            "linear_accel_m_s2": 0.2,
            "yaw_accel_rad_s2": 0.6,
            "steer_rate_rad_s": 0.6,
            "wheel_torque_nm": 12,
            "steer_torque_nm": 20,
        },
        "tool": "Original fingers plus connected opening fore/aft lips and bottom ledges; no weld, magnet, mocap or attachment",
        "support": "Pickup foot and post reach floor; lower staging shelf has grounded legs and continuous brackets into existing inspection bench",
        "steering_bearing_gaps_m": {
            "kingpin_radial": 0.001,
            "kingpin_axial": 0.0005,
            "wheel_stub_axial": 0.001,
        },
        "physics": "Earth gravity; no body gravity compensation; arm gravity feedforward uses the finite joint actuators",
        "limits": [
            "Rigid flat high-bay aisle only; soil bins and tiltbed remain untraversed rigid context",
            "No granular soil, sinkage, slip calibration, vehicle dynamics identification, NASA hardware replication or native Isaac validation",
            "The generic 25 g box, tool cradle, track, steering geometry and lower inspection shelf are authored estimates, not source-observed NASA equipment",
            "Finite sampled direct checks and per-step native contacts do not establish all possible motions",
        ],
    }
    # Keep a floor/soil-supported optical target when the rover starts on the aisle.
    board = w.find("body[@name='rover__mast_registration_fixture']")
    board.set("pos", "-2.7 .6 .3")
    board.set("euler", f"0 0 {math.atan2(.6-START[1],-2.7-(START[0]-.22))}")
    item["mast_inspection"][
        "scope"
    ] = "Grounded-rover pan target remains available; historical stand-supported mast qualification is a separate revision"
    item["mast_inspection"]["limitations"].append(
        "The rover base now moves through ground contact; historical anchored-base pose/trajectory hashes do not validate this revision"
    )
    _use_ground_datum(rover)
