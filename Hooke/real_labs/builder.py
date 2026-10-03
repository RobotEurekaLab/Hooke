"""Compose ten evidence-labelled laboratory layouts and mechanical instruments."""

import math
import xml.etree.ElementTree as ET

import numpy as np

from real_labs.catalog import scene
from real_labs.equipment import add_equipment
from real_labs.furnishings import add_reference_features
from real_labs.reuse import attach_asset
from real_labs.specimens import add_specimens
from real_labs.architecture import build_bench, build_room


def values(items):
    return " ".join(str(float(value)) for value in items)


def box(parent, name, size, pos, rgba, **kwargs):
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type="box",
        size=values(size),
        pos=values(pos),
        rgba=values(rgba),
        **kwargs,
    )


def camera(world, name, position, target, fovy=48):
    z = np.asarray(position) - target
    z /= np.linalg.norm(z)
    x = np.cross((0, 0, 1), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(
        world,
        "camera",
        name=name,
        pos=values(position),
        xyaxes=values(np.r_[x, y]),
        fovy=str(fovy),
    )


def bench(world, item, accent):
    w, d, h = item["size"]
    parent = ET.SubElement(
        world,
        "body",
        name=item["id"],
        pos=values(item["pos"]),
        euler=f"0 0 {item.get('yaw', 0)}",
    )
    box(
        parent,
        item["id"] + "_top",
        (w / 2, d / 2, 0.025),
        (0, 0, h - 0.025),
        (0.16, 0.19, 0.21, 1),
    )
    box(
        parent,
        item["id"] + "_cabinet",
        (w / 2 - 0.045, d / 2 - 0.045, h / 2 - 0.09),
        (0, 0, h / 2 - 0.015),
        (0.82, 0.84, 0.83, 1),
    )
    count = max(2, int(w / 0.55))
    for index in range(count):
        x = -w / 2 + (index + 0.5) * w / count
        for side in (-1, 1):
            box(
                parent,
                f"{item['id']}_front_{index}_{side}",
                (w / count / 2 - 0.012, 0.007, h / 2 - 0.12),
                (x, side * (d / 2 - 0.033), h / 2 - 0.01),
                (*accent[:3], 1),
            )
            box(
                parent,
                f"{item['id']}_handle_{index}_{side}",
                (0.08, 0.018, 0.006),
                (x, side * (d / 2 - 0.022), h - 0.14),
                (0.64, 0.68, 0.70, 1),
            )
    for x in (-w / 2 + 0.07, w / 2 - 0.07):
        for y in (-d / 2 + 0.07, d / 2 - 0.07):
            box(
                parent,
                f"{item['id']}_leg_{x}_{y}",
                (0.03, 0.03, 0.055),
                (x, y, 0.055),
                (0.15, 0.17, 0.19, 1),
            )


def room(world, definition):
    w, d, h = definition["size"]
    colour = definition["wall_color"]
    accent = definition["accent"]
    style = definition["style"]
    box(
        world,
        "floor",
        (w / 2, d / 2, 0.08),
        (0, 0, -0.08),
        (0.52, 0.56, 0.56, 1),
        friction=".9 .01 .001",
    )
    box(world, "back_wall", (w / 2, 0.08, h / 2), (0, d / 2 + 0.08, h / 2), colour)
    box(world, "left_wall", (0.08, d / 2, h / 2), (-w / 2 - 0.08, 0, h / 2), colour)
    for x in np.arange(-w / 2, w / 2, 0.8):
        box(
            world,
            f"floor_line_x_{x}",
            (0.003, d / 2, 0.0005),
            (x, 0, 0.0005),
            (0.42, 0.46, 0.47, 1),
            contype="0",
            conaffinity="0",
        )
    for y in np.arange(-d / 2, d / 2, 0.8):
        box(
            world,
            f"floor_line_y_{y}",
            (w / 2, 0.003, 0.0005),
            (0, y, 0.0005),
            (0.42, 0.46, 0.47, 1),
            contype="0",
            conaffinity="0",
        )
    box(
        world,
        "skirting_back",
        (w / 2, 0.018, 0.045),
        (0, d / 2 - 0.01, 0.045),
        (0.22, 0.26, 0.29, 1),
    )
    box(
        world,
        "wall_service_strip",
        (w / 2 - 0.1, 0.035, 0.05),
        (0, d / 2 - 0.04, 1.18),
        accent,
    )
    for index, x in enumerate(np.linspace(-w / 2 + 0.45, w / 2 - 0.45, 8)):
        box(
            world,
            f"wall_socket_{index}",
            (0.047, 0.012, 0.06),
            (x, d / 2 - 0.084, 1.18),
            (0.94, 0.94, 0.90, 1),
        )
        for offset in (-0.014, 0.014):
            box(
                world,
                f"socket_hole_{index}_{offset}",
                (0.004, 0.004, 0.014),
                (x + offset, d / 2 - 0.098, 1.18),
                (0.15, 0.17, 0.18, 1),
            )
    # The front/right walls and ceiling are intentionally cut away for overview views.
    box(
        world,
        "window_frame",
        (0.95, 0.036, 0.51),
        (-w * 0.23, d / 2 - 0.04, 2.02),
        (0.26, 0.32, 0.36, 1),
    )
    box(
        world,
        "window_glass",
        (0.90, 0.018, 0.46),
        (-w * 0.23, d / 2 - 0.082, 2.02),
        (0.46, 0.67, 0.75, 0.34),
    )
    box(
        world,
        "window_mullion",
        (0.018, 0.025, 0.46),
        (-w * 0.23, d / 2 - 0.11, 2.02),
        (0.30, 0.34, 0.36, 1),
    )
    box(
        world,
        "door_frame",
        (0.53, 0.045, 1.12),
        (w / 2 - 0.68, d / 2 - 0.04, 1.12),
        (0.38, 0.43, 0.45, 1),
    )
    box(
        world,
        "exit_door",
        (0.48, 0.023, 1.08),
        (w / 2 - 0.68, d / 2 - 0.09, 1.10),
        (0.74, 0.79, 0.79, 1),
    )
    box(
        world,
        "door_window",
        (0.20, 0.014, 0.33),
        (w / 2 - 0.68, d / 2 - 0.12, 1.61),
        (0.27, 0.42, 0.49, 0.62),
    )
    box(
        world,
        "door_handle",
        (0.009, 0.029, 0.09),
        (w / 2 - 0.29, d / 2 - 0.13, 1.02),
        (0.48, 0.54, 0.56, 1),
    )
    for i, x in enumerate((-w * 0.24, w * 0.24)):
        box(
            world,
            f"ceiling_light_{i}",
            (0.72, 0.15, 0.022),
            (x, 0.1, h - 0.20),
            (0.96, 0.97, 1, 1),
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            world,
            "light",
            name=f"light_{i}",
            pos=values((x, -0.4, h - 0.35)),
            dir="0 0 -1",
            diffuse=".65 .65 .65",
            directional="false",
            castshadow="true",
        )
    if style in ("robotics", "testhall"):
        for x in (-w * 0.30, w * 0.30):
            box(
                world,
                f"safety_lane_{x}",
                (0.025, d * 0.34, 0.001),
                (x, -0.1, 0.002),
                (0.93, 0.69, 0.13, 1),
                contype="0",
                conaffinity="0",
            )
    if style == "testhall":
        box(
            world,
            "regolith_bed",
            (w * 0.29, d * 0.24, 0.10),
            (0, 0.4, 0.10),
            (0.55, 0.43, 0.30, 1),
            friction="1.1 .02 .003",
        )
        for side in (-1, 1):
            box(
                world,
                f"soil_bin_side_{side}",
                (0.05, d * 0.24, 0.15),
                (side * w * 0.29, 0.4, 0.15),
                (0.55, 0.59, 0.60, 1),
            )
        rng = np.random.default_rng(21)
        for i in range(80):
            xyz = (
                rng.uniform(-w * 0.26, w * 0.26),
                rng.uniform(-d * 0.2, d * 0.2) + 0.4,
                0.21,
            )
            ET.SubElement(
                world,
                "geom",
                name=f"regolith_pebble_{i}",
                type="ellipsoid",
                pos=values(xyz),
                size=values(rng.uniform(0.012, 0.045, 3)),
                rgba=".40 .35 .29 1",
            )


def accessories(world, definition):
    """Workstation essentials outside each main instrument's reserved footprint."""
    for i, b in enumerate(definition["benches"]):
        if not b.get("accessories", True):
            continue
        w, d, h = b["size"]
        local = ET.SubElement(
            world,
            "body",
            name=f"bench_supplies_{i}",
            pos=values(b["pos"]),
            euler=f"0 0 {b.get('yaw',0)}",
        )
        x, y = w / 2 - 0.15, d / 2 - 0.10
        for j in range(3):
            colour = (0.90, 0.89, 0.83, 1) if j % 2 else (0.58, 0.36, 0.17, 1)
            ET.SubElement(
                local,
                "geom",
                name=f"bottle_{i}_{j}",
                type="cylinder",
                size=".025 .055",
                pos=values((x - j * 0.072, y, h + 0.055)),
                rgba=values(colour),
            )
            ET.SubElement(
                local,
                "geom",
                name=f"cap_{i}_{j}",
                type="cylinder",
                size=".019 .009",
                pos=values((x - j * 0.072, y, h + 0.116)),
                rgba=".1 .23 .39 1",
            )
        box(
            local,
            f"tray_{i}",
            (0.105, 0.07, 0.009),
            (-w / 2 + 0.16, d / 2 - 0.11, h + 0.009),
            (0.76, 0.79, 0.82, 1),
        )
        for j in range(5):
            ET.SubElement(
                local,
                "geom",
                name=f"vial_{i}_{j}",
                type="cylinder",
                size=".009 .021",
                pos=values((-w / 2 + 0.09 + j * 0.028, d / 2 - 0.11, h + 0.039)),
                rgba=".65 .81 .82 .65",
            )


def build_scene(identifier):
    definition = scene(identifier)
    root = ET.Element("mujoco", model=identifier)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=".002",
        integrator="implicitfast",
        gravity="0 0 -9.81",
        iterations="60",
    )
    defaults = ET.SubElement(root, "default")
    ET.SubElement(
        defaults, "geom", friction=".8 .01 .001", solref=".008 1", density="600"
    )
    ET.SubElement(defaults, "joint", damping="2", armature=".005")
    ET.SubElement(root, "asset")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="800")
    ET.SubElement(
        visual,
        "headlight",
        ambient=".4 .4 .4",
        diffuse=".65 .65 .65",
        specular=".2 .2 .2",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(root, "actuator")
    revised = "architecture" in definition["room"]
    if revised:
        from real_labs.industrial_features import add_features as industrial_features
        from real_labs.wetlab_features import add_features as wetlab_features
        from real_labs.specialist_features import add_features as specialist_features

        build_room(world, definition["room"])
        for item in definition["benches"]:
            build_bench(world, item, definition["room"]["accent"])
        for furnish in (industrial_features, wetlab_features, specialist_features):
            furnish(root, world, definition)
        if definition.get("university_expansion"):
            from real_labs.university_extensions import (
                add_features as university_features,
            )

            university_features(world, definition)
    else:
        room(world, definition["room"])
        for item in definition["benches"]:
            bench(world, item, definition["room"]["accent"])
        add_reference_features(world, definition)
    accessories(world, definition)
    rows = []
    for item in definition["equipment"]:
        if item["kind"] == "rover" and not revised:
            # Authored service pose for independent wheel/arm checks above the soil bin.
            # The free-base variant supports later contact-driven mobility experiments.
            item["pos"][2] = 0.34
            x, y, _ = item["pos"]
            box(
                world,
                "rover_service_stand",
                (0.18, 0.16, 0.265),
                (x, y, 0.265),
                (0.28, 0.32, 0.35, 1),
            )
            item["params"][
                "authored_pose"
            ] = "supported service inspection; not a driving trial"
        if item["kind"] == "reuse":
            record = attach_asset(
                root,
                world,
                item["params"]["asset"],
                item["id"],
                item["pos"],
                item.get("yaw", 0),
            )
        else:
            record = add_equipment(
                root,
                world,
                item["kind"],
                item["id"],
                item["pos"],
                item.get("yaw", 0),
                item.get("params", {}),
            )
        record.update(
            id=item["id"],
            kind=item["kind"],
            pos=item["pos"],
            yaw=item.get("yaw", 0),
            params=item.get("params", {}),
            role=item["role"],
            evidence=item["evidence"],
            reference_urls=item.get("source_urls", []),
        )
        if "support" in item:
            record["support"] = item["support"]
        rows.append(record)
    w, d, h = definition["room"]["size"]
    primary = next(
        item
        for item in definition["equipment"]
        if item["id"] == definition["task"]["primary_equipment"]
    )
    centre = np.asarray(primary["pos"]) + [
        0,
        0,
        0.40 if primary["kind"] not in ("phenotyping_booth", "sequencer") else 0.9,
    ]
    distance = (
        0.70
        if primary["kind"]
        in (
            "microfluidic_station",
            "magnetic_microrobot",
            "sample_processor",
            "aerosol_sampler",
        )
        else 1
    )
    views = dict(
        overview=dict(
            position=(w * 0.77, -d * 1.04, h * 1.06), target=(0, 0.1, 0.8), fovy=48
        ),
        workstation=dict(
            position=centre + np.array([1.75, -2.15, 1.2]) * distance,
            target=centre,
            fovy=43,
        ),
        interior=dict(
            position=(w * 0.30, -d * 0.38, 1.65), target=(0, d * 0.20, 1.25), fovy=65
        ),
    )
    views.update(definition.get("camera", {}))
    for name, view in views.items():
        camera(world, name, view["position"], view["target"], view.get("fovy", 48))
    ET.SubElement(root, "keyframe")
    specimens = add_specimens(root, rows)
    if identifier == "purdue_phenotyping":
        from real_labs.industrial_features import decorate_purdue_equipment

        decorate_purdue_equipment(root)
    manifest = dict(
        definition,
        equipment=rows,
        units="m",
        gravity_m_s2=9.81,
        supported_runtime="MuJoCo CPU joint actuation; no chemical/biological process solver",
        reconstruction_status=(
            "workflow-based design"
            if definition["layout_fidelity"] == "workflow_based_design"
            else "reference-informed prototype"
        )
        + ", not a surveyed or institution-validated twin",
        specimens=specimens,
        camera_names=list(views),
    )
    ET.indent(root, space="  ")
    return root, manifest
