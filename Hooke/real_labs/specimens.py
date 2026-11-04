"""Free specimen carriers settled by contact, not attached by hidden welds."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.carrier_fixtures import add_carrier_pocket


def _support_pose(model, data, item, size):
    """Find the authored horizontal collision surface below a loading target.

    Reset represents a loaded carrier, not a 10 mm drop test. The target site
    may describe its centre or the deck surface, so locate the actual support
    instead of imposing one site-height convention on every instrument.
    """
    site = model.site(item["sample_site"]).id
    rotation = data.site_xmat[site].reshape(3, 3)
    if not np.allclose(rotation[:, 2], (0, 0, 1), atol=1e-6):
        raise ValueError(f"Supported carrier requires an upright site: {item['id']}")
    origin = data.site_xpos[site].copy()
    groups = model.geom_group.copy()
    try:
        # Ignore non-contact decoration: the carrier must have physical support.
        enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_group[:] = np.where(enabled, 0, 1)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            origin,
            np.array([0.0, 0.0, -1.0]),
            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            1,
            -1,
            hit,
        )
    finally:
        model.geom_group[:] = groups
    if distance < 0 or distance > size[2] / 2 + 0.03:
        raise ValueError(f"No nearby contact support for carrier: {item['id']}")
    support = int(hit[0])
    owner = int(model.geom_bodyid[support])
    expected = int(model.site_bodyid[site])
    if owner != expected:
        raise ValueError(f"Carrier support does not move with its site: {item['id']}")
    surface = origin.copy()
    surface[2] -= distance
    pos = surface.copy()
    pos[2] += size[2] / 2 + 0.00005
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return pos, quat, support, surface


def _register_existing_free_specimen(model, data, item, claimed):
    """Validate an authored free carrier; never create or reposition it."""
    spec = item["existing_free_specimen"]
    try:
        body = model.body(spec["body"]).id
        joint = model.joint(spec["freejoint"]).id
        collider = model.geom(spec["collision_geom"]).id
        support = model.geom(spec["support_geom"]).id
        site = model.site(spec["sample_site"]).id
        owner = model.body(item["body"]).id
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Existing free specimen references a missing object"
        ) from error
    if body in claimed:
        raise ValueError("Existing free specimen body is claimed more than once")
    if (
        model.body_jntnum[body] != 1
        or model.jnt_bodyid[joint] != body
        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
        or model.geom_bodyid[collider] != body
        or not model.geom_contype[collider]
        or not model.geom_conaffinity[collider]
        or model.body_mass[body] <= 0
        or not np.isclose(model.body_mass[body], spec["mass_kg"], rtol=0, atol=1e-12)
        or spec["sample_site"] != item["sample_site"]
    ):
        raise ValueError("Existing free specimen ownership, contact or mass mismatch")
    for member in (int(model.site_bodyid[site]), int(model.geom_bodyid[support])):
        while member and member != owner:
            member = int(model.body_parentid[member])
        if member != owner:
            raise ValueError(
                "Existing free specimen site/support belongs to another instrument"
            )
    if (
        model.body_weldid[model.site_bodyid[site]]
        != model.body_weldid[model.geom_bodyid[support]]
    ):
        raise ValueError(
            "Existing free specimen site and support must share a fixed frame"
        )
    if not model.geom_contype[support] or not model.geom_conaffinity[support]:
        raise ValueError("Existing free specimen support contact is disabled")
    for kind, first, second in zip(model.eq_type, model.eq_obj1id, model.eq_obj2id):
        if kind in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD) and body in (
            first,
            second,
        ):
            raise ValueError(
                "Existing free specimen must not have an attachment equality"
            )
    if model.geom_type[collider] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half = model.geom_size[collider, :2]
        dimensions = np.array([2 * radius, 2 * radius, 2 * half])
    elif model.geom_type[collider] == mujoco.mjtGeom.mjGEOM_BOX:
        dimensions = 2 * model.geom_size[collider]
    else:
        raise ValueError(
            "Existing free specimen needs a declared box or cylinder envelope"
        )
    if not np.allclose(
        dimensions, item["sample_size_m"], rtol=0, atol=1e-12
    ) or not np.allclose(dimensions, spec["dimensions_m"], rtol=0, atol=1e-12):
        raise ValueError("Existing free specimen dimensions mismatch")
    # This registration contract declares a horizontal loading surface. Its
    # vertical half-height calculation requires an upright collider; yaw is
    # allowed, but a tilted box/cylinder cannot reuse the same support claim.
    rotation = data.geom_xmat[collider].reshape(3, 3)
    if not np.allclose(rotation[:, 2], (0, 0, 1), rtol=0, atol=1e-9):
        raise ValueError("Existing free specimen must have an upright collider")
    # Validate a real enabled supporting surface, not a name-only declaration.
    origin = data.geom_xpos[collider].copy()
    hit = np.array([-1], np.int32)
    groups = model.geom_group.copy()
    try:
        enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_group[:] = np.where(enabled, 0, 1)
        distance = mujoco.mj_ray(
            model,
            data,
            origin,
            np.array([0.0, 0.0, -1.0]),
            np.array([1, 0, 0, 0, 0, 0], np.uint8),
            1,
            body,
            hit,
        )
    finally:
        model.geom_group[:] = groups
    gap = distance - dimensions[2] / 2
    if (
        hit[0] != support
        or not np.isclose(gap, spec["initial_support_clearance_m"], atol=1e-9, rtol=0)
        or np.linalg.norm(data.site_xpos[site] - data.xpos[body]) > 0.001
    ):
        raise ValueError("Existing free specimen has no matching physical support/site")
    claimed.add(body)
    process = dict(
        sample_site=spec["sample_site"],
        sample_body=spec["body"],
        load_tolerance=0.002,
        max_sample_speed=0.01,
    )
    record = dict(
        spec,
        equipment=item["id"],
        fidelity="Preinstalled rigid inert carrier; no scientific specimen model",
        registration="Existing free body validated without creating or repositioning geometry",
    )
    return process, record


def add_specimens(root, equipment):
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    records = []
    claimed = set()
    for item in equipment:
        if item.get("sample_attachment") != "supported":
            continue
        if "existing_free_specimen" in item:
            process, record = _register_existing_free_specimen(
                model, data, item, claimed
            )
            item["process"] = process
            records.append(record)
            continue
        size = np.asarray(item["sample_size_m"], dtype=float)
        pos, quat, support, surface = _support_pose(model, data, item, size)
        fixture = add_carrier_pocket(root, model, data, item, size, support, surface)
        if fixture:
            pos[2] += fixture["bearing_pad_thickness_m"]
        name = item["id"] + "__specimen"
        body = ET.SubElement(
            root.find("worldbody"),
            "body",
            name=name,
            pos=" ".join(map(str, pos)),
            quat=" ".join(map(str, quat)),
        )
        ET.SubElement(body, "freejoint", name=name + "__free")
        ET.SubElement(
            body,
            "geom",
            name=name + "__collision",
            type="box",
            size=" ".join(map(str, size / 2)),
            rgba=".35 .54 .62 1",
            mass=".005",
            friction="1 .01 .001",
            solref=f"{fixture['contact_time_constant_s'] if fixture else .006} 1",
            solimp=".95 .99 .0001",
        )
        process = dict(
            sample_site=item["sample_site"],
            sample_body=name,
            load_tolerance=max(0.015, float(size[2] / 2 + 0.006)),
        )
        access = next(
            (
                key
                for key in ("door", "front_door_hinge", "lid", "chamber_lid", "flowcell_tray")
                if key in item["actuators"]
            ),
            None,
        )
        if access:
            process.update(access_control=access, closed_value=0, closed_tolerance=0.01)
        item["process"] = process
        records.append(
            dict(
                body=name,
                equipment=item["id"],
                description=item["sample_description"],
                dimensions_m=size.tolist(),
                fidelity="rigid specimen carrier; not a scientific specimen model",
                support_geom=(
                    fixture["bearing_pad_geom"] if fixture else model.geom(support).name
                ),
                initial_support_clearance_m=0.00005,
                mass_kg=0.005,
                mass_basis="generic inert carrier assumption; not manufacturer data",
                retaining_fixture=fixture,
            )
        )
    return records
