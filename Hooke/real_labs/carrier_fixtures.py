"""Original open carrier pockets, inferred for mechanical inspection only.

These are visible contact hardware attached to the supporting stage. Carriers
retain six free degrees of freedom and can be lifted out; no welds are added.
Dimensions describe our surrogate fixture, not a manufacturer's internal part.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# Per-side clearance, wall thickness/height, and bearing-pad thickness, metres.
_POCKETS = {
    "microfluidic_station": (0.00015, 0.0006, 0.0030, 0.0003),
    "isotope_analyzer": (0.00025, 0.0030, 0.0440, 0.0010),
    "aerosol_sampler": (0.00020, 0.0008, 0.0030, 0.0010),
    "deposition_chamber": (0.00020, 0.0010, 0.0080, 0.0010),
}

# Drive force/torque limit and viscous joint damping. These original inspection
# settings replace unrealistically abrupt 120 N/Nm moves of unsecured carriers.
_INSPECTION_DRIVES = {
    "microfluidic_station": {"stage_x": (0.2, 2.5), "stage_y": (0.2, 2.5)},
    # The source-informed Cartesian rack is fixed; its carrier needs no stage drive.
    "isotope_analyzer": {},
    "aerosol_sampler": {"filter_drawer": (0.25, 2.5)},
    "deposition_chamber": {"platen": (0.2, 0.05)},
}


def _inspection_dynamics(root, item):
    """Declare finite drive bandwidth, preserving full joint and control ranges."""
    for semantic, (limit, damping) in _INSPECTION_DRIVES[item["kind"]].items():
        name = item["actuators"][semantic]
        drive = root.find(f".//position[@name='{name}']")
        if drive is None:
            raise ValueError(f"Carrier fixture requires a position drive: {name}")
        drive.set("forcelimited", "true")
        drive.set("forcerange", f"{-limit} {limit}")
        joint = root.find(f".//joint[@name='{item['joints'][semantic]}']")
        joint.set("damping", str(damping))
        rotary_joint = joint.get("type", "hinge") == "hinge"
        if rotary_joint:
            # Moderate PD gains avoid force-limit chatter with the small rotor
            # inertia; stiffness, damping and torque are inspection assumptions.
            drive.set("kp", "6")
            drive.set("kv", ".6")
        item.setdefault("control_contracts", {})[semantic] = dict(
            mode="position",
            units="rad" if rotary_joint else "m",
            force_limit=limit,
            force_units="Nm" if rotary_joint else "N",
            joint_viscous_damping=damping,
            damping_units="Nm/(rad/s)" if rotary_joint else "N/(m/s)",
            position_gain=float(drive.get("kp")),
            velocity_gain=float(drive.get("kv")),
            parameter_basis="original force-limited inspection-drive assumption",
            limitations=[
                "Not a manufacturer-calibrated motion profile",
                "Full authored joint and command ranges retained",
                "Carrier retention depends on physical contact; no attachment constraint",
                "Drive and contact behavior tested in MuJoCo only; native Isaac parity unverified",
            ],
        )
    rotary = item["kind"] in {"isotope_analyzer", "deposition_chamber"}
    step = 0.0005 if rotary else 0.002
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(min(float(option.get("timestep", "0.002")), step)))
    return 0.002 if rotary else 0.006


def add_carrier_pocket(root, model, data, item, size, support, surface):
    dimensions = _POCKETS.get(item["kind"])
    if dimensions is None:
        return None
    clearance, thickness, height, pad = dimensions
    contact_time_constant = _inspection_dynamics(root, item)
    owner = int(model.geom_bodyid[support])
    parent = root.find(f".//body[@name='{model.body(owner).name}']")
    if parent is None:
        raise ValueError(f"Missing carrier fixture parent: {item['id']}")
    site = model.site(item["sample_site"]).id
    frame = data.site_xmat[site].reshape(3, 3)
    body_frame = data.xmat[owner].reshape(3, 3)
    origin = body_frame.T @ (surface - data.xpos[owner])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (body_frame.T @ frame).ravel())
    fixture = ET.SubElement(
        parent,
        "body",
        name=item["id"] + "__carrier_pocket",
        pos=" ".join(map(str, origin)),
        quat=" ".join(map(str, quat)),
    )
    floor = item["id"] + "__carrier_bearing_pad"
    ET.SubElement(
        fixture,
        "geom",
        type="box",
        name=floor,
        pos=f"0 0 {pad / 2}",
        size=f"{size[0] / 2 + clearance + thickness} "
        f"{size[1] / 2 + clearance + thickness} {pad / 2}",
        rgba=".35 .38 .40 1",
        contype="1",
        conaffinity="1",
        density="1200",
        friction=".8 .01 .001",
        solref=f"{contact_time_constant} 1",
        solimp=".95 .99 .0001",
    )
    for axis in (0, 1):
        other = 1 - axis
        half = np.array([0.0, 0.0, height / 2])
        half[axis] = thickness / 2
        half[other] = size[other] / 2 + clearance + thickness
        for sign in (-1, 1):
            center = np.array([0.0, 0.0, pad + height / 2])
            center[axis] = sign * (size[axis] / 2 + clearance + thickness / 2)
            ET.SubElement(
                fixture,
                "geom",
                type="box",
                name=f"{item['id']}__carrier_guide_{axis}_{sign}",
                pos=" ".join(map(str, center)),
                size=" ".join(map(str, half)),
                rgba=".22 .25 .28 1",
                contype="1",
                conaffinity="1",
                density="1200",
                friction=".8 .01 .001",
                solref=f"{contact_time_constant} 1",
                solimp=".95 .99 .0001",
            )
    return dict(
        body=fixture.get("name"),
        design="original open retaining pocket, inferred mechanical fixture",
        lateral_clearance_m=clearance,
        wall_thickness_m=thickness,
        wall_height_m=height,
        bearing_pad_thickness_m=pad,
        bearing_pad_geom=floor,
        contact_time_constant_s=contact_time_constant,
        contact_basis="rigid-carrier contact surrogate; finite-step convergence checked",
        removal="upward lift against contact friction; no hidden attachment",
        manufacturer_part=False,
    )
