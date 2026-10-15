"""Authored dry specimen loading fixture; no OEM or tissue processing claim."""

import xml.etree.ElementTree as ET

import numpy as np
import mujoco

REVISION = "penn-dry-processor-v1"
SOURCE = dict(
    reference="Penn pathology official room still; authored dry enclosed carrier fixture",
    url="https://www.pennmedicine.org/departments-and-centers/department-of-pathology-and-laboratory-medicine",
    dimensions_m=[0.46, 0.90, 0.64],
    dimension_basis="Estimated complete fixture envelope; no identifiable processor or OEM dimensions in cached still",
)
INTERFACE = (
    "specimen_seat",
    (0.09, 0.07, 0.025),
    "supported",
    "free inert cassette in an open carriage pocket",
)


def build_processor(b, base, params):
    """Build physical shell, open-bore rail carriages, door and free cassette."""

    def box(parent, name, pos, half, material="metal"):
        return b.box(
            parent,
            pos,
            half,
            material,
            name=f"{b.name}__{name}",
            collision=True,
            density=0,
        )

    def rod(parent, name, start, end, radius=0.0045):
        return b.rod(
            parent,
            start,
            end,
            radius,
            "bright",
            name=f"{b.name}__{name}",
            collision=True,
            density=0,
        )

    ET.SubElement(base, "inertial", pos="0 0 .20", mass="12", diaginertia=".35 .40 .30")
    # The fixed base is continuously supported by four feet on the worktop.
    for x in (-0.16, 0.16):
        for y in (-0.395, 0.195):
            box(base, f"foot_{x}_{y}", (x, y, 0.01), (0.025, 0.025, 0.01), "rubber")
    box(base, "base", (0, -0.10, 0.032), (0.20, 0.35, 0.012), "shell")
    # Housing floor is below the carrier; rails have an unobstructed open path.
    box(base, "chamber_floor", (0, 0.051, 0.0545), (0.18, 0.174, 0.0105), "metal")
    for x in (-0.18, 0.18):
        box(base, f"side_{x}", (x, 0.051, 0.2025), (0.014, 0.174, 0.1585), "shell")
    box(base, "rear", (0, 0.225, 0.2025), (0.166, 0.012, 0.1585), "shell")
    box(base, "roof", (0, 0.057, 0.373), (0.194, 0.180, 0.012), "shell")
    # Two supported guide rods and real square-bore sleeves, not a solid rod
    # hidden inside a parent-filtered solid bearing.
    for x in (-0.068, 0.068):
        rod(base, f"tray_rail_{x}", (x, -0.42, 0.081), (x, 0.18, 0.081))
        for y in (-0.423, 0.183):
            box(
                base,
                f"tray_rail_end_{x}_{y}",
                (x, y, 0.06475),
                (0.013, 0.007, 0.02075),
                "metal",
            )
    carrier = b.moving(
        base,
        "carrier_translation",
        (0, -0.34, 0),
        (0, 1, 0),
        (0, 0.38),
        mass=0.50,
        kp=1600,
        force=80,
    )
    floor = box(
        carrier, "carrier_floor", (0, 0, 0.113), (0.085, 0.068, 0.008), "bright"
    )
    for x in (-0.068, 0.068):
        for y in (-0.035, 0.035):
            # Bore half-width .00495 versus .0045-radius round shaft.
            for s in (-1, 1):
                box(
                    carrier,
                    f"tray_sleeve_x_{x}_{y}_{s}",
                    (x + s * 0.007475, y, 0.081),
                    (0.002525, 0.019, 0.010),
                    "dark",
                )
                box(
                    carrier,
                    f"tray_sleeve_z_{x}_{y}_{s}",
                    (x, y, 0.081 + s * 0.007475),
                    (0.00495, 0.019, 0.002525),
                    "dark",
                )
            box(
                carrier,
                f"tray_bridge_{x}_{y}",
                (x, y, 0.098),
                (0.009, 0.018, 0.007),
                "metal",
            )
    for x in (-0.051, 0.051):
        box(carrier, f"pocket_side_{x}", (x, 0, 0.133), (0.004, 0.039, 0.012), "blue")
    for y in (-0.043, 0.043):
        box(carrier, f"pocket_end_{y}", (0, y, 0.133), (0.055, 0.004, 0.012), "blue")
    # Door slides outside the shell; its plane never traverses the rear lintel.
    for x in (-0.173, 0.173):
        box(
            base, f"door_column_{x}", (x, -0.155, 0.330), (0.014, 0.016, 0.286), "shell"
        )
    for x in (-0.145, 0.145):
        rod(base, f"door_rail_{x}", (x, -0.159, 0.072), (x, -0.159, 0.616), 0.004)
        for z in (0.065, 0.623):
            box(
                base,
                f"door_rail_end_{x}_{z}",
                (x, -0.159, z),
                (0.014, 0.013, 0.007),
                "metal",
            )
    # Top bridge connects both columns and rod supports.
    box(base, "door_top_bridge", (0, -0.155, 0.623), (0.187, 0.016, 0.007), "shell")
    door = b.moving(
        base,
        "access_door",
        (0, -0.139, 0.220),
        (0, 0, 1),
        (0, 0.255),
        mass=0.45,
        kp=1800,
        force=100,
    )
    for x in (-0.115, 0.115):
        box(door, f"door_side_{x}", (x, 0, 0), (0.008, 0.006, 0.133), "metal")
    for z in (-0.125, 0.125):
        box(door, f"door_cross_{z}", (0, 0, z), (0.107, 0.006, 0.008), "metal")
    box(door, "door_window", (0, 0, 0), (0.107, 0.005, 0.117), "glass")
    for x in (-0.145, 0.145):
        for z in (-0.09, 0.09):
            for s in (-1, 1):
                box(
                    door,
                    f"door_sleeve_x_{x}_{z}_{s}",
                    (x + s * 0.007, -0.02, z),
                    (0.0025, 0.0095, 0.015),
                    "dark",
                )
                box(
                    door,
                    f"door_sleeve_y_{x}_{z}_{s}",
                    (x, -0.02 + s * 0.007, z),
                    (0.0045, 0.0025, 0.015),
                    "dark",
                )
            box(
                door,
                f"door_bracket_{x}_{z}",
                (np.sign(x) * 0.1265, -0.007, z),
                (0.009, 0.012, 0.01),
                "metal",
            )
    box(door, "door_grip", (0, -0.014, -0.125), (0.038, 0.008, 0.006), "dark")
    # Small cabinet-supported controller outside both swept interfaces.
    box(base, "controller", (0.20, 0.115, 0.265), (0.022, 0.07, 0.08), "shell")
    box(
        base, "controller_screen", (0.223, 0.105, 0.284), (0.001, 0.04, 0.032), "screen"
    )
    box(base, "controller_button", (0.224, 0.102, 0.227), (0.002, 0.008, 0.006), "led")
    site = b.site(carrier, "specimen_seat", (0, 0, 0.13355))
    b.site(base, "chamber_center", (0, 0.04, 0.13355))
    # World free body with the same yaw as the fixture, registered by specimens.py.
    quat = np.fromstring(base.get("quat"), sep=" ")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    pos = np.fromstring(base.get("pos"), sep=" ") + rotation.reshape(3, 3) @ np.array(
        [0, -0.34, 0.13355]
    )
    body_name = b.name + "__cassette"
    free_name = body_name + "__free"
    shape_name = body_name + "__collision"
    body = ET.SubElement(
        b.root.find("worldbody"),
        "body",
        name=body_name,
        pos=" ".join(map(str, pos)),
        quat=base.get("quat"),
    )
    ET.SubElement(body, "freejoint", name=free_name)
    ET.SubElement(
        body,
        "geom",
        name=shape_name,
        type="box",
        size=".045 .035 .0125",
        rgba=".68 .25 .31 1",
        mass=".06",
        contype="1",
        conaffinity="1",
        friction="1 .01 .001",
        solref=".006 1",
        solimp=".95 .99 .0001",
    )
    b.metadata.update(
        processor_revision=REVISION,
        existing_free_specimen=dict(
            body=body_name,
            freejoint=free_name,
            collision_geom=shape_name,
            support_geom=floor.get("name"),
            sample_site=site,
            mass_kg=0.06,
            dimensions_m=[0.09, 0.07, 0.025],
            initial_support_clearance_m=0.00005,
        ),
        capabilities=[
            "physical powered access door",
            "contact-conveyed free inert cassette",
            "load into dry chamber and return to external staging",
        ],
        retired_controls={
            key: "Retired mismatched microtome interface; use PennProcessor loading route"
            for key in ("handwheel", "specimen_feed", "section_stroke", "blade_guard")
        },
        coordinated_inspection_targets={
            "access_door": 0.255,
            "carrier_translation": 0.38,
        },
        limitations=[
            "Authored dry mechanical workflow, not an identified processor or OEM replica",
            "Cassette starts supported on the external tray; external accession pickup and tissue processing are not modeled",
            "Powered prismatic joints model ideal compensated guides; no motor internals or processing fluid",
        ],
    )
