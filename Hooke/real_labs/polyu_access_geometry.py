"""Version-three PolyU access fixture with a genuinely free inert coupon.

The installed photograph and manufacturer-family drawings inform the split
front cover. Hinges, drives, clamp guides and low retaining lips are authored
mechanics, not an exact OEM reconstruction or calibrated material model.
"""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.architecture import numbers
from real_labs.equipment import _Builder
from real_labs.mounting import annular_mount
from real_labs.polyu_contact_geometry import build as build_revision_two

OLD = "umt_station__polyu_"
NEW = "umt_access__"
EQUIPMENT = "umt_station"
ACCESS_CONTROLS = ("guard_left", "guard_right", "clamp_front", "clamp_rear")


def _doors(b, base):
    root = b.root

    def box(parent, key, pos, half, mat="metal", **kwargs):
        return b.box(parent, pos, half, mat, collision=True, name=NEW + key, **kwargs)

    def cylinder(parent, key, pos, radius, half, mat="bright", **kwargs):
        return b.cylinder(
            parent, pos, radius, half, mat, collision=True, name=NEW + key, **kwargs
        )

    def point(angle, radius_x=0.2525, radius_y=0.2715, z=0.51):
        return np.array(
            [radius_x * math.cos(angle), 0.02 + radius_y * math.sin(angle), z]
        )

    for side, label in ((-1, "left"), (1, "right")):
        pivot = np.array([side * 0.263, 0.02, 0.51])
        door = b.moving(
            base,
            "guard_" + label,
            pivot,
            (0, 0, side),
            (0, math.radians(100)),
            kind="hinge",
            mass=0.95,
            kp=120,
            force=10,
        )
        door.find("inertial").set("diaginertia", ".045 .045 .030")
        # Authored angular damping uses the declared hinge-axis inertia plus
        # armature, not translational panel mass. Original X/Z gains unchanged.
        b.actuator.find(
            "position[@name='" + NEW + "actuator_guard_" + label + "']"
        ).set("kv", str(2 * math.sqrt(120 * 0.035)))
        for z in (0.25, 0.77):
            box(
                base,
                f"hinge_root_{label}_{z}",
                (side * 0.254, 0.046, z - 0.025),
                (0.016, 0.012, 0.008),
            )
            b.rod(
                base,
                (side * 0.263, 0.020, z - 0.025),
                (side * 0.263, 0.045, z - 0.025),
                0.003,
                "bright",
                collision=True,
                name=NEW + f"hinge_root_link_{label}_{z}",
            )
            cylinder(
                base, f"hinge_pin_{label}_{z}", (side * 0.263, 0.02, z), 0.004, 0.025
            )
            # Fixed thrust collars bound the sleeve axially with 0.6 mm
            # running gaps; the collars are genuinely attached to the pin.
            for end in (-1, 1):
                cylinder(
                    base,
                    f"hinge_collar_{label}_{z}_{end}",
                    (side * 0.263, 0.02, z + end * 0.0146),
                    0.0078,
                    0.004,
                )
            # The pin meets the root; the rotating sleeve has radial/end gaps.
            annular_mount(
                b,
                door,
                f"hinge_{label}_{z}",
                (0, 0, z - 0.51),
                0.0046,
                0.0086,
                0.010,
                axis="z",
                proxies=True,
                material="metal",
                namespace="hinge",
            )
            box(
                door,
                f"hinge_leaf_{label}_{z}",
                (-side * 0.0086, 0.0, z - 0.51),
                (0.0035, 0.003, 0.009),
            )

        a0, a1 = (
            (math.pi, 1.5 * math.pi - 0.009)
            if side < 0
            else (1.5 * math.pi + 0.009, 2 * math.pi)
        )
        count = 24
        vertices = []
        for z in (0.184, 0.836):
            for offset in (0, 0.003):
                for i in range(count + 1):
                    angle = a0 + (a1 - a0) * i / count
                    vertices.append(
                        tuple(point(angle, 0.251 + offset, 0.270 + offset, z) - pivot)
                    )
        stride = count + 1
        faces = []
        for i in range(count):
            for a, c, reverse in (
                (0, stride, False),
                (2 * stride, 3 * stride, True),
                (0, 2 * stride, True),
                (stride, 3 * stride, False),
            ):
                triangles = ((a + i, a + i + 1, c + i + 1), (a + i, c + i + 1, c + i))
                faces.extend(tuple(reversed(f)) if reverse else f for f in triangles)
        for i in (0, count):
            triangles = (
                (i, stride + i, 3 * stride + i),
                (i, 3 * stride + i, 2 * stride + i),
            )
            faces.extend(tuple(reversed(f)) if i else f for f in triangles)
        mesh = NEW + "pane_" + label
        ET.SubElement(
            root.find("asset"),
            "mesh",
            name=mesh,
            vertex=numbers(np.asarray(vertices).ravel()),
            face=" ".join(str(x) for f in faces for x in f),
            maxhullvert="4",
        )
        b.geom(door, "mesh", (), mesh=mesh, material="glass", name=mesh)
        for i in range(count):
            ids = [
                i,
                i + 1,
                stride + i,
                stride + i + 1,
                2 * stride + i,
                2 * stride + i + 1,
                3 * stride + i,
                3 * stride + i + 1,
            ]
            name = NEW + f"pane_{label}_contact_{i}"
            ET.SubElement(
                root.find("asset"),
                "mesh",
                name=name + "_mesh",
                vertex=numbers(np.asarray([vertices[j] for j in ids]).ravel()),
            )
            b.geom(
                door,
                "mesh",
                (),
                mesh=name + "_mesh",
                material="glass",
                collision=True,
                name=name,
                group=3,
                rgba=(0.5, 0.5, 0.5, 0),
            )
        # Slim structural edge frame connects each pane to its two bearings.
        angle = a0 if side < 0 else a1
        edge = point(angle, z=0.51) - pivot
        box(door, f"edge_{label}", edge, (0.002, 0.003, 0.326))
        for z in (0.186, 0.834):
            for i in range(count):
                a = a0 + (a1 - a0) * i / count
                c = a0 + (a1 - a0) * (i + 1) / count
                b.rod(
                    door,
                    point(a, z=z) - pivot,
                    point(c, z=z) - pivot,
                    0.0018,
                    "dark",
                    collision=True,
                    name=NEW + f"edge_{label}_{z}_{i}",
                )
        # Large low circular grips correspond to the front-cover drawing.
        angle = 1.5 * math.pi + side * 0.30
        center = point(angle, z=0.275) - pivot
        normal = np.array([math.cos(angle), math.sin(angle), 0.0])
        grip = ET.SubElement(
            door,
            "body",
            name=NEW + "grip_" + label,
            pos=numbers(center + normal * 0.026),
            quat=numbers((math.cos(angle / 2), 0, 0, math.sin(angle / 2))),
        )
        annular_mount(
            b,
            grip,
            "grip_" + label,
            (0, 0, 0),
            0.015,
            0.024,
            0.004,
            axis="x",
            proxies=True,
            namespace="grip",
            material="dark",
        )
        for zoffset in (-0.017, 0.017):
            b.rod(
                door,
                center + [0, 0, zoffset],
                center + normal * 0.026 + [0, 0, zoffset],
                0.003,
                "bright",
                collision=True,
                name=NEW + f"grip_mount_{label}_{zoffset}",
            )


def _clamps(b, slider):
    def box(parent, key, pos, half, mat="metal", **kwargs):
        return b.box(parent, pos, half, mat, collision=True, name=NEW + key, **kwargs)

    def cylinder(parent, key, pos, radius, half, mat="bright", **kwargs):
        return b.cylinder(
            parent, pos, radius, half, mat, collision=True, name=NEW + key, **kwargs
        )

    for sign, label in ((-1, "front"), (1, "rear")):
        jaw = b.moving(
            slider,
            "clamp_" + label,
            (0, sign * 0.025, 0.032),
            (0, sign, 0),
            (-0.0002, 0.015),
            mass=0.03,
            kp=2000,
            force=5,
        )
        for side in (-1, 1):
            x = side * 0.033
            cylinder(
                slider,
                f"clamp_rail_{label}_{side}",
                (x, sign * 0.043, 0.032),
                0.0025,
                0.024,
                euler=(math.pi / 2, 0, 0),
            )
            for y in (0.016, 0.070):
                box(
                    slider,
                    f"clamp_foot_{label}_{side}_{y}",
                    (x, sign * y, 0.025),
                    (0.005, 0.003, 0.007),
                )
            bearing = ET.SubElement(
                jaw,
                "body",
                name=NEW + f"jaw_bearing_{label}_{side}",
                pos=numbers((x, 0, 0)),
                quat=numbers((math.sqrt(0.5), 0, 0, math.sqrt(0.5))),
            )
            annular_mount(
                b,
                bearing,
                f"jaw_{label}_{side}",
                (0, 0, 0),
                0.0029,
                0.0059,
                0.004,
                axis="x",
                proxies=True,
                namespace="jaw",
                material="dark",
            )
        box(jaw, "jaw_web_" + label, (0, 0, 0), (0.0271, 0.004, 0.003))
        box(
            jaw,
            "jaw_pad_" + label,
            (0, -sign * 0.003, -0.0035),
            (0.018, 0.002, 0.0015),
            "metal",
        )


def _free_coupon(root, base, slider):
    """Place the authored free body from the instrument's initial world frame."""
    world = root.find("worldbody")
    if base not in list(world):
        raise ValueError("PolyU instrument requires a world-mounted base")
    coupon = slider.find("geom[@name='" + OLD + "coupon']")
    surface = slider.find("site[@name='umt_station__site_tribology_coupon']")
    local = np.fromstring(slider.get("pos"), sep=" ") + np.fromstring(
        coupon.get("pos"), sep=" "
    )
    position = np.fromstring(base.get("pos"), sep=" ")
    quaternion = np.fromstring(base.get("quat", "1 0 0 0"), sep=" ")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    position += rotation.reshape(3, 3) @ local
    body = ET.SubElement(
        world,
        "body",
        name=NEW + "free_coupon",
        pos=numbers(position),
        quat=numbers(quaternion),
    )
    ET.SubElement(body, "freejoint", name=NEW + "free_coupon_joint")
    slider.remove(coupon)
    slider.remove(surface)
    coupon.set("pos", "0 0 0")
    coupon.set("mass", ".01728")
    coupon.attrib.pop("density", None)
    surface.set("pos", "0 0 .002")
    body.extend((coupon, surface))
    return body


def _support_and_lips(builder, slider):
    support = slider.find("geom[@name='" + OLD + "specimen_puck']")
    support.set("type", "box")
    support.set("size", ".023 .023 .004")
    for side, sign in (("negative", -1), ("positive", 1)):
        ET.SubElement(
            slider,
            "geom",
            name="umt_retention__stop_" + side,
            type="box",
            pos=f"{sign * .02115:.8f} 0 .027",
            size=".001 .012 .002",
            material="umt_access__mat_metal",
            contype="1",
            conaffinity="1",
            friction=".8 .01 .001",
            mass="0",
        )
    name = EQUIPMENT + "__site_loading_center"
    ET.SubElement(
        slider, "site", name=name, pos="0 0 .028", size=".0005", rgba=".2 .8 .7 1"
    )
    builder.metadata["sites"]["loading_center"] = name


def build(builder, base, params):
    """Build only the explicit v3 contract; no silent legacy task mapping."""
    if params.get("interface_revision") != 3 or builder.name != EQUIPMENT:
        raise ValueError(
            "PolyU access requires explicit interface_revision=3 and umt_station; replay v2 from its archived package"
        )
    build_revision_two(builder, base, dict(params, interface_revision=2))
    root = builder.root
    slider = base.find("body[@name='umt_station__body_coupon_x']")
    for owner in base.iter():
        for geom in list(owner.findall("geom")):
            name = geom.get("name", "")
            if (
                name == OLD + "clear_front"
                or name.startswith(OLD + "guard_contact_")
                or any(
                    name.startswith(OLD + key)
                    for key in (
                        "drive_cover_",
                        "drive_face_",
                        "retaining_post_",
                        "retaining_pad_",
                    )
                )
            ):
                owner.remove(geom)
    free = _free_coupon(root, base, slider)
    access = _Builder(root, "umt_access")
    _doors(access, base)
    _clamps(access, slider)
    _support_and_lips(builder, slider)
    for body in base.iter("body"):
        if not body.get("name", "").startswith(NEW):
            continue
        if body.find("inertial") is None:
            ET.SubElement(
                body, "inertial", pos="0 0 0", mass=".002", diaginertia="1e-7 1e-7 1e-7"
            )
            body.set("gravcomp", "1")
        for geom in body.findall("geom"):
            geom.set("mass", "0")
    for key in ("joints", "actuators", "joint_ranges", "actuator_modes"):
        builder.metadata[key].update(access.metadata[key])
    builder.metadata.update(
        interface_revision=3,
        capabilities=[
            "Contact-supported free inert coupon",
            "Measured guard/clamp release and return",
            "Measured low-load reciprocation with physical retention",
        ],
        access_interface=dict(
            revision=3,
            free_coupon_body=free.get("name"),
            sample_size_m=[0.04, 0.04, 0.004],
            autonomous_pickup=False,
            door_motion_basis="Manufacturer-family panel and hinge topology, estimated dimensions and powered manual-action proxies",
        ),
        existing_free_specimen=dict(
            body=free.get("name"),
            freejoint=NEW + "free_coupon_joint",
            collision_geom=OLD + "coupon",
            support_geom=OLD + "specimen_puck",
            sample_site=EQUIPMENT + "__site_loading_center",
            dimensions_m=[0.04, 0.04, 0.004],
            mass_kg=0.01728,
            mass_basis="Authored aluminium-density inert coupon, not identified experimental material",
            initial_support_clearance_m=0.0,
        ),
    )
    builder.metadata["contact_interface"].update(
        revision=3,
        coordinate_migration="Original X/Z frames, ranges, drives and specimen scale are retained. The coupon is now a free body; loading_center is a separate fixed support datum and tribology_coupon follows its actual top. Old fixed-coupon task semantics are retired, without automatic mapping.",
        control_model="Original X/Z drives plus authored guard/clamp servos. Real support, jaw and low-lip contacts retain the free coupon; no hidden weld or calibrated force controller.",
        positive_retention=dict(
            side_clearance_m=0.00015,
            support_overlap_m=0.001,
            lip_height_above_support_m=0.003,
            lip_top_below_coupon_top_m=0.001,
            dimensions_basis="Authored rectangular-holder estimate, not measured OEM geometry",
            mass_basis="Included in original authored carriage lumped inertia",
        ),
        source_boundary="PolyU installed photograph and Bruker-family patent figures support the exterior/guard concept; actual model identity, internal fixture dimensions and powered actuation remain unverified.",
    )
    builder.metadata["limitations"] = [
        "Reference-informed mechanical prototype, not an exact OEM replica or manufacturer performance model.",
        "Original 8 ms contact response, friction and ideal gravity compensation are authored simulation assumptions; no calibrated friction, wear or deformation.",
        "Doors and clamps expose a supported free coupon and a lifting corridor; autonomous pickup, transfer and replacement are not implemented.",
        "Full joint ranges contain obstructed combinations. Loaded operation requires both guards closed, both jaws engaged and observed ball contact.",
        "The free coupon can retain a sub-millimetre offset inside its fixture clearance; exact sample repositioning is not claimed.",
    ]
