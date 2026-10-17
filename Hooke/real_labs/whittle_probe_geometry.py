"""Connected dry traverse beside the source-informed historical Whittle rig.

The photograph establishes the duct's appearance, not this external fixture.
Guide dimensions and running gaps are authored. Original axis frames, travel,
inertias, reference dimensions and probe endpoints remain unchanged.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.mounting import annular_mount

PREFIX = "primary_apparatus__whittle_"


def _box(b, parent, key, pos, half, material="metal"):
    return b.box(parent, pos, half, material, collision=True, name=PREFIX + key)


def _sleeve(b, parent, key, center, inner, outer, half, axis):
    # Rotate the reusable sleeve's geometry directly into its owning rigid
    # body. A separate fixed body would infer additional moving inertia.
    holder = ET.Element("body")
    annular_mount(
        b,
        holder,
        key,
        (0, 0, 0),
        inner,
        outer,
        half,
        axis="z",
        proxies=True,
        namespace="whittle",
        material="dark",
    )
    for geom in holder.findall("geom"):
        x, y, z = map(float, geom.get("pos", "0 0 0").split())
        if axis == "y":
            x, y, z = x, z, -y
            angle = float(geom.attrib.pop("euler", "0 0 0").split()[2]) / 2
            c, s = math.cos(angle) / math.sqrt(2), math.sin(angle) / math.sqrt(2)
            geom.set("quat", f"{c} {-c} {s} {s}")
        geom.set("pos", " ".join(str(a + v) for a, v in zip(center, (x, y, z))))
        geom.set("density", "0")
        parent.append(geom)


def add_fixture(b, base):
    """Real rail mounts, open sleeves and a connected probe-to-reference path."""
    for i, x in enumerate((1.90, 2.38)):
        for j, y in enumerate((0.12, 1.08)):
            _box(b, base, f"leg_{i}_{j}", (x, y, 0.66), (0.035, 0.035, 0.66))
    _box(b, base, "bed", (2.14, 0.60, 1.35), (0.29, 0.56, 0.03))
    for i, x in enumerate((1.97, 2.31)):
        b.rod(
            base,
            (x, 0.12, 1.42),
            (x, 1.08, 1.42),
            0.013,
            "bright",
            collision=True,
            name=PREFIX + f"rail_y_{i}",
        )
        for j, y in enumerate((0.16, 1.04)):
            _box(b, base, f"rail_y_mount_{i}_{j}", (x, y, 1.402), (0.026, 0.020, 0.022))
    slide = b.moving(
        base,
        "probe_y",
        (2.14, 0.60, 1.46),
        (0, 1, 0),
        (-0.23, 0.23),
        mass=1,
        kp=400,
        force=800,
    )
    _box(b, slide, "carriage", (0, 0, 0), (0.20, 0.15, 0.025), "dark")
    for i, x in enumerate((-0.17, 0.17)):
        for j, y in enumerate((-0.10, 0.10)):
            _sleeve(
                b, slide, f"bearing_y_{i}_{j}", (x, y, -0.04), 0.0136, 0.022, 0.05, "y"
            )
    _box(b, slide, "column", (0.17, 0, 0.26), (0.025, 0.08, 0.24))
    for i, y in enumerate((-0.058, 0.058)):
        b.rod(
            slide,
            (0.125, y, 0.08),
            (0.125, y, 0.46),
            0.006,
            "bright",
            collision=True,
            name=PREFIX + f"rail_z_{i}",
        )
        for j, z in enumerate((0.10, 0.44)):
            _box(b, slide, f"rail_z_mount_{i}_{j}", (0.136, y, z), (0.015, 0.012, 0.02))
    probe = b.moving(
        slide,
        "probe_z",
        (0, 0, 0.27),
        (0, 0, 1),
        (-0.05, 0.06),
        mass=0.25,
        kp=200,
        force=300,
    )
    for i, y in enumerate((-0.058, 0.058)):
        _sleeve(b, probe, f"bearing_z_{i}", (0.125, y, 0), 0.0066, 0.011, 0.04, "z")
        _box(
            b,
            probe,
            f"head_bracket_{i}",
            (0.108, math.copysign(0.047, y), 0),
            (0.010, 0.013, 0.035),
        )
    _box(b, probe, "head", (0.054, 0, 0), (0.061, 0.044, 0.035))
    b.rod(
        probe,
        (0, 0, 0),
        (0, 0, -0.10),
        0.004,
        "bright",
        collision=True,
        name=PREFIX + "needle",
    )
    # The probe is a capsule; its actual lower pole is 4 mm beyond the endpoint.
    b.site(probe, "probe_tip", (0, 0, -0.104))
    _box(b, slide, "reference_pad", (0, 0, 0.0275), (0.045, 0.035, 0.0025), "dark")
    _box(b, slide, "reference", (0, 0, 0.07), (0.045, 0.035, 0.04), "cream")
    for i, x in enumerate((-0.050, 0.050)):
        _box(b, slide, f"reference_clamp_{i}", (x, 0, 0.0625), (0.005, 0.026, 0.0375))
    b.site(slide, "dry_reference", (0, 0, 0.07))
    b.site(slide, "reference_surface", (0, 0, 0.11))
    b.metadata["fixture_support"] = dict(
        revision=1,
        source_status="Authored external dry traverse; photograph supports the historical duct only",
        original_axis_frames_and_travel_preserved=True,
        running_clearance_m=0.0006,
        y_bearing_length_m=0.10,
        z_bearing_length_m=0.08,
        reference_dimensions_m=[0.09, 0.07, 0.08],
        minimum_tip_reference_gap_m=0.006,
        ideal_joint_load_support=True,
        scope="Retained reference and noncontact probe positioning; no calibrated measurement or turbine aerodynamics",
    )


def finish_casing(b, base, bands, contacts):
    """Continuous open shells over unchanged enabled segmented contact geometry."""
    for geom in contacts:
        geom.set("group", "3")
        geom.set("rgba", "0 0 0 0")
    for i, (y, length, radius, material) in enumerate(bands):
        frame = ET.SubElement(
            base,
            "body",
            name=PREFIX + f"casing_{i}_frame",
            pos=f"0 {y} 1.90",
            euler=f"{-math.pi / 2} 0 0",
        )
        annular_mount(
            b,
            frame,
            f"casing_{i}",
            (0, 0, 0),
            radius - 0.035,
            radius + 0.035,
            length / 2,
            axis="z",
            material=material,
            namespace="whittle",
        )
        mesh = b.root.find(f'./asset/mesh[@name="{PREFIX}casing_{i}_shell_mesh"]')
        _casing_normals(mesh, radius)
    b.metadata["casing_presentation"] = dict(
        continuous_open_visual_surfaces=True,
        enabled_contact_cells_preserved=len(contacts),
        note="Original static collision geometry retained. Continuous shells remove artificial cell seams; hidden turbine structure remains absent.",
    )


def _casing_normals(mesh, mid_radius):
    """Keep every triangle while separating flat rim and radial wall normals."""
    values = list(map(float, mesh.get("vertex").split()))
    original = [values[i : i + 3] for i in range(0, len(values), 3)]
    indices = list(map(int, mesh.get("face").split()))
    vertices, normals, faces, lookup = [], [], [], {}
    for offset in range(0, len(indices), 3):
        face = indices[offset : offset + 3]
        heights = [original[i][2] for i in face]
        rim = max(heights) - min(heights) < 1e-12
        for index in face:
            x, y, z = original[index]
            radius = math.hypot(x, y)
            if rim:
                normal = (0, 0, math.copysign(1, z))
            else:
                sign = 1 if radius > mid_radius else -1
                normal = (sign * x / radius, sign * y / radius, 0)
            key = (index, rim)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append((x, y, z))
                normals.append(normal)
            faces.append(lookup[key])
    mesh.set("vertex", numbers([v for p in vertices for v in p]))
    mesh.set("normal", numbers([v for p in normals for v in p]))
    mesh.set("face", " ".join(map(str, faces)))
