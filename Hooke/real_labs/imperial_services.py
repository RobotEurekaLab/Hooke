"""Static exterior service detail from the historical 2018 Bessemer photograph.

The visible density, colors and layered lower-arm coverings are reference-led.
Exact routes, connectors, dimensions and crumples are authored estimates. These
rigid surfaces do not simulate hoses, vacuum, electrical or thermal behavior.
Existing chamber geometry supplies every attachment datum; the separate dry
inspection fixture is neither modified nor used to support these services.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

from real_labs.architecture import numbers


def add_historical_services(b, parent):
    prefix = f"{b.name}__services_"
    old_prefix = f"{b.name}__imperial_"
    asset = b.root.find("asset")
    materials = {}
    for role, color in (
        ("blue", (0.018, 0.15, 0.48, 1)),
        ("warning", (0.72, 0.48, 0.022, 1)),
        ("dark", (0.012, 0.014, 0.018, 1)),
        ("bright", (0.62, 0.64, 0.66, 1)),
        ("guard_red", (0.52, 0.018, 0.012, 1)),
    ):
        name = f"{prefix}detail__mat_{role}"
        ET.SubElement(asset, "material", name=name, rgba=numbers(color))
        materials[role] = name

    def old(key):
        element = parent.find(f"geom[@name='{old_prefix + key}']")
        if element is None:
            raise ValueError(f"Missing historical chamber attachment: {key}")
        return element

    def vector(element, key):
        return np.fromstring(element.get(key), sep=" ")

    def axis_end(key):
        ends = vector(old(key), "fromto").reshape(2, 3)
        direction = ends[1] - ends[0]
        return ends[1], direction / np.linalg.norm(direction)

    def rod(key, start, end, radius, role="bright"):
        geom = b.rod(
            parent,
            tuple(start),
            tuple(end),
            radius,
            role,
            collision=True,
            name=prefix + key,
            density=0,
        )
        geom.set("material", materials[role])
        return geom.get("name")

    def cylinder(key, start, end, radius, role="bright", collision=True):
        geom = b.geom(
            parent,
            "cylinder",
            (radius,),
            material=role,
            fromto=tuple(start) + tuple(end),
            collision=collision,
            name=prefix + key,
            density=0,
        )
        geom.set("material", materials[role])
        return geom.get("name")

    anchors, routes, coverings = [], [], []

    def connector(key, source, position, direction, length=0.045, radius=0.014):
        position, direction = np.asarray(position), np.asarray(direction)
        direction = direction / np.linalg.norm(direction)
        end = position + length * direction
        stem = cylinder(key + "_stem", position - 0.008 * direction, end, radius)
        collar = cylinder(
            key + "_collar",
            end - 0.014 * direction,
            end - 0.004 * direction,
            radius * 1.38,
        )
        anchors.append(
            dict(
                key=key,
                source=source,
                stem=stem,
                collar=collar,
                source_point_m=position.tolist(),
                terminal_m=end.tolist(),
            )
        )
        return end, stem, direction

    def lead(key, first, last, control_a, control_b, role, radius=0.0045):
        start, first_anchor, first_direction = first
        end, last_anchor, last_direction = last
        # Leave and approach each fitting on its exposed side. Round the
        # estimated parked route's corners inside their local convex hulls.
        points = np.array(
            (
                start,
                start + 0.10 * first_direction,
                control_a,
                control_b,
                end + 0.10 * last_direction,
                end,
            )
        )
        names = []
        samples = [points[0]]
        for before, corner, after in zip(points, points[1:], points[2:]):
            incoming, outgoing = corner - before, after - corner
            cut = min(
                0.095, 0.35 * np.linalg.norm(incoming), 0.35 * np.linalg.norm(outgoing)
            )
            a = corner - incoming / np.linalg.norm(incoming) * cut
            c = corner + outgoing / np.linalg.norm(outgoing) * cut
            samples.append(a)
            for t in np.linspace(0, 1, 8)[1:]:
                samples.append((1 - t) ** 2 * a + 2 * (1 - t) * t * corner + t * t * c)
        samples.append(points[-1])
        for index, (previous, current) in enumerate(zip(samples, samples[1:])):
            names.append(rod(f"{key}_span_{index}", previous, current, radius, role))
        routes.append(
            dict(
                key=key,
                first_anchor=first_anchor,
                last_anchor=last_anchor,
                segments=names,
                rounded_route_waypoints_m=points.tolist(),
                radius_m=radius,
            )
        )

    def wrap(key, source, first_fraction, last_fraction):
        branch = old(source)
        ends = vector(branch, "fromto").reshape(2, 3)
        original_radius = float(branch.get("size").split()[0])
        direction = ends[1] - ends[0]
        start = ends[0] + first_fraction * direction
        finish = ends[0] + last_fraction * direction
        axis = direction / np.linalg.norm(direction)
        tangent = np.cross(axis, (0, 0, 1))
        tangent /= np.linalg.norm(tangent)
        other = np.cross(axis, tangent)
        vertices, faces = [], []
        rings, sectors = 11, 32
        # Faceted, closed outer covering with annular ends touching the source
        # body. Its wrinkles are mesh geometry, independent of renderer shaders.
        minimum_radius = original_radius * 1.37 + 0.007
        maximum_radius = minimum_radius + 0.013
        for i in range(rings):
            along = i / (rings - 1)
            centre = start + along * (finish - start)
            for j in range(sectors):
                angle = math.tau * j / sectors
                wrinkle = (
                    0.5
                    + 0.26 * math.sin(7 * angle + i * 2.1)
                    + 0.16 * math.cos(11 * angle - i * 1.7)
                    + 0.08 * math.sin(3 * angle + i * 4.3)
                )
                radius = minimum_radius + 0.013 * wrinkle
                vertices.append(
                    centre
                    + radius * (math.cos(angle) * tangent + math.sin(angle) * other)
                )
        for i in range(rings - 1):
            for j in range(sectors):
                k = (j + 1) % sectors
                a, c = i * sectors + j, i * sectors + k
                d, e = (i + 1) * sectors + k, (i + 1) * sectors + j
                faces.extend(((a, c, d), (a, d, e)))
        # Narrow annular closures reach the original cylindrical exterior.
        for ring, centre in ((0, start), (rings - 1, finish)):
            offset = len(vertices)
            for j in range(sectors):
                angle = math.tau * j / sectors
                vertices.append(
                    centre
                    + original_radius
                    * 0.95
                    * (math.cos(angle) * tangent + math.sin(angle) * other)
                )
            for j in range(sectors):
                k = (j + 1) % sectors
                a, c, d, e = (
                    ring * sectors + j,
                    ring * sectors + k,
                    offset + k,
                    offset + j,
                )
                pair = ((a, c, d), (a, d, e))
                faces.extend(tuple(reversed(f)) if ring == 0 else f for f in pair)
        mesh = prefix + key + "_mesh"
        ET.SubElement(
            asset,
            "mesh",
            name=mesh,
            vertex=numbers(np.array(vertices).ravel()),
            face=" ".join(str(i) for face in faces for i in face),
        )
        visual = b.geom(
            parent,
            "mesh",
            (),
            material="bright",
            mesh=mesh,
            name=prefix + key + "_foil_surface",
            density=0,
        )
        visual.set("material", materials["bright"])
        proxy = cylinder(key + "_outer_contact", start, finish, maximum_radius)
        contact = parent.find(f"geom[@name='{proxy}']")
        contact.set("rgba", "0.5 0.5 0.5 0")
        contact.set("group", "3")
        coverings.append(
            dict(
                source=old_prefix + source,
                visual=visual.get("name"),
                proxy=proxy,
                mesh=mesh,
                vertices=len(vertices),
                triangles=len(faces),
                minimum_radius_m=minimum_radius,
                maximum_radius_m=maximum_radius,
                scope="Estimated wrinkled exterior; conservative cylinder contact envelope, no flexible or thermal foil behavior.",
            )
        )

    for key in ("left", "right"):
        vessel = old(key + "_vessel")
        centre = vector(vessel, "pos")
        vessel_size = vector(vessel, "size")
        x, y = centre[:2]
        top = centre[2] + vessel_size[1]
        cabinet = old(key + "_cabinet")
        cabinet_center = vector(cabinet, "pos")
        cabinet_size = vector(cabinet, "size")
        front_y = cabinet_center[1] - cabinet_size[1]
        # Several unequal small fittings around the lid are visible in the
        # photograph. Their identities and exact plumbing remain unspecified.
        upper = []
        for i, (dx, dy, height) in enumerate(
            (
                (-0.16, -0.10, 0.17),
                (0.10, -0.15, 0.115),
                (0.16, 0.08, 0.085),
                (-0.12, -0.19, 0.105),
                (0.04, -0.19, 0.075),
            )
        ):
            source = old_prefix + key + "_lid"
            upper.append(
                connector(
                    f"{key}_upper_{i}",
                    source,
                    (x + dx, y + dy, top + 0.009),
                    (0, 0, 1),
                    length=height,
                    radius=0.010 + 0.002 * i,
                )
            )
        cabinet_sockets = []
        for i, (dx, z) in enumerate(
            ((-0.27, 0.43), (-0.10, 0.68), (0.12, 0.52), (0.29, 0.77))
        ):
            cabinet_sockets.append(
                connector(
                    f"{key}_panel_{i}",
                    old_prefix + key + "_cabinet",
                    (x + dx, front_y, z),
                    (0, -1, 0),
                    length=0.032,
                )
            )
        lower = {}
        for branch in range(2):
            source = f"{key}_branch_{branch}_end"
            endpoint, direction = axis_end(source)
            tangent = np.cross(direction, (0, 0, 1))
            tangent /= np.linalg.norm(tangent)
            other = np.cross(direction, tangent)
            count = (
                5
                if key == "right" and branch == 0
                else 3 if key == "left" and branch == 1 else 2
            )
            for j in range(count):
                angle = math.tau * j / count
                radius = 0.035 if count == 5 else 0.025 if count == 3 else 0.017
                offset = radius * (math.cos(angle) * tangent + math.sin(angle) * other)
                lower[branch, j] = connector(
                    f"{key}_lower_{branch}_{j}",
                    old_prefix + source,
                    endpoint + offset,
                    direction,
                    length=0.035,
                    radius=0.009,
                )
            # Layered collar and colored retaining disk at the source end.
            cylinder(
                f"{key}_end_retainer_{branch}",
                endpoint - 0.006 * direction,
                endpoint + 0.001 * direction,
                0.035,
                "guard_red",
            )
        # Asymmetric parked loops stay on their own apparatus side. The right
        # machine's low loops stay outside the separate dry fixture's envelope.
        side = -1 if key == "left" else 1
        outer_x = x + side * 0.70
        low_y = y - 0.96 if key == "left" else y - 0.65
        lead(
            key + "_blue_low_0",
            lower[0, 0],
            cabinet_sockets[0],
            (outer_x, low_y + (0.20 if key == "right" else 0), 0.20),
            (x + side * 0.60, front_y - 0.28, 0.22),
            "blue",
            0.005,
        )
        lead(
            key + "_blue_low_1",
            lower[0, 1] if key == "right" else lower[1, 0],
            cabinet_sockets[2],
            (x + (0.90 if key == "right" else 0.45), low_y - 0.12, 0.18),
            (x + (0.85 if key == "right" else 0.35), front_y - 0.18, 0.31),
            "blue",
            0.0045,
        )
        lead(
            key + "_yellow_head",
            upper[0],
            lower[0, 2] if key == "right" else lower[0, 1],
            (x + side * 0.46, y - 0.34, top + 0.36),
            (
                (x + 1.58, y - 0.63, 0.42)
                if key == "right"
                else (outer_x, low_y + 0.13, 0.83)
            ),
            "warning",
            0.003,
        )
        lead(
            key + "_dark_head",
            upper[1],
            cabinet_sockets[1],
            (x + side * 0.43, y - 0.49, top + 0.18),
            (x + side * 0.45, front_y - 0.25, 0.77),
            "dark",
            0.0035,
        )
        lead(
            key + "_blue_side",
            upper[2],
            lower[0, 3] if key == "right" else lower[1, 1],
            (x + (0.93 if key == "right" else 0.87), y + 0.27, top + 0.11),
            (x + (1.55 if key == "right" else 0.83), y - 0.62, 0.44),
            "blue",
            0.004,
        )
        lead(
            key + "_yellow_panel",
            lower[0, 4] if key == "right" else lower[1, 2],
            cabinet_sockets[3],
            (x + (1.55 if key == "right" else 0.30), y - 0.65, 0.33),
            (x + 0.50, front_y - 0.19, 0.73),
            "warning",
            0.0028,
        )
        # Two short metallic capillaries add the photographed layered upper
        # fittings without pretending their obscured scientific roles are known.
        for i in range(2):
            port = key + (
                ("_port_0_cap" if key == "left" else "_port_1_cap")
                if i == 0
                else "_top_0_terminal"
            )
            if i == 0:
                endpoint, direction = axis_end(port)
            else:
                terminal = old(port)
                endpoint = vector(terminal, "pos") + (vector(terminal, "size")[0], 0, 0)
                direction = np.array((1, 0, 0))
            target = connector(
                f"{key}_small_port_{i}",
                old_prefix + port,
                endpoint,
                direction,
                length=0.025,
                radius=0.008,
            )
            a = (
                (x - 0.20, y - 0.70, top + 0.22)
                if key == "left"
                else (x + 1.08, y + 0.40, top + 0.32)
            )
            c = (
                (x - 0.20, y - 0.72, top - 0.04)
                if key == "left"
                else (x + 1.10, y + 0.04, top - 0.14)
            )
            lead(
                f"{key}_metal_tube_{i}",
                upper[3 + i],
                target,
                (a if i == 0 else (x + 0.27, y - 0.30, top + 0.40)),
                (c if i == 0 else (x + 0.27, y + 0.02, top + 0.55)),
                "bright",
                0.0026,
            )

    # Strong foil texture is visible on the near/left lower source assembly.
    # The far/right source retains its bare layered flange hardware.
    wrap("left_lower_wrap", "left_branch_0_body", 0.65, 0.99)
    wrap("left_secondary_wrap", "left_branch_1_body", 0.74, 0.96)
    b.metadata["historical_service_detail"] = dict(
        reference="Historical 2018 Bessemer photograph; visible exterior only",
        anchors=anchors,
        routes=routes,
        coverings=coverings,
        scope="Fixed connected surfaces; route/end identity, dimensions and covering details estimated. Added static surfaces contribute no inertia, preserving the original body mass model. No functional hose, vacuum, electrical or thermal model.",
    )
