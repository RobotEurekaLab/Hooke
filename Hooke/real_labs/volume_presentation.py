"""Explicit, estimated absorption for selected closed optical volumes.

No field means no change. This presentation-only contract never assigns fluid
mass, contact, flow, scattering, or calibrated scientific optical properties.
"""

import math
import struct


def absorption_entries(manifest):
    """Validate all metadata before the Blender implementation changes anything."""
    rows = manifest.get("presentation_volumes", [])
    if not isinstance(rows, list):
        raise ValueError("presentation_volumes must be a list")
    result, names = [], set()
    required = {
        "object",
        "kind",
        "units",
        "absorption_rgb",
        "dimension_basis",
        "scientific_optics_calibrated",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("An absorption entry requires the exact declared fields")
        name = row["object"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Absorption targets must be distinct exact object names")
        if row["kind"] != "estimated_rgb_absorption" or row["units"] != "m^-1":
            raise ValueError(
                "Only estimated RGB absorption in inverse metres is supported"
            )
        if row["scientific_optics_calibrated"] is not False:
            raise ValueError(
                "This presentation contract does not establish calibration"
            )
        if (
            not isinstance(row["dimension_basis"], str)
            or not row["dimension_basis"].strip()
        ):
            raise ValueError("Absorption estimates require an explicit basis")
        sigma = row["absorption_rgb"]
        if not isinstance(sigma, (list, tuple)) or len(sigma) != 3:
            raise ValueError("Absorption requires three coefficients")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x < 0
            for x in sigma
        ):
            raise ValueError("Absorption coefficients must be finite and nonnegative")
        names.add(name)
        absorption_node_values(sigma)
        result.append(dict(row, absorption_rgb=tuple(float(x) for x in sigma)))
    return result


def absorption_node_values(sigma):
    """Return representable sockets for sigma = (1 - Color) * Density.

    Blender sockets use float32. Reject overflow, underflow, or cancellation
    rather than silently erase a positive band. Accepted effective coefficients
    differ by at most 1e-5 relative plus 1e-12 inverse metres absolute.
    """
    density = max(sigma)
    color = tuple(1 - value / density for value in sigma) if density else (1, 1, 1)

    def socket(value):
        try:
            result = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("Absorption exceeds Blender socket precision") from exc
        if not math.isfinite(result):
            raise ValueError("Absorption requires finite Blender socket values")
        return result

    density = socket(density)
    color = tuple(socket(channel) for channel in color)
    for requested, channel in zip(sigma, color):
        effective = (1 - channel) * density
        if (requested > 0 and effective <= 0) or not math.isclose(
            effective, requested, rel_tol=1e-5, abs_tol=1e-12
        ):
            raise ValueError("Absorption ratios exceed Blender socket precision")
    return color, density


def closed_convex_volume(vertices, faces):
    """Check a single convex, outward-wound closed boundary in world metres.

    This intentionally supports convex water containers only. It rejects open,
    disconnected, nonconvex, mirrored and inconsistently wound meshes rather
    than inferring a usable medium from an object's name or bounding box.
    """
    points = [tuple(float(x) for x in p) for p in vertices]
    if len(points) < 4 or any(
        len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points
    ):
        raise ValueError("Optical volume vertices must be finite 3D points")
    centre = tuple(sum(p[i] for p in points) / len(points) for i in range(3))
    span = max(max(p[i] for p in points) - min(p[i] for p in points) for i in range(3))
    tolerance = max(1e-9, span * 1e-7)
    edges, used, volume = {}, set(), 0.0

    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    for face in faces:
        ids = tuple(face)
        if (
            len(ids) < 3
            or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < len(points) for i in ids)
        ):
            raise ValueError("Invalid optical-volume face")
        used.update(ids)
        origin = points[ids[0]]
        normal = [0.0, 0.0, 0.0]
        for i in range(1, len(ids) - 1):
            area = cross(sub(points[ids[i]], origin), sub(points[ids[i + 1]], origin))
            normal = [x + y for x, y in zip(normal, area)]
            volume += dot(sub(origin, centre), area) / 6
        length = math.sqrt(dot(normal, normal))
        if length <= tolerance * tolerance:
            raise ValueError("Degenerate optical-volume face")
        normal = tuple(x / length for x in normal)
        if any(abs(dot(sub(points[i], origin), normal)) > tolerance for i in ids):
            raise ValueError("Optical-volume faces must be planar")
        if any(dot(sub(p, origin), normal) > tolerance for p in points):
            raise ValueError("Optical volume must be convex with outward normals")
        # Every polygon must itself be convex and consistently ordered.
        for k, first in enumerate(ids):
            second, third = ids[(k + 1) % len(ids)], ids[(k + 2) % len(ids)]
            if (
                dot(
                    cross(
                        sub(points[second], points[first]),
                        sub(points[third], points[second]),
                    ),
                    normal,
                )
                < -tolerance * tolerance
            ):
                raise ValueError("Optical-volume face has inconsistent vertex order")
            key = tuple(sorted((first, second)))
            edges.setdefault(key, []).append((first, second))
    if (
        len(used) != len(points)
        or not edges
        or any(len(e) != 2 or e[0] != e[1][::-1] for e in edges.values())
    ):
        raise ValueError("Optical volume must be a closed two-manifold boundary")
    adjacency = {i: set() for i in used}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    reached, pending = set(), [next(iter(used))]
    while pending:
        index = pending.pop()
        if index not in reached:
            reached.add(index)
            pending.extend(adjacency[index] - reached)
    if reached != used or volume <= tolerance**3:
        raise ValueError("Optical volume must be connected and positively oriented")
    return {
        "closed_convex_outward_boundary": True,
        "world_volume_m3": volume,
        "boundary_tolerance_m": tolerance,
    }


def apply_presentation_volumes(manifest):
    """Apply validated object-local absorption; absent metadata is a strict no-op."""
    rows = absorption_entries(manifest)
    if not rows:
        return []
    import bpy

    if (
        manifest.get("units") != "m"
        or bpy.context.scene.unit_settings.scale_length != 1
    ):
        raise ValueError("Absorption requires metre geometry with unit scale one")
    plans = []
    for row in rows:
        obj = bpy.context.scene.objects.get(row["object"])
        if obj is None or obj.type != "MESH":
            raise ValueError("Absorption target must be an existing scene mesh")
        if obj.data.shape_keys is not None:
            raise ValueError(
                "Optical volume requires a static mesh without shape keys"
            )
        if any(modifier.show_render for modifier in obj.modifiers):
            raise ValueError(
                "Optical volume requires an explicit unmodified mesh boundary"
            )
        determinant = obj.matrix_world.to_3x3().determinant()
        if not math.isfinite(determinant) or determinant <= 0:
            raise ValueError(
                "Optical volume requires a finite positive world transform"
            )
        material = obj.active_material
        if material is None or not material.use_nodes or len(obj.material_slots) != 1:
            raise ValueError(
                "Optical volume requires one explicit active node material"
            )
        if material.name.split(".")[0].rsplit("__mat_", 1)[-1] != "clear_water":
            raise ValueError(
                "Absorption is restricted to the explicit clear_water role"
            )
        if "__mat_" not in material.name or any(
            p.material_index != 0 for p in obj.data.polygons
        ):
            raise ValueError(
                "Optical volume requires an unambiguous clear-water surface"
            )
        outputs = [
            n
            for n in material.node_tree.nodes
            if n.type == "OUTPUT_MATERIAL" and n.is_active_output
        ]
        if len(outputs) != 1 or outputs[0].inputs["Volume"].is_linked:
            raise ValueError(
                "Optical volume requires one output with no existing volume shader"
            )
        proof = closed_convex_volume(
            [tuple(obj.matrix_world @ v.co) for v in obj.data.vertices],
            [tuple(p.vertices) for p in obj.data.polygons],
        )
        plans.append((row, obj, material, proof))
    records = []
    for row, obj, original, proof in plans:
        color, density = absorption_node_values(row["absorption_rgb"])
        if density:
            material = original.copy()
            # Primitive meshes can be shared. Never change the DATA material slot.
            obj.material_slots[0].link = "OBJECT"
            obj.material_slots[0].material = material
            node = material.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
            node.name = "Estimated RGB absorption (inverse metres)"
            node.inputs["Color"].default_value = (*color, 1)
            node.inputs["Density"].default_value = density
            output = next(
                n
                for n in material.node_tree.nodes
                if n.type == "OUTPUT_MATERIAL" and n.is_active_output
            )
            material.node_tree.links.new(
                node.outputs["Volume"], output.inputs["Volume"]
            )
        records.append(
            dict(
                row,
                **proof,
                node_color_rgb=color,
                node_density_m_inv=density,
                effective_absorption_rgb_m_inv=tuple(
                    (1 - channel) * density for channel in color
                ),
                socket_precision_tolerance={"relative": 1e-5, "absolute_m_inv": 1e-12},
                applied=bool(density),
                export_scope="Blender shader graph; interchange volume support is format-dependent"
            )
        )
    return records
