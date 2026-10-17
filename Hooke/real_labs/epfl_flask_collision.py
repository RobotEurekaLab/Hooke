"""Convex wall cells for the EPFL apparatus's authored hollow flask.

Cells are derived from the emitted visual lathe mesh, so its triangles remain
covered without a solid proxy across the cavity. Local convexification thickens
the effective wall slightly. This is rigid, empty glass geometry, not calibrated
capacity, glass deformation, fracture, liquid, or chemical compatibility.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull


def parse_shell(mesh, center):
    """Read the actual two-layer pole/ring mesh and validate its angular grid."""
    vertices = np.fromstring(mesh.get("vertex", ""), sep=" ").reshape(-1, 3)
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("The flask center must contain three finite coordinates")
    if len(vertices) < 36 or len(vertices) % 2 or not np.isfinite(vertices).all():
        raise ValueError("Expected finite vertices in equal outer and inner layers")
    half = len(vertices) // 2
    outer, inner = vertices[:half], vertices[half:]
    # Each layer starts with its bowl pole, followed by constant-Z rings.
    segments = next(
        (i - 1 for i in range(2, half)
         if abs(outer[i, 2] - outer[1, 2]) > 1e-10),
        None,
    )
    if not segments or segments < 8 or (half - 1) % segments:
        raise ValueError("Source mesh must have the expected pole/ring topology")
    layers = []
    angles = np.arange(segments) * math.tau / segments
    for layer in (outer, inner):
        rings = [layer[:1]] + list(layer[1:].reshape(-1, segments, 3))
        profile = np.array([
            (np.linalg.norm(ring[0, :2] - center[:2]), ring[0, 2] - center[2])
            for ring in rings
        ])
        if profile[0, 0] > 2e-10 or np.any(np.diff(profile[:, 1]) <= 0):
            raise ValueError("Source profile must start on its axis and increase in Z")
        for ring, (radius, z) in zip(rings[1:], profile[1:]):
            expected = np.column_stack((
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
                np.full(segments, center[2] + z),
            ))
            if not np.allclose(ring, expected, rtol=0, atol=2e-10):
                raise ValueError("Source rings must share an angular grid and center")
        layers.append((rings, profile))
    return dict(
        vertices=vertices, segments=segments, center=center,
        outer_rings=layers[0][0], inner_rings=layers[1][0],
        outer_profile=layers[0][1], inner_profile=layers[1][1],
    )


def chord_deviation(profile, first, last):
    """Maximum source-station distance from a profile band's endpoint chord."""
    a, b = profile[first], profile[last]
    delta = b - a
    u = np.clip((profile[first:last + 1] - a) @ delta / (delta @ delta), 0, 1)
    return float(np.max(np.linalg.norm(
        profile[first:last + 1] - (a + u[:, None] * delta), axis=1,
    )))


def profile_bands(shell, tolerance):
    """Use the validated greedy grouping on both actual surface profiles."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Profile tolerance must be finite and nonnegative")
    profiles = (shell["outer_profile"], shell["inner_profile"])
    bands = []
    first = 0
    while first < len(profiles[0]) - 1:
        last = first + 1
        while (last + 1 < len(profiles[0])
               and max(chord_deviation(p, first, last + 1) for p in profiles)
               <= tolerance):
            last += 1
        bands.append((first, last))
        first = last
    return bands


def _named(parent, tag, name):
    return next((item for item in parent.findall(tag)
                 if item.get("name") == name), None)


def add_hollow_flask_colliders(root, body, visual_mesh_name, *, center,
                              prefix="hollow_flask", angular_sectors=32,
                              profile_tolerance=0.00015):
    """Append enabled, invisible wall cells; return metadata, shell, and hulls.

    Coordinates are local to ``body``. This low-level helper leaves existing
    geoms intact; ``install_flask_collision`` checks the visual geom and removes
    the legacy solid proxies. Defaults preserve the qualified 736-cell geometry.
    """
    asset = root.find("asset")
    mesh = _named(asset, "mesh", visual_mesh_name) if asset is not None else None
    if mesh is None:
        raise ValueError("Visible flask shell mesh not found")
    for attribute, identity in (("scale", (1, 1, 1)), ("refpos", (0, 0, 0)),
                                ("refquat", (1, 0, 0, 0))):
        if attribute in mesh.attrib:
            values = np.fromstring(mesh.get(attribute), sep=" ")
            if values.shape != (len(identity),) or not np.array_equal(values, identity):
                raise ValueError("Source mesh transforms must be baked before collision generation")
    shell = parse_shell(mesh, center)
    segments = shell["segments"]
    if (not isinstance(angular_sectors, int) or angular_sectors < 3
            or segments % angular_sectors):
        raise ValueError("Angular sectors must divide the visual ring tessellation")
    stride = segments // angular_sectors
    bands = profile_bands(shell, profile_tolerance)
    names = {item.get("name") for item in root.iter("mesh")}
    names.update(item.get("name") for item in root.iter("geom"))
    if any(f"{prefix}_wall_{band:02}_{sector:02}" in names
           for band in range(len(bands)) for sector in range(angular_sectors)):
        raise ValueError("Flask collision cells are already installed or names collide")

    cells, meshes, geoms = [], [], []
    for band, (first, last) in enumerate(bands):
        for sector in range(angular_sectors):
            indices = np.arange(sector * stride, (sector + 1) * stride + 1) % segments
            vertices = np.concatenate([
                ring if len(ring) == 1 else ring[indices]
                for key in ("outer_rings", "inner_rings")
                for ring in shell[key][first:last + 1]
            ])
            vertices = np.unique(vertices, axis=0)
            hull = ConvexHull(vertices)
            faces = []
            for simplex, equation in zip(hull.simplices, hull.equations):
                a, b, c = vertices[simplex]
                faces.append(simplex if np.dot(np.cross(b - a, c - a), equation[:3]) > 0
                             else simplex[::-1])
            name = f"{prefix}_wall_{band:02}_{sector:02}"
            meshes.append(ET.Element(
                "mesh", name=name,
                vertex=" ".join(f"{v:.15g}" for v in vertices.ravel()),
                face=" ".join(str(int(v)) for face in faces for v in face),
            ))
            geoms.append(ET.Element(
                "geom", name=name, type="mesh", mesh=name, pos="0 0 0", quat="1 0 0 0",
                contype="1", conaffinity="1", group="3", rgba="0 0 0 0", density="0", mass="0",
                condim="3", margin="0", gap="0", priority="0", solmix="1",
                friction=".4 .002 .0001", solref=".002 1", solimp=".9999 .9999 .001",
            ))
            cells.append(dict(name=name, vertices=vertices, planes=hull.equations,
                              band=band, sector=sector))
    # Build every hull before changing the scene, including before removing proxies.
    asset.extend(meshes)
    body.extend(geoms)
    metadata = dict(
        source_mesh_name=visual_mesh_name,
        source_mesh_sha256=hashlib.sha256(ET.tostring(mesh)).hexdigest(),
        source_vertices=len(shell["vertices"]), source_angular_segments=segments,
        source_profile_stations=len(shell["outer_profile"]),
        angular_sectors=angular_sectors, profile_bands=bands, cell_count=len(cells),
        maximum_meridian_chord_deviation_m=max(
            chord_deviation(p, a, b)
            for p in (shell["outer_profile"], shell["inner_profile"]) for a, b in bands
        ),
        maximum_angular_chord_deficit_m=float(np.max(shell["outer_profile"][:, 0]))
        * (1 - math.cos(math.pi / angular_sectors)),
        maximum_vertex_outer_radius_m=float(np.max(np.linalg.norm(
            shell["vertices"][:len(shell["vertices"]) // 2, :2] - shell["center"][:2],
            axis=1,
        ))),
        scope=("Each convex cell contains the actual source mesh triangles in its "
               "sector/profile band. Conservative local convexification; "
               "no full solid proxy across the cavity."),
    )
    return metadata, shell, cells


def install_flask_collision(b, base, key="main", *, center=None,
                            angular_sectors=32, profile_tolerance=0.00015):
    """Replace this flask's three solid proxies after its visual shell is made.

    ``center`` is the bulb center in ``base`` coordinates. When omitted, infer
    the axis from the shell pole and its height from the exact equator. The
    visual shell must have unit scale, identity placement, and disabled contact.
    The returned metadata is serializable; no geometry arrays are retained.

    The static support body retains the flask. Isolated force-probe qualification
    used a 200 microsecond timestep and does not establish unrestricted access
    past other apparatus, or equivalent contact behavior at larger timesteps.
    """
    namespace = f"{b.name}__assembly_{key}"
    visual_name = namespace + "_flask_shell"
    mesh_name = namespace + "_flask_shell_mesh"
    visual = _named(base, "geom", visual_name)
    if visual is None or visual.get("mesh") != mesh_name:
        raise ValueError("Expected this flask's visible shell on the supplied base")
    for attribute, identity in (("pos", (0, 0, 0)), ("quat", (1, 0, 0, 0)),
                                ("euler", (0, 0, 0))):
        if attribute in visual.attrib:
            values = np.fromstring(visual.get(attribute), sep=" ")
            if values.shape != (len(identity),) or not np.array_equal(values, identity):
                raise ValueError("The visual flask geom must have identity placement")
    if any(attribute in visual.attrib for attribute in ("axisangle", "xyaxes", "zaxis", "fromto")):
        raise ValueError("The visual flask geom must have identity placement")
    if visual.get("contype") != "0" or visual.get("conaffinity") != "0":
        raise ValueError("The visible flask mesh must not supply solid convex contact")
    if center is None:
        asset = b.root.find("asset")
        mesh = _named(asset, "mesh", mesh_name) if asset is not None else None
        if mesh is None:
            raise ValueError("Visible flask shell mesh not found")
        vertices = np.fromstring(mesh.get("vertex", ""), sep=" ").reshape(-1, 3)
        if len(vertices) < 36 or len(vertices) % 2:
            raise ValueError("Expected equal outer and inner visual layers")
        outer = vertices[:len(vertices) // 2]
        radii = np.linalg.norm(outer[:, :2] - outer[0, :2], axis=1)
        center = (*outer[0, :2], outer[int(np.argmax(radii)), 2])
    metadata, _, _ = add_hollow_flask_colliders(
        b.root, base, mesh_name, center=center, prefix=namespace + "_flask_collision",
        angular_sectors=angular_sectors, profile_tolerance=profile_tolerance,
    )
    removed = []
    for suffix in ("_bulb", "_neck", "_shoulder_proxy"):
        proxy = _named(base, "geom", namespace + suffix)
        if proxy is not None:
            base.remove(proxy)
            removed.append(proxy.get("name"))
    metadata.update(removed_solid_proxies=removed, visual_geom_name=visual_name,
                    collision_prefix=namespace + "_flask_collision",
                    body_local_bulb_center_m=[float(value) for value in center])
    return metadata
