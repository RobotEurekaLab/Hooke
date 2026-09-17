"""Generated surface detail and distant scenery, separate from mission physics.

NASA photographs guide the palettes; generated textures and landforms are
illustrative, not registered orbital imagery or measured geological assets.
"""

from dataclasses import asdict, dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class Appearance:
    soil_rgb: tuple[int, int, int]
    rock_rgb: tuple[int, int, int]
    sky_top: str
    sky_bottom: str
    sun_direction: str
    sunlight: str
    ambient: str
    reference: str


APPEARANCES = {
    "lunar": Appearance(
        (105, 103, 99),
        (87, 85, 81),
        "0 0 0",
        "0 0 0",
        "-.55 .35 -.65",
        "1 .98 .94",
        ".12 .12 .12",
        "https://science.nasa.gov/moon/image-galleries/views-from-the-lunar-surface/",
    ),
    "martian": Appearance(
        (151, 118, 87),
        (110, 100, 88),
        ".57 .44 .35",
        ".79 .66 .53",
        "-.55 .35 -.65",
        "1 .93 .83",
        ".25 .23 .21",
        "https://www.jpl.nasa.gov/images/pia26644-nasas-perseverance-rover-at-falbreen/",
    ),
}


def noise(rng, size, cells):
    field = zoom(
        rng.normal(size=(cells, cells)),
        size / cells,
        order=3,
        mode="grid-wrap",
        grid_mode=True,
    )
    return (field - field.mean()) / max(float(field.std()), 1e-6)


def write_textures(directory, world, size=2048):
    """A ten-metre material tile with mineral patches and millimetre grains."""
    rng = np.random.default_rng(530 if world == "lunar" else 531)
    appearance = APPEARANCES[world]
    broad = noise(rng, size, 12)
    medium = noise(rng, size, 48)
    fine = noise(rng, size, 256)
    grains = rng.normal(size=(size, size))
    value = 8 * broad + 5 * medium + 3 * fine + 2 * grains
    if world == "martian":
        yy = np.linspace(0, 1, size)[:, None]
        # Colour bands suggest wind-sorted grains; no soil displacement occurs.
        ripple = np.sin(2 * math.pi * (68 * yy + 0.4 * medium))
        value += 3 * ripple
    else:
        value -= 9 * np.maximum(fine - 1.8, 0)
    for name, color, amplitude in (
        ("soil", appearance.soil_rgb, 1.0),
        ("rock", appearance.rock_rgb, 1.6),
    ):
        pixels = np.clip(np.asarray(color) + amplitude * value[..., None], 0, 255)
        Image.fromarray(pixels.astype(np.uint8)).save(directory / f"{name}.png")


def write_mesh(path, points, faces, uv, normals=None):
    with path.open("w") as stream:
        for point in points:
            stream.write("v " + " ".join(f"{v:.6f}" for v in point) + "\n")
        for coordinate in uv:
            stream.write(f"vt {coordinate[0]:.6f} {coordinate[1]:.6f}\n")
        if normals is not None:
            for normal in normals:
                stream.write("vn " + " ".join(f"{v:.6f}" for v in normal) + "\n")
        for face in faces:
            tokens = [
                (
                    f"{i + 1}/{i + 1}/{i + 1}"
                    if normals is not None
                    else f"{i + 1}/{i + 1}"
                )
                for i in face
            ]
            stream.write("f " + " ".join(tokens) + "\n")


def add_surface_visual(world, assets, mission, height, directory):
    """Smooth shading over the collision heightfield's unchanged float samples."""
    half = mission.terrain_half_size_m
    minimum, span = float(height.min()), float(np.ptp(height))
    elevation = ((height - minimum) / span).astype(np.float32).astype(
        float
    ) * span + minimum
    rows, columns = height.shape
    x, y = np.meshgrid(
        np.linspace(-half, half, columns), np.linspace(-half, half, rows)
    )
    dy, dx = np.gradient(elevation, 2 * half / (rows - 1), 2 * half / (columns - 1))
    normals = np.c_[-dx.ravel(), -dy.ravel(), np.ones(height.size)]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    grid = np.arange(height.size).reshape(height.shape)
    a, b, c, d = (
        grid[:-1, :-1].ravel(),
        grid[:-1, 1:].ravel(),
        grid[1:, :-1].ravel(),
        grid[1:, 1:].ravel(),
    )
    faces = np.vstack((np.c_[a, b, d], np.c_[a, d, c]))
    points = np.c_[x.ravel(), y.ravel(), elevation.ravel()]
    # Match MuJoCo's metre-based soil UVs, including the OpenGL vertical axis.
    uv = np.c_[x.ravel() / 10 + 0.5, -y.ravel() / 10 + 0.5]
    path = directory / "surface-visual.obj"
    write_mesh(path, points, faces, uv, normals)
    ET.SubElement(
        assets,
        "mesh",
        name="surface_visual",
        file=str(path.resolve()),
        inertia="shell",
        maxhullvert="64",
    )
    ET.SubElement(
        world,
        "geom",
        name="surface_visual",
        type="mesh",
        mesh="surface_visual",
        material="soil",
        contype="0",
        conaffinity="0",
        mass="0",
    )


def add_horizon(world, assets, mission, height, directory):
    """Join the collision map's exact border to a distant visual mesh."""
    count = 256
    angle = np.linspace(0, math.tau, count, endpoint=False)
    radial = np.geomspace(1, 24, 24)[:, None]
    half = mission.terrain_half_size_m
    edge = half / np.maximum(np.abs(np.cos(angle)), np.abs(np.sin(angle)))
    x, y = radial * edge * np.cos(angle), radial * edge * np.sin(angle)
    axis = np.linspace(-half, half, height.shape[0])
    ground = RegularGridInterpolator((axis, axis), height)
    boundary = ground(np.c_[np.clip(y[0], -half, half), np.clip(x[0], -half, half)])
    distance = np.hypot(x, y)
    relief = (
        0.45 * np.sin(x / 39 + np.sin(y / 67))
        + 0.3 * np.cos(y / 53 + x / 87)
        + 0.15 * np.sin(x / 13) * np.cos(y / 17)
        + 0.1 * np.sin(x / 5 + y / 8)
    )
    z = 4 + (12 if mission.world == "lunar" else 18) * relief
    if mission.world == "lunar":
        for cx, cy, radius in ((-105, 85, 45), (110, 150, 65), (225, -140, 85)):
            r = np.hypot(x - cx, y - cy) / radius
            z += 9 * np.exp(-(((r - 1) / 0.16) ** 2)) - 12 * np.exp(-((r / 0.7) ** 4))
    else:
        # Irregular, flat-topped erosional remnants in the distant landscape.
        for cx, cy, rx, ry, top in (
            (-185, 195, 95, 65, 35),
            (170, 240, 80, 110, 43),
            (300, -150, 100, 75, 32),
            (-290, -200, 100, 75, 37),
        ):
            r = np.hypot((x - cx) / rx, (y - cy) / ry)
            z += top / (1 + np.exp(np.clip(12 * (r - 1), -60, 60)))
    weight = np.clip((radial - 1) / 1.5, 0, 1)
    weight = weight**2 * (3 - 2 * weight)
    z = (1 - weight) * boundary + weight * z
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]
    faces = []
    for row in range(len(radial) - 1):
        for column in range(count):
            a = row * count + column
            b = row * count + (column + 1) % count
            c, d = a + count, b + count
            faces.extend(((a, c, d), (a, d, b)))
    path = directory / "horizon.obj"
    write_mesh(path, points, faces, points[:, :2] / 10)
    ET.SubElement(
        assets,
        "mesh",
        name="distant_landscape",
        file=str(path.resolve()),
        inertia="shell",
        maxhullvert="64",
    )
    ET.SubElement(
        world,
        "geom",
        name="distant_landscape",
        type="mesh",
        mesh="distant_landscape",
        material="soil",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    return {
        "visual_extent_m": float(2 * distance.max()),
        "collision_extent_m": 2 * half,
        "horizon": "generated visual scenery outside the collision map; not navigable",
    }


def add_gravel(world, assets, mission, height, directory):
    rng = np.random.default_rng(745)
    variants = 10
    for index in range(variants):
        points = rng.normal(size=(22, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        points *= rng.uniform(0.65, 1.1, (len(points), 1))
        points[:, 2] *= rng.uniform(0.4, 0.8)
        hull = ConvexHull(points)
        faces = hull.simplices.copy()
        for i, face in enumerate(faces):
            a, b, c = points[face]
            if np.dot(np.cross(b - a, c - a), hull.equations[i, :3]) < 0:
                faces[i] = face[::-1]
        path = directory / f"gravel-{index}.obj"
        write_mesh(path, points, faces, points[:, :2] / 2 + 0.5)
        ET.SubElement(
            assets, "mesh", name=f"gravel_mesh_{index}", file=str(path.resolve())
        )
    axis = np.linspace(
        -mission.terrain_half_size_m, mission.terrain_half_size_m, height.shape[0]
    )
    ground = RegularGridInterpolator((axis, axis), height)
    for index in range(160):
        extent = 18 if index < 100 else 37
        x, y = rng.uniform(-extent, extent, 2)
        if -4 < x < 11 and abs(y) < 2.4:
            y = math.copysign(2.4 + abs(y), y)
        if np.hypot(x + 2.8, y - 3.5) < 2.8:
            continue
        radius = rng.uniform(0.025, 0.13) if index < 100 else rng.uniform(0.15, 0.55)
        z = float(ground([[y, x]])[0]) + radius * 0.2
        # Instance scaling lives on the mesh asset, supported by both renderers.
        mesh = f"gravel_instance_{index}"
        ET.SubElement(
            assets,
            "mesh",
            name=mesh,
            file=str((directory / f"gravel-{index % variants}.obj").resolve()),
            scale=f"{radius} {radius} {radius}",
        )
        yaw = rng.uniform(-math.pi, math.pi)
        ET.SubElement(
            world,
            "geom",
            name=f"gravel_{index}",
            type="mesh",
            mesh=mesh,
            material="regolith_rock",
            pos=f"{x} {y} {z}",
            quat=f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}",
            contype="0",
            conaffinity="0",
            mass="0",
        )


def add_appearance(world, assets, mission, height, directory):
    write_textures(directory, mission.world)
    for name, material in (("soil", "soil"), ("rock", "regolith_rock")):
        ET.SubElement(
            assets,
            "texture",
            name=f"generated_{name}",
            type="2d",
            file=str((directory / f"{name}.png").resolve()),
        )
        ET.SubElement(
            assets,
            "material",
            name=material,
            texture=f"generated_{name}",
            texuniform="true",
            texrepeat=".2 .2" if name == "soil" else "1 1",
            rgba="1 1 1 1",
            specular=".05",
            shininess="0",
        )
    report = add_horizon(world, assets, mission, height, directory)
    add_surface_visual(world, assets, mission, height, directory)
    add_gravel(world, assets, mission, height, directory)
    return {
        **report,
        "palette_reference": APPEARANCES[mission.world].reference,
        "textures": "generated mineral patches and grains; no photographic map registration",
        "scenery_collision": False,
        "appearance": asdict(APPEARANCES[mission.world]),
    }
