"""Estimated presentation properties, separate from simulated physical materials.

Only explicit authored material roles receive a surface profile. These values
are visual defaults, not measurements or manufacturer material specifications.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SurfaceProfile:
    name: str
    metallic: float
    roughness: float
    specular: float = 0.35
    coat: float = 0.0
    texture_scale: float = 0.0
    bump_distance: float = 0.0
    ior: float | None = None
    transmission: float | None = None
    color: tuple[float, float, float, float] | None = None


PROFILES = {
    # Explicit photographic presentation roles, not measured optical spectra.
    "clear_acrylic": SurfaceProfile(
        "clear_acrylic", 0.0, 0.025, specular=0.5, ior=1.49,
        transmission=1.0, color=(1.0, 1.0, 1.0, 1.0),
    ),
    "annulus_water": SurfaceProfile(
        "annulus_water", 0.0, 0.045, specular=0.5, ior=1.333,
        transmission=0.25, color=(0.008, 0.065, 0.018, 1.0),
    ),
    "turbid_water": SurfaceProfile(
        "turbid_water", 0.0, 0.055, specular=0.5, ior=1.333,
        transmission=0.08, color=(0.025, 0.038, 0.012, 1.0),
    ),
    # Explicit opt-in: colored liquids and legacy glass keep their own finishes.
    # This is a still, neutral interface, not measured aquarium-water optics.
    "clear_water": SurfaceProfile(
        "clear_water",
        0.0,
        0.018,
        specular=0.5,
        ior=1.333,
        transmission=1.0,
        color=(1.0, 1.0, 1.0, 1.0),
    ),
    "satin_metal": SurfaceProfile("satin_metal", 1.0, 0.30, texture_scale=900),
    "machined_metal": SurfaceProfile("machined_metal", 1.0, 0.23, texture_scale=1100),
    "copper": SurfaceProfile("copper", 1.0, 0.28, texture_scale=850),
    "paint": SurfaceProfile(
        "paint", 0.0, 0.43, coat=0.12, texture_scale=800, bump_distance=0.000018
    ),
    "rubber": SurfaceProfile(
        "rubber", 0.0, 0.72, specular=0.20, texture_scale=1100, bump_distance=0.000012
    ),
    "dark_housing": SurfaceProfile(
        "dark_housing", 0.0, 0.48, texture_scale=900, bump_distance=0.000012
    ),
    "display": SurfaceProfile("display", 0.0, 0.21, coat=0.30),
    "optical_face": SurfaceProfile("optical_face", 0.0, 0.14, coat=0.35),
}

_ROLES = {
    "clear_acrylic": "clear_acrylic",
    "annulus_water": "annulus_water",
    "turbid_water": "turbid_water",
    "clear_water": "clear_water",
    "metal": "satin_metal",
    "bright": "machined_metal",
    "copper": "copper",
    "shell": "paint",
    "cream": "paint",
    "blue": "paint",
    "cyan": "paint",
    "orange": "paint",
    "warning": "paint",
    "guard_red": "paint",
    "rubber": "rubber",
    "dark": "dark_housing",
    "screen": "display",
    "lens": "optical_face",
}


def surface_profile(material_name, object_name, color):
    """Resolve authored roles without inferring bare metal from paint color."""
    name = material_name.split(".")[0]
    if "__mat_" in name:
        role = name.rsplit("__mat_", 1)[1]
        profile = _ROLES.get(role)
        # Unknown explicit roles (including glass, sample and plant) stay intact.
        return PROFILES.get(profile)
    if name in {"metal", "bronze", "rubber"}:
        return PROFILES[
            {"metal": "satin_metal", "bronze": "copper", "rubber": "rubber"}[name]
        ]
    # Some architecture parts have explicit metal names but no material role.
    # Colored rails may be painted; their authored color must not become metal.
    channels = tuple(float(value) for value in color[:3])
    neutral = max(channels) - min(channels) < 0.12
    if neutral and any(token in object_name for token in ("_steel_", "_aluminium_")):
        return PROFILES["satin_metal"]
    if neutral and "_rubber_" in object_name and max(channels) < 0.25:
        return PROFILES["rubber"]
    return None


def inspection_softbox_visibility(manifest):
    """Resolve an explicit presentation-only choice for synthetic fill lights.

    Hiding those emitters in glossy/transmission rays retains their illumination
    and the actual ceiling fixtures. No flag changes geometry or real lamps.
    """
    settings = [item["inspection_softbox_visible_in_glass"]
                for item in manifest.get("equipment", [])
                if "inspection_softbox_visible_in_glass" in item]
    if any(type(value) is not bool for value in settings):
        raise ValueError("Inspection softbox visibility must be boolean")
    if len(set(settings)) > 1:
        raise ValueError("Conflicting scene inspection softbox visibility")
    return settings[0] if settings else True


def primitive_face_smoothing(geometry_type, face_vertices):
    """Smooth analytic curved primitives while retaining flat cylinder end caps.

    Mesh geometry keeps its exported normals; this is not a mesh repair pass.
    """
    if geometry_type in (2, 3, 4):
        return True
    if geometry_type == 5:
        return face_vertices == 4
    return None


def fixture_emitter_bounds(vertices, clearance=0.015):
    """Place a downward presentation light below the actual fixture envelope.

    Input corners are in world metres. Using the lower surface rather than a
    fixed offset from the centre prevents thick or tilted fixtures shadowing
    their own area light. Its footprint remains an estimated lighting proxy.
    """
    corners = [tuple(float(value) for value in point) for point in vertices]
    if not corners or any(len(point) != 3 for point in corners):
        raise ValueError("Fixture bounds require three-dimensional corners")
    if not all(math.isfinite(value) for point in corners for value in point):
        raise ValueError("Fixture bounds must be finite")
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("Emitter clearance must be positive and finite")
    lower = tuple(min(point[axis] for point in corners) for axis in range(3))
    upper = tuple(max(point[axis] for point in corners) for axis in range(3))
    position = (
        (lower[0] + upper[0]) / 2,
        (lower[1] + upper[1]) / 2,
        lower[2] - clearance,
    )
    footprint = tuple(
        max(0.05, (upper[axis] - lower[axis]) * 0.92) for axis in range(2)
    )
    return position, footprint
