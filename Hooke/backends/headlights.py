"""Camera-local inspection fill derived from the source headlight settings.

MuJoCo's OpenGL coefficients have no physical radiometric units. This mapping
improves inspection visibility; it does not reproduce the OpenGL light model.
Only the camera being captured may illuminate the scene.
"""

from contextlib import contextmanager
import math
from numbers import Real
import os
from pathlib import Path
import xml.etree.ElementTree as ET


# mj_defaultVisual defaults, independently checked against the installed MuJoCo.
_DEFAULTS = {"active": True, "ambient": [0.1] * 3,
             "diffuse": [0.4] * 3, "specular": [0.5] * 3}
_CHANNELS = ("ambient", "diffuse", "specular")


def _valid_coefficient(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, Real)
                and math.isfinite(value) and value >= 0)
    except OverflowError:
        return False


def validate_source_headlight(value):
    """Return a detached JSON-compatible source record, rejecting ambiguity."""
    if not isinstance(value, dict) or set(value) != set(_DEFAULTS):
        raise ValueError("Source headlight requires active, ambient, diffuse and specular")
    if type(value["active"]) is not bool:
        raise ValueError("Source headlight active must be a boolean")
    result = {"active": value["active"]}
    for name in _CHANNELS:
        channel = value[name]
        if (not isinstance(channel, (list, tuple)) or len(channel) != 3
                or not all(_valid_coefficient(v) for v in channel)):
            raise ValueError(f"Source headlight {name} must contain three finite nonnegative numbers")
        result[name] = [float(v) for v in channel]
    return result


def source_headlight_from_model(model):
    """Preserve the compiled setting, including explicitly disabled headlights."""
    headlight = model.vis.headlight
    return validate_source_headlight({"active": bool(headlight.active),
                                      **{name: getattr(headlight, name).tolist()
                                         for name in _CHANNELS}})


def resolve_source_headlight(metadata, archive_dir):
    """Read new metadata or a legacy archive's self-contained canonical MJCF.

    Missing/invalid source evidence is an error, not an invented light setting.
    A canonical MJCF without a headlight element uses the actual MuJoCo defaults.
    """
    if "source_headlight" in metadata:
        return validate_source_headlight(metadata["source_headlight"]), "scene.json/source_headlight"
    root = ET.parse(Path(archive_dir) / "scene.xml").getroot()
    if root.tag != "mujoco" or root.find(".//include") is not None:
        raise ValueError("Legacy headlight settings require a self-contained canonical MJCF")
    element = root.find("./visual/headlight")
    value = {key: list(item) if isinstance(item, list) else item
             for key, item in _DEFAULTS.items()}
    if element is not None:
        active = element.get("active", "1")
        if active not in ("0", "1"):
            raise ValueError("MJCF headlight active must be 0 or 1")
        value["active"] = active == "1"
        for name in _CHANNELS:
            if name in element.attrib:
                value[name] = [float(v) for v in element.get(name).split()]
    return validate_source_headlight(value), "scene.xml/visual/headlight"


def headlight_enabled_from_environment():
    """Allow reproduction of captures made without the inspection mapping."""
    mode = os.environ.get("HOOKE_ISAAC_HEADLIGHT", "source")
    if mode not in ("source", "off"):
        raise ValueError("HOOKE_ISAAC_HEADLIGHT must be source or off")
    return mode == "source"


class CameraHeadlights:
    """Author camera-mounted lights and isolate their use in render transactions.

    Lamps start disabled. Callers must drain their renderer's frame latency
    inside ``capture(camera_id)`` before reading that camera's pixels. The
    transaction changes USD light attributes only and never steps physics.
    """

    def __init__(self, stage, camera_paths, source, *, enabled=True, provenance=None):
        if type(enabled) is not bool:
            raise ValueError("Headlight enabled must be a boolean")
        self.source = validate_source_headlight(source)
        self.provenance = provenance
        self.camera_paths = dict(camera_paths)
        self.lights = {}
        self.intensity = 0.0
        self.color = [0.0] * 3
        self.specular = 0.0
        self.enabled = enabled and self.source["active"]
        if not self.enabled:
            return
        # Ambient joins diffuse as a directional inspection fill. It does not
        # become an isotropic OpenGL ambient term. The scale is deliberately
        # recorded rather than presented as a physical unit conversion.
        combined = [a + d for a, d in zip(self.source["ambient"], self.source["diffuse"])]
        peak = max(combined)
        if peak == 0:
            return
        self.color = [v / peak for v in combined]
        self.intensity = 100000.0 * peak / 1.05
        self.specular = max(self.source["specular"]) / peak
        if not all(math.isfinite(v) for v in (*self.color, self.intensity, self.specular)):
            raise ValueError("Source headlight exceeds the finite native inspection mapping")
        from pxr import Gf, UsdGeom, UsdLux

        if len(set(self.camera_paths.values())) != len(self.camera_paths):
            raise ValueError("Each headlight camera must have a distinct USD path")
        for camera, camera_path in self.camera_paths.items():
            parent = stage.GetPrimAtPath(camera_path)
            if not parent or not parent.IsA(UsdGeom.Camera):
                raise ValueError(f"Headlight parent must be a USD camera: {camera_path}")
            path = camera_path + "/HookeInspectionHeadlight"
            if stage.GetPrimAtPath(path):
                raise ValueError(f"Headlight path already exists: {path}")
        for camera, camera_path in self.camera_paths.items():
            path = camera_path + "/HookeInspectionHeadlight"
            light = UsdLux.DiskLight.Define(stage, path)
            light.CreateRadiusAttr(0.12)
            light.CreateNormalizeAttr(False)
            light.CreateIntensityAttr(0.0)
            light.CreateColorAttr(Gf.Vec3f(*self.color))
            light.CreateDiffuseAttr(1.0)
            light.CreateSpecularAttr(self.specular)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.1, 0.1))
            self.lights[camera] = light.GetIntensityAttr()

    @contextmanager
    def capture(self, camera_id):
        """Activate only this camera's lamp, restoring even after capture errors."""
        if camera_id not in self.camera_paths:
            raise ValueError(f"Unknown headlight camera: {camera_id}")
        previous = {camera: attr.Get() for camera, attr in self.lights.items()}
        try:
            for camera, attr in self.lights.items():
                attr.Set(self.intensity if camera == camera_id else 0.0)
            yield
        finally:
            for camera, value in previous.items():
                self.lights[camera].Set(value)

    def report(self):
        return {"profile": "source_headlight_inspection_v1", "enabled": self.enabled,
                "source": validate_source_headlight(self.source), "source_provenance": self.provenance,
                "camera_paths": dict(self.camera_paths), "light_count": len(self.lights),
                "type": "UsdLux.DiskLight", "intensity": self.intensity,
                "color": list(self.color), "radius_m": 0.12, "normalize": False,
                "diffuse_multiplier": 1.0, "specular_multiplier": self.specular,
                "local_translation_m": [0.0, 0.1, 0.1],
                "transaction": "Only the captured camera illuminates; lamp values restore on exit",
                "photometrically_calibrated": False,
                "scope": "Inspection fill derived from source ambient/diffuse coefficients; ambient is directional and specular color is not independently reproduced. No OpenGL or scientific illumination parity claim."}
