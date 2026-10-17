"""Source headlight preservation and isolated native inspection transactions."""

from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.headlights import (CameraHeadlights, headlight_enabled_from_environment,
                                resolve_source_headlight, source_headlight_from_model,
                                validate_source_headlight)

try:
    from pxr import Gf, Usd, UsdGeom, UsdLux
except ImportError:
    Usd = None


SOURCE = {"active": True, "ambient": [0.4] * 3, "diffuse": [0.65] * 3,
          "specular": [0.2] * 3}


class HeadlightMetadata(unittest.TestCase):
    def resolve(self, xml):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "scene.xml").write_text(xml)
            return resolve_source_headlight({}, folder)[0]

    @unittest.skipIf(mujoco is None, "Run with installed MuJoCo bindings")
    def test_legacy_defaults_match_compiled_mujoco(self):
        for xml in ("<mujoco/>", '<mujoco><visual><headlight diffuse=".8 .7 .6"/></visual></mujoco>'):
            source = source_headlight_from_model(mujoco.MjModel.from_xml_string(xml))
            resolved = self.resolve(xml)
            self.assertEqual(resolved["active"], source["active"])
            for name in ("ambient", "diffuse", "specular"):
                np.testing.assert_allclose(resolved[name], source[name], rtol=1e-7)

    @unittest.skipIf(mujoco is None, "Run with installed MuJoCo bindings")
    def test_export_keeps_compiled_channels_and_disabled_state(self):
        from backends.baseline import write_snapshot
        xml = '<mujoco><visual><headlight active="0" ambient=".1 .2 .3" diffuse=".8 .7 .6" specular=".4 .5 .6"/></visual></mujoco>'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source = path / "original.xml"
            source.write_text(xml)
            spec = mujoco.MjSpec.from_file(str(source))
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            metadata = write_snapshot(SimpleNamespace(spec=spec, model=model, data=data,
                default_scene=source, task="headlight_fixture", task_info={}), path)
            expected = source_headlight_from_model(model)
            self.assertFalse(expected["active"])
            self.assertEqual(metadata["source_headlight"], expected)
            self.assertEqual(json.loads((path / "scene.json").read_text())["source_headlight"], expected)
            self.assertEqual(resolve_source_headlight(metadata, path), (expected, "scene.json/source_headlight"))

    def test_metadata_precedence_detachment_and_missing_evidence(self):
        metadata = {"source_headlight": deepcopy(SOURCE)}
        result, origin = resolve_source_headlight(metadata, "/does/not/exist")
        result["ambient"][0] = 42
        self.assertEqual(metadata["source_headlight"]["ambient"][0], 0.4)
        self.assertEqual(origin, "scene.json/source_headlight")
        with self.assertRaises(FileNotFoundError):
            resolve_source_headlight({}, "/does/not/exist")
        with self.assertRaises(ValueError):
            self.resolve('<mujoco><include file="missing.xml"/></mujoco>')

    def test_rejects_malformed_records_and_xml(self):
        malformed = [None, [], {}, {**SOURCE, "active": 1}, {**SOURCE, "extra": 1}]
        for value in ([], [1, 2], "123", [True, 0, 0], [-1, 0, 0],
                      [float("nan"), 0, 0], [float("inf"), 0, 0], [10**1000, 0, 0]):
            malformed.append({**SOURCE, "ambient": value})
        for value in malformed:
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                validate_source_headlight(value)
        for attributes in ('active="false"', 'ambient=".1 .2"', 'diffuse="nan 0 0"'):
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                self.resolve(f'<mujoco><visual><headlight {attributes}/></visual></mujoco>')

    def test_environment_mode_is_explicit(self):
        for mode, expected in (("source", True), ("off", False)):
            with patch.dict(os.environ, {"HOOKE_ISAAC_HEADLIGHT": mode}):
                self.assertEqual(headlight_enabled_from_environment(), expected)
        with patch.dict(os.environ, {"HOOKE_ISAAC_HEADLIGHT": "false"}):
            with self.assertRaises(ValueError):
                headlight_enabled_from_environment()

    def test_disabled_or_black_headlight_does_not_touch_stage(self):
        for source, enabled in ((SOURCE, False), ({**SOURCE, "active": False}, True),
                                ({**SOURCE, "ambient": [0]*3, "diffuse": [0]*3}, True)):
            rig = CameraHeadlights(None, {0: "/Camera"}, source, enabled=enabled)
            with rig.capture(0):
                self.assertEqual(rig.lights, {})
            self.assertEqual(rig.report()["light_count"], 0)
        with self.assertRaises(ValueError):
            CameraHeadlights(None, {}, SOURCE, enabled=1)


class HeadlightTransactions(unittest.TestCase):
    def rig(self):
        class Attribute:
            def __init__(self, value): self.value = value
            def Get(self): return self.value
            def Set(self, value): self.value = value
        rig = object.__new__(CameraHeadlights)
        rig.camera_paths = {0: "/CameraA", 1: "/CameraB"}
        rig.lights = {0: Attribute(17.0), 1: Attribute(29.0)}
        rig.intensity = 100000.0
        return rig

    def test_cameras_never_cross_illuminate_and_restore_on_failure(self):
        rig = self.rig()
        with self.assertRaisesRegex(RuntimeError, "renderer failed"):
            with rig.capture(0):
                self.assertEqual([a.Get() for a in rig.lights.values()], [100000., 0.])
                with rig.capture(1):
                    self.assertEqual([a.Get() for a in rig.lights.values()], [0., 100000.])
                self.assertEqual([a.Get() for a in rig.lights.values()], [100000., 0.])
                raise RuntimeError("renderer failed")
        self.assertEqual([a.Get() for a in rig.lights.values()], [17., 29.])

    def test_unknown_camera_leaves_every_light_unchanged(self):
        rig = self.rig()
        with self.assertRaises(ValueError):
            with rig.capture(99):
                self.fail("Unknown camera entered capture")
        self.assertEqual([a.Get() for a in rig.lights.values()], [17., 29.])

    def actual_capture(self):
        """Exercise production orchestration without importing the GPU runtime."""
        path = Path(__file__).resolve().parents[1] / "Hooke/backends/isaac_runtime.py"
        tree = ast.parse(path.read_text())
        native = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "NativeScene")
        capture = next(node for node in native.body if isinstance(node, ast.FunctionDef)
                       and node.name == "capture")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[capture], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["capture"]

    def test_runtime_reads_each_camera_under_only_its_own_light(self):
        from PIL import Image
        rig = self.rig()
        rendered = []
        observed = []
        def pixels(index):
            expected = [100000., 0.] if index == 0 else [0., 100000.]
            self.assertEqual([a.Get() for a in rig.lights.values()], expected)
            observed.append(index)
            return np.full((32, 32, 4), 50 + index * 100, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            scene = SimpleNamespace(headlights=rig, output=Path(folder), frame_count=0,
                render_settings=SimpleNamespace(width=32, height=32),
                world=SimpleNamespace(render=lambda: rendered.append([a.Get() for a in rig.lights.values()])),
                cameras={index: SimpleNamespace(get_rgba=lambda i=index: pixels(i)) for index in (0, 1)})
            self.actual_capture()(scene)
            self.assertEqual(observed, [0, 1])
            self.assertTrue(rendered)
            self.assertTrue(all(value in ([100000., 0.], [0., 100000.]) for value in rendered))
            self.assertEqual(scene.frame_count, 1)
            for index in (0, 1):
                image = np.asarray(Image.open(Path(folder) / f"camera_{index}/00000.png"))
                self.assertTrue(np.all(image == 50 + index * 100))
        self.assertEqual([a.Get() for a in rig.lights.values()], [17., 29.])

    def test_runtime_camera_error_restores_lights(self):
        rig = self.rig()
        def fail():
            raise RuntimeError("camera unavailable")
        scene = SimpleNamespace(headlights=rig, frame_count=0,
            world=SimpleNamespace(render=lambda: None), cameras={0: SimpleNamespace(get_rgba=fail)})
        with self.assertRaisesRegex(RuntimeError, "camera unavailable"):
            self.actual_capture()(scene)
        self.assertEqual([a.Get() for a in rig.lights.values()], [17., 29.])
        self.assertEqual(scene.frame_count, 0)


@unittest.skipIf(Usd is None, "Run with installed Isaac USD bindings")
class HeadlightUSD(unittest.TestCase):
    def cameras(self):
        stage = Usd.Stage.CreateInMemory()
        for name in ("A", "B"):
            UsdGeom.Camera.Define(stage, f"/Camera{name}")
        return stage

    def test_camera_mount_moves_and_rotates_without_remounting(self):
        stage = self.cameras()
        rig = CameraHeadlights(stage, {0: "/CameraA", 1: "/CameraB"}, SOURCE)
        lamp = stage.GetPrimAtPath("/CameraA/HookeInspectionHeadlight")
        camera = UsdGeom.Xformable(stage.GetPrimAtPath("/CameraA"))
        camera.AddTranslateOp().Set(Gf.Vec3d(1., 2., 3.))
        camera.AddRotateZOp().Set(90.)
        actual = UsdGeom.XformCache().GetLocalToWorldTransform(lamp).ExtractTranslation()
        np.testing.assert_allclose(actual, [.9, 2., 3.1], atol=1e-12)
        self.assertEqual([a.Get() for a in rig.lights.values()], [0., 0.])
        with rig.capture(1):
            self.assertEqual([a.Get() for a in rig.lights.values()], [0., 100000.])
        self.assertEqual([a.Get() for a in rig.lights.values()], [0., 0.])
        light = UsdLux.DiskLight(lamp)
        self.assertAlmostEqual(light.GetRadiusAttr().Get(), .12, places=6)
        self.assertFalse(light.GetNormalizeAttr().Get())

    def test_invalid_parent_is_rejected_before_authoring_any_lamp(self):
        stage = self.cameras()
        with self.assertRaises(ValueError):
            CameraHeadlights(stage, {0: "/CameraA", 1: "/Missing"}, SOURCE)
        self.assertFalse(stage.GetPrimAtPath("/CameraA/HookeInspectionHeadlight"))
        with self.assertRaises(ValueError):
            CameraHeadlights(stage, {0: "/CameraA", 1: "/CameraA"}, SOURCE)


if __name__ == "__main__":
    unittest.main()
