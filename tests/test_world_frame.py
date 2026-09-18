"""Rebasing preserves source coordinates, mounted views and physical anchors."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.world_frame import WorldFrame

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from backends.usd_scene import SceneBridge
except ImportError:
    Usd = None


class Coordinates(unittest.TestCase):
    def test_source_roundtrip_and_force_lever_are_preserved(self):
        frame = WorldFrame([.01, -.07, .995])
        points = np.array([[.003, -.064, 1.099], [.004, -.063, 1.10]])
        native = frame.to_native(points)
        np.testing.assert_allclose(frame.to_source(native), points, atol=1e-15)
        np.testing.assert_allclose(native[1] - native[0], points[1] - points[0], atol=1e-15)
        force = np.array([20., -30., 40.]) * 1e-9
        np.testing.assert_allclose(np.cross(native[1] - native[0], force),
                                   np.cross(points[1] - points[0], force), atol=1e-23)
        np.testing.assert_array_equal(WorldFrame().to_native(points), points)

    def test_invalid_origin_cannot_enter_the_solver(self):
        for origin in (1., [1., 2.], [[1., 2., 3.]], [0., np.nan, 0.], [0., 0., np.inf]):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                WorldFrame(origin)


@unittest.skipIf(Usd is None, "Run with installed Isaac USD bindings")
class ExportCoordinates(unittest.TestCase):
    def test_all_geometry_cameras_sites_and_joint_anchors_translate_together(self):
        fixture = os.environ.get("HOOKE_TEST_USD_SOURCE")
        if not fixture:
            self.skipTest("Set HOOKE_TEST_USD_SOURCE to a compiled source archive")
        origin = np.array([.01, -.07, .995])
        with tempfile.TemporaryDirectory() as output:
            bridges = []
            for i, offset in enumerate((np.zeros(3), origin)):
                folder = Path(output) / str(i)
                folder.mkdir()
                bridge = SceneBridge(Usd.Stage.CreateInMemory(), fixture, folder,
                                     dict(world_origin_m=offset.tolist()))
                bridge.build()
                bridges.append(bridge)
            caches = [UsdGeom.XformCache() for _ in bridges]
            paths = set(bridges[0].geom_paths.values()) | set(bridges[0].camera_paths.values())
            paths.update(bridges[0].body_paths[int(body)] + f"/site{i}"
                         for i, body in enumerate(bridges[0].m["site_bodyid"]))
            self.assertTrue(paths and bridges[0].free_bodies and bridges[0].joint_paths)
            for path in paths:
                poses = [cache.GetLocalToWorldTransform(bridge.stage.GetPrimAtPath(path))
                         for bridge, cache in zip(bridges, caches)]
                np.testing.assert_allclose(np.asarray(poses[1].ExtractTranslation()),
                                           np.asarray(poses[0].ExtractTranslation()) - origin,
                                           atol=1e-12, rtol=0., err_msg=path)
                np.testing.assert_allclose(np.asarray(poses[1])[:3, :3],
                                           np.asarray(poses[0])[:3, :3], atol=1e-12, err_msg=path)
            joints = [prim.GetPath() for prim in bridges[0].stage.Traverse()
                      if prim.IsA(UsdPhysics.Joint) and not str(prim.GetPath()).split("/")[-1].startswith("rootAnchor")]
            self.assertGreater(len(joints), len(bridges[0].joint_paths))
            for path in joints:
                for side in (0, 1):
                    points = []
                    for bridge, cache in zip(bridges, caches):
                        joint = UsdPhysics.Joint(bridge.stage.GetPrimAtPath(path))
                        targets = getattr(joint, f"GetBody{side}Rel")().GetTargets()
                        local = getattr(joint, f"GetLocalPos{side}Attr")().Get() or Gf.Vec3f(0.)
                        transform = (cache.GetLocalToWorldTransform(bridge.stage.GetPrimAtPath(targets[0]))
                                     if targets else Gf.Matrix4d(1.))
                        points.append(np.asarray(transform.Transform(Gf.Vec3d(local))))
                    np.testing.assert_allclose(points[1], points[0] - origin, atol=2e-7,
                                               rtol=0., err_msg=f"{path}, side {side}")


if __name__ == "__main__":
    unittest.main()
