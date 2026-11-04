"""Review visibility must be opt-in and must preserve the physical USD scene."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from backends.visual_profiles import (
    REAL_LABS_CUTAWAY,
    apply_visibility_profile,
    source_visibility_profile,
)

try:
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
except ImportError:
    Usd = None


def metadata():
    return {
        "visual_profile": REAL_LABS_CUTAWAY,
        "task_info": {"camera_mapping": {"room": "overview", "detail": "workstation"}},
        "names": {
            "camera": ["overview", "workstation", "interior", "mounted"],
            "geom": ["arch_roof_slab", "arch_cutaway_wall_front", "equipment_panel", "arch_luminaire_0"],
        },
    }


class VisibilityProfileTests(unittest.TestCase):
    def test_unrequested_robot_or_microscopy_view_is_unchanged(self):
        source = {"task_info": {"camera_mapping": {"task": "microscope_camera"}}}
        result = source_visibility_profile(source)
        self.assertIsNone(result["name"])
        self.assertEqual(result["hidden_geom_ids"], [])
        self.assertFalse(apply_visibility_profile(None, {}, result)["applied"])

    def test_exact_cutaway_names_preserve_equipment_and_input(self):
        source = metadata()
        source["names"]["geom"].extend(["equipment_roof", "wall_back", "prefix_arch_roof_slab", None])
        original = deepcopy(source)
        result = source_visibility_profile(source)
        self.assertEqual(result["hidden_geom_ids"], [0, 1, 3])
        self.assertEqual(result["cameras"], ["overview", "workstation"])
        self.assertEqual(source, original)

    def test_mixed_scientific_or_missing_camera_is_rejected(self):
        for camera in ("interior", "mounted", "not_authored"):
            with self.subTest(camera=camera):
                source = metadata()
                source["task_info"]["camera_mapping"]["detail"] = camera
                with self.assertRaisesRegex(ValueError, "overview/workstation"):
                    source_visibility_profile(source)

    def test_profile_requires_explicit_review_scene(self):
        source = metadata()
        source["visual_profile"] = "hide_everything"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            source_visibility_profile(source)
        for mapping in ({}, {"detail": None}):
            source = metadata()
            source["task_info"]["camera_mapping"] = mapping
            with self.assertRaisesRegex(ValueError, "explicit camera mapping"):
                source_visibility_profile(source)
        source = metadata()
        source["names"]["geom"] = ["equipment_roof"]
        with self.assertRaisesRegex(ValueError, "no authored cutaway geometry"):
            source_visibility_profile(source)


@unittest.skipIf(Usd is None, "Requires existing USD Python bindings; no simulator or GPU is started")
class VisibilityUSDTests(unittest.TestCase):
    def stage(self):
        stage = Usd.Stage.CreateInMemory()
        paths = {}
        for index in range(4):
            path = f"/World/g{index}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(0.2 + index)
            cube.AddTranslateOp().Set((index, 0, 1))
            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(2 + index)
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
            paths[index] = path
        filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(paths[0]))
        filtered.CreateFilteredPairsRel().SetTargets([paths[2]])
        UsdLux.DomeLight.Define(stage, "/World/Ambient").CreateIntensityAttr(450)
        UsdGeom.Camera.Define(stage, "/World/camera").CreateFocalLengthAttr(24)
        return stage, paths

    def physical_fingerprint(self, stage):
        return {
            str(prim.GetPath()): {
                "type": prim.GetTypeName(),
                "active": prim.IsActive(),
                "schemas": list(prim.GetAppliedSchemas()),
                "attributes": {
                    attr.GetName(): repr(attr.Get()) for attr in prim.GetAttributes()
                    if attr.GetName() != "visibility"
                },
                "relationships": {
                    rel.GetName(): list(map(str, rel.GetTargets())) for rel in prim.GetRelationships()
                },
            }
            for prim in stage.Traverse()
        }

    def test_cutaway_changes_only_visibility_and_retains_colliders(self):
        stage, paths = self.stage()
        before = self.physical_fingerprint(stage)
        report = apply_visibility_profile(stage, paths, source_visibility_profile(metadata()))
        self.assertEqual(before, self.physical_fingerprint(stage))
        self.assertEqual(report["hidden_geom_paths"], [paths[0], paths[1], paths[3]])
        for index, path in paths.items():
            prim = stage.GetPrimAtPath(path)
            self.assertTrue(prim.IsActive())
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            visibility = UsdGeom.Imageable(prim).GetVisibilityAttr().Get()
            self.assertEqual(visibility, "inherited" if index == 2 else "invisible")

    def test_default_profile_does_not_author_any_usd_change(self):
        stage, paths = self.stage()
        before = stage.GetRootLayer().ExportToString()
        apply_visibility_profile(stage, paths, source_visibility_profile({}))
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_invalid_mapping_cannot_partially_hide_a_stage(self):
        stage, paths = self.stage()
        paths[3] = "/World/not_converted"
        before = stage.GetRootLayer().ExportToString()
        with self.assertRaisesRegex(ValueError, "not imageable"):
            apply_visibility_profile(stage, paths, source_visibility_profile(metadata()))
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)


if __name__ == "__main__":
    unittest.main()
