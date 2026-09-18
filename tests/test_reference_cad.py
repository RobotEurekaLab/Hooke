"""Read back the independent STEP and verify its dimensioned outside envelope."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))


@unittest.skipUnless(importlib.util.find_spec('OCP'), 'OpenCascade is an optional offline CAD dependency')
class ObjectiveOutline(unittest.TestCase):
    @unittest.skipUnless((ROOT/'temp/microscopy_research/openframe/openFrame CAD/OF-AD-OBH-M25-60.stp').is_file(),
                         'Pinned open hardware is an optional local source fixture')
    def test_nominal_m25_derivative_clears_the_thread_envelope_and_preserves_original(self):
        import hashlib
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.gp import gp_Pnt
        from microscopy.cad_import import convert
        from microscopy.reference_cad import M25_MOUNT_SOURCE, write_nominal_m25_mount

        original = ROOT/'temp/microscopy_research/openframe/openFrame CAD/OF-AD-OBH-M25-60.stp'
        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            source = Path(folder)/'nominal.step'
            evidence = write_nominal_m25_mount(original, source)
            report = convert(source, Path(folder)/'mesh', .02)
            self.assertEqual(len(report['parts']), 1)
            self.assertTrue(report['parts'][0]['watertight'])
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), M25_MOUNT_SOURCE)
            for path, state in ((original, TopAbs_IN), (source, TopAbs_OUT)):
                reader = STEPControl_Reader()
                reader.ReadFile(str(path))
                reader.TransferRoots()
                solid = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID).Current()
                self.assertEqual(BRepClass3d_SolidClassifier(solid, gp_Pnt(12, 0, 3), 1e-7).State(), state)
                self.assertEqual(BRepClass3d_SolidClassifier(solid, gp_Pnt(15, 0, 3), 1e-7).State(), TopAbs_IN)
            self.assertFalse(evidence['thread_fit_verified'])
            self.assertFalse(evidence['fabrication_validated'])
            self.assertIn('CERN-OHL-P-2.0', evidence['rights'])

    def test_focus_export_has_separate_fixed_and_moving_solids_and_real_hole_patterns(self):
        from microscopy.cad_import import convert, inspect_cylinders
        from microscopy.reference_cad import write_focus_stage

        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            source = Path(folder)/'focus.step'
            evidence = write_focus_stage(source)
            report = convert(source, Path(folder)/'mesh', .02)
            self.assertEqual(len(report['parts']), 2)
            self.assertTrue(all(part['watertight'] for part in report['parts']))
            np.testing.assert_allclose(report['cad_bounds_mm'], [-22.5, -24, -15, 22.5, 24, 0], atol=1e-6)
            metrology = inspect_cylinders(source)
            for x in (-18.5, 18.5):
                for y in (-18.5, 18.5):
                    self.assertTrue(any(abs(c['radius_mm']-1.45)<1e-6
                        and np.allclose(np.array(c['point_mm'])[:2], [x,y]) for c in metrology['cylinders']))
            for x in (-12.5, 12.5):
                self.assertTrue(any(abs(c['radius_mm']-1)<1e-6
                    and np.allclose(np.array(c['point_mm'])[:2], [x,12.5])
                    and c['bounds_mm'][2]<-14.9 for c in metrology['cylinders']))
            self.assertEqual(evidence['part_roles'], ['fixed_base','moving_carriage'])
            self.assertIn('base_thickness', evidence['estimates'])
            self.assertFalse(evidence['manufacturer_cad'])
            with self.assertRaises(FileExistsError):
                write_focus_stage(source)

    def test_exported_objective_dimensions_hollow_body_and_declared_estimates(self):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.gp import gp_Pnt
        from microscopy.cad_import import convert, inspect_cylinders
        from microscopy.reference_cad import write_objective

        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            source = Path(folder)/'objective.step'
            evidence = write_objective(source)
            report = convert(source, Path(folder)/'mesh', .02)
            self.assertEqual(len(report['parts']), 1)
            self.assertTrue(report['parts'][0]['watertight'])
            np.testing.assert_allclose(report['cad_bounds_mm'], [-13.5, -13.5, -5., 13.5, 13.5, 49.4], atol=1e-6)
            cylinders = inspect_cylinders(source)['cylinders']
            thread = next(c for c in cylinders if abs(c['radius_mm']-12.5) < 1e-6 and c['bounds_mm'][2] < -1)
            np.testing.assert_allclose(np.array(thread['bounds_mm'])[[2, 5]], [-5, 0], atol=1e-6)
            reader = STEPControl_Reader()
            reader.ReadFile(str(source))
            reader.TransferRoots()
            solid = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID).Current()
            self.assertEqual(BRepClass3d_SolidClassifier(solid, gp_Pnt(0, 0, 20), 1e-7).State(), TopAbs_OUT)
            self.assertEqual(BRepClass3d_SolidClassifier(solid, gp_Pnt(5, 0, 20), 1e-7).State(), TopAbs_IN)
            self.assertFalse(evidence['manufacturer_cad'])
            self.assertIn('internal_bore_diameter', evidence['estimates'])
            self.assertIn('Glass elements', evidence['omitted'])
            self.assertEqual(evidence['verified_dimensions']['length_excluding_threads'], 49.4)
            with self.assertRaises(FileExistsError):
                write_objective(source)


if __name__ == '__main__':
    unittest.main()
