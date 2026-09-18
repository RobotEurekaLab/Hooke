"""Independent optics and real finite support, without vendor replica claims."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))

from microscopy.illumination_design import layout

FIXTURE = ROOT/'temp/microscopy_research/cad_meshes_surface_normals'


class OpticalDesign(unittest.TestCase):
    def test_first_order_conjugate_na_and_invalid_separation(self):
        for separation in (140., 155., 180.):
            design = layout(separation)
            thickness, index = design['lens_centre_thickness'], design['assumed_lens_index']
            u = design['led_plane_z']-design['lens_flat_plane_z']-thickness
            v, focal = design['lens_to_sample'], design['equivalent_focal_length']
            propagate = lambda distance: np.array([[1., distance], [0., 1.]])
            surface = np.array([[1., 0.], [-1/focal, 1.]])
            transfer = propagate(v)@propagate(thickness/index)@surface@propagate(u)
            self.assertAlmostEqual(transfer[0, 1], 0., places=10)
            self.assertAlmostEqual(np.sin(np.arctan(design['aperture_radius']/v)), .25)
            self.assertGreater(v, 25.)
        for invalid in (float('nan'), float('inf'), -1., 20.):
            with self.assertRaises(ValueError):
                layout(invalid)


@unittest.skipUnless(importlib.util.find_spec('OCP'), 'Optional offline OpenCascade dependency')
class IlluminatorCAD(unittest.TestCase):
    def test_original_hollow_housing_spherical_lens_and_closed_tessellation(self):
        from microscopy.cad_import import convert, inspect_cylinders
        from microscopy.illumination_design import write_illuminator
        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            source = Path(folder)/'lamp.step'
            reference = write_illuminator(source)
            report = convert(source, Path(folder)/'mesh', .02)
            self.assertEqual(len(report['parts']), 8)
            self.assertTrue(all(part['watertight'] for part in report['parts']))
            self.assertFalse(reference['manufacturer_cad'])
            design = reference['design']
            np.testing.assert_allclose(report['cad_bounds_mm'],
                [-22, -22, design['lens_flat_plane_z']-2, 22, 22, 44], atol=1e-6)
            lens = report['parts'][3]
            np.testing.assert_allclose(np.asarray(lens['cad_bounds_mm']).ravel(),
                [-12.7, -12.7, design['lens_flat_plane_z'], 12.7, 12.7,
                 design['lens_flat_plane_z']+design['lens_centre_thickness']], atol=1e-6)
            cylinders = inspect_cylinders(source)['cylinders']
            self.assertTrue(any(abs(c['radius_mm']-13.)<1e-8 and c['bounds_mm'][2]<-100
                                for c in cylinders))
            with self.assertRaises(FileExistsError):
                write_illuminator(source)


@unittest.skipUnless((FIXTURE/'independent-critical-illuminator-v2/manifest.json').is_file()
                     and importlib.util.find_spec('mujoco'), 'Complete optional runtime fixture')
class SupportAssembly(unittest.TestCase):
    def environment(self):
        return patch.dict(os.environ, HOOKE_MICROSCOPY_ASSETS='cad', HOOKE_MICROSCOPY_OPTICS='assembled',
                          HOOKE_MICROSCOPY_ASSET_ROOT=str(FIXTURE))

    def test_a_rod_that_misses_its_finite_bore_is_rejected_before_physics(self):
        import copy
        from microscopy.cad_assets import CadAssets
        from microscopy.tasks import make_task
        read = CadAssets.interfaces

        def shortened(cad, name):
            report = copy.deepcopy(read(cad, name))
            if name == 'openframe/OF-TI-ARM':
                for surface in report['cylinders']:
                    if abs(surface['radius_mm']-8)<1e-8:
                        surface['bounds_mm'][5] = 100.
            return report

        with self.environment(), patch.object(CadAssets, 'interfaces', shortened):
            with self.assertRaisesRegex(ValueError, 'finite clamping bore'):
                make_task('suction_injection')

    def test_an_unsupported_barrel_diameter_is_rejected_before_physics(self):
        import copy
        from microscopy.cad_assets import CadAssets
        from microscopy.tasks import make_task
        read = CadAssets.read

        def wrong_diameter(cad, name, expected_hash=None):
            report = read(cad, name, expected_hash)
            if name == 'independent-critical-illuminator-v2':
                report = copy.deepcopy(report)
                report['dimensional_reference']['design']['barrel_diameter'] = 44.
            return report

        with self.environment(), patch.object(CadAssets, 'read', wrong_diameter):
            with self.assertRaisesRegex(ValueError, 'original optical/mechanical design'):
                make_task('suction_injection')

    def test_actual_rod_spans_bore_and_condenser_faces_sample_without_changing_encoders(self):
        import mujoco
        from microscopy.cad_assets import asset_evidence
        from microscopy.scene import ORIGIN, BENCH_TOP
        from microscopy.tasks import make_task
        with self.environment():
            task = make_task('suction_injection')
            task.reset(0)
        microscope = asset_evidence(task.model)['microscope']
        evidence = microscope['illumination_assembly']
        self.assertAlmostEqual(evidence['rod_bore_engagement_m'], .0285)
        self.assertGreater(evidence['rod_front_to_clamp_span_m'][0], 0)
        self.assertLess(evidence['rod_front_to_clamp_span_m'][1], .2)
        self.assertAlmostEqual(evidence['led_sleeve_radial_clearance_m'], .000025)
        self.assertLess(evidence['axis_alignment_error_m'], 1e-10)
        arm = next(p for p in microscope['components'] if p['component']=='OF-TI-ARM')
        direction = np.asarray(arm['rotation'])[:, 2]
        self.assertLess(direction[0], -.9)
        self.assertLess(direction[1], -.2)
        self.assertAlmostEqual(np.linalg.norm(np.asarray(arm['position_m'])[:2]), .031)
        geom_id = task.model.geom('microscope_cad_openframe_OF_TI_ARM_0').id
        mesh_id = task.model.geom_dataid[geom_id]
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, task.model.mesh_quat[mesh_id])
        input_to_mesh = matrix.reshape(3, 3).T
        for axial in (0., .2):
            mesh_point = input_to_mesh@(np.array([0.,0.,axial])-task.model.mesh_pos[mesh_id])
            actual = task.data.geom_xpos[geom_id]+task.data.geom_xmat[geom_id].reshape(3,3)@mesh_point
            expected = np.r_[ORIGIN[:2], BENCH_TOP]+np.asarray(arm['position_m'])+direction*axial
            np.testing.assert_allclose(actual, expected, atol=1e-8)
        point = np.asarray(evidence['position_m'])+[0,0,
            evidence['original_design']['design']['lens_flat_plane_z']*1e-3]
        self.assertAlmostEqual(point[2]+BENCH_TOP-ORIGIN[2], evidence['condenser_front_to_glass_m'])
        self.assertGreater(evidence['condenser_front_to_glass_m'], .025)
        self.assertFalse(evidence['threaded_fit_verified'])
        self.assertFalse(evidence['microscope_image_illumination_calibrated'])


if __name__ == '__main__':
    unittest.main()
