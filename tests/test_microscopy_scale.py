"""Measure rendered and compiled sample dimensions against the SI model."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from microscopy.cell_scene import NEEDLE_BACK
from microscopy.cell_tasks import CellInjection
from microscopy.suction_tasks import SuctionInjection
from microscopy.tasks import CalibrationPush as Push


class MicroscopyScaleTests(unittest.TestCase):
    def test_cell_footprint_in_pixels_matches_the_three_dimensional_sample(self):
        for cls in (CellInjection, SuctionInjection):
            with self.subTest(operation=cls.operation):
                task = cls.Expert(cls.load())
                task.reset(0)
                state = task.mechanics.image_state(task.data)
                state["focus_m"] = 0.
                # Phase-contrast minima lie inside or outside an object because
                # of interference. Use the filled tracer channel to measure
                # the nominal specimen footprint, independently of its halos.
                state["delivered_pl"] = .5
                image = np.asarray(task.microscope.render(
                    state, annotate=False, channel="fluorescence"), dtype=float)[..., 1]
                centre = np.asarray(state["centres_m"][0])
                calibration = task.microscope.calibration
                pixel = np.asarray(calibration.pixel(centre[:2]))
                measured = []
                for axis in (0, 1):
                    profile = image[int(round(pixel[1])), :] if axis == 0 else image[:, int(round(pixel[0]))]
                    distances = np.abs(np.arange(calibration.pixels)-pixel[axis])*calibration.pixel_size_m
                    radius = task.mechanics.cell_radii[axis]
                    background = np.percentile(profile, 5)
                    threshold = background+.1*(profile.max()-background)
                    footprint = np.flatnonzero((distances < radius*1.25) & (profile > threshold))
                    self.assertGreater(len(footprint), 20, "A resolved tracer footprint must be present")
                    measured.append((footprint[-1]-footprint[0])*calibration.pixel_size_m)
                np.testing.assert_allclose(measured, 2*task.model.geom("cell_0_shell").size[:2],
                                           rtol=0., atol=2*calibration.pixel_size_m)
                np.testing.assert_allclose(task.model.geom("cell_0_shell").size,
                                           task.mechanics.cell_radii, rtol=0., atol=1e-15)
                np.testing.assert_allclose(task.model.geom("cell_0_nucleus").size,
                                           task.mechanics.nuclear_radii, rtol=0., atol=1e-15)
                if cls is SuctionInjection:
                    np.testing.assert_allclose(task.runtime_visuals()[0]["size"],
                                               task.model.geom("cell_0_shell").size, rtol=0., atol=1e-15)

    def test_capillary_and_bead_do_not_inherit_cell_camera_magnification(self):
        cell = CellInjection.Expert(CellInjection.load())
        cell.reset(0)
        state = cell.public_state()
        geom = cell.model.geom("cell_hollow_pipette")
        mesh = cell.model.mesh(int(geom.dataid[0]))
        start, count = int(mesh.vertadr[0]), int(mesh.vertnum[0])
        rotation = cell.data.geom_xmat[geom.id].reshape(3, 3)
        vertices = cell.model.mesh_vert[start:start+count]@rotation.T+cell.data.geom_xpos[geom.id]
        axial = (vertices-cell.data.site_xpos[cell.model.site("injector_tcp").id])@NEEDLE_BACK
        self.assertAlmostEqual(float(np.ptp(axial)), state["sample_scale"]["exposed_capillary_length_m"],
                               delta=2e-9)
        self.assertLess(max(state["sample_scale"]["diameters_m"]), 100e-6)
        self.assertLess(state["sample_scale"]["needle_tip_outer_diameter_m"], 2e-6)
        bead = Push.Expert(Push.load())
        bead.reset(0)
        self.assertEqual(bead.public_state()["sample_scale"]["kind"], "calibration_bead")
        self.assertAlmostEqual(2*bead.model.geom("bead_push_geom").size[0], .0012)
        self.assertGreater(bead.microscope.calibration.field_width_m,
                           cell.microscope.calibration.field_width_m*50)
        self.assertFalse(state["calibration"]["physical_optics_calibrated"])


if __name__ == "__main__":
    unittest.main()
