"""Pipette mouth metrology and hydraulic consistency in the compiled scene."""

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.pipettes import INJECTION_PROFILE, PipetteProfile, pipette_back
from microscopy.tasks import make_task


class PipetteGeometryTests(unittest.TestCase):
    def test_compiled_mouth_and_fine_neck_keep_micrometre_dimensions(self):
        task = make_task("suction_injection")
        task.reset(0)
        geom = task.model.geom("cell_hollow_pipette")
        mesh = task.model.mesh(int(geom.dataid[0]))
        start, count = int(mesh.vertadr[0]), int(mesh.vertnum[0])
        rotation = task.data.geom_xmat[geom.id].reshape(3, 3)
        vertices = task.model.mesh_vert[start:start+count]@rotation.T+task.data.geom_xpos[geom.id]
        delta = vertices-task.data.site_xpos[task.model.site("injector_tcp").id]
        back = pipette_back("injector")
        axial = delta@back
        radial = np.linalg.norm(delta-axial[:, None]*back, axis=1)
        for distance, outer, inner in INJECTION_PROFILE.sections:
            ring = radial[abs(axial-distance) < 2e-9]
            self.assertEqual(len(ring), 96)
            self.assertAlmostEqual(float(ring.max()), outer, delta=2e-9)
            self.assertAlmostEqual(float(ring.min()), inner, delta=2e-9)
        self.assertAlmostEqual(2*radial[abs(axial) < 2e-9].max(), 1.2e-6, delta=2e-9)
        self.assertLess(2*radial[axial < .300001e-3].max(), 4.01e-6)
        angles = task.public_state()["sample_scale"]["pipette_elevation_deg"]
        self.assertAlmostEqual(angles["holder"], 30.)
        self.assertAlmostEqual(angles["injector"], 35.)
        for tool in ("holder", "injector"):
            proxy = task.model.geom(tool+"_tip")
            self.assertEqual(proxy.group[0], 3)
            self.assertEqual(proxy.contype[0], 1)
        for tool in ("holder", "injector"):
            body = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_BODY, tool+"_needle_holder")
            if body >= 0:
                holder_axis = task.data.xmat[body].reshape(3, 3)[:, 2]
                np.testing.assert_allclose(holder_axis, pipette_back(tool), atol=1e-8)

    def test_flow_reduces_to_cylinder_and_cone_with_continuous_radius_limit(self):
        viscosity, length, radius = .001, .001, 1e-6
        cylinder = PipetteProfile(((0., 2*radius, radius), (length, 2*radius, radius)))
        expected = 8*viscosity*length/(np.pi*radius**4)
        self.assertAlmostEqual(cylinder.resistance(viscosity)/expected, 1.)
        near_cylinder = PipetteProfile(((0., 2*radius, radius),
                                       (length, 2*radius, radius*(1+1e-10))))
        self.assertAlmostEqual(near_cylinder.resistance(viscosity)/expected, 1.)
        cone = PipetteProfile(((0., 2*radius, radius), (length, 20*radius, 10*radius)))
        analytic = 8*viscosity/np.pi*(radius**-3-(10*radius)**-3)/(3*9*radius/length)
        self.assertAlmostEqual(cone.resistance(viscosity)/analytic, 1.)
        task = make_task("cell_injection")
        task.reset(0)
        self.assertEqual(task.mechanics.resistance_pa_s_m3, INJECTION_PROFILE.resistance(viscosity))


if __name__ == "__main__":
    unittest.main()
