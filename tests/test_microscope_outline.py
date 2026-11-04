"""Physical datums and joint ownership of the original inclined carousel."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))
from microscopy.cad_assets import asset_evidence
from microscopy.reference_objective import installed_dimensions
from microscopy.tasks import make_task
from microscopy.turret_geometry import build_geometry


class MicroscopeOutlineTests(unittest.TestCase):
    def test_carousel_is_closed_and_preserves_the_selected_axis(self):
        vertices, faces, sockets, axes = build_geometry()
        shape = trimesh.Trimesh(vertices, faces, process=False)
        self.assertTrue(shape.is_watertight)
        self.assertTrue(shape.is_winding_consistent)
        self.assertGreater(shape.volume, 0.)
        np.testing.assert_allclose(sockets[0], 0., atol=1e-15)
        np.testing.assert_allclose(axes[0], [0., 0., 1.], atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1., atol=1e-15)
        self.assertTrue(np.all(sockets[1:, 2] < 0.))
        self.assertTrue(np.all(axes[1:, 0] < 0.))
        self.assertLess(vertices[:, 2].min(), -.030)
        self.assertLessEqual(vertices[:, 2].max(), 1e-15)

    def test_focus_carries_turret_and_socket_axes_without_moving_specimen(self):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS="reference",
                HOOKE_MICROSCOPY_STAND="te2000-s-reference", HOOKE_MICROSCOPY_STAGE="reference",
                HOOKE_MICROSCOPY_OPTICS="estimated", HOOKE_MICROSCOPY_MANIPULATOR="auto")):
            task = make_task("suction_injection")
            task.reset(0)
        model, data = task.model, task.data
        evidence = asset_evidence(model)["objective"]
        axes = np.asarray(evidence["mount_socket_axes"])
        for index, slot in enumerate(evidence["legacy_occupied_slots"]):
            actual = data.xmat[model.body(f"reference_objective_socket_{index}").id].reshape(3, 3)[:, 2]
            np.testing.assert_allclose(actual, axes[slot], atol=1e-9)
        address = model.joint("focus").qposadr[0]
        data.qpos[address] = 0.
        mujoco.mj_forward(model, data)
        fixed = data.geom_xpos[model.geom("focus_mount_fixed_outline").id].copy()
        sample = data.site_xpos[model.site("sample_plane").id].copy()
        moving = ("nosepiece", "focus_mount_carrier_cap", "focus_mount_spindle", "reference_40x_nose")
        initial = {name: data.geom_xpos[model.geom(name).id].copy() for name in moving}
        for position in (-.001, 0., .001):
            data.qpos[address] = position
            mujoco.mj_forward(model, data)
            for name in moving:
                np.testing.assert_allclose(data.geom_xpos[model.geom(name).id]-initial[name],
                                           [0., 0., position], atol=1e-9)
            np.testing.assert_allclose(data.geom_xpos[model.geom("focus_mount_fixed_outline").id], fixed)
            np.testing.assert_allclose(data.site_xpos[model.site("sample_plane").id], sample)
            dimensions = installed_dimensions(model, data)
            self.assertAlmostEqual(dimensions["geometric_working_distance_m"], .0031-26e-6-position,
                                   delta=2e-9)
            arm = model.geom("focus_mount_carrier_cap")
            lower = data.geom_xpos[arm.id, 2]-arm.size[2]
            self.assertGreater(lower, evidence["focus_mount"]["stand_attachment_plane_world_z_m"])


if __name__ == "__main__":
    unittest.main()
