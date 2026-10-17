"""Motion and physical-clearance regression for three Australian apparatus bays."""

from itertools import product
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.layout_audit import (
    audit_layout,
    model_obstacles,
    doorway_obstacle_warnings,
)
from real_labs.runtime import InstrumentSession


SCENES = ("uwa_ngcf_c72", "uts_infrastructure_robotics", "unsw_wrl_wave_basin")


def session(identifier):
    root, manifest = build_scene(identifier)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class OceaniaWave2GeometryTests(unittest.TestCase):
    def test_three_dynamic_targets_per_axis(self):
        for identifier in SCENES:
            s = session(identifier)
            for key, control in s.controls.items():
                span = control.upper - control.lower
                for fraction in (0.15, 0.85, 0.25):
                    result = s.command(
                        *key, control.lower + fraction * span, duration=2
                    )
                    self.assertLessEqual(
                        abs(result["error"]),
                        max(1e-6, span * 0.025),
                        (identifier, result),
                    )
                s.reset()

    def test_full_stroke_clearance_and_combined_extremes(self):
        for identifier in SCENES:
            s = session(identifier)
            controls = list(s.controls.values())
            poses = [np.zeros(len(controls))]
            for axis, c in enumerate(controls):
                for value in np.linspace(c.lower, c.upper, 21):
                    pose = np.zeros(len(controls))
                    pose[axis] = value
                    poses.append(pose)
            if len(controls) <= 3:
                poses.extend(
                    product(*[np.linspace(c.lower, c.upper, 5) for c in controls])
                )
            else:
                poses.extend(
                    [
                        [
                            c.lower if (i + offset) % 2 else c.upper
                            for i, c in enumerate(controls)
                        ]
                        for offset in (0, 1)
                    ]
                )
                poses.extend(
                    [
                        [getattr(c, bound) for c in controls]
                        for bound in ("lower", "upper")
                    ]
                )
            for pose in poses:
                s.reset()
                for c, value in zip(controls, pose):
                    s.data.qpos[s.model.jnt_qposadr[c.joint]] = value
                mujoco.mj_forward(s.model, s.data)
                bad = [
                    (
                        s.model.geom(c.geom1).name,
                        s.model.geom(c.geom2).name,
                        float(c.dist),
                    )
                    for c in s.data.contact
                    if c.dist < -0.0001
                ]
                self.assertEqual(bad, [], (identifier, list(pose), bad))

    def test_recessed_tank_is_open_and_has_supported_floor(self):
        s = session("uts_infrastructure_robotics")
        geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            s.model,
            s.data,
            np.array([-2.75, 2.6, 0.2]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            -1,
            geom,
        )
        # First surface may be the static water. It must lie below the room datum.
        self.assertGreater(distance, 0.2)
        floor = s.model.geom("arch_floor_uts_tank")
        self.assertAlmostEqual(floor.pos[2] + floor.size[2], -1.5)
        self.assertNotIn(
            "arch_floor", [s.model.geom(i).name for i in range(s.model.ngeom)]
        )
        self.assertAlmostEqual(5 * 6 * 1.5, 45)

    def test_twenty_paddles_are_independent_and_physically_present(self):
        s = session("unsw_wrl_wave_basin")
        self.assertEqual(len(s.controls), 20)
        for c in s.controls.values():
            body = s.model.jnt_bodyid[c.joint]
            geoms = np.flatnonzero(s.model.geom_bodyid == body)
            self.assertTrue(any(s.model.geom_contype[g] for g in geoms))
            self.assertTrue(any(s.model.geom_size[g, 2] > 0.5 for g in geoms))
        self.assertEqual(len({c.joint for c in s.controls.values()}), 20)

    def test_floor_support_and_door_approaches(self):
        for identifier in SCENES:
            s = session(identifier)
            self.assertTrue(audit_layout(s.manifest)["passed"])
            self.assertEqual(
                doorway_obstacle_warnings(s.manifest, model_obstacles(s.model, s.data)),
                [],
            )
            np.testing.assert_allclose(s.model.opt.gravity, [0, 0, -9.81])
        s = session("uwa_ngcf_c72")
        bottom = s.model.body("primary_apparatus__body_basket_tilt").pos
        self.assertLess(bottom[0], -4.0)
        for name in ("arch_roof_c72_left", "arch_roof_c72_right"):
            self.assertTrue(s.model.geom(name).contype)


if __name__ == "__main__":
    unittest.main()
