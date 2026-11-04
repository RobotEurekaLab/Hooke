"""Physical guides, retained true-size marker, and full-stroke dry operation."""

from copy import deepcopy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


class EpflMicroscopeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("epfl_microbiorobotics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        m = cls.model
        cls.joints = [
            m.joint("magnetic_microrobot__joint_" + name).id
            for name in ("focus", "stage_x", "stage_y")
        ]
        groups = set(m.body_weldid[m.jnt_bodyid[cls.joints]])
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        cls.pairs = np.array(
            [
                (a, b)
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (
                    m.body_weldid[m.geom_bodyid[a]] in groups
                    or m.body_weldid[m.geom_bodyid[b]] in groups
                )
            ],
            dtype=int,
        )

    def data(self):
        d = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d)
        return d

    def geom(self, name):
        return self.model.geom("magnetic_microrobot__epfl_" + name).id

    def assert_clear(self, d, exhaustive=False):
        m = self.model
        pairs = self.pairs
        if not exhaustive:
            rotation = d.geom_xmat.reshape(-1, 3, 3)
            center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
            half = np.einsum("nij,nj->ni", np.abs(rotation), m.geom_aabb[:, 3:])
            a, b = pairs.T
            lower = np.linalg.norm(
                np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
            )
            pairs = pairs[lower < 0.001]
        minimum = 0.001
        for a, b in pairs:
            distance = mujoco.mj_geomDistance(m, d, int(a), int(b), 0.001, None)
            self.assertGreaterEqual(
                distance, 0.00049, (m.geom(int(a)).name, m.geom(int(b)).name, distance)
            )
            minimum = min(minimum, distance)
        return minimum

    def test_original_controls_marker_and_microscope_camera(self):
        m, d = self.model, self.data()
        for joint, origin, axis, limits in zip(
            self.joints,
            ((0, 0, 0.291), (0, 0, 0.324), (0, 0, 0.021)),
            ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            ((-0.003, 0.004), (-0.057, 0.057), (-0.0365, 0.0365)),
        ):
            np.testing.assert_allclose(m.body_pos[m.jnt_bodyid[joint]], origin)
            np.testing.assert_allclose(m.jnt_axis[joint], axis)
            np.testing.assert_allclose(m.jnt_range[joint], limits)
        marker = self.geom("micro_marker")
        self.assertAlmostEqual(m.geom_size[marker, 0], 0.00006)
        self.assertAlmostEqual(m.geom_size[marker, 1], 0.0002)
        np.testing.assert_allclose(d.geom_xpos[marker], (-0.65, -0.25, 1.162))
        camera = m.camera("magnetic_microrobot__microscope_camera").id
        np.testing.assert_allclose(m.cam_pos[camera], (0, 0, 0.53))
        self.assertEqual(m.cam_fovy[camera], 11)
        self.assertEqual(m.geom_bodyid[marker], m.jnt_bodyid[self.joints[2]])

    def test_open_guide_clearances_and_connected_supports(self):
        m, d = self.model, self.data()
        for side in (-1, 1):
            rail = self.geom(f"x_rail_{side}")
            for end in (-1, 1):
                for wall in ("top", "bottom", "side"):
                    suffixes = (
                        (f"{side}_{end}",)
                        if wall != "side"
                        else (f"{side}_{end}_-1", f"{side}_{end}_1")
                    )
                    for suffix in suffixes:
                        shoe = self.geom(f"x_shoe_{wall}_{suffix}")
                        self.assertAlmostEqual(
                            mujoco.mj_geomDistance(m, d, rail, shoe, 0.01, None),
                            0.0005,
                            places=8,
                        )
                pedestal = self.geom(f"x_pedestal_{side}_{end}")
                self.assertAlmostEqual(
                    d.geom_xpos[pedestal, 2] + m.geom_size[pedestal, 2],
                    d.geom_xpos[rail, 2] - m.geom_size[rail, 2],
                )
                self.assertAlmostEqual(
                    d.geom_xpos[pedestal, 2] - m.geom_size[pedestal, 2], 0.930
                )
            yrail = self.geom(f"y_rail_{side}")
            for wall in (
                f"top_{side}",
                f"bottom_{side}",
                f"side_{side}_-1",
                f"side_{side}_1",
            ):
                self.assertAlmostEqual(
                    mujoco.mj_geomDistance(
                        m, d, yrail, self.geom("y_shoe_" + wall), 0.01, None
                    ),
                    0.0005,
                    places=8,
                )
        stem = self.geom("focus_stem")
        for wall in ("side_-1", "side_1", "front", "back"):
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(
                    m, d, stem, self.geom("focus_guide_" + wall), 0.01, None
                ),
                0.0005,
                places=8,
            )
        glass = self.geom("sample_glass")
        for side in (-1, 1):
            seat = self.geom(f"slide_seat_{side}")
            pad = self.geom(f"slide_keeper_pad_{side}")
            self.assertAlmostEqual(
                d.geom_xpos[seat, 2] + m.geom_size[seat, 2],
                d.geom_xpos[glass, 2] - m.geom_size[glass, 1],
            )
            self.assertAlmostEqual(
                d.geom_xpos[pad, 2] - m.geom_size[pad, 2],
                d.geom_xpos[glass, 2] + m.geom_size[glass, 1],
            )
        for name in (
            "focus_stem",
            "focus_arm",
            "objective_face_0",
            "x_beam_1",
            "micro_marker",
        ):
            geom = self.geom(name)
            self.assertEqual((m.geom_contype[geom], m.geom_conaffinity[geom]), (1, 1))

    def test_full_three_axis_travel_includes_parent_filtered_interfaces(self):
        m, d = self.model, self.data()
        addresses = m.jnt_qposadr[self.joints]
        # Every enabled cross-weld pair, including native parent-filtered
        # interfaces, receives a direct distance query at every sampled pose.
        for pose in itertools.product(
            *[np.linspace(*m.jnt_range[j], 5) for j in self.joints]
        ):
            d.qpos[addresses] = pose
            mujoco.mj_forward(m, d)
            self.assert_clear(d, exhaustive=True)

    def test_actual_simultaneous_extrema_and_return_without_reset(self):
        m, d = self.model, self.data()
        addresses = m.jnt_qposadr[self.joints]
        actuators = [
            m.actuator("magnetic_microrobot__actuator_" + name).id
            for name in ("focus", "stage_x", "stage_y")
        ]
        marker = self.geom("micro_marker")
        home = d.geom_xpos[marker].copy()
        steps = int(np.ceil(0.8 / m.opt.timestep))
        total = 0
        for target in ((-0.003, -0.057, -0.0365), (0.004, 0.057, 0.0365), (0, 0, 0)):
            d.ctrl[actuators] = target
            for _ in range(steps):
                mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                self.assert_clear(d)
                total += 1
            np.testing.assert_allclose(d.qpos[addresses], target, atol=2e-6)
            np.testing.assert_allclose(
                d.geom_xpos[marker] - home, (target[1], target[2], 0), atol=2e-6
            )
        self.assertAlmostEqual(d.time, total * m.opt.timestep, places=9)
        self.assertFalse(any(w.number for w in d.warning))

    def test_external_probe_reacts_to_focus_and_x_carriage_surfaces(self):
        for name in ("objective_face_0", "x_beam_-1"):
            root = deepcopy(self.root)
            d = self.data()
            geom = self.geom(name)
            half_height = self.model.geom_size[
                geom, 1 if name.startswith("objective") else 2
            ]
            position = d.geom_xpos[geom].copy()
            position[2] += half_height + 0.00095
            body = ET.SubElement(
                root.find("worldbody"),
                "body",
                name="contact_probe",
                pos=" ".join(map(str, position)),
            )
            ET.SubElement(body, "freejoint", name="contact_probe_joint")
            ET.SubElement(
                body,
                "geom",
                name="contact_probe_geom",
                type="sphere",
                size=".001",
                mass=".001",
                contype="1",
                conaffinity="1",
            )
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            state = mujoco.MjData(model)
            mujoco.mj_forward(model, state)
            probe = model.geom("contact_probe_geom").id
            surface = model.geom("magnetic_microrobot__epfl_" + name).id
            matching = [
                i
                for i, c in enumerate(state.contact)
                if {c.geom1, c.geom2} == {probe, surface}
            ]
            self.assertTrue(matching, name)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, state, matching[0], wrench)
            self.assertGreater(wrench[0], 0)

    def test_frame_butt_joints_preserve_solid_union_without_duplicate_faces(self):
        m = self.model
        groups = [
            [f"x_{part}_{side}" for part in ("beam", "outer_tie") for side in (-1, 1)],
            [f"y_{part}_{side}" for part in ("side", "crossmember") for side in (-1, 1)]
            + [f"coil_upright_arm_{side}_{sx}" for side in (-1, 1) for sx in (-1, 1)],
            ["focus_arm", "focus_rear_arm"],
        ]
        for names in groups:
            geoms = [self.geom(name) for name in names]
            self.assertEqual(len(set(m.geom_bodyid[geoms])), 1)
            new_center, new_size = m.geom_pos[geoms].copy(), m.geom_size[geoms].copy()
            old_center, old_size = new_center.copy(), new_size.copy()
            for i, name in enumerate(names):
                if name.startswith("x_outer_tie"):
                    old_size[i, 1] = 0.162
                elif name.startswith("y_crossmember"):
                    old_size[i, 0] = 0.125
                elif name.startswith("coil_upright_arm"):
                    old_center[i, 1] = np.sign(old_center[i, 1]) * 0.060
                    old_size[i, 1] = 0.021
                elif name == "focus_arm":
                    old_center[i, 1], old_size[i, 1] = 0.078, 0.042
            # Partition all space covered by these axis-aligned boxes at
            # every old/new face coordinate. Occupancy is constant in each
            # open cell; equality proves the closed finite solid unions match.
            axes = [
                np.unique(
                    np.concatenate(
                        (
                            old_center[:, k] - old_size[:, k],
                            old_center[:, k] + old_size[:, k],
                            new_center[:, k] - new_size[:, k],
                            new_center[:, k] + new_size[:, k],
                        )
                    )
                )
                for k in range(3)
            ]
            for point in itertools.product(*[(a[:-1] + a[1:]) / 2 for a in axes]):
                old_inside = np.any(
                    np.all(
                        np.abs(np.array(point) - old_center) < old_size + 1e-14, axis=1
                    )
                )
                new_inside = np.any(
                    np.all(
                        np.abs(np.array(point) - new_center) < new_size + 1e-14, axis=1
                    )
                )
                self.assertEqual(old_inside, new_inside, (names, point))
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    overlap = np.minimum(
                        new_center[i] + new_size[i], new_center[j] + new_size[j]
                    ) - np.maximum(
                        new_center[i] - new_size[i], new_center[j] - new_size[j]
                    )
                    self.assertLessEqual(
                        np.prod(np.maximum(overlap, 0)), 1e-18, (names[i], names[j])
                    )

    def test_sole_connects_all_original_feet_to_enabled_chassis(self):
        m, d = self.model, self.data()
        sole = self.geom("chassis_sole")
        core = m.geom("magnetic_microrobot__box_7").id
        self.assertEqual((m.geom_contype[sole], m.geom_conaffinity[sole]), (1, 1))
        self.assertAlmostEqual(
            mujoco.mj_geomDistance(m, d, sole, core, 0.01, None), 0, places=8
        )
        for index in range(1, 5):
            foot = m.geom(f"magnetic_microrobot__cylinder_{index}").id
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(m, d, sole, foot, 0.01, None), 0, places=8
            )
            self.assertAlmostEqual(
                d.geom_xpos[foot, 2] + m.geom_size[foot, 1],
                d.geom_xpos[sole, 2] - m.geom_size[sole, 2],
            )


if __name__ == "__main__":
    unittest.main()
