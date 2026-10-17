"""Supported fixed source context, with untouched towing mechanics."""

from copy import copy
import itertools
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.berkeley_tank_motion import TankClearance, cylinder_box_gap
from real_labs.berkeley_hall_context import APPEARANCE_PREFIXES
from real_labs.builder import build_scene
from real_labs.layout_audit import doorway_obstacle_warnings, model_obstacles
from real_labs.presentation import fixture_emitter_bounds

SCENE = "berkeley_richmond_towing_tank"


def is_context(name):
    return "berkeley_context_" in name


def boxes(model, data):
    rotation = data.geom_xmat.reshape(-1, 3, 3)
    centers = data.geom_xpos + np.einsum("nij,nj->ni", rotation, model.geom_aabb[:, :3])
    half = np.einsum("nij,nj->ni", abs(rotation), model.geom_aabb[:, 3:])
    return centers, half


def unsupported(model, data):
    """Return new pieces lacking a surface path to an existing fixed anchor.

    Query cap and attachment tolerance must be distinct. Aligned cylinder/box
    queries use the existing analytic bound for known native distance errors.
    Intentional fixed-member laps are acceptable; this is an attachment graph,
    not a claim that all fixed overlaps are absent.
    """
    enabled = np.flatnonzero(
        ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        & (model.body_weldid[model.geom_bodyid] == 0)
    )
    context = {i for i in enabled if is_context(model.geom(i).name)}
    reached = set(enabled) - context
    center, half = boxes(model, data)
    edges = []
    for a in sorted(context):
        gaps = np.linalg.norm(
            np.maximum(abs(center[a] - center[enabled]) - half[a] - half[enabled], 0),
            axis=1,
        )
        for b in enabled[gaps <= 0.00005]:
            if b == a or (b in context and b < a):
                continue
            gap = mujoco.mj_geomDistance(model, data, int(a), int(b), 0.005, None)
            ta, tb = model.geom_type[a], model.geom_type[b]
            if ta == mujoco.mjtGeom.mjGEOM_CYLINDER and tb == mujoco.mjtGeom.mjGEOM_BOX:
                analytic = cylinder_box_gap(model, data, a, b)
                if analytic is not None:
                    gap = analytic
            elif (
                tb == mujoco.mjtGeom.mjGEOM_CYLINDER and ta == mujoco.mjtGeom.mjGEOM_BOX
            ):
                analytic = cylinder_box_gap(model, data, b, a)
                if analytic is not None:
                    gap = analytic
            if gap <= 0.00005:
                edges.append((a, b))
    while True:
        more = {
            b if a in reached else a
            for a, b in edges
            if (a in reached) != (b in reached)
        }
        if not more:
            break
        reached.update(more)
    return {model.geom(i).name for i in context - reached}


class BerkeleyHallContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene(SCENE)
        cls.root = root
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def state(self):
        m = copy(self.model)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d

    def visible_rgba(self, model, data, geom):
        """Inspect MuJoCo's resolved material color without an OpenGL context."""
        scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 100)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 1
        mujoco.mjv_updateScene(model, data, option, None, mujoco.MjvCamera(),
                              mujoco.mjtCatBit.mjCAT_ALL, scene)
        matching = [scene.geoms[i].rgba.copy() for i in range(scene.ngeom)
                    if scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and scene.geoms[i].objid == geom]
        self.assertEqual(len(matching), 1)
        return matching[0]

    def test_all_new_shapes_contact_and_have_real_fixed_attachment_paths(self):
        m, d = self.state()
        ids = [i for i in range(m.ngeom) if is_context(m.geom(i).name)]
        self.assertGreater(len(ids), 100)
        self.assertTrue(np.all(m.geom_contype[ids] == 1))
        self.assertTrue(np.all(m.geom_conaffinity[ids] == 1))
        self.assertTrue(np.all(m.body_weldid[m.geom_bodyid[ids]] == 0))
        self.assertEqual(unsupported(m, d), set())
        # A real detached exterior must fail, even if all its masks stay on.
        case = m.geom("berkeley_context_bench_case").id
        m.geom_pos[case, 2] += 0.1
        mujoco.mj_forward(m, d)
        self.assertIn("berkeley_context_bench_case", unsupported(m, d))

    def test_complete_original_instrument_and_camera_arrays_are_preserved(self):
        with patch("real_labs.berkeley_hall_context.add_context", lambda *_: None):
            root, _ = build_scene(SCENE)
        baseline = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        current = self.model
        self.assertEqual(baseline.nbody, current.nbody)
        for name in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "body_gravcomp",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
            "dof_damping",
            "dof_armature",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "actuator_ctrlrange",
            "cam_pos",
            "cam_quat",
            "cam_fovy",
        ):
            np.testing.assert_array_equal(
                getattr(baseline, name), getattr(current, name), err_msg=name
            )
        for i in range(baseline.ngeom):
            old = baseline.geom(i)
            new = current.geom(old.name)
            for field in (
                "pos",
                "quat",
                "size",
                "rgba",
                "friction",
                "solref",
                "solimp",
                "contype",
                "conaffinity",
                "margin",
            ):
                if field == "rgba" and old.name.startswith(APPEARANCE_PREFIXES):
                    continue  # Explicit source-toned fixed exterior colors only.
                np.testing.assert_array_equal(
                    getattr(old, field),
                    getattr(new, field),
                    err_msg=old.name + ":" + field,
                )
        self.assertEqual(current.nu, 2)
        self.assertEqual(current.nv, 10)
        np.testing.assert_array_equal(
            current.camera("source").pos, current.camera("reference").pos
        )
        np.testing.assert_array_equal(
            current.camera("source").quat, current.camera("reference").quat
        )
        self.assertEqual(
            current.camera("source").fovy, current.camera("reference").fovy
        )

    def test_dense_source_relief_and_water_are_physically_honest(self):
        m, d = self.state()
        # Names retain side before fold; finite inward ribs must actually be
        # visible from the aisle, not outside an opaque backing sheet.
        folds = [
            i
            for i in range(m.ngeom)
            if "_wall_" in m.geom(i).name and "fold_" in m.geom(i).name
        ]
        self.assertGreater(len(folds), 700)
        for i in folds:
            self.assertLess(abs(d.geom_xpos[i, 0]) - m.geom_size[i, 0], 3.251)
            self.assertGreater(abs(d.geom_xpos[i, 0]) + m.geom_size[i, 0], 3.251)
        roof = [
            i for i in range(m.ngeom) if "berkeley_context_roof_fold_" in m.geom(i).name
        ]
        self.assertEqual(len(roof), 778)
        for i in roof:
            self.assertGreater(m.geom_size[i, 2], 0.005)
        water = m.geom("towing_facility__box_79").id
        np.testing.assert_allclose(
            m.geom_size[water], [1.198, 31.995, 0.003], atol=1e-14
        )
        np.testing.assert_allclose(
            self.visible_rgba(m, d, water), [.17, .22, .11, .92], atol=2e-8
        )
        self.assertEqual(
            m.material(int(m.geom_matid[water])).name, "towing_facility__mat_turbid_water"
        )
        self.assertEqual((m.geom_contype[water], m.geom_conaffinity[water]), (0, 0))
        self.assertAlmostEqual(d.geom_xpos[water, 2], 1.67)

    def test_equipment_phase_really_assigns_only_fixed_tank_finishes(self):
        m, d = self.state()
        colors = {
            "towing_facility__tank_sidewall_4": (0.13, 0.19, 0.12, 1),
            "towing_facility__tank_sidewall_40": (0.13, 0.19, 0.12, 1),
            "towing_facility__tank_endwall_76": (0.13, 0.19, 0.12, 1),
            "towing_facility__tank_endwall_78": (0.13, 0.19, 0.12, 1),
            "towing_facility__service_walkway_81": (0.47, 0.44, 0.35, 1),
        }
        for name, color in colors.items():
            geom = m.geom(name).id
            np.testing.assert_allclose(m.geom_rgba[geom], color, rtol=0, atol=2e-8)
            self.assertEqual(m.geom_matid[geom], -1)
            self.assertEqual(m.body_weldid[m.geom_bodyid[geom]], 0)
            self.assertEqual((m.geom_contype[geom], m.geom_conaffinity[geom]), (1, 1))
        # Static green-brown water uses a material; the default per-geom color
        # is not its displayed RGBA. This does not enable fluid collision.
        water = m.geom("towing_facility__box_79").id
        self.assertGreaterEqual(m.geom_matid[water], 0)
        np.testing.assert_allclose(
            self.visible_rgba(m, d, water), [.17, .22, .11, .92], atol=2e-8
        )

    def test_context_clears_conservative_whole_travel_envelope(self):
        m, d = self.state()
        context = np.array([i for i in range(m.ngeom) if is_context(m.geom(i).name)])
        moving = np.flatnonzero(
            (m.body_weldid[m.geom_bodyid] != 0)
            & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        )
        center, half = boxes(m, d)
        fixed_min, fixed_max = (
            center[context] - half[context],
            center[context] + half[context],
        )
        # Bounding spheres around each moving body's origin cover every wheel
        # angle. Add full 8 m carriage travel, full 350 mm immersion even for
        # bodies that do not immerse, and 10 mm extra tracking/journal padding.
        # This is conservative spatial separation, not hydrodynamic validation.
        minimum = float("inf")
        for i in moving:
            body = int(m.geom_bodyid[i])
            origin = d.xpos[body]
            radius = np.linalg.norm(d.geom_xpos[i] - origin) + m.geom_rbound[i]
            low = origin - radius + np.array([0, -0.01, -0.36])
            high = origin + radius + np.array([0, 8.01, 0.01])
            separation = np.maximum(np.maximum(low - fixed_max, fixed_min - high), 0)
            minimum = min(minimum, float(np.linalg.norm(separation, axis=1).min()))
        self.assertGreater(minimum, 0.05)
        # Also execute the existing uncensored direct checker at coupled ends
        # and intermediate configurations with actual passive wheel coordinates.
        checker = TankClearance(m, d)
        travel = m.joint("towing_facility__joint_carriage_travel").qposadr[0]
        immersion = m.joint("towing_facility__joint_model_immersion").qposadr[0]
        for q, z in itertools.product((0, 2.17, 5.61, 8), (0, 0.193, 0.35)):
            d.qpos[travel], d.qpos[immersion] = q, z
            for side, end in itertools.product(range(2), repeat=2):
                key = f"towing_facility__berkeley_wheel_{side}_{end}"
                d.qpos[m.joint(key + "_spin").qposadr[0]] = 17.31 + side
                d.qpos[m.joint(key + "_radial_play").qposadr[0]] = 0.0003
            mujoco.mj_forward(m, d)
            checker.check(d)

    def test_new_portal_keeps_the_existing_entry_approach_open(self):
        m, d = self.state()
        self.assertEqual(
            doorway_obstacle_warnings(self.manifest, model_obstacles(m, d)), []
        )
        # The first candidate's genuine doorway obstruction must remain
        # detectable, rather than disappear through masks or a weaker audit.
        column = m.geom("berkeley_context_end_column_-34.0_-1").id
        m.geom_pos[column, 1] += 0.6
        mujoco.mj_forward(m, d)
        warnings = doorway_obstacle_warnings(self.manifest, model_obstacles(m, d))
        self.assertTrue(
            any(m.geom(column).name in item["intersecting_bounds"] for item in warnings)
        )

    def test_existing_luminaire_role_has_clear_downward_emitters(self):
        m, d = self.state()
        fixtures = [
            i
            for i in range(m.ngeom)
            if m.geom(i).name.startswith("arch_luminaire_berkeley_context_")
        ]
        self.assertEqual(len(fixtures), 32)
        for i in fixtures:
            rotation = d.geom_xmat[i].reshape(3, 3)
            center, half = m.geom_aabb[i, :3], m.geom_aabb[i, 3:]
            corners = [
                d.geom_xpos[i] + rotation @ (center + np.array(sign) * half)
                for sign in itertools.product((-1, 1), repeat=3)
            ]
            emitter, footprint = fixture_emitter_bounds(corners)
            self.assertAlmostEqual(min(v[2] for v in corners) - emitter[2], 0.015)
            self.assertGreater(min(footprint), 0)
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                m, d, np.array(emitter), np.array([0.0, 0.0, -1.0]), None, 1, -1, hit
            )
            self.assertGreater(distance, 0.75, m.geom(i).name)
            self.assertGreaterEqual(hit[0], 0)
            self.assertNotIn("lamp_", m.geom(int(hit[0])).name)

    def test_aisle_floor_contacts_and_existing_source_camera_are_clear(self):
        m, d = self.state()
        cabinet = m.geom("berkeley_context_cabinet_top")
        self.assertGreater(d.geom_xpos[cabinet.id, 0] - cabinet.size[0] - 0.6, 0.85)
        for i in range(m.ngeom):
            name = m.geom(i).name
            if "berkeley_context_cabinet_foot_" in name:
                self.assertAlmostEqual(d.geom_xpos[i, 2] - m.geom_size[i, 2], 0.85)
            elif "berkeley_context_stool_foot_" in name:
                self.assertAlmostEqual(d.geom_xpos[i, 2] - m.geom_size[i, 1], 0.85)
        # Every new object must clear a continuous 0.85 m wide aisle above
        # the authored raised walkway, not only the cabinet's worktop.
        centers, half = boxes(m, d)
        corridor_low = np.array([0.6, -32.0, 0.85])
        corridor_high = np.array([1.45, 32.0, 2.50])
        for i in range(m.ngeom):
            if is_context(m.geom(i).name):
                gap = np.maximum(
                    np.maximum(
                        corridor_low - centers[i] - half[i],
                        centers[i] - half[i] - corridor_high,
                    ),
                    0,
                )
                self.assertGreater(np.linalg.norm(gap), 0.02, m.geom(i).name)
        camera = m.camera("reference").id
        origin = d.cam_xpos[camera]
        direction = -d.cam_xmat[camera].reshape(3, 3)[:, 2]
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(m, d, origin, direction, None, 1, -1, hit)
        self.assertGreater(distance, 5.0)
        self.assertFalse(is_context(m.geom(int(hit[0])).name))


if __name__ == "__main__":
    unittest.main()
