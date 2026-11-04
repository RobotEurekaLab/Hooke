"""Mounting paths, filtered-pair clearance and truthful glovebox access limits."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.glovebox_details import RUNNING_CLEARANCE_M
from real_labs.runtime import InstrumentSession


MODULES = ("glovebox_west_1", "glovebox_west_2", "glovebox_east_1", "glovebox_east_2")


def session():
    root, manifest = build_scene("eth_mougel_inorganic")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def named(model, module, suffix):
    return model.geom(f"{module}__glovebox_{suffix}").id


def distance(s, a, b):
    return mujoco.mj_geomDistance(s.model, s.data, a, b, 10, None)


def fixed_at(model, module, pos, size):
    body = model.body(module).id
    return next(
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        and np.allclose(model.geom_pos[g], pos)
        and np.allclose(model.geom_size[g], size)
    )


class MougelGloveboxGeometryTests(unittest.TestCase):
    def test_original_motion_sample_and_camera_contract(self):
        s = session()
        self.assertEqual((s.model.njnt, s.model.nu), (8, 8))
        for module, xy in zip(
            MODULES, ((-1.9, -2.25), (-1.9, 2.4), (1.9, -2.25), (1.9, 2.4))
        ):
            lid = s.controls[module, "pass_through_lid"]
            tray = s.controls[module, "vial_carriage"]
            np.testing.assert_allclose([lid.lower, lid.upper], [0, 1.25])
            np.testing.assert_allclose([tray.lower, tray.upper], [-0.24, 0.24])
            site = s.model.site(f"{module}__site_inert_vial").id
            np.testing.assert_allclose(s.data.site_xpos[site], [*xy, 1.13])
            item = s.equipment[module]
            self.assertEqual(item["sample_attachment"], "clamped")
            np.testing.assert_allclose(item["sample_size_m"], [0.072, 0.072, 0.11])
            self.assertIn("0.5 mm", item["assembly_adaptations"]["authored"])
            self.assertIn("does not reach", item["assembly_adaptations"]["limits"])
        self.assertEqual(
            {s.model.camera(i).name for i in range(s.model.ncam)},
            {"overview", "interior", "workstation", "rear_gloveboxes"},
        )

    def test_load_path_and_open_bearing_clearance_through_full_travel(self):
        s = session()
        m = s.model
        for module in MODULES:
            deck = fixed_at(m, module, (0, 0, 0.93), (1.32, 0.55, 0.035))
            plate = named(m, module, "carriage_plate")
            for row, end in itertools.product(range(2), repeat=2):
                pedestal = named(m, module, f"rail_{row}_pedestal_{end}")
                guide = named(m, module, f"guide_{row}")
                self.assertAlmostEqual(distance(s, pedestal, deck), 0, places=6)
                self.assertAlmostEqual(distance(s, pedestal, guide), 0, places=6)
                self.assertEqual(m.geom_contype[guide], 1)
            roof = fixed_at(m, module, (0, 0, 2.045), (1.35, 0.57, 0.075))
            accessory = fixed_at(m, module, (-0.82, 0.2, 2.235), (0.23, 0.28, 0.11))
            pad = named(m, module, "upper_accessory_pad")
            self.assertAlmostEqual(distance(s, pad, roof), 0, places=6)
            self.assertAlmostEqual(distance(s, pad, accessory), 0, places=6)
            control = s.controls[module, "vial_carriage"]
            for q in np.linspace(control.lower, control.upper, 25):
                s.data.qpos[m.jnt_qposadr[control.joint]] = q
                mujoco.mj_forward(m, s.data)
                for row, shoe in itertools.product(range(2), repeat=2):
                    prefix = f"{module}__glovebox_bearing_{row}_{shoe}_proxy_"
                    proxies = [
                        g for g in range(m.ngeom) if m.geom(g).name.startswith(prefix)
                    ]
                    self.assertEqual(len(proxies), 24)
                    self.assertLess(min(distance(s, plate, g) for g in proxies), 0)
                    guide = named(m, module, f"guide_{row}")
                    self.assertAlmostEqual(
                        min(distance(s, guide, g) for g in proxies),
                        RUNNING_CLEARANCE_M,
                        places=6,
                    )
                    self.assertGreater(min(distance(s, deck, g) for g in proxies), 0.03)

    def test_hinge_is_connected_with_a_real_pin_bore(self):
        s = session()
        m = s.model
        for module in MODULES:
            mount = named(m, module, "hinge_mount")
            pin = named(m, module, "hinge_pin")
            barrel = [named(m, module, f"barrel_collision_{i}") for i in range(32)]
            self.assertLess(min(distance(s, mount, g) for g in barrel), 0)
            for side in (-1, 1):
                lug = named(m, module, f"hinge_lug_{side}")
                self.assertLess(distance(s, lug, mount), 0)
                self.assertLess(distance(s, lug, pin), 0)
            proxies = [
                g
                for g in range(m.ngeom)
                if m.geom(g).name.startswith(f"{module}__glovebox_hinge_hub_proxy_")
            ]
            lid = named(m, module, "lid_disc")
            self.assertLess(min(distance(s, lid, g) for g in proxies), 0)
            control = s.controls[module, "pass_through_lid"]
            for q in np.linspace(0, 1.25, 33):
                s.data.qpos[m.jnt_qposadr[control.joint]] = q
                mujoco.mj_forward(m, s.data)
                self.assertAlmostEqual(
                    min(distance(s, pin, g) for g in proxies),
                    RUNNING_CLEARANCE_M,
                    places=6,
                )

    def test_full_travel_direct_geometry_includes_parent_filtered_pairs(self):
        s = session()
        m = s.model
        for control in s.controls.values():
            body = m.jnt_bodyid[control.joint]
            moving = np.flatnonzero((m.geom_bodyid == body) & (m.geom_contype != 0))
            fixed = np.flatnonzero((m.geom_bodyid != body) & (m.geom_contype != 0))
            for q in np.linspace(control.lower, control.upper, 33):
                s.reset()
                s.data.qpos[m.jnt_qposadr[control.joint]] = q
                mujoco.mj_forward(m, s.data)
                for first in moving:
                    separation = np.linalg.norm(
                        s.data.geom_xpos[fixed] - s.data.geom_xpos[first], axis=1
                    )
                    near = fixed[
                        separation < m.geom_rbound[fixed] + m.geom_rbound[first] + 0.001
                    ]
                    for second in near:
                        gap = distance(s, int(first), int(second))
                        self.assertGreaterEqual(
                            gap,
                            -1e-5,
                            (q, m.geom(first).name, m.geom(second).name, gap),
                        )

    def test_passage_stays_open_and_front_gloves_stay_presentation_only(self):
        s = session()
        m = s.model
        collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        m.geom_group[:] = collides.astype(np.int32)
        group = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        for module in MODULES:
            body = m.body(module).id
            rotation = s.data.xmat[body].reshape(3, 3)
            origin = s.data.xpos[body].copy()
            hit = np.array([-1], dtype=np.int32)
            control = s.controls[module, "pass_through_lid"]
            for value, closed in ((0, True), (1.25, False)):
                s.data.qpos[m.jnt_qposadr[control.joint]] = value
                mujoco.mj_forward(m, s.data)
                for y, z in itertools.product(
                    np.linspace(-0.036, 0.036, 7), np.linspace(-0.055, 0.055, 7)
                ):
                    start = origin + rotation @ np.array([2.8, y, 1.36 + z])
                    gap = mujoco.mj_ray(
                        m,
                        s.data,
                        start,
                        rotation @ np.array([-1.0, 0, 0]),
                        group,
                        True,
                        -1,
                        hit,
                    )
                    self.assertEqual(
                        0 <= gap < 1.48, closed, (module, value, y, z, gap)
                    )
                    # The presentation meshes must also have real apertures;
                    # a convex cylinder would conceal the valid collision bore.
                    for suffix in ("barrel_shell", "flange_0_shell", "flange_1_shell"):
                        shell = named(m, module, suffix)
                        visual_hit = mujoco.mj_rayMesh(
                            m,
                            s.data,
                            shell,
                            start,
                            rotation @ np.array([-1.0, 0, 0]),
                        )
                        self.assertEqual(visual_hit, -1, (module, suffix, y, z))
            for glove in range(2):
                prefix = f"{module}__glovebox_glove_{glove}"
                geoms = [g for g in range(m.ngeom) if m.geom(g).name.startswith(prefix)]
                self.assertTrue(geoms)
                for g in geoms:
                    self.assertEqual(int(m.geom_bodyid[g]), body)
                    self.assertEqual((m.geom_contype[g], m.geom_conaffinity[g]), (0, 0))
                    mesh = m.geom_dataid[g]
                    vertices = m.mesh_vert[
                        m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh]
                        + m.mesh_vertnum[mesh]
                    ]
                    world = (
                        vertices @ s.data.geom_xmat[g].reshape(3, 3).T
                        + s.data.geom_xpos[g]
                    )
                    if "west" in module:
                        self.assertLessEqual(world[:, 0].max(), -0.652)
                    else:
                        self.assertGreaterEqual(world[:, 0].min(), 0.652)
            # Retained collision-solid pane is a deliberate capability boundary.
            for x in (-0.60, 0.60):
                start = origin + rotation @ np.array([x, -1.5, 1.44])
                gap = mujoco.mj_ray(
                    m,
                    s.data,
                    start,
                    rotation @ np.array([0.0, 1, 0]),
                    group,
                    True,
                    -1,
                    hit,
                )
                self.assertTrue(0 < gap < 1.1)
                self.assertEqual(m.geom_type[hit[0]], mujoco.mjtGeom.mjGEOM_BOX)

    def test_measured_full_reversals_contacts_hold_and_reset(self):
        s = session()
        m = s.model
        for key, control in s.controls.items():
            s.reset()
            for fraction in (0.15, 0.85, 0.25, 0, 1, 0.5):
                target = control.lower + fraction * (control.upper - control.lower)
                s.data.ctrl[control.actuator] = target
                for _ in range(round(2 / m.opt.timestep)):
                    mujoco.mj_step(m, s.data)
                    self.assertFalse([c for c in s.data.contact if c.dist < -1e-5], key)
                    self.assertTrue(np.isfinite(s.data.qpos).all())
                self.assertLess(
                    abs(s._measured(control) - target),
                    0.025 * (control.upper - control.lower),
                )
                for other in s.controls.values():
                    if other.actuator != control.actuator:
                        self.assertLess(abs(s._measured(other)), 1e-5)
            self.assertFalse(any(w.number for w in s.data.warning))
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, m.qpos0)
        self.assertEqual(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
