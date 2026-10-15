"""Passive photo details retain the validated PICE registration mechanism."""

import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.copenhagen_source_geometry import MACHINE_PREFIX, ROOM_PREFIX
from real_labs.runtime import InstrumentSession

SCENE = "copenhagen_pice_ice_workroom"


def load(with_context=True):
    if with_context:
        root, manifest = build_scene(SCENE)
    else:
        with patch("real_labs.copenhagen_source_geometry.add_machine_context"), patch(
            "real_labs.copenhagen_source_geometry.add_room_services"
        ):
            root, manifest = build_scene(SCENE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def new_geoms(model):
    return np.array(
        [
            g
            for g in range(model.ngeom)
            if model.geom(g).name.startswith((MACHINE_PREFIX, ROOM_PREFIX))
        ]
    )


def full_motion_envelope_gap(model, data):
    """Cover continuous hinge travel using a Lipschitz padding between samples.

    Moving geometry has one fixed-axis Z hinge or one affine Y slide. A point
    at radius r moves at most r*delta_angle, so a half-sample angular padding
    bounds all intermediate XY positions; Z is invariant. Slider endpoints
    bound all its intermediate positions. Include visual and enabled geometry.
    """
    low, high = [], []
    for angle in np.linspace(0, 1.1, 23):
        for feed in (0, 0.152):
            data.qpos[:] = (angle, feed)
            mujoco.mj_forward(model, data)
            rotation = data.geom_xmat.reshape(-1, 3, 3)
            center = data.geom_xpos + np.einsum(
                "nij,nj->ni", rotation, model.geom_aabb[:, :3]
            )
            half = np.einsum("nij,nj->ni", abs(rotation), model.geom_aabb[:, 3:])
            low.append(center - half)
            high.append(center + half)
    low, high = np.min(low, axis=0), np.max(high, axis=0)
    hinge = model.joint("ice_station__joint_wheel_guard").id
    guard = np.flatnonzero(model.geom_bodyid == model.jnt_bodyid[hinge])
    pivot = data.xanchor[hinge, :2]
    radius = np.linalg.norm(
        np.maximum(abs(low[guard, :2] - pivot), abs(high[guard, :2] - pivot)), axis=1
    )
    low[guard, :2] -= radius[:, None] * 0.025
    high[guard, :2] += radius[:, None] * 0.025
    moving = np.flatnonzero(model.body_weldid[model.geom_bodyid] != 0)
    added = new_geoms(model)
    gap = np.linalg.norm(
        np.maximum(
            np.maximum(
                low[added, None] - high[moving], low[moving] - high[added, None]
            ),
            0,
        ),
        axis=2,
    )
    i, j = np.unravel_index(gap.argmin(), gap.shape)
    return float(gap[i, j]), (model.geom(added[i]).name, model.geom(moving[j]).name)


class CopenhagenSourceContextTests(unittest.TestCase):
    def test_every_added_contact_surface_reaches_original_fixed_support(self):
        session = load()
        model, data = session.model, session.data
        added = set(new_geoms(model))
        self.assertGreater(len(added), 100)
        self.assertTrue(
            all(model.geom_contype[g] and model.geom_conaffinity[g] for g in added)
        )
        anchors = {
            g
            for g in range(model.ngeom)
            if g not in added
            and model.geom_contype[g]
            and model.body_weldid[model.geom_bodyid[g]] == 0
        }
        reached, pending = set(anchors), set(added)
        while pending:
            connected = {
                g
                for g in pending
                if any(
                    mujoco.mj_geomDistance(model, data, g, a, 1e-6, None) <= 1e-7
                    for a in reached
                )
            }
            if not connected:
                break
            reached.update(connected)
            pending.difference_update(connected)
        self.assertFalse(pending, [model.geom(g).name for g in pending])
        # The source's parked lead is actually on the floor, not floating above it.
        floor = model.geom("arch_floor").id
        for g in added:
            if "floor_lead_loop" in model.geom(g).name:
                self.assertAlmostEqual(
                    mujoco.mj_geomDistance(model, data, g, floor, 0.1, None),
                    0,
                    places=8,
                )

    def test_added_details_clear_continuous_guard_and_feed_travel(self):
        session = load()
        gap, pair = full_motion_envelope_gap(session.model, session.data)
        self.assertGreater(gap, 0.025, pair)

    def test_original_shapes_controls_and_actual_return_are_unchanged(self):
        old, new = load(False), load()
        a, b = old.model, new.model
        fields = (
            "geom_type",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_contype",
            "geom_conaffinity",
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "geom_margin",
            "geom_gap",
            "geom_bodyid",
            "geom_rgba",
            "geom_group",
        )
        for g in range(a.ngeom):
            name = a.geom(g).name
            other = b.geom(name).id
            for field in fields:
                np.testing.assert_array_equal(
                    getattr(a, field)[g],
                    getattr(b, field)[other],
                    err_msg=f"{name}:{field}",
                )
        for field in (
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "jnt_type",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
            "dof_damping",
            "dof_armature",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_trnid",
            "cam_pos",
            "cam_quat",
            "cam_fovy",
            "site_pos",
            "site_size",
        ):
            np.testing.assert_array_equal(
                getattr(a, field), getattr(b, field), err_msg=field
            )
        initial = new.data.qpos.copy()
        for control, target in (
            ("wheel_guard", 0),
            ("core_feed_v2", 0.08),
            ("core_feed_v2", 0),
            ("wheel_guard", 0.75),
        ):
            old.command("ice_station", control, target, duration=1.0)
            new.command("ice_station", control, target, duration=1.0)
            for field in ("qpos", "qvel", "ctrl", "qfrc_actuator", "qfrc_constraint"):
                np.testing.assert_array_equal(
                    getattr(old.data, field), getattr(new.data, field), err_msg=field
                )
            address = b.jnt_qposadr[new.controls["ice_station", control].joint]
            self.assertLess(abs(new.data.qpos[address] - target), 0.001)
        self.assertLess(np.max(abs(new.data.qpos - initial)), 0.001)
        self.assertFalse(any(new.data.warning.number))


if __name__ == "__main__":
    unittest.main()
