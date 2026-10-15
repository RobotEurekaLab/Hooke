"""Source-informed fixed context with preserved Whittle dry-traverse physics."""

from itertools import product
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene

SCENE = "cambridge_whittle_low_speed_turbine"


def compiled(context=True):
    if context:
        root, manifest = build_scene(SCENE)
    else:
        with patch("real_labs.whittle_context.add_apparatus_context"), patch(
            "real_labs.whittle_context.add_hall_context"
        ):
            root, manifest = build_scene(SCENE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, manifest


def bounds(model, data):
    rotation = data.geom_xmat.reshape(-1, 3, 3)
    center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, model.geom_aabb[:, :3])
    half = np.einsum("nij,nj->ni", abs(rotation), model.geom_aabb[:, 3:])
    return center - half, center + half


def context_ids(model):
    return np.array(
        [
            i
            for i in range(model.ngeom)
            if "whittle_context" in model.geom(i).name
            and model.geom_contype[i] | model.geom_conaffinity[i]
        ],
        dtype=int,
    )


def disconnected(model, data):
    """Contact graph to original fixed geometry; query cap exceeds tolerance."""
    enabled = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
    new = set(map(int, context_ids(model)))
    reached = {model.geom("arch_floor").id}
    # An old fixed collider can itself be unsupported. Traverse all fixed
    # enabled geometry from the actual floor, then assess the new components.
    left = {
        int(i) for i in enabled if model.body_weldid[model.geom_bodyid[i]] == 0
    } - reached
    low, high = bounds(model, data)
    while left:
        attached = set()
        candidates = np.array(sorted(reached))
        for i in left:
            near = candidates[
                np.max(
                    np.maximum(low[candidates] - high[i], low[i] - high[candidates]),
                    axis=1,
                )
                <= 1e-7
            ]
            if any(
                mujoco.mj_geomDistance(model, data, i, int(j), 0.001, None) <= 1e-7
                for j in near
            ):
                attached.add(i)
        if not attached:
            break
        reached |= attached
        left -= attached
    return [model.geom(i).name for i in sorted(left & new)]


class WhittleContextTests(unittest.TestCase):
    def test_original_physics_and_cameras_preserved(self):
        before, _, _ = compiled(False)
        after, _, _ = compiled()
        for prefix in ("body_", "jnt_", "dof_", "actuator_", "cam_"):
            for key in dir(before):
                old = getattr(before, key)
                if (
                    key.startswith(prefix)
                    and isinstance(old, np.ndarray)
                    and key
                    not in {
                        "body_bvhadr",
                        "body_bvhnum",
                        "body_geomadr",
                        "body_geomnum",
                    }
                ):
                    np.testing.assert_array_equal(old, getattr(after, key), err_msg=key)
        physical_fields = (
            "type",
            "size",
            "pos",
            "quat",
            "contype",
            "conaffinity",
            "friction",
            "solref",
            "solimp",
            "margin",
            "gap",
            "bodyid",
        )
        for i in range(before.ngeom):
            j = after.geom(before.geom(i).name).id
            for field in physical_fields:
                np.testing.assert_array_equal(
                    getattr(before, "geom_" + field)[i],
                    getattr(after, "geom_" + field)[j],
                    err_msg=before.geom(i).name + ":" + field,
                )
        self.assertEqual(before.nbody, after.nbody)
        self.assertEqual(after.njnt, 2)
        self.assertGreater(len(context_ids(after)), 1200)

    def test_all_new_enabled_shapes_are_supported_and_detachment_fails(self):
        model, data, _ = compiled()
        self.assertEqual(disconnected(model, data), [])
        detached = model.geom("whittle_context_control_blank_module_0").id
        model.geom_pos[detached, 0] += 10
        mujoco.mj_forward(model, data)
        self.assertIn(model.geom(detached).name, disconnected(model, data))

    def test_reducer_is_required_and_services_stay_outside_solids(self):
        model, data, _ = compiled()
        for i in context_ids(model):
            if "inlet_reducer" in model.geom(int(i)).name:
                model.geom_contype[i] = model.geom_conaffinity[i] = 0
        mujoco.mj_forward(model, data)
        self.assertTrue(any("band_1" in name for name in disconnected(model, data)))
        model, data, _ = compiled()
        casing = [
            i
            for i in range(model.ngeom)
            if model.geom(i).name.startswith("primary_apparatus__box_")
            and model.geom_group[i] == 3
            and model.geom_contype[i]
        ]
        self.assertEqual(len(casing), 288)
        for index in range(4):
            lead = model.geom(f"whittle_context_control_harness_{index}").id
            self.assertGreater(
                min(
                    mujoco.mj_geomDistance(model, data, lead, i, 1, None)
                    for i in casing
                ),
                0.005,
            )
        column = model.geom("whittle_context_hall_column_3.36").id
        duct = model.geom("arch_roof_whittle_duct").id
        self.assertGreater(
            mujoco.mj_geomDistance(model, data, column, duct, 1, None), 0.039
        )

    def test_named_stair_and_service_connections(self):
        m, d, _ = compiled()
        pairs = []
        for x in (-2.42, -1.68):
            for i in range(4):
                pairs.append((f"stair_tread_bearing_{x}_{i}", f"stair_stringer_{x}"))
            for y in (-0.58, 0.39):
                pairs.append((f"stair_rail_post_{x}_{y}", f"stair_stringer_{x}"))
                pairs.append((f"stair_rail_post_{x}_{y}", f"stair_handrail_{x}"))
            pairs.extend(
                [
                    (f"stair_stringer_{x}", f"stair_foot_front_{x}"),
                    (f"stair_stringer_{x}", f"stair_rear_leg_{x}"),
                ]
            )
        for first, second in pairs:
            gap = mujoco.mj_geomDistance(
                m,
                d,
                m.geom("whittle_context_" + first).id,
                m.geom("whittle_context_" + second).id,
                1,
                None,
            )
            self.assertLessEqual(gap, 1e-7, (first, second, gap))
        for x in (1.28, 2.38):
            hanger = m.geom(f"arch_roof_whittle_context_crossduct_hanger_{x}").id
            duct = m.geom("arch_roof_whittle_context_crossduct").id
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, hanger, duct, 1, None), 1e-7
            )

    def test_screen_is_open_connected_enabled_weave(self):
        m, d, _ = compiled()
        wires = [i for i in context_ids(m) if "screen_" in m.geom(int(i)).name]
        self.assertEqual(len(wires), 386)
        for i in wires:
            self.assertEqual(m.geom_type[i], mujoco.mjtGeom.mjGEOM_CAPSULE)
            self.assertAlmostEqual(m.geom_size[i, 0], 0.0012)
        vertical = m.geom("whittle_context_screen_vertical_0").id
        horizontal = m.geom("whittle_context_screen_horizontal_0").id
        self.assertAlmostEqual(
            mujoco.mj_geomDistance(m, d, vertical, horizontal, 0.01, None),
            -0.0001,
            places=8,
        )
        # A midpoint between wires remains an opening through the screen plane.
        point = np.array([0.0075, -2.14, 1.9075])
        group = np.zeros(6, dtype=np.uint8)
        group[0] = 1
        # Explicit wire geometry, rather than scene hits behind it, proves the gap.
        for i in wires:
            local = d.geom_xmat[i].reshape(3, 3).T @ (point - d.geom_xpos[i])
            segment = np.array(
                [0, 0, np.clip(local[2], -m.geom_size[i, 1], m.geom_size[i, 1])]
            )
            self.assertGreater(
                np.linalg.norm(local - segment) - m.geom_size[i, 0], 0.003
            )

    def test_new_fixed_geometry_is_clear_of_complete_slide_envelope(self):
        m, d, _ = compiled()
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        moving = enabled[m.body_weldid[m.geom_bodyid[enabled]] != 0]
        new = context_ids(m)
        lows, highs = [], []
        for y, z in product((-0.23, 0.23), (-0.05, 0.06)):
            d.qpos[:] = [y, z]
            mujoco.mj_forward(m, d)
            low, high = bounds(m, d)
            lows.append(low[moving])
            highs.append(high[moving])
        swept_low, swept_high = np.min(lows, axis=0), np.max(highs, axis=0)
        separation = np.maximum(
            np.maximum(low[new, None] - swept_high, swept_low - high[new, None]), 0
        )
        self.assertGreater(np.linalg.norm(separation, axis=2).min(), 0.15)
        # Translational extrema enclose every allowed pose. Also query actual
        # parent-filtered pairs at an independent coupled interior pose.
        d.qpos[:] = [0.137, -0.021]
        mujoco.mj_forward(m, d)
        actual = min(
            mujoco.mj_geomDistance(m, d, int(a), int(b), 2, None)
            for a in new
            for b in moving
        )
        self.assertGreater(actual, 0.15)


if __name__ == "__main__":
    unittest.main()
