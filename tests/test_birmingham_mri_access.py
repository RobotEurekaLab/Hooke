"""Open MRI-shaped phantom-table supports and coupled dry motion checks."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.birmingham_mri_geometry import PREFIX

SCENE = "birmingham_chbh_mri"


def session():
    root, manifest = build_scene(SCENE)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(m, manifest)


def pairs(m):
    enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
    weld = m.body_weldid[m.geom_bodyid]
    return np.array(
        [
            (a, b)
            for i, a in enumerate(enabled)
            for b in enabled[i + 1 :]
            if weld[a] != weld[b] and (weld[a] or weld[b])
        ],
        dtype=int,
    )


def clearance(m, d, p):
    r = d.geom_xmat.reshape(-1, 3, 3)
    c = d.geom_xpos + np.einsum("nij,nj->ni", r, m.geom_aabb[:, :3])
    h = np.einsum("nij,nj->ni", abs(r), m.geom_aabb[:, 3:])
    a, b = p.T
    lower = np.linalg.norm(np.maximum(abs(c[a] - c[b]) - h[a] - h[b], 0), axis=1)
    selected = p[lower < 0.002 + 1e-10]
    return min(
        [0.002]
        + [
            float(mujoco.mj_geomDistance(m, d, int(a), int(b), 0.002, None))
            for a, b in selected
        ]
    ), len(selected)


def distance(m, d, a, b):
    return float(
        mujoco.mj_geomDistance(
            m, d, m.geom(PREFIX + a).id, m.geom(PREFIX + b).id, 1, None
        )
    )


def loaded_route(s):
    m, d = s.model, s.data
    check = pairs(m)
    executed = queries = 0
    minimum = 0.002
    legs = []

    def state():
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        v = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, v, spec)
        return {
            "time_s": float(d.time),
            "qpos": d.qpos.tolist(),
            "qvel": d.qvel.tolist(),
            "ctrl": d.ctrl.tolist(),
            "qfrc_applied": d.qfrc_applied.tolist(),
            "integration_state": v.tolist(),
            "state_kind": int(spec),
        }

    initial = state()
    # Full endpoints and reversals, with small explicit generalized loads.
    # The retained phantom moves with the original1.2kg ideal-joint carriage.
    for target, force in [
        ((0.06, 0.85), (0.18, -0.30)),
        ((0, 0), (-0.18, 0.30)),
        ((0.009, 0.7225), (0.18, 0.30)),
        ((0.051, 0.2125), (-0.18, -0.30)),
        ((0, 0), (0, 0)),
    ]:
        before = state()
        d.qfrc_applied[:] = force
        d.ctrl[:] = target
        for _ in range(round(3 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            executed += 1
            gap, count = clearance(m, d, check)
            queries += count
            minimum = min(minimum, gap)
            if gap < 0.0005998:
                raise AssertionError(("clearance", gap, d.qpos.copy()))
            if any(c.dist < -1e-7 for c in d.contact):
                raise AssertionError("native penetration")
        error = np.abs(d.qpos - target)
        if any(error > np.array([0.06, 0.85]) * 0.025):
            raise AssertionError(("tracking", error))
        legs.append(
            {
                "target": list(target),
                "generalized_force": list(force),
                "before": before,
                "after": state(),
                "error_m": error.tolist(),
            }
        )
    return {
        "passed": True,
        "executed_steps": executed,
        "checked_steps": executed,
        "direct_queries": queries,
        "pair_count": len(check),
        "minimum_clearance_lower_bound_m": minimum,
        "bound_cap_m": 0.002,
        "initial_state": initial,
        "legs": legs,
        "final_state": state(),
        "reset_performed": False,
        "scope": "Small authored generalized loads on an inert retained phantom table, ideal bearings, no MRI or patient procedure",
    }


class MRIRepairTests(unittest.TestCase):
    def test_preserved_controls_and_phantom(self):
        s = session()
        m = s.model
        for key, pos, axis, limits, mass in [
            ("table_height", (0, 0, 0.72), (0, 0, 1), (0, 0.06), 1.5),
            ("table_insert", (0, -0.92, 0.15), (0, 1, 0), (0, 0.85), 1.2),
        ]:
            c = s.controls["scanner", key]
            j = c.joint
            i = m.jnt_bodyid[j]
            np.testing.assert_allclose(m.body_pos[i], pos)
            np.testing.assert_allclose(m.jnt_axis[j], axis)
            np.testing.assert_allclose(m.jnt_range[j], limits)
            self.assertEqual(m.body_mass[i], mass)
            np.testing.assert_allclose(m.body_inertia[i], [mass * 0.006] * 3)
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], 300)
            np.testing.assert_allclose(m.actuator_forcerange[c.actuator], [-120, 120])
        np.testing.assert_allclose(
            m.geom(PREFIX + "inert_phantom").size[:2], [0.093, 0.147]
        )
        np.testing.assert_allclose(
            m.site("scanner__site_inert_mri_phantom").pos, [0, 0.38, 0.15]
        )
        self.assertEqual(
            [m.camera(i).name for i in range(m.ncam)],
            ["overview", "workstation", "interior"],
        )

    def test_open_stem_bearings_and_connected_retainers(self):
        s = session()
        m, d = s.model, s.data
        for side in (-1, 1):
            self.assertLessEqual(
                distance(m, d, "table_foot", f"lift_wall_x_{side}"), 1e-8
            )
            self.assertLessEqual(
                distance(m, d, "table_foot", f"lift_wall_y_{side}"), 1e-8
            )
            for i in range(2):
                self.assertLessEqual(
                    distance(m, d, f"retaining_post_{i}_{side}", f"cradle_side_{side}"),
                    1e-8,
                )
                self.assertLessEqual(
                    distance(
                        m, d, f"retaining_post_{i}_{side}", f"retaining_bridge_{i}"
                    ),
                    1e-8,
                )
                self.assertLessEqual(
                    distance(
                        m, d, f"retaining_fastener_{i}_{side}", f"retaining_bridge_{i}"
                    ),
                    1e-8,
                )
        for i in range(2):
            for j in range(2):
                self.assertLessEqual(
                    distance(m, d, f"rail_end_mount_{i}_{j}", f"rail_bed_{i}"), 1e-8
                )
                self.assertLessEqual(
                    distance(m, d, f"rail_end_mount_{i}_{j}", f"rail_{i}"), 1e-8
                )
                self.assertLessEqual(
                    distance(m, d, f"carriage_bearing_{i}_{j}_crown", "table_board"),
                    1e-8,
                )
        # Real centerline bore remains open in the presentation mesh too.
        for i, j in itertools.product(range(2), repeat=2):
            g = m.geom(PREFIX + f"carriage_bearing_{i}_{j}_shell").id
            rail = m.geom(PREFIX + f"rail_{i}").id
            axis = d.geom_xmat[rail].reshape(3, 3)[:, 2]
            self.assertLess(
                mujoco.mj_rayMesh(m, d, g, d.geom_xpos[rail] - 0.8 * axis, axis), 0
            )

    def test_full_coupled_travel_and_enabled_obstacle_detection(self):
        s = session()
        m, d = s.model, s.data
        p = pairs(m)
        for height, insert in itertools.product(
            np.linspace(0, 0.06, 13), np.linspace(0, 0.85, 17)
        ):
            d.qpos[:] = [height, insert]
            mujoco.mj_forward(m, d)
            self.assertGreaterEqual(clearance(m, d, p)[0], 0.0005998, (height, insert))
            # 20 mm minimum physical engagement of the rectangular stem.
            self.assertGreaterEqual(0.64 - (0.72 + height - 0.065 - 0.095), 0.01999999)
            for local_y in (-0.50, -0.26):
                self.assertGreater(-0.92 + insert + local_y - 0.045, -1.83 + 0.020)
                self.assertLess(-0.92 + insert + local_y + 0.045, -0.21 - 0.020)
        g = m.geom(PREFIX + "lift_wall_x_1").id
        m.geom_pos[g, 0] -= 0.003
        mujoco.mj_forward(m, d)
        self.assertLess(clearance(m, d, p)[0], 0)

    def test_loaded_actual_full_range_return(self):
        s = session()
        r = loaded_route(s)
        self.assertEqual(r["checked_steps"], round(15 / s.model.opt.timestep))
        self.assertLess(np.max(abs(s.data.qpos)), 1e-6)
        self.assertAlmostEqual(s.data.time, 15, places=8)


if __name__ == "__main__":
    unittest.main()
