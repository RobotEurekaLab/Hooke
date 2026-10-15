"""Open Turner bearing/supports and measured bounded-speed specimen access."""

from copy import copy, deepcopy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.cambridge_motion import (
    AccessObserver,
    CambridgeAccess,
    inspect_cambridge,
    SPEED_LIMITS,
)
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class CambridgeAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("cambridge_schofield_turner")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_frames_drives_and_full_ranges_are_preserved(self):
        s = self.session()
        m = s.model
        np.testing.assert_array_equal(m.jnt_range, [[-0.2, 0.2], [-0.15, 0.15]])
        np.testing.assert_array_equal(m.jnt_axis, [[0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_equal(
            m.body("turner_rotor__body_maintenance_index").pos, [0, 0, 1.02]
        )
        np.testing.assert_array_equal(
            m.body("turner_rotor__body_specimen_carriage").pos, [4.04, 0, 0.272]
        )
        np.testing.assert_array_equal(m.actuator_forcerange, [[-350, 350], [-120, 120]])
        np.testing.assert_array_equal(m.actuator_gainprm[:, 0], [500, 220])
        self.assertEqual(m.njnt, 2)
        self.assertEqual(
            s.manifest["camera"]["workstation"],
            dict(position=[5.65, -2.45, 1.9], target=[4.05, 0, 1.15], fovy=48),
        )

    def test_full_combined_travel_direct_pairs_and_native_contacts(self):
        s = self.session()
        observer = AccessObserver(s)
        for q in product(np.linspace(-0.2, 0.2, 9), np.linspace(-0.15, 0.15, 13)):
            s.data.qpos[:] = q
            mujoco.mj_forward(s.model, s.data)
            observer.check(s.model, s.data)
            self.assertFalse([c for c in s.data.contact if c.dist < -1e-7])
        self.assertEqual(observer.checked, 117)
        self.assertGreaterEqual(observer.minimum, 0.00049)
        self.assertGreater(len(observer.pairs), 5000)

    def test_bearing_is_watertight_visible_annulus_with_open_physical_bore(self):
        s = self.session()
        m, d = s.model, s.data
        surface = next(
            x
            for x in self.root.findall("./asset/mesh")
            if x.get("name") == "turner_rotor__turner_bearing_surface_mesh"
        )
        faces = np.fromstring(surface.get("face"), sep=" ", dtype=int).reshape(-1, 3)
        vertices = np.fromstring(surface.get("vertex"), sep=" ").reshape(-1, 3)
        edges = {}
        for a, b, c in faces:
            for pair in ((a, b), (b, c), (c, a)):
                key = tuple(sorted(pair))
                edges[key] = edges.get(key, 0) + 1
        self.assertEqual(set(edges.values()), {2})
        volume = (
            np.einsum(
                "ij,ij->i",
                vertices[faces[:, 0]],
                np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
            ).sum()
            / 6
        )
        self.assertGreater(volume, 0)
        self.assertGreater(np.linalg.norm(vertices[:, :2], axis=1).min(), 0.251)
        hub = m.geom("turner_rotor__turner_rotating_hub").id
        for i in range(48):
            race = m.geom("turner_rotor__turner_bearing_contact_" + str(i)).id
            self.assertEqual(m.geom_contype[race], 1)
            self.assertGreater(
                mujoco.mj_geomDistance(m, d, hub, race, 0.01, None), 0.0013
            )

    def test_platform_shoes_swing_arms_and_sample_have_actual_mounts(self):
        s = self.session()
        m, d = s.model, s.data

        def gap(a, b):
            return mujoco.mj_geomDistance(
                m,
                d,
                m.geom("turner_rotor__turner_" + a).id,
                m.geom("turner_rotor__turner_" + b).id,
                0.1,
                None,
            )

        for x in (-1.12, 1.12):
            for side in ("front", "rear"):
                self.assertAlmostEqual(
                    gap(f"member_{side}_upper_top", f"platform_bracket_{x}"),
                    0, places=7,
                )
                self.assertAlmostEqual(
                    gap(f"platform_{side}", f"platform_bracket_{x}"), 0, places=7
                )
        for end in (-1, 1):
            for y, side in ((-0.23, "front"), (0.23, "rear")):
                key = f"parked_swing_{end}_{y}"
                self.assertLessEqual(gap(f"parked_swing_platform_{end}", key + "_arm"), 0)
                self.assertLessEqual(gap(key + "_arm", key + "_pivot"), 0)
                wall = f"member_{side}_upper_{'rear' if y < 0 else 'front'}"
                self.assertLessEqual(gap(wall, key + "_pivot"), 0)
                for px in (3.15, 4.50):
                    mount = f"{key}_torsion_mount_{px}"
                    self.assertLessEqual(gap(wall, mount), 0)
                    self.assertLessEqual(gap(key + "_torsion_bar", mount), 0)
        self.assertAlmostEqual(gap("carrier_plate", "inert_sample"), 0, places=7)
        for travel in (-0.15, 0, 0.15):
            d.qpos[1] = travel
            mujoco.mj_forward(m, d)
            for x in (-0.2, 0.2):
                for y in (-0.27, 0.27):
                    self.assertAlmostEqual(
                        gap("carrier_plate", f"shoe_{x}_{y}_top"), 0, places=7
                    )
                    self.assertGreater(gap(f"rail_{y}", f"shoe_{x}_{y}_top"), 0.00049)
                    self.assertGreater(
                        gap(f"member_{'front' if y < 0 else 'rear'}_upper_top",
                            f"shoe_{x}_{y}_bottom"), 0.00049
                    )
                    rail = m.geom(f"turner_rotor__turner_rail_{y}")
                    shoe = m.geom(f"turner_rotor__turner_shoe_{x}_{y}_top")
                    self.assertLess(
                        abs(d.geom_xpos[shoe.id, 0] - d.geom_xpos[rail.id, 0])
                        + shoe.size[0],
                        rail.size[0],
                    )

    def test_four_members_have_open_bores_and_published_outer_section(self):
        s = self.session()
        m, d = s.model, s.data
        for side, y in (("front", -0.48), ("rear", 0.48)):
            for level, z in (("lower", -0.155), ("upper", 0.095)):
                ids = [m.geom(f"turner_rotor__turner_member_{side}_{level}_{face}").id
                       for face in ("front", "rear", "top", "bottom")]
                low = (m.geom_pos[ids] - m.geom_size[ids]).min(axis=0)
                high = (m.geom_pos[ids] + m.geom_size[ids]).max(axis=0)
                np.testing.assert_allclose(high - low, [9.44, 0.4, 0.2], atol=1e-12)
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    m, d, np.array([-4.719, y, z + 1.02]), np.array([1., 0., 0.]),
                    None, 1, -1, hit,
                )
                self.assertTrue(distance < 0 or distance > 9.438,
                                (side, level, distance, m.geom(int(hit[0])).name))
        scope = s.equipment["turner_rotor"]["turner_exterior_scope"]
        self.assertEqual(scope["principal_members"], 4)
        self.assertIn("rigidly parked", scope["swing_platforms"])
        self.assertEqual(m.njnt, 2, "The static swing exterior must not add a false actuator")

    def test_both_bearings_share_continuous_shaft_and_supported_open_mount(self):
        s = self.session()
        m, d = s.model, s.data
        shaft = m.geom("turner_rotor__turner_rotating_hub")
        lower = d.geom_xpos[shaft.id, 2] - shaft.size[1]
        upper = d.geom_xpos[shaft.id, 2] + shaft.size[1]
        for key, low, high in (("bearing", .576, .750), ("upper_bearing", 1.720, 1.894)):
            self.assertLess(lower, high)
            self.assertGreater(upper, low)
            for i in range(48):
                race = m.geom(f"turner_rotor__turner_{key}_contact_{i}")
                self.assertEqual(m.geom_contype[race.id], 1)
                self.assertGreater(mujoco.mj_geomDistance(m, d, shaft.id, race.id, .01, None), .0013)
        for x in (-.92, .92):
            for y in (-1.10, 1.10):
                foot = m.geom(f"turner_rotor__turner_upper_mount_foot_{x}_{y}")
                column = m.geom(f"turner_rotor__turner_upper_mount_column_{x}_{y}")
                plate = m.geom("turner_rotor__turner_upper_mount_plate_" + ("front" if y < 0 else "rear"))
                self.assertAlmostEqual(d.geom_xpos[foot.id, 2] - foot.size[2], 0)
                for first, second in ((foot.id, column.id), (column.id, plate.id)):
                    self.assertAlmostEqual(mujoco.mj_geomDistance(m, d, first, second, .01, None), 0, places=7)

    def test_surface_site_matches_true_inert_sample_top(self):
        s = self.session()
        m = s.model
        sample = m.geom("turner_rotor__turner_inert_sample")
        site = m.site("turner_rotor__site_inert_soil_model")
        self.assertEqual(site.bodyid, sample.bodyid)
        self.assertAlmostEqual(site.pos[2], sample.pos[2] + sample.size[2])
        task = CambridgeAccess(s)
        view = task._observe_surface()
        self.assertTrue(view["visible"])
        self.assertEqual(view["clear_vertical_access_rays"], 25)

    def test_actual_bounded_speed_round_trip_and_return_without_reset(self):
        s = self.session()
        signature = _model_signature(s.model)
        report = inspect_cambridge(s)
        self.assertEqual(report["status"], "succeeded", report)
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 15)
        self.assertTrue(report["observation"]["visible"])
        self.assertEqual(report["observation"]["clear_vertical_access_rays"], 25)
        self.assertGreater(report["observation"]["qpos"][1], 0.119)
        np.testing.assert_allclose(report["measured_return_error"], [0, 0], atol=0.0002)
        for phase in report["phases"]:
            measured = phase["direct_clearance"]
            self.assertTrue(measured["coverage_complete"])
            self.assertEqual(measured["checked_steps"], phase["physics_steps"])
            self.assertTrue(
                np.all(np.array(measured["maximum_observed_speed"]) <= SPEED_LIMITS)
            )
            self.assertLessEqual(
                measured["maximum_observed_coupled_point_speed_bound_m_s"], 0.155
            )
        self.assertEqual(_model_signature(s.model), signature)

    def test_altered_geometry_physics_masks_and_metadata_reject_before_mutation(self):
        mutations = [
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom("turner_rotor__turner_rotating_hub").id, 0
            ),
            lambda s: s.model.geom_pos.__setitem__(
                (s.model.geom("turner_rotor__turner_rail_-0.27").id, 2), 0.5
            ),
            lambda s: s.model.body_mass.__setitem__(
                s.model.body("turner_rotor__body_maintenance_index").id, 80
            ),
            lambda s: s.model.opt.gravity.__setitem__(2, 0),
            lambda s: s.equipment["turner_rotor"]["sites"].__setitem__(
                "inert_soil_model", "wrong"
            ),
        ]
        for bit in (
            "mjDSBL_CONTACT",
            "mjDSBL_CONSTRAINT",
            "mjDSBL_LIMIT",
            "mjDSBL_ACTUATION",
        ):
            mutations.append(
                lambda s, bit=bit: setattr(
                    s.model.opt, "disableflags", int(getattr(mujoco.mjtDisableBit, bit))
                )
            )
        for change in mutations:
            s = self.session()
            change(s)
            before = (
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            )
            with self.assertRaises(ValueError):
                CambridgeAccess(s)
            for first, last in zip(
                before, (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time)
            ):
                np.testing.assert_array_equal(first, last)

    def test_foreign_or_replaced_state_and_postconstruction_changes_reject(self):
        for mode in ("foreign", "replacement", "mask"):
            s = self.session()
            task = CambridgeAccess(s)
            if mode == "foreign":
                s.data = mujoco.MjData(copy(s.model))
            elif mode == "replacement":
                s.data = mujoco.MjData(s.model)
            else:
                s.model.geom_conaffinity[
                    s.model.geom("turner_rotor__turner_rotating_hub").id
                ] = 0
            before = (
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            )
            with self.assertRaises(ValueError):
                task.move([-0.01, 0.1])
            for a, b in zip(
                before, (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time)
            ):
                np.testing.assert_array_equal(a, b)

    def test_external_held_axis_load_fails_with_hold_partial_checks_and_no_reset(self):
        s = self.session()
        s.data.qfrc_applied[0] = 100
        result = CambridgeAccess(s).move([0, 0.1])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(s.data.time, 0)
        self.assertGreater(s.data.qpos[0], 0)
        self.assertFalse(result["direct_clearance"]["coverage_complete"])
        self.assertIsNone(
            result["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
        )
        np.testing.assert_allclose(
            s.data.ctrl, np.clip(s.data.qpos, [-0.2, -0.15], [0.2, 0.15])
        )
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, [0, 0])

    def test_observer_failure_reports_completed_checks_without_false_success(self):
        s = self.session()
        count = 0

        def stop(m, d):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("read_only_failure_probe")

        result = CambridgeAccess(s).move([0, 0.02], step_observer=stop)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["physics_steps"], 3)
        self.assertEqual(result["direct_clearance"]["checked_steps"], 3)
        self.assertTrue(result["direct_clearance"]["coverage_complete"])
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])

    def test_loaded_registration_failure_does_not_claim_task_success_from_complete_checks(
        self,
    ):
        s = self.session()
        s.data.qfrc_applied[0] = 0.2
        result = CambridgeAccess(s).move([0, 0.002])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"], "turner_registration_not_settled")
        self.assertTrue(result["direct_clearance"]["coverage_complete"])
        self.assertTrue(result["hold_requested"])
        self.assertGreater(s.data.qpos[0], 0.0002)
        self.assertFalse(result["reset_performed"])

    def test_nonfinite_and_out_of_range_targets_do_not_step(self):
        for target in ([np.nan, 0], [0, 0.16], [0.21, 0]):
            s = self.session()
            task = CambridgeAccess(s)
            with self.assertRaises(ValueError):
                task.move(target)
            self.assertEqual(s.data.time, 0)
            np.testing.assert_array_equal(s.data.qpos, [0, 0])


if __name__ == "__main__":
    unittest.main()
