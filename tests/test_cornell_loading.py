"""Open Cornell loading assembly and measured inspection regression."""

from copy import copy, deepcopy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.cornell_loading import CornellLoading, LoadingObserver, inspect_cornell
from real_labs.motion import PlanningError, _model_signature
from real_labs.runtime import InstrumentSession
from real_labs.usability import camera_target_screening


class CornellLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("cornell_schlom_mbe")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_axes_coupon_and_site_are_retained(self):
        s = self.session()
        np.testing.assert_array_equal(s.model.jnt_range, [[0, 1.25], [0, 0.2]])
        np.testing.assert_array_equal(s.model.jnt_axis, [[0, 0, -1], [-1, 0, 0]])
        for key, pos in (
            ("load_lock_door", [1.125, -0.335, 1.22]),
            ("transfer_rod", [1.82, 0.07, 1.22]),
        ):
            np.testing.assert_allclose(s.model.body("oxide_mbe__body_" + key).pos, pos)
        coupon = s.model.geom("oxide_mbe__loading_coupon")
        site = s.model.site("oxide_mbe__site_substrate")
        np.testing.assert_allclose(coupon.size, [0.021, 0.019, 0.002])
        np.testing.assert_allclose(site.pos, coupon.pos + [0, 0, 0.002])
        self.assertEqual(site.bodyid, coupon.bodyid)
        self.assertEqual(coupon.contype, 1)

    def test_continuous_shell_meshes_have_no_open_or_internal_faces(self):
        meshes = [
            m
            for m in self.root.findall("./asset/mesh")
            if "__loading_" in m.get("name") and "_surface_mesh" in m.get("name")
        ]
        self.assertGreater(len(meshes), 5)
        for mesh in meshes:
            faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
            edges = {}
            for a, b, c in faces:
                for edge in ((a, b), (b, c), (c, a)):
                    k = tuple(sorted(edge))
                    edges[k] = edges.get(k, 0) + 1
            self.assertEqual(set(edges.values()), {2}, mesh.get("name"))
            vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
            signed_volume = (
                np.einsum(
                    "ij,ij->i",
                    vertices[faces[:, 0]],
                    np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
                ).sum()
                / 6
            )
            self.assertGreater(signed_volume, 0, mesh.get("name"))

    def test_full_travel_direct_pairs_and_enabled_contacts(self):
        s = self.session()
        observer = LoadingObserver(s)
        for hatch, travel in product(np.linspace(0, 1.25, 9), np.linspace(0, 0.2, 11)):
            s.data.qpos[:] = [hatch, travel]
            mujoco.mj_forward(s.model, s.data)
            observer.check(s.model, s.data)
            self.assertFalse([c for c in s.data.contact if c.dist < -1e-7])
        self.assertEqual(observer.checked, 99)
        self.assertGreater(observer.minimum, 0.00025)

    def test_real_hinge_and_guide_supports(self):
        s = self.session()
        m, d = s.model, s.data
        leaf = m.geom("oxide_mbe__loading_hinge_leaf").id
        disc = m.geom("oxide_mbe__loading_door_disc").id
        carrier = m.geom("oxide_mbe__loading_carrier").id
        shaft = m.geom("oxide_mbe__loading_transfer_shaft").id
        coupon = m.geom("oxide_mbe__loading_coupon").id
        for q in ([0, 0], [0.7, 0.1], [1.25, 0.2]):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            self.assertLess(mujoco.mj_geomDistance(m, d, leaf, disc, 0.1, None), 0)
            self.assertLess(mujoco.mj_geomDistance(m, d, carrier, shaft, 0.1, None), 0)
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(m, d, carrier, coupon, 0.1, None), 0, places=7
            )
        for x in (1.5, 1.65):
            stand = m.geom(f"oxide_mbe__loading_guide_stand_{x}")
            self.assertAlmostEqual(stand.pos[2] - stand.size[2], 0.965)
            self.assertAlmostEqual(stand.pos[2] + stand.size[2], 1.22 - 0.045)

    def test_camera_sees_actual_coupon_only_after_opening(self):
        s = self.session()
        for q, visible in (([0, 0], False), ([1.1, 0.04], True)):
            s.data.qpos[:] = q
            mujoco.mj_forward(s.model, s.data)
            screen = camera_target_screening(s.model, s.data, s.manifest)[0]
            self.assertEqual("loading_inspection" in screen["candidate_views"], visible)
            if visible:
                camera = next(
                    c for c in screen["cameras"] if c["camera"] == "loading_inspection"
                )
                self.assertEqual(
                    camera["first_hit"]["name"], "oxide_mbe__loading_coupon"
                )
                self.assertGreater(camera["estimated_target_diameter_px"], 60)

    def test_real_round_trip_has_complete_coverage_and_no_reset(self):
        s = self.session()
        original = _model_signature(s.model)
        report = inspect_cornell(s)
        self.assertEqual(report["status"], "succeeded", report)
        self.assertTrue(report["observation"]["visible"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 1)
        self.assertLess(np.max(np.abs(report["measured_return_error"])), 0.003)
        for phase in report["phases"]:
            direct = phase["direct_clearance"]
            self.assertTrue(direct["coverage_complete"])
            self.assertEqual(direct["checked_steps"], phase["physics_steps"])
            self.assertGreater(direct["whole_leg_clearance_lower_bound_m"], 0.00025)
        self.assertEqual(_model_signature(s.model), original)

    def test_external_load_failure_preserves_actual_state_and_reports_partial_checks(
        self,
    ):
        s = self.session()
        s.data.qfrc_applied[1] = 100
        task = CornellLoading(s)
        result = task.move([1.1, 0])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(s.data.time, 0)
        self.assertGreater(s.data.qpos[1], 0)
        self.assertFalse(result["direct_clearance"]["coverage_complete"])
        self.assertIsNone(
            result["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
        )
        np.testing.assert_allclose(
            s.data.ctrl, np.clip(s.data.qpos, [0, 0], [1.25, 0.2])
        )
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, [0, 0])

    def test_observer_failure_keeps_checked_coverage_without_false_success(self):
        s = self.session()
        count = 0

        def stop(model, data):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("external_readonly_observer_stop")

        report = CornellLoading(s).move([0.2, 0], step_observer=stop)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["physics_steps"], 3)
        self.assertTrue(report["direct_clearance"]["coverage_complete"])
        self.assertTrue(report["hold_requested"])

    def test_disabled_essential_geometry_rejects_without_state_change(self):
        names = [
            "chamber_contact_0_0",
            "rod_sleeve_contact_0_0",
            "hinge_pin",
            "moving_knuckle_contact_0_0",
            "coupon",
            "carrier",
            "transfer_shaft",
            "left_branch_side_-1",
            "door_disc",
        ]
        for name in names:
            for mask in ("geom_contype", "geom_conaffinity"):
                s = self.session()
                s.data.time = 0.123
                geom = s.model.geom("oxide_mbe__loading_" + name).id
                getattr(s.model, mask)[geom] = 0
                before = (
                    s.data.qpos.copy(),
                    s.data.qvel.copy(),
                    s.data.ctrl.copy(),
                    s.data.time,
                )
                with self.assertRaises(ValueError):
                    CornellLoading(s)
                for actual, expected in zip(
                    (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time), before
                ):
                    np.testing.assert_array_equal(actual, expected)

    def test_disabled_flags_shape_mapping_and_postconstruction_mutation_reject(self):
        for flag in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
        ):
            s = self.session()
            s.model.opt.disableflags |= int(flag)
            with self.assertRaises(ValueError):
                CornellLoading(s)
            self.assertEqual(s.data.time, 0)
        s = self.session()
        s.model.geom("oxide_mbe__loading_coupon").size[0] *= 2
        with self.assertRaises(ValueError):
            CornellLoading(s)
        s = self.session()
        s.controls["oxide_mbe", "transfer_rod"] = s.controls[
            "oxide_mbe", "load_lock_door"
        ]
        with self.assertRaises(ValueError):
            CornellLoading(s)
        s = self.session()
        task = CornellLoading(s)
        s.model.geom("oxide_mbe__loading_transfer_shaft").contype[:] = 0
        with self.assertRaises(ValueError):
            task.move([0.2, 0])
        self.assertEqual(s.data.time, 0)

    def test_wrong_state_and_nonhome_start_reject(self):
        s = self.session()
        s.data = mujoco.MjData(copy(s.model))
        with self.assertRaises(ValueError):
            CornellLoading(s)
        s = self.session()
        s.data.qpos[0] = 0.2
        mujoco.mj_forward(s.model, s.data)
        with self.assertRaises(PlanningError):
            inspect_cornell(s)
        self.assertEqual(s.data.time, 0)
        self.assertEqual(s.data.qpos[0], 0.2)

    def test_misbound_sample_and_camera_metadata_reject(self):
        s = self.session()
        s.equipment["oxide_mbe"]["sample_site"] = "oxide_mbe__site_door_handle"
        with self.assertRaises(ValueError):
            CornellLoading(s)
        self.assertEqual(s.data.time, 0)
        s = self.session()
        s.manifest["camera"]["loading_inspection"]["position"][0] += 0.1
        with self.assertRaises(ValueError):
            CornellLoading(s)
        self.assertEqual(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
