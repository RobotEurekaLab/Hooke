"""Detailed, inert mechanical access for the reused miniature centrifuge."""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs import mini_centrifuge
from real_labs.builder import build_scene
from real_labs.mini_centrifuge_contacts import ContactCheck
from real_labs.mini_centrifuge_fixture import (
    build_inspection_scene,
    create_inspection_session,
    export_inspection_scene,
)
from real_labs.mini_centrifuge_motion import MiniInspection
from real_labs.runtime import InstrumentSession
from real_labs.reuse import MODEL_ROOT
from real_labs.reuse_drives import MINI_CENTRIFUGE


def state(session):
    return [
        getattr(session.data, key).copy()
        for key in ("qpos", "qvel", "ctrl", "xfrc_applied", "qfrc_applied")
    ] + [float(session.data.time)]


class MiniCentrifugeAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = create_inspection_session("rochester_genomics")

    def session(self):
        return InstrumentSession(
            copy.copy(self.original.model), copy.deepcopy(self.original.manifest)
        )

    def unchanged(self, before, session):
        for expected, actual in zip(before, state(session)):
            np.testing.assert_array_equal(actual, expected)

    def test_final_resources_and_original_asset_hashes(self):
        metadata, names, bodies, vertices, offsets = mini_centrifuge._resources()
        mini_centrifuge._verify_upstream(metadata)
        self.assertEqual(len(names), 1398)
        self.assertEqual(np.bincount(bodies).tolist(), [117, 733, 548])
        self.assertEqual(offsets[-1], len(vertices))
        self.assertFalse(vertices.flags.writeable)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in mini_centrifuge.RESOURCE_DIR.iterdir():
                (root / path.name).write_bytes(path.read_bytes())
            with (root / "contacts.npz").open("ab") as stream:
                stream.write(b"modified")
            mini_centrifuge._resources.cache_clear()
            try:
                with mock.patch.object(mini_centrifuge, "RESOURCE_DIR", root):
                    with self.assertRaisesRegex(ValueError, "resource hash changed"):
                        mini_centrifuge._resources()
            finally:
                mini_centrifuge._resources.cache_clear()

    def test_original_inertias_and_joint_frames_are_preserved(self):
        source = mujoco.MjModel.from_xml_path(str(MODEL_ROOT / MINI_CENTRIFUGE))
        adapted = self.original.model
        for name in ("body", "lid", "rotor"):
            old, new = source.body(name).id, adapted.body("centrifuge__" + name).id
            for field in (
                "body_pos",
                "body_quat",
                "body_mass",
                "body_inertia",
                "body_ipos",
                "body_iquat",
            ):
                np.testing.assert_allclose(
                    getattr(adapted, field)[new],
                    getattr(source, field)[old],
                    atol=1e-12,
                )
        for name in ("lid", "rotor"):
            old, new = source.joint(name).id, adapted.joint("centrifuge__" + name).id
            for field in ("jnt_pos", "jnt_axis", "jnt_range"):
                np.testing.assert_allclose(
                    getattr(adapted, field)[new],
                    getattr(source, field)[old],
                    atol=1e-12,
                )

    def test_all_three_assets_keep_controls_and_precision_is_conditional(self):
        for scene in mini_centrifuge.SCENES:
            with self.subTest(scene=scene):
                root, manifest = build_scene(scene)
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                session = InstrumentSession(model, manifest)
                rotor, lid = (
                    session.controls["centrifuge", name] for name in ("rotor", "lid")
                )
                self.assertEqual(model.opt.ccd_tolerance, 1e-10)
                self.assertEqual((lid.lower, lid.upper), (0, 1.712))
                self.assertEqual(rotor.mode, "velocity")
                np.testing.assert_allclose(
                    [rotor.lower, rotor.upper], [-math.tau, math.tau]
                )
                self.assertFalse(model.jnt_limited[rotor.joint])
                interface = session.equipment["centrifuge"]["mechanical_interface"]
                self.assertEqual(interface["contact_count"], 1398)
                self.assertEqual(len(interface["port_sites"]), 8)
                self.assertNotIn("inspection_variant", manifest)
        root, _ = build_scene("mit_paoc_rotating_fluids")
        self.assertNotEqual(
            float(root.find("option").get("ccd_tolerance", "1e-6")), 1e-10
        )

    def test_portable_variant_remains_free_and_source_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = export_inspection_scene("rochester_genomics", folder)
            path = Path(folder)
            self.assertEqual(
                hashlib.sha256((path / "scene.xml").read_bytes()).hexdigest(),
                manifest["scene_sha256"],
            )
            session = InstrumentSession.from_files(
                path / "scene.xml", path / "manifest.json"
            )
            operation = MiniInspection(session)
            model = session.model
            joint = model.body_jntadr[operation.gauge]
            self.assertEqual(model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual(model.body_jntnum[operation.gauge], 1)
            np.testing.assert_allclose(
                session.data.xpos[operation.gauge],
                session.data.site_xpos[operation.park],
                atol=1e-12,
            )
            for name in operation.fixture["gauge_geoms"]:
                geom = model.geom(name).id
                self.assertEqual(model.geom_contype[geom], 1)
                self.assertEqual(model.geom_conaffinity[geom], 1)

    def test_foreign_state_and_changed_essential_contacts_reject_without_mutation(self):
        for mutation in ("foreign", "disabled", "moved", "metadata"):
            with self.subTest(mutation=mutation):
                session = self.session()
                if mutation == "foreign":
                    session.data = mujoco.MjData(copy.copy(session.model))
                elif mutation == "metadata":
                    session.manifest["inspection_variant"]["mass_kg"] *= 2
                else:
                    geom = session.model.geom(
                        session.manifest["inspection_variant"]["gauge_geoms"][0]
                    ).id
                    if mutation == "disabled":
                        session.model.geom_contype[geom] = 0
                        session.model.geom_conaffinity[geom] = 0
                    else:
                        session.model.geom_pos[geom, 0] += 0.001
                before = state(session)
                with self.assertRaises(ValueError):
                    MiniInspection(session)
                self.unchanged(before, session)

    def test_cached_state_replacement_rejects_before_stepping(self):
        session = self.session()
        operation = MiniInspection(session)
        old = session.data
        session.data = mujoco.MjData(session.model)
        before = state(session)
        with self.assertRaisesRegex(ValueError, "state was replaced"):
            operation.run()
        self.unchanged(before, session)
        self.assertEqual(old.time, 0)

    def test_all_free_gauge_pairs_survive_envelope_pruning(self):
        session = self.session()
        operation = MiniInspection(session)
        check, model = operation.contacts, session.model
        excluded = check.excluded_pairs
        self.assertFalse(np.any(model.geom_bodyid[excluded] == operation.gauge))
        # Every excluded pair has a strict positive separating interval over
        # the whole authored hinge envelope, independently of collision masks.
        first, second = excluded.T
        gap = np.maximum(
            check.envelope_lower[first] - check.envelope_upper[second],
            check.envelope_lower[second] - check.envelope_upper[first],
        )
        self.assertTrue(np.all(np.max(gap, axis=1) > 0))
        for lid_angle, rotor_angle in ((0.217, 2.94), (0.937, -1.88), (1.712, 0.51)):
            private = mujoco.MjData(model)
            private.qpos[:] = session.data.qpos
            private.qpos[model.jnt_qposadr[operation.lid.joint]] = lid_angle
            private.qpos[model.jnt_qposadr[operation.rotor.joint]] = rotor_angle
            mujoco.mj_forward(model, private)
            for body in (check.lid, check.rotor):
                geoms = np.flatnonzero(model.geom_bodyid == body)
                for geom in geoms:
                    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                        continue
                    mesh = model.geom_dataid[geom]
                    vertices = model.mesh_vert[
                        model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh]
                        + model.mesh_vertnum[mesh]
                    ]
                    world = (
                        vertices @ private.geom_xmat[geom].reshape(3, 3).T
                        + private.geom_xpos[geom]
                    )
                    self.assertTrue(
                        np.all(world.min(0) >= check.envelope_lower[geom] - 2e-8)
                    )
                    self.assertTrue(
                        np.all(world.max(0) <= check.envelope_upper[geom] + 2e-8)
                    )

    def test_disturbed_payload_has_incomplete_coverage_and_hold_without_reset(self):
        session = self.session()
        operation = MiniInspection(session)
        joint = session.model.body_jntadr[operation.gauge]
        dof = session.model.jnt_dofadr[joint]
        session.data.qfrc_applied[dof] = 100
        report = operation.run()
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["executed_steps"], report["completed_checks"])
        self.assertFalse(report["coverage_complete"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertTrue(report["model_signature_unchanged"])
        self.assertGreater(session.data.time, 0)
        self.assertEqual(session.data.qfrc_applied[dof], 100)
        np.testing.assert_array_equal(session.data.xfrc_applied[operation.gauge], 0)

    def test_retained_obstacle_is_checked_even_when_source_binding_rejects_it(self):
        root, manifest = build_inspection_scene("rochester_genomics")
        mount = root.find(".//body[@name='centrifuge__mount']")
        ET.SubElement(
            mount,
            "geom",
            name="retained_spacer",
            type="box",
            pos=".06 0 .11310644",
            size=".004 .004 .035",
            contype="1",
            conaffinity="1",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        session = InstrumentSession(model, manifest)
        with self.assertRaises(ValueError):
            MiniInspection(session)
        interface = session.equipment["centrifuge"]["mechanical_interface"]
        checker = ContactCheck(
            model, session.data, interface, manifest["inspection_variant"]
        )
        obstacle = model.geom("retained_spacer").id
        self.assertTrue(np.any(checker.pairs == obstacle))
        lid = session.controls["centrifuge", "lid"]
        session.data.ctrl[lid.actuator] = 1.712
        failed = False
        for _ in range(2000):
            mujoco.mj_step(model, session.data)
            mujoco.mj_forward(model, session.data)
            try:
                checker.check(session.data)
            except RuntimeError as error:
                self.assertIn("contact_limit", str(error))
                failed = True
                break
        self.assertTrue(
            failed, "The physical spacer must block the original lid closure"
        )


if __name__ == "__main__":
    unittest.main()
