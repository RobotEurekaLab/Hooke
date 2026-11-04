"""Exercise the Sutter reference through compiled scenes and real motor motion."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from microscopy.cad_assets import asset_evidence
from microscopy.kinematics import tool_commands
from microscopy.pneumatics import hose_visuals
from microscopy.scene import HOMES, ORIGIN
from microscopy.tasks import make_task


def mesh_world(task, name):
    """Recover the compiled visual mesh, including MuJoCo's mesh transform."""
    model, data = task.model, task.data
    geom_id = model.geom(name).id
    mesh_id = model.geom_dataid[geom_id]
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start:start + count]
    world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    start, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    return world, model.mesh_face[start:start + count]


def hose_endpoint(shape, sign):
    axis = np.asarray(shape["mat"]).reshape(3, 3)[:, 2]
    return np.asarray(shape["pos"]) + sign * shape["size"][1] * axis


class SutterWorkstationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {
            "HOOKE_MICROSCOPY_ASSETS": "reference",
            "HOOKE_MICROSCOPY_STAND": "te2000-s-reference",
            "HOOKE_MICROSCOPY_STAGE": "reference",
            "HOOKE_MICROSCOPY_OPTICS": "estimated",
            "HOOKE_MICROSCOPY_MANIPULATOR": "sutter-mp285-reference",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def task(self, operation="suction_injection", seed=0):
        task = make_task(operation)
        task.reset(seed)
        return task

    def test_public_profile_tracks_installed_assemblies_and_retains_provenance(self):
        for operation, tools in (("cell_injection", {"injector"}),
                                 ("suction_injection", {"injector", "holder"})):
            with self.subTest(operation=operation):
                task = self.task(operation)
                state = task.public_state()
                report = task.microscopy_report()
                self.assertEqual(state["manipulator_profile"], "sutter-mp285-reference")
                self.assertEqual(state["stand_profile"], "te2000-s-reference")
                assemblies = report["assets"]["assemblies"]
                self.assertEqual({item["tool"] for item in assemblies}, tools)
                self.assertEqual(report["assets"], asset_evidence(task.model))
                for item in assemblies:
                    self.assertEqual(item["profile"], state["manipulator_profile"])
                    self.assertFalse(item["official_cad"])
                    self.assertFalse(item["one_to_one_verified"])
                    self.assertTrue(item["source_urls"])
                    self.assertTrue(item["estimates"])
                    for joint in item["chain"]:
                        self.assertEqual(task.model.joint(joint).type, mujoco.mjtJoint.mjJNT_SLIDE)
                json.dumps(state, allow_nan=False)
                json.dumps(report, allow_nan=False)

    def test_serial_xyz_commands_reach_world_targets_and_report_actual_encoders(self):
        task = self.task()
        rng = np.random.default_rng(2026)
        targets = np.vstack(([.3e-3, -.2e-3, .1e-3], rng.uniform(-.6e-3, .6e-3, (4, 3)), np.zeros(3)))
        for tool in ("injector", "holder"):
            with self.subTest(tool=tool):
                tip = task.model.site(tool + "_tcp").id
                node = int(task.model.site_bodyid[tip])
                ancestors = []
                while node:
                    start, count = task.model.body_jntadr[node], task.model.body_jntnum[node]
                    ancestors.extend(task.model.joint(i).name for i in range(start, start + count))
                    node = int(task.model.body_parentid[node])
                self.assertEqual(ancestors, [tool + "_" + axis for axis in "zyx"])
                other = "holder" if tool == "injector" else "injector"
                other_start = task.data.site(other + "_tcp").xpos.copy()
                stage_start = task.data.site("sample_plane").xpos.copy()
                for displacement in targets:
                    commands = tool_commands(task.model, tool, displacement)
                    task.command(commands, .4)
                    np.testing.assert_allclose(task.data.site_xpos[tip],
                        ORIGIN + HOMES[tool] + displacement, atol=2e-8, rtol=0.)
                    feedback = task.instrument_drives.feedback(task.data)
                    for name, target in commands.items():
                        measured = task.data.qpos[task.model.joint(name).qposadr[0]]
                        self.assertEqual(feedback[name], measured)
                        self.assertAlmostEqual(measured, target, delta=2e-8)
                    np.testing.assert_allclose(task.data.site(other + "_tcp").xpos, other_start, atol=2e-8, rtol=0.)
                    np.testing.assert_allclose(task.data.site("sample_plane").xpos, stage_start, atol=2e-8, rtol=0.)
        self.assertEqual(task.model.neq, 0)
        self.assertEqual(task.model.nmocap, 0)
        self.assertFalse(np.any(task.data.warning.number))

    def test_compiled_shaft_and_distal_glass_follow_each_application_layout(self):
        for operation in ("cell_injection", "suction_injection"):
            task = self.task(operation)
            tools = ("injector", "holder") if operation == "suction_injection" else ("injector",)
            for tool in tools:
                with self.subTest(operation=operation, tool=tool):
                    holder = task.model.body(tool + "_needle_holder").id
                    shaft = task.data.xmat[holder].reshape(3, 3)[:, 2]
                    self.assertAlmostEqual(np.rad2deg(np.arcsin(shaft[2])),
                                           35. if tool == "holder" else 45., delta=1e-6)
                    name = "cell_hollow_pipette" if tool == "injector" else "holding_hollow_pipette"
                    world, _ = mesh_world(task, name)
                    vertices = world - task.data.site(tool + "_tcp").xpos
                    near = vertices[np.linalg.norm(vertices, axis=1) < .25e-3]
                    # Fit the visible distal glass, independently of its metadata.
                    _, _, axes = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
                    terminal = axes[0] * np.sign(axes[0] @ near.mean(axis=0))
                    actual = task.mechanics.needle_back if tool == "injector" else task.mechanics.holder_back
                    np.testing.assert_allclose(terminal, actual, atol=2e-5, rtol=0.)
                    if operation == "suction_injection":
                        self.assertAlmostEqual(terminal[2], 0., delta=2e-5)
                        self.assertGreater(np.linalg.norm(shaft - terminal), .5)
                    else:
                        np.testing.assert_allclose(terminal, shaft, atol=2e-5, rtol=0.)
                    # The wide rear ring must terminate at the holder's chuck.
                    offset = world - task.data.xpos[holder]
                    base_ring = offset[np.abs(offset @ shaft) < 2e-9]
                    self.assertEqual(len(base_ring), 96)
                    np.testing.assert_allclose(base_ring.mean(axis=0), 0., atol=2e-9, rtol=0.)

    def test_compiled_glass_mouths_have_micrometre_bores_without_solid_caps(self):
        task = self.task()
        for tool, name, inner, outer in (
                ("injector", "cell_hollow_pipette", .25e-6, .60e-6),
                ("holder", "holding_hollow_pipette", 4e-6, 6e-6)):
            with self.subTest(tool=tool):
                world, faces = mesh_world(task, name)
                vertices = world - task.data.site(tool + "_tcp").xpos
                direction = task.mechanics.needle_back if tool == "injector" else task.mechanics.holder_back
                at_mouth = np.abs(vertices @ direction) < 2e-9
                self.assertEqual(np.count_nonzero(at_mouth), 96)
                radii = np.linalg.norm(vertices[at_mouth], axis=1)
                np.testing.assert_allclose(np.sort(radii)[:48], inner, atol=2e-9, rtol=0.)
                np.testing.assert_allclose(np.sort(radii)[48:], outer, atol=2e-9, rtol=0.)
                rim = vertices[faces[np.all(at_mouth[faces], axis=1)]]
                area = np.linalg.norm(np.cross(rim[:, 1] - rim[:, 0], rim[:, 2] - rim[:, 0]), axis=1).sum() / 2
                self.assertAlmostEqual(area / (np.pi * (outer**2 - inner**2)), 1., delta=.01)
                # The collision sphere is hidden; it must not visually plug the bore.
                self.assertEqual(task.model.geom(tool + "_tip").group, 3)
                self.assertLess(2 * outer, task.sample_scale()["diameters_m"][0])

    def test_hose_endpoints_follow_holder_ports_during_real_motor_motion(self):
        task = self.task()
        for displacement in (np.zeros(3), np.array([.35e-3, -.25e-3, .15e-3])):
            for tool in ("injector", "holder"):
                task.command(tool_commands(task.model, tool, displacement), .4)
            before = [value.copy() for value in (task.data.qpos, task.data.qvel, task.data.ctrl, task.model.body_mass)]
            hoses = hose_visuals(task.model, task.data)
            self.assertEqual(len(hoses), 96)
            for index, tool in enumerate(("injector", "holder")):
                route = hoses[index * 48:(index + 1) * 48]
                np.testing.assert_allclose(hose_endpoint(route[-1], 1),
                    task.data.site(tool + "_pneumatic_port").xpos, atol=1e-12, rtol=0.)
                np.testing.assert_allclose(hose_endpoint(route[0], -1),
                    task.data.site("pneumatic_" + tool + "_anchor0").xpos, atol=1e-12, rtol=0.)
                for previous, following in zip(route, route[1:]):
                    np.testing.assert_allclose(hose_endpoint(previous, 1), hose_endpoint(following, -1), atol=1e-12, rtol=0.)
            for previous, actual in zip(before, (task.data.qpos, task.data.qvel, task.data.ctrl, task.model.body_mass)):
                np.testing.assert_array_equal(previous, actual)

    def test_both_image_guided_experts_complete_with_actual_contact_and_focused_dose(self):
        for operation in ("cell_injection", "suction_injection"):
            with self.subTest(operation=operation):
                task = self.task(operation, seed=4)
                task.execute()
                self.assertTrue(task.check(), task.microscopy_report())
                mechanics = task.mechanics
                trace = mechanics.interaction_trace()
                delivery = np.diff(trace["delivered_m3"], prepend=0.) > 0
                self.assertTrue(delivery.any())
                self.assertTrue(np.all(trace["inside"][delivery]))
                self.assertTrue(np.all(trace["punctured"][delivery]))
                np.testing.assert_allclose(trace["normalized_distance"],
                    np.linalg.norm((trace["tip_world_m"] - trace["contact_cell_world_m"]) / mechanics.cell_radii, axis=1),
                    atol=1e-12, rtol=0.)
                # Compensation can flow during the moving puncture approach;
                # the pressure pulse is delivered at the selected focal plane.
                pulse = delivery & (trace["command_pressure_pa"] > 0)
                self.assertTrue(pulse.any())
                focus_error = trace["tip_world_m"][pulse, 2] - trace["focus_plane_z_m"][pulse]
                self.assertLess(np.abs(focus_error).max(), .45e-6)
                self.assertAlmostEqual(trace["delivered_m3"][-1] * 1e15, .5)
                self.assertAlmostEqual(task.observation_focus[-1]["observed_focus_m"], 0., delta=1e-8)
                if operation == "suction_injection":
                    self.assertLess(mechanics.seal_time_s, mechanics.puncture_time_s)
                    self.assertLess(mechanics.withdrawal_time_s, mechanics.release_time_s)
                    self.assertGreater(mechanics.maximum_holding_force_n, 20e-9)
                    self.assertLess(mechanics.maximum_displacement_m, 1e-6)
                    self.assertAlmostEqual(task.observation_focus[0]["observed_focus_m"], 0., delta=1e-8)
                self.assertFalse(np.any(task.data.warning.number))
                json.dumps(task.microscopy_report(), allow_nan=False)

    def test_occluded_holding_channel_cannot_support_puncture_or_deliver_dose(self):
        task = self.task()
        mechanics = task.mechanics
        centre = mechanics.position_m - ORIGIN
        radius = mechanics.cell_radii[0]
        task.move("holder", centre + mechanics.holder_back * radius, .7)
        mechanics.holding_pressure_pa = -1500.
        mechanics.holding_occluded = True
        task.wait(.2)
        self.assertLess(mechanics.actual_holding_pressure_pa, -1400.)
        self.assertEqual(mechanics.holding_capacity_n, 0.)
        self.assertFalse(mechanics.sealed)
        self.assertFalse(mechanics.established)
        self.assertEqual(mechanics.hold_s, 0.)
        centre = mechanics.position_m - ORIGIN
        task.move("injector", centre + mechanics.needle_back * (radius - 2.5e-6), .7)
        task.data.userdata[0] = 5000.
        task.wait(.3)
        self.assertGreater(mechanics.maximum_displacement_m, 1e-6)
        self.assertFalse(mechanics.punctured)
        self.assertEqual(mechanics.ledger.state("cell").volume_m3, 0.)
        self.assertFalse(task.microscopy_checks()["holding_established"])
        self.assertFalse(task.check())


if __name__ == "__main__":
    unittest.main()
