"""Motion planning retains real obstacles, measured failure and stale-state checks."""

from copy import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.motion import (
    JointMotion,
    PlanningError,
    _model_signature,
    inspect_nasa_cache,
    inspect_point_and_return,
    inspect_waterloo_handoff,
)
from real_labs.runtime import InstrumentSession


ARM_CONTROLS = (
    "base_yaw",
    "shoulder",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "tool_yaw",
)


def waterloo():
    root, manifest = build_scene("waterloo_robohub")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def fixture():
    xml = """<mujoco><compiler angle="radian" autolimits="true"/>
    <option gravity="0 0 0" timestep=".002" integrator="implicitfast"/>
    <asset><mesh name="tetra" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"/></asset>
    <worldbody>
      <geom name="obstacle" type="box" pos=".6 0 .25" size=".12 .14 .20"/>
      <geom name="mesh_obstacle" type="mesh" mesh="tetra" pos="3 0 .25"/>
      <site name="static_site" pos="0 0 .25"/>
      <body name="loose_obstacle" pos="4 0 .25"><freejoint name="loose"/>
        <geom name="loose_geom" type="sphere" size=".04" mass=".1"/>
      </body>
      <body name="cart" pos="0 0 .25">
        <joint name="x" type="slide" axis="1 0 0" range="0 1.2" damping="2"/>
        <joint name="y" type="slide" axis="0 1 0" range="-.6 .6" damping="2"/>
        <geom name="tool" type="sphere" size=".035" mass=".1"/>
        <site name="tcp" size=".002"/>
        <site name="offset_tcp" pos="0 .1 0" size=".002"/>
      </body>
    </worldbody>
    <contact><pair geom1="tool" geom2="obstacle"/>
      <exclude body1="cart" body2="loose_obstacle"/></contact>
    <actuator>
      <position name="x_drive" joint="x" kp="80" kv="3" ctrlrange="0 1.2"/>
      <position name="y_drive" joint="y" kp="80" kv="3" ctrlrange="-.6 .6"/>
    </actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    manifest = {
        "equipment": [
            {
                "id": "cart",
                "joints": {"x": "x", "y": "y"},
                "actuators": {"x": "x_drive", "y": "y_drive"},
                "sites": {"tcp": "tcp"},
            }
        ]
    }
    return InstrumentSession(model, manifest)


class CheckedJointMotionTests(unittest.TestCase):
    def test_native_badacc_reset_is_reported_with_zero_checked_steps(self):
        badacc = int(mujoco.mjtWarning.mjWARN_BADQACC)
        for start_time, previous_count in ((1.0, 0), (0.0, 0), (0.0, 1), (0.0, 7)):
            with self.subTest(start_time=start_time, previous_count=previous_count):
                session = fixture()
                session.data.time = start_time
                session.data.warning[badacc].number = previous_count
                # Match the previous warning's DOF too: a repeated native reset
                # need not change either count or lastinfo at time zero.
                dof = session.model.joint("x").dofadr[0]
                session.data.warning[badacc].lastinfo = dof
                session.data.qfrc_applied[dof] = 1e25
                motion = JointMotion(session, "cart", ("x", "y"))
                observations = []
                result = motion.execute(
                    motion.plan_joint_target([0, 0]),
                    step_observer=lambda m, d: observations.append(d.time),
                )
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["physics_steps"], 1)
                self.assertEqual(result["checked_steps"], 0)
                self.assertFalse(result["coverage_complete"])
                self.assertFalse(observations)
                self.assertTrue(result["reset_performed"])
                self.assertTrue(result["automatic_reset_detected"])
                self.assertFalse(result["explicit_reset_performed"])
                self.assertTrue(result["hold_requested"])
                self.assertEqual(session.data.time, session.model.opt.timestep)
                self.assertEqual(session.data.qfrc_applied[dof], 0)
                event = result["physics_events"][0]
                self.assertIn("qfrc_applied", event["reset_cleared_inputs"])

    def test_warning_without_autoreset_does_not_claim_reset(self):
        session = fixture()
        session.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        session.data.qfrc_applied[session.model.joint("x").dofadr[0]] = 1e25
        motion = JointMotion(session, "cart", ("x", "y"))
        result = motion.execute(motion.plan_joint_target([0, 0]))
        self.assertEqual(result["failure"], "physics_warning")
        self.assertFalse(result["reset_performed"])
        self.assertFalse(result["automatic_reset_detected"])
        self.assertEqual(result["checked_steps"], 0)
        self.assertEqual(
            session.data.qfrc_applied[session.model.joint("x").dofadr[0]], 1e25
        )

    def test_stale_bad_warning_at_zero_and_unresolvable_clock_remain_unknown(self):
        for case in ("stale_warning", "large_time"):
            with self.subTest(case=case):
                session = fixture()
                if case == "stale_warning":
                    session.data.warning[
                        int(mujoco.mjtWarning.mjWARN_BADQACC)
                    ].number = 1
                else:
                    session.data.time = 1e30
                motion = JointMotion(session, "cart", ("x", "y"))
                result = motion.execute(motion.plan_joint_target([0, 0]))
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["reset_performed"])
                self.assertFalse(result["automatic_reset_detected"])
                self.assertFalse(result["explicit_reset_performed"])
                self.assertEqual(result["physics_steps"], 1)
                self.assertEqual(result["checked_steps"], 0)
                self.assertFalse(result["coverage_complete"])
                self.assertTrue(result["hold_requested"])

    def test_historical_warning_at_advancing_clock_does_not_change_motion(self):
        states = []
        for warning in (False, True):
            session = fixture()
            session.data.time = 1
            if warning:
                session.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 7
            motion = JointMotion(session, "cart", ("x", "y"))
            observed = []
            result = motion.execute(
                motion.plan_joint_target([0.001, 0]),
                step_observer=lambda m, d: observed.append(
                    np.concatenate((d.qpos, d.qvel, d.ctrl, [d.time]))
                ),
            )
            self.assertEqual(result["status"], "succeeded")
            self.assertFalse(result["reset_performed"])
            self.assertFalse(result["physics_events"])
            self.assertEqual(result["physics_steps"], result["checked_steps"])
            self.assertTrue(result["coverage_complete"])
            states.append(observed)
        np.testing.assert_array_equal(*states)

    def test_nonfinite_time_rejects_before_applying_controls(self):
        for value in (np.nan, np.inf):
            session = fixture()
            motion = JointMotion(session, "cart", ("x", "y"))
            plan = motion.plan_joint_target([0.1, 0])
            session.data.time = value
            before = (
                session.data.qpos.copy(),
                session.data.qvel.copy(),
                session.data.ctrl.copy(),
            )
            with self.assertRaisesRegex(PlanningError, "finite simulation state"):
                motion.execute(plan)
            for array, expected in zip(
                (session.data.qpos, session.data.qvel, session.data.ctrl), before
            ):
                np.testing.assert_array_equal(array, expected)
            self.assertTrue(
                np.isnan(session.data.time)
                if np.isnan(value)
                else session.data.time == value
            )

    def test_exact_boundary_endpoint_is_preserved_without_relaxing_limits(self):
        session = fixture()
        motion = JointMotion(session, "cart", ("x", "y"))
        session.data.qpos[motion.qadr] = [0, 0.599999999]
        session.data.ctrl[motion.actuators] = [0, 0.599999999]
        mujoco.mj_forward(session.model, session.data)
        # The previous arithmetic produces -0.6000000000000001, outside the
        # strict legal range, even though the requested endpoint is exactly -0.6.
        self.assertLess(0.599999999 + (-0.6 - 0.599999999), -0.6)
        before = session.data.qpos.copy(), session.data.ctrl.copy(), session.data.time
        plan = motion.plan_joint_target([0, -0.6])
        np.testing.assert_array_equal(plan.waypoints[-1], [0, -0.6])
        np.testing.assert_array_equal(session.data.qpos, before[0])
        np.testing.assert_array_equal(session.data.ctrl, before[1])
        self.assertEqual(session.data.time, before[2])
        with self.assertRaises(ValueError):
            motion.plan_joint_target([0, np.nextafter(-0.6, -np.inf)])

    def test_edge_endpoint_fix_preserves_interior_checks_and_obstacles(self):
        session = fixture()
        motion = JointMotion(session, "cart", ("x", "y"))
        for first, second in (
            ([0, 0], [0.125, 0.25]),
            ([0, 0.25], [0, -0.25]),
            ([0, 0], [1.0, 0]),
        ):
            first, second = np.asarray(first), np.asarray(second)
            count = max(
                1, int(np.ceil(np.max(abs(second - first) / motion.resolution)))
            )
            expected = []
            clear = True
            for fraction in np.linspace(0, 1, count + 1):
                point = first + fraction * (second - first)
                expected.append(point)
                if motion.configuration_contacts(point):
                    clear = False
                    break
            with patch.object(
                motion, "configuration_contacts", wraps=motion.configuration_contacts
            ) as calls:
                self.assertEqual(motion._edge_clear(first, second), clear)
            observed = [call.args[0] for call in calls.call_args_list]
            np.testing.assert_array_equal(observed, expected)
        self.assertFalse(clear, "The real box obstacle must remain blocking")

    def test_small_welded_tool_clearance_matches_recompiled_body_margins(self):
        for existing_margin in (0.0, 0.025):
            with self.subTest(existing_margin=existing_margin):
                root = ET.fromstring(
                    f"""<mujoco><compiler autolimits="true"/>
                <option gravity="0 0 0"/>
                <worldbody>
                  <geom name="tiny_obstacle" type="sphere" size=".008" pos=".03 0 .25"/>
                  <body name="cart" pos="0 0 .25">
                    <inertial pos="0 0 0" mass=".1" diaginertia=".001 .001 .001"/>
                    <joint name="x" type="slide" axis="1 0 0" range="-.1 .1"/>
                    <body name="welded_tool">
                      <geom name="tip" type="sphere" size=".012" margin="{existing_margin}"/>
                      <site name="tcp"/>
                    </body>
                  </body>
                </worldbody>
                <actuator><position name="drive" joint="x" kp="80" kv="3" ctrlrange="-.1 .1"/></actuator>
                </mujoco>"""
                )
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                session = InstrumentSession(
                    model,
                    {
                        "equipment": [
                            {
                                "id": "cart",
                                "joints": {"x": "x"},
                                "actuators": {"x": "drive"},
                                "sites": {"tcp": "tcp"},
                            }
                        ]
                    },
                )
                signature = _model_signature(model)
                margins = model.geom_margin.copy(), model.body_margin.copy()
                distance = mujoco.mj_geomDistance(
                    model,
                    session.data,
                    model.geom("tip").id,
                    model.geom("tiny_obstacle").id,
                    1,
                    None,
                )
                self.assertAlmostEqual(distance, 0.010)
                motion = JointMotion(session, "cart", ("x",), clearance_m=0.015)
                contacts = motion.configuration_contacts([0])
                self.assertTrue(contacts)
                self.assertAlmostEqual(contacts[0]["distance_m"], distance)
                root.find(".//geom[@name='tip']").set(
                    "margin", str(max(existing_margin, 0.015))
                )
                reference = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                state = mujoco.MjData(reference)
                mujoco.mj_forward(reference, state)
                self.assertTrue(any(c.dist < 0.015 for c in state.contact))
                np.testing.assert_array_equal(
                    motion.model.body_margin, reference.body_margin
                )
                np.testing.assert_array_equal(model.geom_margin, margins[0])
                np.testing.assert_array_equal(model.body_margin, margins[1])
                self.assertEqual(signature, _model_signature(model))

    def test_read_only_observer_sees_every_executed_step_and_settling(self):
        session = fixture()
        motion = JointMotion(session, "cart", ("x", "y"))
        observations = []

        def observe(model, data):
            self.assertIs(model, session.model)
            self.assertIs(data, session.data)
            observations.append(float(data.time))

        report = motion.execute(
            motion.plan_joint_target([0.1, 0]), step_observer=observe
        )
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(len(observations), report["physics_steps"])
        np.testing.assert_allclose(
            np.diff([0, *observations]), session.model.opt.timestep
        )
        # The quintic's 6.25 s trajectory is followed by actual settling steps.
        self.assertGreater(report["physics_steps"], 6.25 / session.model.opt.timestep)

    def test_observer_runtime_failure_preserves_actual_state_and_requests_hold(self):
        session = fixture()
        motion = JointMotion(session, "cart", ("x", "y"))
        observations = []

        def observe(model, data):
            observations.append(data.qpos.copy())
            if len(observations) == 3:
                raise RuntimeError("measured_observer_failure")

        report = motion.execute(
            motion.plan_joint_target([0.1, 0]), step_observer=observe
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"], "measured_observer_failure")
        self.assertEqual(report["physics_steps"], 3)
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        np.testing.assert_array_equal(session.data.qpos, observations[-1])
        np.testing.assert_allclose(
            session.data.ctrl[motion.actuators], motion.measured()
        )
        self.assertAlmostEqual(session.data.time, 3 * session.model.opt.timestep)

    def test_contact_override_preserves_required_private_clearance(self):
        session = fixture()
        session.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
        session.model.opt.o_margin = 0
        signature = _model_signature(session.model)
        motion = JointMotion(session, "cart", ("x", "y"), clearance_m=0.015)
        # Tool-to-box surface gap is 10 mm, smaller than required 15 mm.
        self.assertTrue(motion.configuration_contacts([0.435, 0]))
        with self.assertRaisesRegex(PlanningError, "goal violates collision clearance"):
            motion.plan_joint_target([0.435, 0])
        self.assertEqual(session.model.opt.o_margin, 0)
        self.assertEqual(signature, _model_signature(session.model))
        self.assertEqual(session.data.time, 0)

    def test_nonselected_descendant_control_under_another_equipment_is_monitored(self):
        original = fixture()
        manifest = {
            "equipment": [
                {
                    "id": "cart",
                    "joints": {"x": "x"},
                    "actuators": {"x": "x_drive"},
                    "sites": {"tcp": "tcp"},
                },
                {"id": "aux", "joints": {"y": "y"}, "actuators": {"y": "y_drive"}},
            ]
        }
        session = InstrumentSession(original.model, manifest)
        control = session.controls["aux", "y"]
        session.data.qfrc_applied[session.model.jnt_dofadr[control.joint]] = 5
        motion = JointMotion(session, "cart", ("x",))
        result = motion.execute(motion.plan_joint_target([0.15]))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"], "uncommanded_control_drift")
        self.assertGreater(result["held_control_drift"]["aux.y"], 0.03)
        self.assertEqual(session.data.ctrl[control.actuator], 0)
        self.assertTrue(result["hold_requested"])

    def test_round_trip_uses_the_requested_offset_site_in_both_directions(self):
        session = fixture()
        motion = JointMotion(session, "cart", ("x", "y"))
        report = inspect_point_and_return(motion, "offset_tcp", [0.15, 0.1, 0.25])
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["returning"]["tool_site"], "offset_tcp")
        np.testing.assert_allclose(
            report["returning"]["target_position_m"], [0, 0.1, 0.25]
        )
        self.assertLess(report["final_tool_return_error_m"], 0.001)
        self.assertLess(report["returning"]["cartesian_error_m"], 0.001)

    def test_legacy_cache_inspection_rejects_grounded_rover_before_motion(self):
        root, manifest = build_scene("nasa_planetary")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        session = InstrumentSession(model, manifest)
        signature = _model_signature(model)
        before = session.data.qpos.copy(), session.data.qvel.copy(), session.data.ctrl.copy(), session.data.time
        for cache in (1, 2, 3):
            with self.assertRaisesRegex(ValueError, "rover_traverse_and_stage"):
                inspect_nasa_cache(session, cache=cache)
        for actual, expected in zip((session.data.qpos, session.data.qvel, session.data.ctrl), before[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(session.data.time, before[3])
        self.assertEqual(signature, _model_signature(model))

    def test_waterloo_useful_hover_round_trip_preserves_limits_and_contacts(self):
        session = waterloo()
        signature = _model_signature(session.model)
        ranges = session.model.jnt_range.copy()
        masks = session.model.geom_contype.copy()
        result = inspect_waterloo_handoff(session)
        self.assertEqual(result["status"], "succeeded", result["outward"]["failure"])
        np.testing.assert_allclose(result["target_position_m"], [3, 0.35, 0.93])
        self.assertAlmostEqual(result["endpoint_shoulder_target_rad"], 1.155)
        self.assertLess(result["outward"]["cartesian_error_m"], 0.001)
        self.assertLess(result["final_tool_return_error_m"], 0.001)
        for leg in (result["outward"], result["returning"]):
            self.assertFalse(leg["collisions"])
            self.assertFalse(leg["held_control_drift"])
            self.assertTrue(leg["direct_checks"]["coverage_complete"])
            self.assertEqual(
                leg["physics_steps"], leg["direct_checks"]["fully_checked_steps"]
            )
            self.assertGreaterEqual(
                leg["direct_checks"]["whole_leg_external_clearance_lower_bound_m"], 0.01
            )
            self.assertFalse(leg["reset_performed"])
            self.assertGreater(leg["physics_steps"], len(leg["telemetry"]))
            self.assertEqual(
                leg["collision_sample_period_s"], session.model.opt.timestep
            )
        self.assertFalse(result["grasp_attempted"])
        self.assertFalse(result["object_transfer_validated"])
        self.assertEqual(signature, _model_signature(session.model))
        np.testing.assert_array_equal(ranges, session.model.jnt_range)
        np.testing.assert_array_equal(masks, session.model.geom_contype)
        session.reset()
        self.assertEqual(session.data.time, 0)
        np.testing.assert_array_equal(session.data.qpos, session.model.qpos0)

    def test_waterloo_isolated_obstructed_shoulder_goal_stays_invalid(self):
        session = waterloo()
        from real_labs.robot_arm_motion import ArmMotion

        motion = ArmMotion(session)
        target = motion.measured()
        target[1] = 0.825
        before = session.data.qpos.copy(), session.data.ctrl.copy()
        with self.assertRaisesRegex(PlanningError, "goal violates collision clearance"):
            motion.plan_joint_target(target)
        np.testing.assert_array_equal(session.data.qpos, before[0])
        np.testing.assert_array_equal(session.data.ctrl, before[1])
        self.assertEqual(session.data.time, 0)

    def test_rrt_checks_interior_of_path_and_routes_around_obstacle(self):
        session = fixture()
        motion = JointMotion(session, "cart", ("x", "y"))
        self.assertFalse(motion.configuration_contacts([0, 0]))
        self.assertFalse(motion.configuration_contacts([1.2, 0]))
        self.assertTrue(motion.configuration_contacts([0.6, 0]))
        plan = motion.plan_joint_target([1.2, 0], seed=7)
        self.assertEqual(plan.method, "rrt_with_checked_shortcuts")
        self.assertGreater(len(plan.waypoints), 2)
        self.assertGreater(max(abs(q[1]) for q in plan.waypoints), 0.19)
        for first, second in zip(plan.waypoints, plan.waypoints[1:]):
            for fraction in np.linspace(0, 1, 101):
                q = np.array(first) + fraction * (np.array(second) - first)
                self.assertFalse(motion.configuration_contacts(q))
        result = motion.execute(plan)
        self.assertEqual(result["status"], "succeeded", result["failure"])
        self.assertFalse(result["collisions"])
        np.testing.assert_allclose(
            result["final_joint_positions"], [1.2, 0], atol=0.0002
        )

    def test_full_model_signature_covers_options_meshes_and_pair_filters(self):
        session = fixture()
        model = session.model
        original = _model_signature(model)
        self.assertEqual(original, _model_signature(copy(model)))
        mutations = (
            lambda m: setattr(
                m.opt, "disableflags", int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            ),
            lambda m: setattr(m.opt, "timestep", 0.001),
            lambda m: m.mesh_vert.__setitem__((0, 0), m.mesh_vert[0, 0] + 0.001),
            lambda m: m.geom_conaffinity.__setitem__(0, 0),
            lambda m: m.geom_pos.__setitem__((0, 0), 0.3),
            lambda m: m.pair_margin.__setitem__(0, 0.005),
            lambda m: m.exclude_signature.__setitem__(0, 0),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                motion = JointMotion(session, "cart", ("x", "y"))
                plan = motion.plan_joint_target([0.1, 0])
                mutate(model)
                self.assertNotEqual(plan.model_signature, _model_signature(model))
                with self.assertRaisesRegex(PlanningError, "Model changed"):
                    motion.execute(plan)
                self.assertEqual(session.data.time, 0)
                session = fixture()
                model = session.model
        for flag in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        ):
            model.opt.disableflags = int(flag)
            with self.assertRaisesRegex(PlanningError, "must remain enabled"):
                JointMotion(session, "cart", ("x", "y"))

    def test_stale_nonselected_state_and_planning_settings_are_rejected(self):
        session = fixture()
        motion = JointMotion(session, "cart", ("x", "y"))
        plan = motion.plan_joint_target([0.1, 0])
        for altered in (
            replace(plan, clearance_m=0.001),
            replace(plan, interpolation_steps=(0.2, 0.2)),
        ):
            with self.assertRaisesRegex(PlanningError, "settings differ"):
                motion.execute(altered)
        loose = session.model.joint("loose").qposadr[0]
        session.data.qpos[loose] += 0.001
        with self.assertRaisesRegex(PlanningError, "world state changed"):
            motion.execute(plan)
        self.assertEqual(session.data.time, 0)
        np.testing.assert_array_equal(session.data.ctrl, [0, 0])
        with self.assertRaisesRegex(ValueError, "selected mechanism"):
            motion.plan_to_point("static_site", [0, 0, 0.25])

    def test_explicit_collision_pairs_keep_their_effect_with_zero_geom_masks(self):
        session = fixture()
        tool = session.model.geom("tool").id
        session.model.geom_contype[tool] = 0
        session.model.geom_conaffinity[tool] = 0
        motion = JointMotion(session, "cart", ("x", "y"))
        # The obstacle ends at x=.48; a tool centred at .435 leaves 10 mm.
        # A pair margin overrides geom margins, so it must be checked too.
        self.assertTrue(motion.configuration_contacts([0.435, 0]))
        self.assertEqual(session.model.pair_margin[0], 0)
        self.assertEqual(session.model.geom_contype[tool], 0)

    def test_measured_actuation_failure_requests_hold_without_reset(self):
        session = fixture()
        session.model.actuator_forcelimited[:] = True
        session.model.actuator_forcerange[:] = 0
        motion = JointMotion(session, "cart", ("x", "y"))
        plan = motion.plan_joint_target([0.2, 0])
        result = motion.execute(plan)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"], "trajectory_tracking_limit")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(session.data.time, 0)
        np.testing.assert_array_equal(session.data.ctrl, motion.measured())
        self.assertLess(motion.measured()[0], 0.001)
        session.reset()
        self.assertEqual(session.data.time, 0)

    def test_moving_nonselected_body_aborts_stationary_world_execution(self):
        session = fixture()
        loose_dof = session.model.joint("loose").dofadr[0]
        session.data.qfrc_applied[loose_dof] = 1
        motion = JointMotion(session, "cart", ("x", "y"))
        result = motion.execute(motion.plan_joint_target([0.2, 0]))
        self.assertEqual(result["failure"], "stationary_world_moved")
        self.assertTrue(result["hold_requested"])
        self.assertGreater(result["maximum_stationary_geometry_displacement_m"], 0.001)
        self.assertGreater(session.data.time, 0)


if __name__ == "__main__":
    unittest.main()
