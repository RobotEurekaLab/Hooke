"""Finite CPU jobs, scene exclusion, bounded memory and measured API results."""

from copy import deepcopy
from pathlib import Path
from threading import Event
from time import monotonic
from unittest.mock import patch
import json
import sys
import unittest

from flask import Flask
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.runtime import InstrumentSession
from webui import real_labs_api
from test_real_labs_runtime import MANIFEST, MODEL


class OperationApiTests(unittest.TestCase):
    def setUp(self):
        self.sessions = {}
        self.releases = []
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.register_blueprint(real_labs_api.bp)
        self.client = self.app.test_client()
        definitions = {key: {"id": key} for key in ("alpha", "beta", "gamma")}
        patches = {
            "catalog": patch.object(real_labs_api, "scenes", return_value=definitions),
            "build": patch.object(
                real_labs_api, "_build_session", side_effect=self.build
            ),
            "metadata": patch.object(
                real_labs_api.operations,
                "available_operations",
                return_value=[
                    dict(
                        id="cycle",
                        label="Cycle",
                        equipment_id="reader",
                        description="Test cycle",
                    )
                ],
            ),
            "policy": patch.object(
                real_labs_api.operations, "control_policy", return_value=[]
            ),
            "validate": patch.object(
                real_labs_api.operations,
                "validate_operation",
                side_effect=self.validate,
            ),
            "run": patch.object(real_labs_api.operations, "run_operation"),
        }
        for name, item in patches.items():
            setattr(self, name, item.start())
            self.addCleanup(item.stop)
        self.addCleanup(self.finish_workers)

    def build(self, identifier):
        manifest = deepcopy(MANIFEST)
        manifest["id"] = identifier
        session = InstrumentSession(mujoco.MjModel.from_xml_string(MODEL), manifest)
        self.sessions[identifier] = session
        return session

    @staticmethod
    def validate(session, operation):
        if operation != "cycle":
            raise ValueError("Unknown operation")

    def blocked_runner(self, *, success=True):
        started, release = Event(), Event()
        self.releases.append(release)

        def run(session, operation):
            started.set()
            if not release.wait(10):
                raise TimeoutError("Test did not release operation")
            session.advance(10)
            return {
                "success": success,
                "scene": session.manifest["id"],
                "physics_steps": 10,
            }

        return run, started, release

    def wait_terminal(self, scene="alpha", timeout=5):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            response = self.client.get(f"/api/real-labs/{scene}/operation")
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            if body["job"]["status"] != "running":
                return body
            Event().wait(0.01)
        self.fail("Operation did not terminate")

    def finish_workers(self):
        for release in self.releases:
            release.set()
        cache = self.app.extensions.get("real_labs_sessions")
        if cache:
            with cache.lock:
                running = [
                    key
                    for key, entry in cache.items.items()
                    if entry.job and entry.job["status"] == "running"
                ]
            for scene in running:
                self.wait_terminal(scene)

    def start(self, scene="alpha"):
        response = self.client.post(
            f"/api/real-labs/{scene}/operation", json={"operation": "cycle"}
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        self.assertEqual(response.get_json()["job"]["status"], "running")
        return response.get_json()["job"]

    def test_metadata_and_invalid_requests_do_not_move_physics(self):
        self.assertEqual(
            self.client.get("/api/real-labs/alpha/operation").status_code, 404
        )
        self.build.assert_not_called()
        state = self.client.get("/api/real-labs/alpha/state").get_json()
        self.assertEqual(state["operations"][0]["id"], "cycle")
        self.assertEqual(state["control_policy"], [])
        initial = self.sessions["alpha"].data.qpos.copy()
        for body in (
            None,
            [],
            {},
            {"operation": 2},
            {"operation": ""},
            {"operation": "cycle", "extra": 1},
            {"operation": "unknown"},
        ):
            with self.subTest(body=body):
                response = self.client.post("/api/real-labs/alpha/operation", json=body)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(
            self.client.post(
                "/api/real-labs/absent/operation", json={"operation": "cycle"}
            ).status_code,
            404,
        )
        self.run.assert_not_called()
        self.assertEqual(self.sessions["alpha"].data.time, 0)
        np.testing.assert_array_equal(self.sessions["alpha"].data.qpos, initial)

    def test_running_job_excludes_same_scene_and_polls_without_inspection(self):
        self.run.side_effect, started, release = self.blocked_runner()
        job = self.start()
        self.assertTrue(started.wait(2))
        session = self.sessions["alpha"]
        with patch.object(session, "inspect", wraps=session.inspect) as inspect:
            response = self.client.get("/api/real-labs/alpha/operation")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("state", response.get_json())
            responses = [
                self.client.get("/api/real-labs/alpha/state"),
                self.client.post("/api/real-labs/alpha/reset"),
                self.client.post(
                    "/api/real-labs/alpha/command",
                    json={"equipment": "reader", "control": "door", "value": 0.1},
                ),
                self.client.post(
                    "/api/real-labs/alpha/operation", json={"operation": "cycle"}
                ),
            ]
            for response in responses:
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.get_json()["job"], job)
            inspect.assert_not_called()
        release.set()
        final = self.wait_terminal()
        self.assertEqual(final["job"]["status"], "succeeded")
        self.assertEqual(final["job"]["id"], job["id"])
        self.assertEqual(final["job"]["result"]["physics_steps"], 10)
        self.assertGreater(final["state"]["time_s"], 0)
        self.assertEqual(self.client.get("/api/real-labs/alpha/state").status_code, 200)

    def test_other_scene_remains_usable_and_only_idle_entry_is_evicted(self):
        self.run.side_effect, started, release = self.blocked_runner()
        self.start()
        self.assertTrue(started.wait(2))
        response = self.client.post(
            "/api/real-labs/beta/command",
            json={"equipment": "reader", "control": "door", "value": 0.1},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.client.get("/api/real-labs/beta/operation").status_code, 404
        )
        self.assertEqual(self.client.get("/api/real-labs/gamma/state").status_code, 200)
        cache = self.app.extensions["real_labs_sessions"]
        self.assertEqual(set(cache.items), {"alpha", "gamma"})
        self.assertEqual(
            self.client.get("/api/real-labs/alpha/operation").get_json()["job"][
                "status"
            ],
            "running",
        )
        release.set()
        self.assertEqual(self.wait_terminal()["job"]["result"]["scene"], "alpha")

    def test_two_running_entries_reject_third_without_build_or_eviction(self):
        runner, started, release = self.blocked_runner()
        self.run.side_effect = runner
        alpha, beta = self.start("alpha"), self.start("beta")
        self.assertNotEqual(alpha["id"], beta["id"])
        for path, method, body in (
            ("state", "get", None),
            ("reset", "post", {}),
            ("operation", "post", {"operation": "cycle"}),
            (
                "command",
                "post",
                {"equipment": "reader", "control": "door", "value": 0.1},
            ),
        ):
            response = getattr(self.client, method)(
                f"/api/real-labs/gamma/{path}", json=body
            )
            self.assertEqual(response.status_code, 409)
        self.assertEqual(self.build.call_count, 2)
        self.assertEqual(
            set(self.app.extensions["real_labs_sessions"].items), {"alpha", "beta"}
        )
        release.set()
        self.assertEqual(self.wait_terminal("alpha")["job"]["result"]["scene"], "alpha")
        self.assertEqual(self.wait_terminal("beta")["job"]["result"]["scene"], "beta")

    def test_reported_failure_preserves_final_state_and_explicit_reset_is_separate(
        self,
    ):
        self.run.side_effect, started, release = self.blocked_runner(success=False)
        self.start()
        self.assertTrue(started.wait(2))
        release.set()
        final = self.wait_terminal()
        self.assertEqual(final["job"]["status"], "failed")
        self.assertFalse(final["job"]["result"]["success"])
        self.assertIn("error", final["job"])
        self.assertGreater(final["state"]["time_s"], 0)
        self.assertEqual(self.sessions["alpha"].data.time, final["state"]["time_s"])
        self.assertEqual(
            self.client.post("/api/real-labs/alpha/reset").get_json()["time_s"], 0
        )
        reset_poll = self.client.get("/api/real-labs/alpha/operation").get_json()
        self.assertEqual(reset_poll["job"], final["job"])
        self.assertEqual(reset_poll["state"]["time_s"], 0)
        self.assertEqual(reset_poll["completed_state"], final["completed_state"])
        response = self.client.post(
            "/api/real-labs/alpha/command",
            json={"equipment": "reader", "control": "door", "value": 0.1},
        )
        self.assertEqual(response.status_code, 200)
        command_poll = self.client.get("/api/real-labs/alpha/operation").get_json()
        self.assertEqual(command_poll["state"], response.get_json()["state"])
        self.assertEqual(command_poll["completed_state"], final["completed_state"])

    def test_terminal_poll_rejects_concurrent_synchronous_physics_access(self):
        self.run.return_value = {"success": True}
        self.start()
        self.wait_terminal()
        cache = self.app.extensions["real_labs_sessions"]
        entry = cache.acquire("alpha")
        try:
            response = self.client.get("/api/real-labs/alpha/operation")
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.get_json()["job"]["status"], "succeeded")
            self.assertNotIn("state", response.get_json())
        finally:
            entry.lock.release()

    def test_runner_exception_after_motion_is_failed_without_reset(self):
        def fail(session, operation):
            session.advance(5)
            raise RuntimeError("Measured route interrupted")

        self.run.side_effect = fail
        self.start()
        final = self.wait_terminal()
        self.assertEqual(final["job"]["status"], "failed")
        self.assertIn("Measured route interrupted", final["job"]["error"])
        self.assertGreater(final["state"]["time_s"], 0)
        self.assertEqual(self.sessions["alpha"].data.time, final["state"]["time_s"])

    def test_nonboolean_success_never_reports_success_and_keeps_diagnostics(self):
        for value in (None, 1, "true"):
            with self.subTest(value=value):
                self.run.return_value = {"success": value, "physics_steps": 0}
                self.start()
                final = self.wait_terminal()
                self.assertEqual(final["job"]["status"], "failed")
                self.assertEqual(final["job"]["result"]["success"], value)
                self.assertIn("explicit boolean", final["job"]["error"])
                self.assertEqual(final["state"]["time_s"], 0)

    @staticmethod
    def strict_json(response):
        def reject_constant(value):
            raise ValueError(f"Nonstandard JSON constant: {value}")

        return json.loads(
            response.get_data(as_text=True), parse_constant=reject_constant
        )

    def test_failed_nonfinite_report_remains_browser_readable(self):
        self.run.return_value = {
            "success": False,
            "failure": "Contact measurement failed",
            "measurements": [np.float64("nan"), np.array([float("inf"), 2.0])],
        }
        self.start()
        self.wait_terminal()
        final = self.strict_json(self.client.get("/api/real-labs/alpha/operation"))
        self.assertEqual(final["job"]["status"], "failed")
        self.assertEqual(final["job"]["result"]["measurements"], [None, [None, 2.0]])
        self.assertEqual(len(final["job"]["nonfinite_report_fields"]), 2)
        self.assertIn("Contact measurement failed", final["job"]["error"])

    def test_success_report_cannot_override_nonfinite_final_physics(self):
        def corrupt(session, operation):
            session.data.time = float("nan")
            session.data.qpos[:] = float("nan")
            session.data.qvel[:] = float("inf")
            session.data.ctrl[:] = float("inf")
            return {"success": True}

        self.run.side_effect = corrupt
        self.start()
        self.wait_terminal()
        final = self.strict_json(self.client.get("/api/real-labs/alpha/operation"))
        self.assertEqual(final["job"]["status"], "failed")
        self.assertEqual(
            set(final["job"]["invalid_state_fields"]), {"time", "qpos", "qvel", "ctrl"}
        )
        self.assertIn("Nonfinite final simulation state", final["job"]["error"])
        for snapshot in (final["state"], final["completed_state"]):
            self.assertTrue(snapshot["invalid_physics_state"])
            self.assertIsNone(snapshot["time_s"])
        current = self.strict_json(self.client.get("/api/real-labs/alpha/state"))
        self.assertTrue(current["invalid_physics_state"])
        # Serializing diagnostics must not repair or reset the actual failed state.
        self.assertTrue(np.isnan(self.sessions["alpha"].data.qpos).all())
        self.assertTrue(np.isinf(self.sessions["alpha"].data.ctrl).all())

    def test_nonfinite_report_cannot_claim_success_with_finite_physics(self):
        self.run.return_value = {"success": True, "measurement": float("nan")}
        self.start()
        self.wait_terminal()
        final = self.strict_json(self.client.get("/api/real-labs/alpha/operation"))
        self.assertEqual(final["job"]["status"], "failed")
        self.assertIsNone(final["job"]["result"]["measurement"])
        self.assertIn("Nonfinite operation report", final["job"]["error"])

    def test_worker_rechecks_preflight_and_retains_unmodified_state_on_failure(self):
        self.validate.side_effect = [None, ValueError("State is not ready")]
        self.start()
        final = self.wait_terminal()
        self.assertEqual(final["job"]["status"], "failed")
        self.assertIn("State is not ready", final["job"]["error"])
        self.assertEqual(final["state"]["time_s"], 0)
        self.run.assert_not_called()

    def test_thread_start_failure_releases_scene_and_records_failure(self):
        with patch.object(
            real_labs_api.Thread,
            "start",
            side_effect=RuntimeError("Thread unavailable"),
        ):
            response = self.client.post(
                "/api/real-labs/alpha/operation", json={"operation": "cycle"}
            )
        self.assertEqual(response.status_code, 503)
        final = self.wait_terminal()
        self.assertEqual(final["job"]["status"], "failed")
        self.assertEqual(final["state"]["time_s"], 0)
        self.assertEqual(self.client.get("/api/real-labs/alpha/state").status_code, 200)

    def test_synchronous_access_is_pinned_and_busy_without_a_job(self):
        self.client.get("/api/real-labs/alpha/state")
        cache = self.app.extensions["real_labs_sessions"]
        entry = cache.acquire("alpha")
        try:
            response = self.client.get("/api/real-labs/alpha/state")
            self.assertEqual(response.status_code, 409)
            self.assertNotIn("job", response.get_json())
            for scene in ("beta", "gamma"):
                self.assertEqual(
                    self.client.get(f"/api/real-labs/{scene}/state").status_code, 200
                )
            self.assertIn("alpha", cache.items)
        finally:
            entry.lock.release()


class LeedsOperationApiTests(unittest.TestCase):
    def test_real_guard_and_complete_loaded_slide_through_http(self):
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(real_labs_api.bp)
        client = app.test_client()
        base = "/api/real-labs/leeds_environmental_tribology"
        state = client.get(base + "/state").get_json()
        self.assertEqual(
            {item["id"] for item in state["operations"]}, {"loaded_slide", "retract"}
        )
        self.assertEqual(state["control_policy"][0]["upper"], 0.012)
        entry = app.extensions["real_labs_sessions"].items[
            "leeds_environmental_tribology"
        ]
        initial = entry.session.data.qpos.copy()
        rejected = client.post(
            base + "/command",
            json={
                "equipment": "tribology",
                "control": "contact_approach",
                "value": 0.02125,
            },
        )
        self.assertEqual(rejected.status_code, 400, rejected.get_json())
        np.testing.assert_array_equal(entry.session.data.qpos, initial)
        self.assertEqual(entry.session.data.time, 0)
        self.assertEqual(
            client.post(
                base + "/operation", json={"operation": "plate_loading"}
            ).status_code,
            400,
        )
        response = client.post(base + "/operation", json={"operation": "loaded_slide"})
        self.assertEqual(response.status_code, 202, response.get_json())
        deadline = monotonic() + 60
        final = None
        while monotonic() < deadline:
            final = client.get(base + "/operation").get_json()
            if final["job"]["status"] != "running":
                break
            Event().wait(0.02)
        self.assertEqual(final["job"]["status"], "succeeded", final["job"])
        report = final["job"]["result"]
        self.assertTrue(report["success"])
        self.assertEqual(report["physics_steps"], 5125)
        self.assertEqual(report["completed_checks"], 5125)
        self.assertGreater(report["loaded_relative_path_m"], 0.035)
        self.assertFalse(report["reset_performed"])
        self.assertAlmostEqual(final["state"]["time_s"], 10.25)
        self.assertEqual(client.post(base + "/reset").get_json()["time_s"], 0)


if __name__ == "__main__":
    unittest.main()
