"""Image-guided experts using ordinary actuator commands and actual contacts."""

import math

import mujoco
import numpy as np

from microscopy.mechanics import MicroscopyMechanics
from microscopy.kinematics import tool_commands
from microscopy.optics import Microscope
from microscopy.scene import HOMES, ORIGIN, PICK_TARGET, PUSH_TARGET, RADIUS, WELL, build_xml
from simulation import Manager
from task import Expert, SCENE_ROOT, Task


class MicroscopyTask(Task):
    default_scene = SCENE_ROOT / "microscopy_workstation.xml"
    default_task = "microscopy_push"
    time_limit = 45.0
    early_stop = False
    operation = "push"
    native_physics_options = {"contact_offset_m": 1e-6, "solver_velocity_iterations": 16}
    mechanics_type = MicroscopyMechanics
    microscope_type = Microscope
    volume_unit = "nL"
    maximum_target_volume = 200.

    @classmethod
    def load(cls, scene=None):
        return super().load(scene) if scene is not None else mujoco.MjSpec.from_string(build_xml())

    def __init__(self, spec):
        self.mechanics = self.mechanics_type()
        self.microscope = self.microscope_type()
        super().__init__(Manager.from_spec(spec, [self.mechanics]))
        self.drives = {self.model.actuator(i).name.removesuffix("_drive"): i for i in range(self.model.nu)}
        from microscopy.drive_control import InstrumentDrives
        self.instrument_drives = InstrumentDrives(self.model)
        if self.instrument_drives.parallel:
            self.native_physics_options = dict(self.native_physics_options,
                soft_connect_constraints=False,
                world_origin_m=self.model.body("injector_platform").pos.tolist())
        from microscopy.cad_assets import asset_evidence
        self.command_limits = {name: list(limits) for name, limits in self.instrument_drives.limits.items()}
        for name, limits in asset_evidence(self.model).get("stage", {}).get("operating_limits_m", {}).items():
            hardware = self.command_limits[name]
            self.command_limits[name] = [max(hardware[0], limits[0]), min(hardware[1], limits[1])]

    def reset(self, seed=None):
        self.manager.reset()
        rng = np.random.default_rng(seed)
        for name in self.mechanics.samples:
            address = self.model.joint(name + "_free").qposadr[0]
            self.data.qpos[address:address+2] += rng.uniform(-.00010, .00010, 2)
        for side in ("a", "b"):
            if "jaw_"+side not in self.drives:
                continue
            self.data.qpos[self.model.joint("jaw_" + side).qposadr[0]] = .0013
            self.data.ctrl[self.drives["jaw_" + side]] = .0013
        initial_focus = float(rng.uniform(.00025, .00045))
        self.data.qpos[self.model.joint("focus").qposadr[0]] = initial_focus
        self.data.ctrl[self.drives["focus"]] = initial_focus
        self.data.userdata[:] = [0, 100]
        mujoco.mj_forward(self.model, self.data)
        self.mechanics.reset(self.data)
        self.completed = self.focused = False
        self.focus_scan = []
        self.phase_history = []
        self.task_info = dict(
            prefix=f"Motorized inverted microscopy: {self.operation}", seed=seed,
            camera_mapping={"image": "instrument_closeup", "world": "workstation_overview"},
            state_indices=list(range(self.model.nq)), action_indices=list(range(self.model.nu)),
            display_only=False, microscopy_operation=self.operation,
            imaging_model="Gaussian defocus with calibrated orthographic image formation",
        )
        return self.task_info

    def microscope_image(self, *, annotate=True):
        return self.microscope.render(self.mechanics.image_state(self.data), annotate=annotate)

    def runtime_visuals(self):
        from microscopy.pneumatics import hose_visuals
        return hose_visuals(self.model, self.data)

    def capture_microscopy(self, output, index):
        from microscopy.artifacts import microscope_frame
        microscope_frame(self, output, index)

    def public_state(self):
        from microscopy.cad_assets import asset_evidence
        evidence = asset_evidence(self.model)
        return dict(
            time_s=float(self.data.time), operation=self.operation,
            phase=self.phase_history[-1]["phase"] if self.phase_history else "ready",
            encoders_m=self.instrument_drives.feedback(self.data),
            pressure_pa=float(self.data.userdata[0]),
            target_nl=float(self.data.userdata[1]),
            delivered_nl=self.mechanics.ledger.state("well").volume_m3 * 1e12,
            instrument_profile=evidence["profile"],
            manipulator_profile=self.instrument_drives.profile,
            stand_profile=evidence.get("microscope", {}).get("stand_profile", "auto"),
            stage_profile=evidence.get("stage", {}).get("profile", "reference"),
            actuator_limits_m=self.instrument_drives.limits,
            command_limits_m=self.command_limits,
            sample_scale=dict(kind="calibration_bead", diameters_m=[2*RADIUS]*3,
                              length_unit="m", geometry_display_gain=1.),
            calibration=self.microscope.public_calibration(),
        )

    def microscopy_checks(self):
        data, mechanics = self.data, self.mechanics
        stage = np.array([data.qpos[self.model.joint("stage_x").qposadr[0]],
                          data.qpos[self.model.joint("stage_y").qposadr[0]]])
        checks = dict(episode_completed=self.completed, autofocus=self.focused,
                      finite_state=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()))
        if self.operation == "push":
            position = data.xpos[mechanics.samples["bead_push"]][:2] - ORIGIN[:2] - stage
            checks.update(probe_contact_50ms=mechanics.push_contact_s >= .05,
                          bead_at_target=bool(np.linalg.norm(position - PUSH_TARGET) < .00035))
        elif self.operation == "pick_place":
            position = data.xpos[mechanics.samples["bead_pick"]][:2] - ORIGIN[:2] - stage
            checks.update(bilateral_grasp_50ms=mechanics.bilateral_contact_s >= .05,
                          contact_lift_800um=mechanics.lift_m >= .0008,
                          bead_at_target=bool(np.linalg.norm(position - PICK_TARGET) < .00035),
                          released=not mechanics.grasped)
        else:
            volume = mechanics.ledger.state("well").volume_m3 * 1e12
            checks.update(delivered_100nl=bool(abs(volume - 100) < .05),
                          pressure_off=bool(data.userdata[0] == 0),
                          conserved_volume=bool(abs(mechanics.ledger.total_m3
                                                    - mechanics.ledger.initial_total_m3) < 1e-20))
        return checks

    def check(self):
        return all(self.microscopy_checks().values())

    def microscopy_report(self):
        from microscopy.cad_assets import asset_evidence
        return dict(self.mechanics.report(self.data), checks=self.microscopy_checks(),
                    operation=self.operation, focus_scan=self.focus_scan,
                    assets=asset_evidence(self.model),
                    controller="Color-fiducial image feedback and actuator encoders; no object pose commands.",
                    limitations=["Uncalibrated Gaussian defocus; no wave-optics PSF.",
                                 "No cell membrane, adhesion, liquid drag or free-surface CFD.",
                                 "No claim of nanometre positioning or real instrument fidelity."])


class MicroscopyExpert(MicroscopyTask, Expert):
    def phase(self, name):
        self.phase_history.append(dict(phase=name, time_s=float(self.data.time)))

    def wait(self, seconds):
        for _ in range(math.ceil(seconds / self.dt)):
            self.step_and_log({})

    def validate_commands(self, commands):
        for name, target in commands.items():
            if name not in self.command_limits or type(target) not in (int, float) or not math.isfinite(target):
                raise ValueError("Unknown actuator or non-finite command")
            if not self.instrument_drives.limits[name][0] <= target <= self.instrument_drives.limits[name][1]:
                raise ValueError(f"{name} command exceeds travel limit")
            if not self.command_limits[name][0] <= target <= self.command_limits[name][1]:
                raise ValueError(f"{name} command exceeds the installed operating envelope; objective retraction is required for larger stage travel")
        self.instrument_drives.controls(self.data, commands)

    def command(self, commands, seconds=.2):
        self.validate_commands(commands)
        self.instrument_drives.set_targets(self.data, commands)
        self.wait(seconds)

    def move(self, tool, position, seconds=.5, *, focus_m=None):
        commands = tool_commands(self.model, tool, np.asarray(position)-HOMES[tool])
        if focus_m is not None:
            commands["focus"] = focus_m
        self.validate_commands(commands)
        destination = np.array(list(commands.values()))
        names = list(commands)
        targets = self.instrument_drives.targets(self.data)
        initial = np.array([targets[name] for name in names])
        steps = math.ceil(seconds / self.dt)
        for i in range(steps):
            t = (i + 1) / steps
            values = initial + (destination - initial) * t * t * (3 - 2 * t)
            self.instrument_drives.set_targets(self.data, dict(zip(names, values)))
            self.step_and_log({})
        if self.volume_unit == "pL" and tool in self.instrument_drives.parallel:
            self.settle_microposition(commands)
        else:
            self.wait(.12)

    def settle_microposition(self, desired):
        """Correct parallel endpoint error through ordinary motor commands.

        Feedback is the simulated mechanism position; this convergence
        threshold is not a measured instrument accuracy.
        """
        # Let the closed mechanism settle before feeding endpoint error back.
        # Correcting its transient lag immediately can excite the passive rods.
        self.wait(.25)
        record = dict(targets_m=desired.copy(), iterations=0, samples=[],
                      feedback_window_s=.05, error_statistic="mean simulated position",
                      mean_error_limit_m=5e-8, rms_jitter_limit_m=3e-7, peak_jitter_limit_m=8e-7)
        for iteration in range(11):
            # A single sample can cross the tolerance while the endpoint is
            # still oscillating. Sample actual physics at fixed motor targets;
            # cell mechanics continue to use each instantaneous position.
            positions = []
            for _ in range(math.ceil(record["feedback_window_s"] / self.dt)):
                self.step_and_log({})
                feedback = self.instrument_drives.feedback(self.data)
                positions.append([feedback[name] for name in desired])
            positions = np.asarray(positions)
            mean = positions.mean(axis=0)
            rms_jitter = float(positions.std(axis=0).max())
            peak_jitter = float(np.abs(positions - mean).max())
            errors = {name: value - mean[i] for i, (name, value) in enumerate(desired.items())}
            maximum = max(abs(value) for value in errors.values())
            record["samples"].append(dict(time_s=float(self.data.time), errors_m=errors.copy(),
                rms_jitter_m=rms_jitter, peak_jitter_m=peak_jitter,
                instantaneous_error_m=float(np.abs(positions[-1] - list(desired.values())).max())))
            if iteration == 0:
                record["initial_error_m"] = maximum
            record.update(iterations=iteration, final_error_m=maximum)
            if (maximum <= record["mean_error_limit_m"] and rms_jitter <= record["rms_jitter_limit_m"]
                    and peak_jitter <= record["peak_jitter_limit_m"]):
                self.motion_feedback.append(record)
                return
            if iteration == 10:
                self.motion_feedback.append(record)
                raise RuntimeError(f"Parallel microposition did not settle ({maximum * 1e6:.3f} µm mean residual, "
                                   f"{rms_jitter * 1e6:.3f} µm RMS jitter); motion stopped")
            targets = self.instrument_drives.targets(self.data)
            self.command({name: float(targets[name] + np.clip(.25 * error, -5e-7, 5e-7))
                          for name, error in errors.items()}, .15)

    def autofocus(self):
        self.completed = False
        self.focus_scan = []
        self.phase("autofocus")
        for focus in np.linspace(-.0006, .0006, 13):
            self.command({"focus": float(focus)}, .15)
            score = self.microscope.sharpness(self.microscope_image(annotate=False))
            self.focus_scan.append(dict(focus_m=float(focus), sharpness=score))
        best = max(self.focus_scan, key=lambda row: row["sharpness"])
        self.command({"focus": best["focus_m"]}, .2)
        self.focused = abs(float(self.data.qpos[self.model.joint("focus").qposadr[0]])) < .000075

    def locate(self, sample):
        return self.microscope.locate(self.microscope_image(annotate=False), sample)

    def push(self):
        self.phase("locate_bead")
        xy = self.locate("bead_push")
        self.move("probe", [xy[0] - RADIUS - .00025, xy[1], .003])
        self.move("probe", [xy[0] - RADIUS - .00025, xy[1], RADIUS])
        self.phase("contact_push")
        for _ in range(7):
            observed = self.locate("bead_push")
            error = PUSH_TARGET - observed
            if np.linalg.norm(error) < .00015:
                break
            direction = error / np.linalg.norm(error)
            start = observed - direction * (RADIUS + .00025)
            self.move("probe", [*start, .002], .3)
            self.move("probe", [*start, RADIUS], .3)
            end = observed + direction * min(.0008, np.linalg.norm(error))
            contact = end - direction * (RADIUS + .00010)
            self.move("probe", [*contact, RADIUS], .7)
        self.phase("probe_retreat")
        self.move("probe", HOMES["probe"])
        self.wait(.4)

    def pick_place(self):
        self.phase("locate_bead")
        xy = self.locate("bead_pick")
        self.move("gripper", [*xy, .003])
        self.move("gripper", [*xy, RADIUS])
        self.phase("close_microgripper")
        self.command({"jaw_a": .00025, "jaw_b": .00025} if "jaw_b" in self.drives
                     else {"jaw_a": -.00009}, .5)
        self.phase("contact_lift")
        self.move("gripper", [*xy, .0025], .7)
        self.phase("place_bead")
        self.move("gripper", [*PICK_TARGET, .0025], .8)
        self.move("gripper", [*PICK_TARGET, RADIUS], .7)
        self.phase("release")
        self.command({name: .0013 for name in ("jaw_a", "jaw_b") if name in self.drives}, .4)
        self.move("gripper", [*PICK_TARGET, .004])
        self.wait(.4)

    def injection(self):
        self.phase("align_capillary")
        self.move("injector", [*WELL, .003])
        self.move("injector", [*WELL, .00045])
        self.phase("pressure_injection")
        self.data.userdata[:] = [12000, 100]
        self.wait(6.0)
        self.data.userdata[0] = 0
        self.phase("capillary_retreat")
        self.move("injector", HOMES["injector"])

    def execute(self):
        self.wait(.2)
        self.autofocus()
        getattr(self, self.operation)()
        self.phase("complete")
        self.completed = True
        self.finish()


def operation_task(operation):
    class Operation(MicroscopyTask):
        pass
    class OperationExpert(MicroscopyExpert, Operation):
        pass
    Operation.operation = OperationExpert.operation = operation
    Operation.default_task = OperationExpert.default_task = "microscopy_" + operation
    Operation.Expert = OperationExpert
    return Operation


CalibrationPush = operation_task("push")
CalibrationPickPlace = operation_task("pick_place")
CalibrationInjection = operation_task("injection")


def make_task(operation):
    if operation == "suction_injection":
        from microscopy.suction_tasks import SuctionInjection
        cls = SuctionInjection
    elif operation == "cell_injection":
        from microscopy.cell_tasks import CellInjection
        cls = CellInjection
    elif operation == "injection":
        from microscopy.cell_tasks import CellDoseInjection
        cls = CellDoseInjection
    else:
        from microscopy.cell_handling_tasks import CellPush, CellPickPlace
        cls = {"push": CellPush, "pick_place": CellPickPlace}[operation]
    return cls.Expert(cls.load())
