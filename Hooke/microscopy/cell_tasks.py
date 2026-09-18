"""Image-guided adherent-cell phantom injection, using actual motor motion."""

import mujoco
import numpy as np

from microscopy.cell_mechanics import CellMechanics
from microscopy.cell_optics import CellMicroscope
from microscopy.cell_scene import (CELL_CENTER, CELL_RADII, NEEDLE_BACK,
                                   NEEDLE_LENGTH_M, TIP_INNER_M, build_xml)
from microscopy.tasks import MicroscopyExpert, MicroscopyTask


class CellInjection(MicroscopyTask):
    operation = "cell_injection"
    default_task = "microscopy_cell_injection"
    mechanics_type = CellMechanics
    microscope_type = CellMicroscope
    volume_unit = "pL"
    maximum_target_volume = 1.
    cell_center = CELL_CENTER
    task_prefix = "Adherent-cell phantom cytoplasmic injection"
    specimen_kind = "adherent_cell_phantom"
    injection_depth_m = 2e-6

    @classmethod
    def load(cls, scene=None):
        return super().load(scene) if scene is not None else mujoco.MjSpec.from_string(build_xml())

    def reset(self, seed=None):
        self.manager.reset()
        rng = np.random.default_rng(seed)
        shift = rng.uniform(-3e-6, 3e-6, 2)
        for name in ("cell_0_shell", "cell_0_nucleus"):
            self.model.geom(name).pos[:] = self.cell_center+[*shift, 0.]
        self.model.site("cell_center").pos[:] = self.cell_center+[*shift, 0.]
        initial_focus = float(rng.uniform(2e-6, 4e-6))
        self.data.qpos[self.model.joint("focus").qposadr[0]] = initial_focus
        self.data.ctrl[self.drives["focus"]] = initial_focus
        self.data.userdata[:] = [0., .5]
        mujoco.mj_forward(self.model, self.data)
        self.mechanics.reset(self.data)
        self.completed = self.focused = False
        self.focus_scan, self.phase_history = [], []
        self.motion_feedback = []
        self.task_info = dict(prefix=self.task_prefix, seed=seed,
                              camera_mapping={"image": "instrument_closeup", "world": "workstation_overview",
                                              "sample": "cell_detail"},
                              state_indices=list(range(self.model.nq)), action_indices=list(range(self.model.nu)),
                              display_only=False, microscopy_operation=self.operation,
                              imaging_model="Estimated scalar phase contrast with a separate synthetic fluorescence observation",
                              fidelity="reduced_uncalibrated", volume_unit="pL")
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "objective_detail") >= 0:
            self.task_info["camera_mapping"]["objective"] = "objective_detail"
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "detection_path_detail") >= 0:
            self.task_info["camera_mapping"]["detection"] = "detection_path_detail"
        return self.task_info

    def public_state(self):
        from microscopy.cad_assets import asset_evidence
        evidence = asset_evidence(self.model)
        return dict(time_s=float(self.data.time), operation=self.operation, volume_unit="pL",
                    instrument_profile=evidence["profile"],
                    manipulator_profile=self.instrument_drives.profile,
                    stand_profile=evidence.get("microscope", {}).get("stand_profile", "auto"),
                    stage_profile=evidence.get("stage", {}).get("profile", "reference"),
                    phase=self.phase_history[-1]["phase"] if self.phase_history else "ready",
                    encoders_m=self.instrument_drives.feedback(self.data),
                    actuator_limits_m=self.instrument_drives.limits,
                    command_limits_m=self.command_limits,
                    pressure_pa=float(self.data.userdata[0]), target_pl=float(self.data.userdata[1]),
                    delivered_pl=self.mechanics.ledger.state("cell").volume_m3*1e15,
                    actual_pressure_pa=self.mechanics.actual_pressure_pa,
                    compensation_pa=self.mechanics.compensation_pa,
                    force_nn=self.mechanics.force_n*1e9, punctured=self.mechanics.punctured,
                    needle_clogged=self.mechanics.needle_clogged,
                    withdrawn=self.mechanics.withdrawn,
                    sample_scale=self.sample_scale(),
                    calibration=self.optical_calibration())

    def optical_calibration(self):
        calibration = self.microscope.public_calibration()
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "reference_40x_nose") >= 0:
            from microscopy.reference_objective import installed_dimensions
            objective = installed_dimensions(self.model, self.data)
            calibration.update(reference_objective=objective,
                virtual_sensor_width_m=objective["nominal_magnification"]*calibration["field_width_m"],
                virtual_sensor_basis="Declared object-space field times nominal 40X; not measured camera hardware")
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "collection_nominal_sensor_plane") >= 0:
            from microscopy.reference_detection import installed_dimensions
            detection = installed_dimensions(self.model, self.data)
            if not np.isclose(calibration["field_width_m"], detection["nominal_object_field_width_m"], rtol=0, atol=1e-12):
                raise ValueError("Synthetic field does not match the declared reference sensor and magnification")
            calibration["reference_detection_path"] = detection
        return calibration

    def sample_scale(self):
        """Nominal SI dimensions; neither camera zoom nor a biological calibration."""
        from microscopy.pipettes import INJECTION_PROFILE, pipette_back
        angles = {tool: float(np.rad2deg(np.arcsin(pipette_back(tool)[2])))
                  for tool in ("injector", "holder")
                  if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tool+"_tcp") >= 0}
        return dict(kind=self.specimen_kind, length_unit="m", geometry_display_gain=1.,
                    diameters_m=(2*self.mechanics.cell_radii).tolist(),
                    nuclear_diameters_m=(2*self.mechanics.nuclear_radii).tolist(),
                    needle_tip_outer_diameter_m=2*float(self.model.geom("injector_tip").size[0]),
                    needle_tip_inner_diameter_m=2*TIP_INNER_M,
                    exposed_capillary_length_m=NEEDLE_LENGTH_M,
                    pipette_elevation_deg=angles,
                    injection_profile_m=[list(section) for section in INJECTION_PROFILE.sections],
                    pipette_profile_calibrated=False,
                    biological_dimensions_calibrated=False)

    def microscopy_checks(self):
        m = self.mechanics
        return dict(episode_completed=self.completed, autofocus=self.focused,
                    finite_state=bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()),
                    membrane_contact=bool(m.contact_s >= .01), membrane_deformation=bool(m.maximum_indentation_m >= 1e-6),
                    membrane_puncture=m.punctured, intracellular_dose=bool(abs(m.ledger.state("cell").volume_m3*1e15-.5) < .005),
                    needle_withdrawn=m.withdrawn, pressure_off=bool(self.data.userdata[0] == 0 and m.compensation_pa == 0
                                                                  and m.actual_pressure_pa < 1.),
                    no_overdepth=not m.overdepth, nucleus_avoided=not m.nuclear_contact,
                    conserved_volume=bool(abs(m.ledger.total_m3-m.ledger.initial_total_m3) < 2e-25))

    def microscopy_report(self):
        from microscopy.cad_assets import asset_evidence
        return dict(self.mechanics.report(self.data), checks=self.microscopy_checks(), operation=self.operation,
                    focus_scan=self.focus_scan,
                    sample_scale=self.sample_scale(), calibration=self.optical_calibration(),
                    assets=asset_evidence(self.model),
                    controller="Separate synthetic nuclear-fluorescence image centroid, nominal ray geometry and simulated mechanism-position feedback; no measured hardware accuracy.",
                    motion_feedback=self.motion_feedback,
                    valve_model="Ideal dose-controlled outlet shutoff; upstream pressure has a first-order response.",
                    limitations=["Uncalibrated scalar membrane relaxation and threshold rupture, not a resolved soft-cell solver.",
                                 "No resealing, survival, biological response, liquid CFD or real dose/force calibration.",
                                 "Nominal 3D cell geometry; deformation and tracer are shown in synthetic microscopy.",
                                 "Scalar thin-object phase contrast with empirical axial blur; no measured optics or fluorescence intensity calibration.",
                                 "Statistical visual cell contours and organelles; the mechanical contact envelope remains an ellipsoid.",
                                 "No claim of nanometre positioning or measured biological accuracy."])


class CellInjectionExpert(MicroscopyExpert, CellInjection):
    def locate(self, sample):
        image = self.microscope.render(self.mechanics.image_state(self.data),
                                       annotate=False, channel="fluorescence")
        return self.microscope.locate(image, sample)

    def autofocus(self):
        self.completed = False
        self.focus_scan = []
        self.phase("autofocus")
        for focus in np.linspace(-4e-6, 4e-6, 17):
            self.command({"focus": float(focus)}, .12)
            self.focus_scan.append(dict(focus_m=float(focus), sharpness=self.microscope.sharpness(
                self.microscope_image(annotate=False))))
        best = max(self.focus_scan, key=lambda row: row["sharpness"])
        self.command({"focus": best["focus_m"]}, .2)
        self.focused = bool(abs(self.data.qpos[self.model.joint("focus").qposadr[0]]) < .35e-6)

    def cell_injection(self):
        self.phase("locate_cell")
        xy = self.locate("cell")
        centre = np.array([*xy, CELL_CENTER[2]])
        ray = 1/np.linalg.norm(NEEDLE_BACK/CELL_RADII)
        self.move("injector", centre+NEEDLE_BACK*(ray+120e-6), .8)
        self.mechanics.compensation_pa = 500.
        self.phase("approach_cell")
        self.move("injector", centre+NEEDLE_BACK*(ray+10e-6), .7)
        self.phase("membrane_contact")
        self.move("injector", centre+NEEDLE_BACK*(ray-.6e-6), .5)
        self.phase("puncture_cell")
        self.move("injector", centre+NEEDLE_BACK*(ray-self.injection_depth_m), .7)
        if not self.mechanics.punctured:
            raise RuntimeError("Membrane puncture was not established; injection aborted")
        self.phase("cytoplasm_injection")
        self.data.userdata[0] = 5000.
        for _ in range(2500):
            self.step_and_log({})
            if self.mechanics.ledger.state("cell").volume_m3 >= .5e-15-1e-28:
                break
        self.data.userdata[0] = 0.
        self.mechanics.compensation_pa = 0.
        self.phase("withdraw_cell_needle")
        self.move("injector", centre+NEEDLE_BACK*(ray+20e-6), .6)
        self.wait(.5)
        self.phase("tracer_review")


CellInjection.Expert = CellInjectionExpert


class CellDoseInjection(CellInjection):
    """Keep the original injection URL with a cellular picolitre experiment."""
    operation = "injection"
    default_task = "microscopy_injection"


class CellDoseInjectionExpert(CellInjectionExpert, CellDoseInjection):
    injection = CellInjectionExpert.cell_injection


CellDoseInjection.Expert = CellDoseInjectionExpert
