"""Contact evidence and a pressure-driven conservative microchamber model."""

import numpy as np

from liquid_transfer import VolumeLedger, VolumeState
from microscopy.scene import BEAD_DAMPING, ORIGIN, RADIUS
from science.flow import CapillaryFlow, CapillaryParameters
from simulation import System


class MicroscopyMechanics(System):
    def _reload(self, model):
        self.samples = {name: model.body(name).id for name in ("bead_push", "bead_pick")}
        self.sample_geoms = {name: model.geom(name + "_geom").id for name in self.samples}
        self.tips = {name: model.site(name + "_tcp").id for name in ("probe", "gripper", "injector")}
        self.probe = model.geom("probe_tip").id
        self.pads = {model.geom("jaw_a_pad").id, model.geom("jaw_b_pad").id}
        self.well = model.site("well_center").id

    def _reset(self, data):
        self.ledger = VolumeLedger({
            "supply": VolumeState(1e-9, 1e-9),
            "well": VolumeState(0, 2e-10),
        })
        self.flow = CapillaryFlow(self.ledger, "supply", "well",
                                  CapillaryParameters(radius_m=20e-6, length_m=.04))
        self.initial = {name: data.xpos[b].copy() for name, b in self.samples.items()}
        self.push_contact_s = self.bilateral_contact_s = self.lift_m = 0.0
        self.grasped = False
        self.time_s = float(data.time)
        self.blocked_injection_s = 0.0

    def contacts(self, data, sample):
        geom = self.sample_geoms[sample]
        return {int(b if a == geom else a) for a, b in data.contact.geom if geom in (a, b)}

    def _update(self, data):
        dt = float(data.time) - self.time_s
        if dt <= 0:
            return
        self.time_s = float(data.time)
        if self.probe in self.contacts(data, "bead_push"):
            self.push_contact_s += dt
        self.grasped = self.pads <= self.contacts(data, "bead_pick")
        if self.grasped:
            self.bilateral_contact_s += dt
            height = float(data.xpos[self.samples["bead_pick"], 2] - self.initial["bead_pick"][2])
            self.lift_m = max(self.lift_m, height)
        pressure = float(data.userdata[0])
        target = data.site_xpos[self.well]
        tip = data.site_xpos[self.tips["injector"]]
        aligned = np.linalg.norm(tip[:2] - target[:2]) < .0008 and -.0002 < tip[2] - target[2] < .0005
        if pressure and aligned:
            remaining = max(0.0, float(data.userdata[1]) * 1e-12 - self.ledger.state("well").volume_m3)
            p = self.flow.parameters
            maximum = remaining * 8 * p.viscosity_pa_s * p.length_m / (np.pi * p.radius_m**4 * dt)
            self.flow.step(dt, min(pressure, maximum))
        elif pressure:
            self.blocked_injection_s += dt

    def image_state(self, data):
        stage = np.array([data.qpos[self.model.joint("stage_x").qposadr[0]],
                          data.qpos[self.model.joint("stage_y").qposadr[0]], 0.0])
        return dict(
            time_s=float(data.time),
            stage_m=stage[:2].tolist(),
            focus_m=float(data.qpos[self.model.joint("focus").qposadr[0]]),
            samples={name: (data.xpos[b] - ORIGIN).tolist() for name, b in self.samples.items()},
            tools={name: (data.site_xpos[site] - ORIGIN).tolist() for name, site in self.tips.items()},
            jaw_centres_m=[(data.geom_xpos[self.model.geom(f"jaw_{side}_pad").id]-ORIGIN).tolist()
                           for side in ("a", "b")],
            delivered_nl=self.ledger.state("well").volume_m3 * 1e12,
        )

    def report(self, data):
        return dict(
            push_contact_s=self.push_contact_s,
            bilateral_contact_s=self.bilateral_contact_s,
            contact_lift_m=self.lift_m,
            held_by_both_jaws=self.grasped,
            delivered_nl=self.ledger.state("well").volume_m3 * 1e12,
            blocked_injection_s=self.blocked_injection_s,
            capillary=self.flow.report(),
            bead_diameter_m=2 * RADIUS,
            bead_passive_resistance=dict(linear_kg_s=BEAD_DAMPING, angular_kg_m2_s=BEAD_DAMPING,
                                         fidelity="uncalibrated_demo_parameter"),
            scope="Rigid colored calibration beads; dry contact; filled capillary volume model.",
        )
