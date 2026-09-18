"""Logical instrument controls independent of slide or parallel motor topology."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.parallel_kinematics import CAD_GEOMETRY, ROTOR_OFFSET_RAD, forward, inverse


class InstrumentDrives:
    def __init__(self, model):
        self.model = model
        self.physical = {model.actuator(i).name.removesuffix("_drive"): i for i in range(model.nu)}
        self.parallel = {}
        self.limits = {name: model.actuator_ctrlrange[i].tolist() for name, i in self.physical.items()}
        reference = mujoco.MjData(model)
        mujoco.mj_kinematics(model, reference)
        for tool in ("probe", "gripper", "injector", "holder"):
            names = [f"{tool}_motor_{i}" for i in range(3)]
            if not all(name in self.physical for name in names):
                continue
            base = model.body(tool + "_base")
            rotation = Rotation.from_quat(base.quat[[1, 2, 3, 0]]).as_matrix()
            site = model.site(tool + "_tcp").id
            self.parallel[tool] = dict(actuators=[self.physical[name] for name in names], rotation=rotation,
                                       site=site, reference=reference.site_xpos[site].copy())
            for name in names:
                self.limits.pop(name)
            self.limits.update({f"{tool}_{axis}": [-.012, .012] for axis in "xyz"})
        self.profile = "parallel-v4" if self.parallel else "linear"

    def targets(self, data):
        targets = {name: float(data.ctrl[self.physical[name]]) for name in self.limits if name in self.physical}
        for tool, assembly in self.parallel.items():
            position = assembly["rotation"] @ forward(data.ctrl[assembly["actuators"]] + ROTOR_OFFSET_RAD, CAD_GEOMETRY)
            targets.update({f"{tool}_{axis}": float(value) for axis, value in zip("xyz", position)})
        return targets

    def feedback(self, data):
        values = {name: float(data.qpos[self.model.joint(name).qposadr[0]])
                  for name in self.limits if name in self.physical}
        for tool, assembly in self.parallel.items():
            position = data.site_xpos[assembly["site"]] - assembly["reference"]
            values.update({f"{tool}_{axis}": float(value) for axis, value in zip("xyz", position)})
        return values

    def controls(self, data, commands):
        """Prepare all motor targets before committing any controls or pressure."""
        controls = data.ctrl.copy()
        for name, value in commands.items():
            if name in self.physical and name in self.limits:
                controls[self.physical[name]] = value
        initial = self.targets(data)
        for tool, assembly in self.parallel.items():
            names = [f"{tool}_{axis}" for axis in "xyz"]
            if not any(name in commands for name in names):
                continue
            position = [commands.get(name, initial[name]) for name in names]
            local = assembly["rotation"].T @ position
            controls[assembly["actuators"]] = inverse(local, CAD_GEOMETRY) - ROTOR_OFFSET_RAD
        return controls

    def set_targets(self, data, commands):
        data.ctrl[:] = self.controls(data, commands)
