"""Contact, travel, collection and return evidence from actual stepped state."""

import numpy as np

from contact_state import body_geoms, touching
from simulation import System
from surface.scene import ARM_PREFIX, BIN_CENTER


class SamplingProgress(System):
    def _configure(self, mission):
        self.mission = mission

    def _reload(self, model):
        self.rover = model.body("rover").id
        self.rover_velocity = int(model.joint("rover_free").dofadr[0])
        self.sample = model.body("field_sample").id
        self.sample_geoms = body_geoms(model, self.sample)
        self.bin_geoms = body_geoms(model, model.body("sample_bin").id)
        self.left = body_geoms(model, model.body(ARM_PREFIX + "2f85:left_pad").id)
        self.right = body_geoms(model, model.body(ARM_PREFIX + "2f85:right_pad").id)

    def _reset(self, data):
        self.previous_time = float(data.time)
        self.previous_position = data.xpos[self.rover, :2].copy()
        self.initial_sample_height = float(data.xpos[self.sample, 2])
        self.distance_m = 0.0
        self.max_lift_m = 0.0
        self.grasp_contact_s = 0.0
        self.bin_stable_s = 0.0
        self.visited_collection_site = False
        self.lifted_in_grasp = False
        self.collected = False
        self.events = []

    def gripped(self, data):
        return touching(data, self.sample_geoms, self.left) and touching(
            data, self.sample_geoms, self.right
        )

    def bin_position(self, data):
        rotation = data.xmat[self.rover].reshape(3, 3)
        return (
            rotation.T @ (data.xpos[self.sample] - data.xpos[self.rover]) - BIN_CENTER
        )

    def inside_bin(self, data):
        x, y, z = self.bin_position(data)
        return bool(abs(x) < 0.115 and abs(y) < 0.095 and 0.005 < z < 0.14)

    def event(self, data, name):
        self.events.append(dict(event=name, simulation_s=float(data.time)))

    def _update(self, data):
        now = float(data.time)
        dt = now - self.previous_time
        if dt <= 0:
            return
        position = data.xpos[self.rover, :2]
        self.distance_m += float(np.linalg.norm(position - self.previous_position))
        self.previous_position = position.copy()
        self.previous_time = now
        at_site = np.linalg.norm(position - self.mission.collection_stop) < 0.3
        if at_site and not self.visited_collection_site:
            self.visited_collection_site = True
            self.event(data, "collection_site_reached")
        gripped = self.gripped(data)
        if gripped:
            self.grasp_contact_s += dt
            rise = float(data.xpos[self.sample, 2] - self.initial_sample_height)
            self.max_lift_m = max(self.max_lift_m, rise)
            if (
                self.visited_collection_site
                and rise > 0.10
                and not self.lifted_in_grasp
            ):
                self.lifted_in_grasp = True
                self.event(data, "rock_lifted_in_bilateral_grasp")
        seated = (
            self.inside_bin(data)
            and touching(data, self.sample_geoms, self.bin_geoms)
            and not gripped
        )
        self.bin_stable_s = self.bin_stable_s + dt if seated else 0.0
        if self.lifted_in_grasp and self.bin_stable_s >= 0.5 and not self.collected:
            self.collected = True
            self.event(data, "rock_deposited_in_rover_bin")

    def checks(self, data):
        home_distance = np.linalg.norm(data.xpos[self.rover, :2] - self.mission.home)
        velocity = data.qvel[self.rover_velocity : self.rover_velocity + 6]
        return dict(
            collection_site_reached=self.visited_collection_site,
            traveled_at_least_9m=self.distance_m >= 9.0,
            bilateral_grasp_at_least_50ms=self.grasp_contact_s >= 0.05,
            sample_lifted_at_least_10cm=self.lifted_in_grasp,
            sample_deposited=self.collected,
            sample_currently_in_bin=self.inside_bin(data),
            sample_released=not self.gripped(data),
            returned_within_20cm=bool(home_distance < 0.20),
            rover_stopped=bool(
                np.linalg.norm(velocity[:3]) < 0.05
                and np.linalg.norm(velocity[3:]) < 0.1
            ),
            bin_contact_stable_500ms=self.bin_stable_s >= 0.5,
        )

    def report(self, data):
        return dict(
            world=self.mission.world,
            robot="six_wheel_rover_with_UR5e_and_Robotiq_2F85",
            traveled_m=self.distance_m,
            max_sample_lift_m=self.max_lift_m,
            bilateral_grasp_s=self.grasp_contact_s,
            rover_position_m=data.xpos[self.rover].tolist(),
            sample_in_bin=self.inside_bin(data),
            checks=self.checks(data),
            events=list(self.events),
            sample_geometry="NASA JSC Apollo sample via SRB; illustrative analog on Mars",
            wheel_driven=True,
            grasp_attachment=False,
            specimen_properties="illustrative 0.1 kg; no geological composition claim",
            limitations=[
                "rigid terrain; no soil deformation, dust or gas dynamics",
                "scripted navigation with state feedback; no perception-based autonomy",
                "humanoid walking is not implemented",
            ],
        )
