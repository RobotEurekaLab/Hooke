"""Mission geometry and limits, independent of either physics engine."""

from dataclasses import dataclass

from worlds.profiles import WORLDS


@dataclass(frozen=True)
class SurfaceMission:
    world: str
    label: str
    home: tuple[float, float] = (0.0, 0.0)
    collection_stop: tuple[float, float] = (5.5, 0.0)
    sample_position: tuple[float, float, float] = (6.4, -0.15, 0.004)
    terrain_half_size_m: float = 40.0
    wheel_radius_m: float = 0.19
    half_track_m: float = 0.42
    speed_limit_m_s: float = 0.5

    @property
    def task_name(self):
        return f"space_{self.world}_surface_sampling"

    @property
    def environment(self):
        return WORLDS[self.world]

    def environment_report(self):
        profile = self.environment.report()
        profile.pop("workstation")
        profile.pop("protected_module")
        profile["label"] = self.label
        profile["support"] = "wheel_ground_contact_on_rigid_terrain"
        profile["sample_storage"] = "free_specimen_in_physical_rover_bin"
        profile["implemented"] = [
            "rigid_body_gravity",
            "collision",
            "wheel_actuation",
            "radiative_thermal_witness",
        ]
        return profile


MISSIONS = {
    "lunar": SurfaceMission("lunar", "月面陨坑探测与岩样采集"),
    "martian": SurfaceMission("martian", "火星河谷探测与岩样采集"),
}
