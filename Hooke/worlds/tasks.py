"""Display/physics qualification scenes; no scientific expert is claimed."""

from archetypes.static_display import StaticDisplaySpec, make_static_task
from worlds.environment import EnvironmentSystem
from worlds.profiles import WORLDS


def make_world_task(name, *, scene_file=None, task_name=None, asset_camera=False):
    profile = WORLDS[name]
    base, _ = make_static_task(
        StaticDisplaySpec(
            name=task_name or profile.task_name,
            scene_file=scene_file or f"space_{name}.gen.xml",
            prompt=profile.label,
            camera="world_overview",
        )
    )

    class StaticDisplayTask(base):
        def __init__(self, spec):
            super().__init__(spec)
            self.environment = EnvironmentSystem(profile=profile)
            self.environment.reload(self.model)
            self.manager.set_systems([self.environment])

        def reset(self, seed=None):
            super().reset(seed)
            joint = self.model.joint("free_cartridge_joint")
            self.data.qvel[joint.dofadr[0]] = 0.06
            self.task_info.update(
                display_only=True,
                camera_mapping={
                    "image": "world_overview",
                    "bench": "experiment_closeup",
                },
                space_environment=profile.report(),
            )
            if asset_camera:
                self.task_info["camera_mapping"]["asset"] = "asset_closeup"
            return self.task_info

        def check(self):
            return False

    class WorldDisplayExpert(StaticDisplayTask):
        def execute(self):
            self.finish()

    StaticDisplayTask.Expert = WorldDisplayExpert
    return StaticDisplayTask


OrbitalWorkstation = make_world_task("orbital")
LunarWorkstation = make_world_task("lunar")
MartianWorkstation = make_world_task("martian")
