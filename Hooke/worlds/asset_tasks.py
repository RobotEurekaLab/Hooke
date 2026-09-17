"""Separate opt-in tasks for the external research asset scene variants."""

from worlds.tasks import make_world_task


def asset_task(name):
    return make_world_task(
        name,
        scene_file=f"space_{name}_assets.gen.xml",
        task_name=f"space_{name}_assets_workstation",
        asset_camera=True,
    )


OrbitalAssetsWorkstation = asset_task("orbital")
LunarAssetsWorkstation = asset_task("lunar")
MartianAssetsWorkstation = asset_task("martian")
