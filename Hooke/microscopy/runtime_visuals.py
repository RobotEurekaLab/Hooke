"""Renderer-neutral visual shapes for analytical microscopy state.

These shapes have no collider, mass or actuator. Instrument dynamics and the
cell model remain independent of rendering in both supported backends.
"""

import mujoco
import numpy as np


def apply_visuals(scene, geometry):
    for shape in geometry:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Microscopy visuals exceed renderer capacity")
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], shape["type"], np.asarray(shape["size"]),
                           np.asarray(shape["pos"]), np.asarray(shape["mat"]).ravel(),
                           np.asarray(shape["rgba"], dtype=np.float32))
        if "surface" in shape:
            surface = shape["surface"]
            scene.geoms[scene.ngeom].specular = float(np.mean(surface["specular_color"]))
            scene.geoms[scene.ngeom].shininess = 1-float(surface["roughness"])
        scene.ngeom += 1


def update_scene(task, renderer, camera):
    from backends.source_renderer import center_directional_shadows
    renderer.update_scene(task.data, camera=camera)
    center_directional_shadows(renderer, task.model)
    if callable(getattr(task, "runtime_visuals", None)):
        apply_visuals(renderer.scene, task.runtime_visuals())
