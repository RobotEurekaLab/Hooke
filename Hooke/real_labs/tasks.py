"""Scene-picker adapter; instrument controls are provided by /real-labs."""

import mujoco

from real_labs.builder import build_scene
from real_labs.catalog import scenes
from simulation import Manager
from task import Expert, Task
import xml.etree.ElementTree as ET


def make_task(identifier):
    class LaboratoryTask(Task):
        default_task = "real_lab_" + identifier
        default_scene = None
        time_limit = 30.0
        early_stop = False

        @classmethod
        def load(cls, scene=None):
            if scene is not None:
                return mujoco.MjSpec.from_file(str(scene))
            root, _ = build_scene(identifier)
            return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))

        def __init__(self, spec):
            super().__init__(Manager.from_spec(spec, []))

        def reset(self, seed=None):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.task_info = dict(
                prefix=scenes()[identifier]["title"]
                + " — reference-informed prototype",
                camera_mapping={"image": "overview"},
                state_indices=[],
                action_indices=[],
                seed=seed,
                interaction_url="/real-labs?scene=" + identifier,
                fidelity="Direct instrument joint control; autonomous experiment not implemented",
            )
            return self.task_info

        def check(self):
            return False  # Rendering an equipment layout is not experiment completion.

    class LaboratoryExpert(LaboratoryTask, Expert):
        def execute(self):
            raise NotImplementedError(
                "Use /real-labs instrument controls or real_labs.export mechanical validation; "
                "no autonomous scientific-workflow expert exists for this reference scene"
            )

    LaboratoryTask.Expert = LaboratoryExpert
    return LaboratoryTask


for _identifier in scenes():
    globals()[_identifier] = make_task(_identifier)
