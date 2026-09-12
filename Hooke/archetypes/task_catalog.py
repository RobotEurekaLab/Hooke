"""A small, hand-verified catalog of atomic Hooke/AutoBio tasks, for Phase I
Step 4 (protocol -> task-sequence synthesis) and Phase II (the long-horizon
compositional benchmark) to build on.

This deliberately does *not* reuse `task.py`'s `get_task_class()` registry:
several of its entries reference files that don't exist on disk at all
(`insert.py`, `pickup.py`, `screw_loose_topp.py`,
`screw_tighten_topp.py`, `vortex_mixer.py` -- noted in Step 1's log). Rather
than silently propagate that breakage into a catalog an LLM will be asked to
compose from, this catalog only lists tasks we've actually run successfully
in this session (see private/technical-log.md for when/how each was
verified) and points directly at the real module + class.

Each entry:
  name          -- catalog key, used both by the LLM prompt and to resolve
                    back to a runnable Task/Expert pair
  description    -- natural-language description an LLM (or a human) uses to
                    pick this task for a given protocol step
  category       -- AutoBio's own primitive taxonomy bucket (transfer,
                    conditioning, separation, combination, measurement,
                    preservation), for grouping/filtering
  module, cls    -- where to import the Task class from
  robot          -- native robot rig the task's scene was authored for (e.g.
                    "ur5e", "aloha"); see webui/robot_registry.py for the
                    other robots selectable in the UI (arm-mount swaps for
                    ur5e-native tasks, or floor-mount bystanders for any
                    task)
  camera         -- which of the scene's cameras to render the preview from
                    (matches task_info['camera_mapping']['image'])
  task_override  -- for classes whose execute()/reset() branch on `self.task`
                    to support more than one named task (see
                    mani_thermal_cycler.py's close vs. open), the string to
                    assign to `expert.task` after construction
"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class CatalogEntry:
    name: str
    description: str
    category: str
    module: str
    cls: str
    robot: str  # which robot rig the task's scene was authored for (see webui/README.md)
    camera: str
    task_override: str | None = None

    def load_classes(self):
        module = __import__(self.module, fromlist=[self.cls])
        task_cls = getattr(module, self.cls)
        return task_cls, task_cls.Expert

    def make_expert(self):
        task_cls, expert_cls = self.load_classes()
        expert = expert_cls(task_cls.load())
        if self.task_override:
            expert.task = self.task_override
        return expert


CATALOG: dict[str, CatalogEntry] = {
    entry.name: entry for entry in [
        CatalogEntry(
            name="pickup_centrifuge_tube",
            description="Pick up a single centrifuge tube from its rack with a dual-arm (Aloha) gripper.",
            category="transfer",
            module="pickup_centrifuge_tube", cls="Pickup", robot="aloha", camera="table_cam_front",
        ),
        CatalogEntry(
            name="thermal_cycler_close",
            description="Close the lid of the Bio-Rad C1000 thermal cycler (lever + screw knob).",
            category="conditioning",
            module="mani_thermal_cycler", cls="ThermalCyclerManipulate", robot="ur5e", camera="table_cam_left",
            task_override="thermal_cycler_close",
        ),
        CatalogEntry(
            name="thermal_cycler_open",
            description="Open the lid of the Bio-Rad C1000 thermal cycler (lever + screw knob).",
            category="conditioning",
            module="mani_thermal_cycler", cls="ThermalCyclerManipulate", robot="ur5e", camera="table_cam_left",
            task_override="thermal_cycler_open",
        ),
        CatalogEntry(
            name="centrifuge_5430_close_lid",
            description="Close and lock the lid of the Eppendorf 5430 centrifuge (grip lever, rotate, engage lock).",
            category="conditioning",
            module="mani_centrifuge_5430", cls="Centrifuge5430Manipulate", robot="ur5e", camera="table_cam_left",
        ),
        CatalogEntry(
            name="centrifuge_5910_lid_close",
            description="Close and lock the lid of the Eppendorf 5910 centrifuge (grip lever, rotate, engage lock).",
            category="conditioning",
            module="mani_centrifuge_5910", cls="Centrifuge5910Manipulate", robot="ur5e", camera="table_cam_left",
        ),
        CatalogEntry(
            name="insert_centrifuge_5430",
            description=(
                "Insert a second centrifuge tube into the centrifuge 5430 rotor slot symmetrically "
                "opposite an already-placed tube (rotor balancing)."
            ),
            category="separation",
            module="load_centrifuge_5430", cls="InsertCentrifuge5430", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="thermal_mixer",
            description="Load a tube into the Eppendorf ThermoMixer C and run a mixing/heating cycle.",
            category="combination",
            module="mani_thermal_mixer", cls="ThermalMixerManipulate", robot="ur5e", camera="table_cam_left",
            task_override="thermal_mixer",
        ),
        CatalogEntry(
            name="pipette",
            description="Pipette liquid from one container to another using a two-armed UR5e pipetting rig.",
            category="transfer",
            module="mani_pipette", cls="Pipette", robot="dual_ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="vortex_mixer",
            description="Vortex-mix a tube's contents using a dual-arm Aloha setup and a Vortex-Genie 2 mixer.",
            category="combination",
            module="mani_vortex_mixer", cls="VortexMixerManipulate", robot="aloha", camera="table_cam_front",
        ),
        CatalogEntry(
            name="centrifuge_mini_close_lid",
            description="Close the lid of the Tiangen T-Gear mini centrifuge.",
            category="conditioning",
            module="mani_centrifuge_mini", cls="CentrifugeMiniManipulate", robot="ur5e", camera="table_cam_left",
        ),
    ]
}


def catalog_prompt_listing() -> str:
    """Renders the catalog as a compact listing suitable for an LLM prompt."""
    lines = []
    for entry in CATALOG.values():
        lines.append(f"- {entry.name} [{entry.category}]: {entry.description}")
    return "\n".join(lines)
