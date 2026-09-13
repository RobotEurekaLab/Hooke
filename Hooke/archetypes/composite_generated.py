"""Task/Expert bindings for cross-instrument composite scenes built via
`archetypes/composite_scene.py`'s `CompositeSceneSpec` archetype -- see
private/technical-log.md for the validation run that produced this (each
pair chained 10/10 both standalone and as a continuous no-reset sequence
via `archetypes/composite_task.py`).

Each pair shares one scene (two physically distinct instruments) and one
Task/Expert class, its two `task_override` values selecting which
instrument's joint/site to operate on for that step -- the same pattern
`mani_thermal_cycler.py` uses for `thermal_cycler_close`/`open`, just
across two different instruments instead of two directions on one.

Only one pair here: a first attempt added five more, picked purely to
cover more discipline labels (biology+materials_science,
food_science+environmental_testing, semiconductor+textile_testing,
forensic_science+geology_mining, microbiology+metallurgy_welding) without
any of them corresponding to a real lab workflow -- e.g. no real
experiment chains a microbiology culture-plate loader with a metallurgy
specimen punch press. Caught on review and removed; see
private/technical-log.md's "Composite batch, corrected" entry. A
composite pair belongs here only if it represents an actual multi-step
protocol a real lab would run, not a device pairing invented to hit a
discipline-diversity count.
"""
from archetypes.composite_scene import CompositeSceneSpec, CompositeInstrumentTarget, make_composite_classes

CHEMISTRY_GENERAL_LAB_SPEC = CompositeSceneSpec(
    name="composite_chemistry_general_lab",
    scene_file="mani_composite_chemistry_general_lab.xml",
    targets=(
        CompositeInstrumentTarget("filter_cartridge_housing_lift", "/filter_cartridge_housing:", "part_joint", 0.0, 0.05),
        CompositeInstrumentTarget("microplate_stacker_eject2", "/microplate_stacker_eject:", "part_joint", 0.0, -0.03),
    ),
)
ChemistryGeneralLabCompositeTask, ChemistryGeneralLabCompositeExpert = make_composite_classes(CHEMISTRY_GENERAL_LAB_SPEC)
