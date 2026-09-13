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
        CatalogEntry(
            name="pickup_reagent_bottle",
            description=(
                "Pick up a chemistry lab reagent bottle from the bench with a single-arm "
                "horizontal side grasp."
            ),
            category="transfer",
            module="mani_reagent_bottle", cls="PickupReagentBottle", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="close_fume_hood",
            description="Grip the handle of a benchtop fume hood's sliding sash and pull it closed.",
            category="conditioning",
            module="mani_fume_hood", cls="OperateFumeHood", robot="ur5e", camera="table_cam_front",
            task_override="close_fume_hood",
        ),

        # --- Visual-only display scenes (archetypes/static_display.py) --
        # real MJCF/collision/render, but no scripted-expert interaction
        # written or verified -- see that module's docstring for why.
        CatalogEntry(
            name="analytical_balance_display",
            description="An analytical balance sitting on the bench (display only, no interaction).",
            category="measurement",
            module="archetypes.static_items", cls="AnalyticalBalanceTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="hot_plate_stirrer_display",
            description="A magnetic hot plate stirrer on the bench (display only, no interaction).",
            category="combination",
            module="archetypes.static_items", cls="HotPlateStirrerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="round_bottom_flask_stand_display",
            description="A round-bottom flask held on a ring stand (display only, no interaction).",
            category="preservation",
            module="archetypes.static_items", cls="RoundBottomFlaskStandTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="tablet_press_display",
            description="A pharmaceutical tablet press (display only, no interaction).",
            category="combination",
            module="archetypes.static_items", cls="TabletPressTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="laminar_flow_hood_display",
            description="A laminar flow hood for sterile pharmaceutical work (display only, no interaction).",
            category="conditioning",
            module="archetypes.static_items", cls="LaminarFlowHoodTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="autoclave_display",
            description="A benchtop autoclave sterilizer (display only, no interaction).",
            category="conditioning",
            module="archetypes.static_items", cls="AutoclaveTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="glovebox_display",
            description="An inert-atmosphere glovebox for battery cell assembly (display only, no interaction).",
            category="preservation",
            module="archetypes.static_items", cls="GloveboxTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="coin_cell_crimper_display",
            description="A manual coin cell crimper for battery assembly (display only, no interaction).",
            category="combination",
            module="archetypes.static_items", cls="CoinCellCrimperTask", robot="ur5e", camera="table_cam_front",
        ),

        # --- second batch (see private/technical-log.md) ---
        CatalogEntry(
            name="burette_stand_display",
            description="A burette on a stand for titration (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="BuretteStandTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="rotary_evaporator_display",
            description="A rotary evaporator with condenser and flasks (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="RotaryEvaporatorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="ph_meter_display",
            description="A benchtop pH meter with probe (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="PhMeterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="vacuum_filtration_setup_display",
            description="A Buchner funnel and vacuum filtration flask (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="VacuumFiltrationSetupTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="desiccator_display",
            description="A desiccator for moisture-sensitive samples (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="DesiccatorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="muffle_furnace_display",
            description="A muffle furnace for high-temperature heating (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="MuffleFurnaceTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="ultrasonic_cleaner_display",
            description="An ultrasonic cleaning bath (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="UltrasonicCleanerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="distillation_apparatus_display",
            description="A distillation setup with boiling flask, condenser, and receiver (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="DistillationApparatusTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="bunsen_burner_display",
            description="A Bunsen burner (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="BunsenBurnerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="uv_vis_spectrophotometer_display",
            description="A UV-Vis spectrophotometer (display only, no interaction).",
            category="chemistry",
            module="archetypes.static_items2", cls="UvVisSpectrophotometerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="capsule_filling_machine_display",
            description="A capsule filling machine (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="CapsuleFillingMachineTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="blister_pack_sealer_display",
            description="A blister pack sealing machine (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="BlisterPackSealerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="tablet_coating_pan_display",
            description="A rotating tablet coating pan (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="TabletCoatingPanTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="freeze_dryer_display",
            description="A freeze dryer (lyophilizer) (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="FreezeDryerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="dissolution_tester_display",
            description="A dissolution tester with sample vessels (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="DissolutionTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="hardness_tester_display",
            description="A tablet hardness tester (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="HardnessTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="vial_filling_line_display",
            description="A vial filling line (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="VialFillingLineTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="ampoule_sealer_display",
            description="An ampoule sealing station (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="AmpouleSealerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="hplc_system_display",
            description="An HPLC (high-performance liquid chromatography) system (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="HplcSystemTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="clean_room_pass_box_display",
            description="A clean room pass box (display only, no interaction).",
            category="pharma",
            module="archetypes.static_items2", cls="CleanRoomPassBoxTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="electrode_coating_machine_display",
            description="A roll-to-roll electrode coating machine (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="ElectrodeCoatingMachineTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="slurry_mixer_display",
            description="An electrode slurry mixer (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="SlurryMixerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="calendering_press_display",
            description="A calendering roll press for electrode film (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="CalenderingPressTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="pouch_cell_sealer_display",
            description="A pouch cell heat sealer (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="PouchCellSealerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="battery_cycler_rack_display",
            description="A battery cycler test rack (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="BatteryCyclerRackTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="cell_stacking_machine_display",
            description="A battery cell stacking machine (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="CellStackingMachineTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="tab_welding_station_display",
            description="A battery tab welding station (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="TabWeldingStationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="electrolyte_filling_station_display",
            description="An electrolyte filling station (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="ElectrolyteFillingStationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="battery_formation_chamber_display",
            description="A battery formation chamber (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="BatteryFormationChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="cell_disassembly_station_display",
            description="A battery cell disassembly station (display only, no interaction).",
            category="battery",
            module="archetypes.static_items2", cls="CellDisassemblyStationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="oscilloscope_display",
            description="A benchtop oscilloscope (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="OscilloscopeTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="function_generator_display",
            description="A benchtop function generator (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="FunctionGeneratorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="optical_breadboard_laser_display",
            description="An optical breadboard with a laser source (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="OpticalBreadboardLaserTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="vacuum_chamber_display",
            description="A vacuum chamber with a viewport (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="VacuumChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="soldering_station_display",
            description="A soldering station (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="SolderingStationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="optical_microscope_display",
            description="An optical microscope (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="OpticalMicroscopeTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="water_bath_display",
            description="A laboratory water bath (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="WaterBathTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="biosafety_cabinet_display",
            description="A biosafety cabinet (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="BiosafetyCabinetTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="orbital_shaker_display",
            description="An orbital shaker (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="OrbitalShakerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="karl_fischer_titrator_display",
            description="A Karl Fischer titrator (display only, no interaction).",
            category="general_lab",
            module="archetypes.static_items2", cls="KarlFischerTitratorTask", robot="ur5e", camera="table_cam_front",
        ),
    ]
}


def catalog_prompt_listing() -> str:
    """Renders the catalog as a compact listing suitable for an LLM prompt."""
    lines = []
    for entry in CATALOG.values():
        lines.append(f"- {entry.name} [{entry.category}]: {entry.description}")
    return "\n".join(lines)
