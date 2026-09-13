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
                    preservation) for most atomic entries, a domain label
                    (chemistry, pharma, nanotechnology, ...) for the
                    display-only breadth batches, or "composite" for an
                    entry that's one step of a multi-instrument chained
                    sequence (archetypes/composite_task.py) rather than a
                    standalone atomic task -- kept distinct from the
                    primitive taxonomy so composite steps don't get
                    silently mixed in with regular atomic tasks of the
                    same nominal primitive when filtering/grouping
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
        CatalogEntry(
            name="push_filling_nozzle_down",
            description="Grip a vial-filling line's dispensing nozzle from above and push it down.",
            category="conditioning",
            module="mani_vial_filling_line", cls="PushFillingNozzle", robot="ur5e", camera="table_cam_front",
        ),
        # --- cross-instrument composite scene (see private/technical-log.md
        # / archetypes/composite_task.py's "not attempted here" note) --
        # both entries share one scene with two physically distinct
        # instruments, switched via task_override the same way
        # thermal_cycler_close/open share one scene with one instrument.
        CatalogEntry(
            name="composite_push_vial_nozzle",
            description=(
                "Step 1 of the cross-instrument composite: push the vial-filling line's "
                "nozzle down (shares a scene with an HPLC injector plunger)."
            ),
            category="composite",
            module="mani_cross_instrument_composite", cls="CrossInstrumentComposite",
            robot="ur5e", camera="table_cam_front", task_override="push_vial_nozzle",
        ),
        CatalogEntry(
            name="composite_push_hplc_plunger",
            description=(
                "Step 2 of the cross-instrument composite: push the HPLC autosampler's "
                "injector plunger down (shares a scene with a vial-filling line)."
            ),
            category="composite",
            module="mani_cross_instrument_composite", cls="CrossInstrumentComposite",
            robot="ur5e", camera="table_cam_front", task_override="push_hplc_plunger",
        ),
        # --- composite batch: cross-instrument pairs must correspond to a
        # real multi-step lab workflow, not just a device pairing chosen to
        # cover more discipline labels -- a first attempt at 5 more pairs
        # here (biology+materials_science, food_science+environmental_testing,
        # semiconductor+textile_testing, forensic_science+geology_mining,
        # microbiology+metallurgy_welding) was caught on review and removed
        # for failing exactly that test (see private/technical-log.md's
        # "Composite batch, corrected" entry) -- only this one survived:
        # filter a sample, then load plates into general-lab automation for
        # further processing, a real and recognizable chemistry/lab-
        # automation sequence.
        CatalogEntry(
            name="composite_filter_cartridge_housing",
            description=(
                "Pull a chemistry filter cartridge housing up, then load plates into a general-lab "
                "microplate stacker -- a real filter-then-aliquot lab sequence."
            ),
            category="composite",
            module="archetypes.composite_generated", cls="ChemistryGeneralLabCompositeTask",
            robot="ur5e", camera="table_cam_front", task_override="filter_cartridge_housing_lift",
        ),
        CatalogEntry(
            name="composite_microplate_stacker_eject",
            description=(
                "Press a general-lab microplate stacker's eject down, the second step after pulling a "
                "chemistry filter cartridge housing up."
            ),
            category="composite",
            module="archetypes.composite_generated", cls="ChemistryGeneralLabCompositeTask",
            robot="ur5e", camera="table_cam_front", task_override="microplate_stacker_eject2",
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
        # --- third batch (see private/technical-log.md) ---
        CatalogEntry(
            name="tensile_testing_machine_display",
            description="A tensile testing machine with a moving crosshead (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="TensileTestingMachineTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="rockwell_hardness_tester_display",
            description="A Rockwell hardness tester (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="RockwellHardnessTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="sem_chamber_display",
            description="A scanning electron microscope sample chamber (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="SemChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="xrd_diffractometer_display",
            description="An X-ray diffractometer with a goniometer arm (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="XrdDiffractometerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="tga_dsc_analyzer_display",
            description="A TGA/DSC thermal analyzer (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="TgaDscAnalyzerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="sputter_coater_display",
            description="A sputter coater with a bell jar (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="SputterCoaterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="hot_press_display",
            description="A hot press for material sample preparation (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="HotPressTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="spin_coater_display",
            description="A spin coater for thin-film deposition (display only, with one operable joint).",
            category="semiconductor",
            module="archetypes.static_items3", cls="SpinCoaterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="wire_bonder_display",
            description="A wire bonder for chip packaging (display only, with one operable joint).",
            category="semiconductor",
            module="archetypes.static_items3", cls="WireBonderTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="wafer_dicing_saw_display",
            description="A wafer dicing saw (display only, with one operable joint).",
            category="semiconductor",
            module="archetypes.static_items3", cls="WaferDicingSawTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="plasma_etcher_display",
            description="A plasma etching chamber (display only, with one operable joint).",
            category="semiconductor",
            module="archetypes.static_items3", cls="PlasmaEtcherTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="wafer_prober_display",
            description="A wafer probing station (display only, with one operable joint).",
            category="semiconductor",
            module="archetypes.static_items3", cls="WaferProberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="wafer_inspection_station_display",
            description="A wafer inspection station (display only, with one operable joint).",
            category="semiconductor",
            module="archetypes.static_items3", cls="WaferInspectionStationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="photolithography_aligner_display",
            description="A photolithography mask aligner (display only, with one operable joint).",
            category="semiconductor",
            module="archetypes.static_items3", cls="PhotolithographyAlignerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="texture_analyzer_display",
            description="A food texture analyzer (display only, with one operable joint).",
            category="food_science",
            module="archetypes.static_items3", cls="TextureAnalyzerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="viscometer_display",
            description="A rotational viscometer (display only, with one operable joint).",
            category="food_science",
            module="archetypes.static_items3", cls="ViscometerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="moisture_analyzer_display",
            description="A moisture analyzer (display only, with one operable joint).",
            category="food_science",
            module="archetypes.static_items3", cls="MoistureAnalyzerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="homogenizer_display",
            description="A laboratory homogenizer (display only, with one operable joint).",
            category="food_science",
            module="archetypes.static_items3", cls="HomogenizerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="brix_refractometer_display",
            description="A Brix refractometer (display only, with one operable joint).",
            category="food_science",
            module="archetypes.static_items3", cls="BrixRefractometerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="pasteurizer_display",
            description="A benchtop pasteurizer tank (display only, with one operable joint).",
            category="food_science",
            module="archetypes.static_items3", cls="PasteurizerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="environmental_chamber_display",
            description="An environmental test chamber (display only, with one operable joint).",
            category="environmental_testing",
            module="archetypes.static_items3", cls="EnvironmentalChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="humidity_chamber_display",
            description="A humidity test chamber (display only, with one operable joint).",
            category="environmental_testing",
            module="archetypes.static_items3", cls="HumidityChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="salt_spray_chamber_display",
            description="A salt spray corrosion test chamber (display only, with one operable joint).",
            category="environmental_testing",
            module="archetypes.static_items3", cls="SaltSprayChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="uv_weathering_chamber_display",
            description="A UV weathering test chamber (display only, with one operable joint).",
            category="environmental_testing",
            module="archetypes.static_items3", cls="UvWeatheringChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="thermal_shock_chamber_display",
            description="A thermal shock test chamber (display only, with one operable joint).",
            category="environmental_testing",
            module="archetypes.static_items3", cls="ThermalShockChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="gel_electrophoresis_rig_display",
            description="A gel electrophoresis rig (display only, with one operable joint).",
            category="biology",
            module="archetypes.static_items3", cls="GelElectrophoresisRigTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="flow_cytometer_display",
            description="A flow cytometer (display only, with one operable joint).",
            category="biology",
            module="archetypes.static_items3", cls="FlowCytometerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="cell_culture_incubator_display",
            description="A cell culture incubator (display only, with one operable joint).",
            category="biology",
            module="archetypes.static_items3", cls="CellCultureIncubatorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="colony_counter_display",
            description="A colony counter (display only, with one operable joint).",
            category="biology",
            module="archetypes.static_items3", cls="ColonyCounterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="microplate_reader_display",
            description="A microplate reader (display only, with one operable joint).",
            category="biology",
            module="archetypes.static_items3", cls="MicroplateReaderTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="cryostat_microtome_display",
            description="A cryostat microtome (display only, with one operable joint).",
            category="biology",
            module="archetypes.static_items3", cls="CryostatMicrotomeTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="spectrofluorometer_display",
            description="A spectrofluorometer (display only, with one operable joint).",
            category="biology",
            module="archetypes.static_items3", cls="SpectrofluorometerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="fdm_3d_printer_display",
            description="An FDM 3D printer (display only, with one operable joint).",
            category="robotics_prototyping",
            module="archetypes.static_items3", cls="Fdm3dPrinterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="laser_cutter_display",
            description="A laser cutter (display only, with one operable joint).",
            category="robotics_prototyping",
            module="archetypes.static_items3", cls="LaserCutterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="cnc_mill_display",
            description="A benchtop CNC mill (display only, with one operable joint).",
            category="robotics_prototyping",
            module="archetypes.static_items3", cls="CncMillTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="conveyor_belt_display",
            description="A conveyor belt (display only, with one operable joint).",
            category="robotics_prototyping",
            module="archetypes.static_items3", cls="ConveyorBeltTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="pick_and_place_machine_display",
            description="A pick-and-place machine (display only, with one operable joint).",
            category="robotics_prototyping",
            module="archetypes.static_items3", cls="PickAndPlaceMachineTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="injection_molding_machine_display",
            description="A benchtop injection molding machine (display only, with one operable joint).",
            category="robotics_prototyping",
            module="archetypes.static_items3", cls="InjectionMoldingMachineTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="vacuum_oven_display",
            description="A vacuum drying oven (display only, with one operable joint).",
            category="materials_science",
            module="archetypes.static_items3", cls="VacuumOvenTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="vibration_test_shaker_display",
            description="A vibration test shaker table (display only, with one operable joint).",
            category="environmental_testing",
            module="archetypes.static_items3", cls="VibrationTestShakerTask", robot="ur5e", camera="table_cam_front",
        ),
        # --- fourth batch (see private/technical-log.md) ---
        CatalogEntry(
            name="atomic_force_microscope_display",
            description="An atomic force microscope (display only, with one operable joint).",
            category="nanotechnology",
            module="archetypes.static_items4", cls="AtomicForceMicroscopeTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="tem_chamber_display",
            description="A transmission electron microscope sample chamber (display only, with one operable joint).",
            category="nanotechnology",
            module="archetypes.static_items4", cls="TemChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="nanoparticle_synthesizer_display",
            description="A nanoparticle synthesis reactor (display only, with one operable joint).",
            category="nanotechnology",
            module="archetypes.static_items4", cls="NanoparticleSynthesizerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="dip_pen_lithography_stage_display",
            description="A dip-pen nanolithography stage (display only, with one operable joint).",
            category="nanotechnology",
            module="archetypes.static_items4", cls="DipPenLithographyStageTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="quartz_crystal_microbalance_display",
            description="A quartz crystal microbalance sensor (display only, with one operable joint).",
            category="nanotechnology",
            module="archetypes.static_items4", cls="QuartzCrystalMicrobalanceTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="langmuir_blodgett_trough_display",
            description="A Langmuir-Blodgett trough for thin-film deposition (display only, with one operable joint).",
            category="nanotechnology",
            module="archetypes.static_items4", cls="LangmuirBlodgettTroughTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="electrospinning_setup_display",
            description="An electrospinning setup for nanofiber production (display only, with one operable joint).",
            category="nanotechnology",
            module="archetypes.static_items4", cls="ElectrospinningSetupTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="fingerprint_fuming_chamber_display",
            description="A cyanoacrylate fingerprint fuming chamber (display only, with one operable joint).",
            category="forensic_science",
            module="archetypes.static_items4", cls="FingerprintFumingChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="alternate_light_source_display",
            description="A forensic alternate light source (display only, with one operable joint).",
            category="forensic_science",
            module="archetypes.static_items4", cls="AlternateLightSourceTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="gunshot_residue_collector_display",
            description="A gunshot residue collection station (display only, with one operable joint).",
            category="forensic_science",
            module="archetypes.static_items4", cls="GunshotResidueCollectorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="luminol_spray_station_display",
            description="A luminol spray station for blood trace detection (display only, with one operable joint).",
            category="forensic_science",
            module="archetypes.static_items4", cls="LuminolSpraySationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="evidence_drying_cabinet_display",
            description="An evidence drying cabinet (display only, with one operable joint).",
            category="forensic_science",
            module="archetypes.static_items4", cls="EvidenceDryingCabinetTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="trace_evidence_vacuum_display",
            description="A trace evidence vacuum collection unit (display only, with one operable joint).",
            category="forensic_science",
            module="archetypes.static_items4", cls="TraceEvidenceVacuumTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="dna_extraction_robot_display",
            description="An automated forensic DNA extraction robot (display only, with one operable joint).",
            category="forensic_science",
            module="archetypes.static_items4", cls="DnaExtractionRobotTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="tensile_fabric_tester_display",
            description="A tensile tester for fabric samples (display only, with one operable joint).",
            category="textile_testing",
            module="archetypes.static_items4", cls="TensileFabricTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="martindale_abrasion_tester_display",
            description="A Martindale fabric abrasion tester (display only, with one operable joint).",
            category="textile_testing",
            module="archetypes.static_items4", cls="MartindaleAbrasionTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="colorfastness_tester_display",
            description="A textile colorfastness tester (display only, with one operable joint).",
            category="textile_testing",
            module="archetypes.static_items4", cls="ColorfastnessTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="fabric_flammability_tester_display",
            description="A fabric flammability tester (display only, with one operable joint).",
            category="textile_testing",
            module="archetypes.static_items4", cls="FabricFlammabilityTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="pilling_tester_display",
            description="A fabric pilling tester drum (display only, with one operable joint).",
            category="textile_testing",
            module="archetypes.static_items4", cls="PillingTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="moisture_wicking_tester_display",
            description="A fabric moisture-wicking tester (display only, with one operable joint).",
            category="textile_testing",
            module="archetypes.static_items4", cls="MoistureWickingTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="yarn_twist_tester_display",
            description="A yarn twist tester (display only, with one operable joint).",
            category="textile_testing",
            module="archetypes.static_items4", cls="YarnTwistTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="distillation_column_display_display",
            description="A benchtop-scale petrochemical distillation column (display only, with one operable joint).",
            category="petrochemical",
            module="archetypes.static_items4", cls="DistillationColumnDisplayTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="catalytic_cracker_model_display",
            description="A bench-scale catalytic cracker model (display only, with one operable joint).",
            category="petrochemical",
            module="archetypes.static_items4", cls="CatalyticCrackerModelTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="flare_stack_monitor_display",
            description="A flare stack emissions monitor (display only, with one operable joint).",
            category="petrochemical",
            module="archetypes.static_items4", cls="FlareStackMonitorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="pipeline_valve_actuator_display",
            description="A pipeline valve actuator (display only, with one operable joint).",
            category="petrochemical",
            module="archetypes.static_items4", cls="PipelineValveActuatorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="oil_water_separator_display",
            description="An oil-water separator (display only, with one operable joint).",
            category="petrochemical",
            module="archetypes.static_items4", cls="OilWaterSeparatorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="viscosity_index_tester_display",
            description="A petroleum viscosity index tester (display only, with one operable joint).",
            category="petrochemical",
            module="archetypes.static_items4", cls="ViscosityIndexTesterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="sulfur_analyzer_display",
            description="A petroleum sulfur content analyzer (display only, with one operable joint).",
            category="petrochemical",
            module="archetypes.static_items4", cls="SulfurAnalyzerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="rock_crusher_display",
            description="A benchtop rock crusher (display only, with one operable joint).",
            category="geology_mining",
            module="archetypes.static_items4", cls="RockCrusherTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="core_sample_splitter_display",
            description="A geological core sample splitter (display only, with one operable joint).",
            category="geology_mining",
            module="archetypes.static_items4", cls="CoreSampleSplitterTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="sieve_shaker_display",
            description="A sieve shaker for particle size analysis (display only, with one operable joint).",
            category="geology_mining",
            module="archetypes.static_items4", cls="SieveShakerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="xrf_rock_analyzer_display",
            description="An XRF rock sample analyzer (display only, with one operable joint).",
            category="geology_mining",
            module="archetypes.static_items4", cls="XrfRockAnalyzerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="drill_core_logger_display",
            description="A drill core logging scanner (display only, with one operable joint).",
            category="geology_mining",
            module="archetypes.static_items4", cls="DrillCoreLoggerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="flotation_cell_display",
            description="A mineral flotation cell (display only, with one operable joint).",
            category="geology_mining",
            module="archetypes.static_items4", cls="FlotationCellTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="sample_pulverizer_display",
            description="A geological sample pulverizer (display only, with one operable joint).",
            category="geology_mining",
            module="archetypes.static_items4", cls="SamplePulverizerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="anaerobic_chamber_display",
            description="An anaerobic culture chamber (display only, with one operable joint).",
            category="microbiology",
            module="archetypes.static_items4", cls="AnaerobicChamberTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="microbial_incubator_shaker_display",
            description="A microbial incubator shaker (display only, with one operable joint).",
            category="microbiology",
            module="archetypes.static_items4", cls="MicrobialIncubatorShakerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="colony_pick_robot_display",
            description="An automated colony-picking robot (display only, with one operable joint).",
            category="microbiology",
            module="archetypes.static_items4", cls="ColonyPickRobotTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="maldi_tof_analyzer_display",
            description="A MALDI-TOF mass spectrometer for microbial identification (display only, with one operable joint).",
            category="microbiology",
            module="archetypes.static_items4", cls="MaldiTofAnalyzerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="biofilm_reactor_display",
            description="A biofilm reactor (display only, with one operable joint).",
            category="microbiology",
            module="archetypes.static_items4", cls="BiofilmReactorTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="petri_dish_stacker_display",
            description="A petri dish stacker (display only, with one operable joint).",
            category="microbiology",
            module="archetypes.static_items4", cls="PetriDishStackerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="gram_stain_station_display",
            description="A gram stain station (display only, with one operable joint).",
            category="microbiology",
            module="archetypes.static_items4", cls="GramStainStationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="arc_welding_station_display",
            description="An arc welding station (display only, with one operable joint).",
            category="metallurgy_welding",
            module="archetypes.static_items4", cls="ArcWeldingStationTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="metallography_polisher_display",
            description="A metallography sample polisher (display only, with one operable joint).",
            category="metallurgy_welding",
            module="archetypes.static_items4", cls="MetallographyPolisherTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="induction_furnace_display",
            description="An induction furnace (display only, with one operable joint).",
            category="metallurgy_welding",
            module="archetypes.static_items4", cls="InductionFurnaceTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="metal_spectrometer_display",
            description="A metal alloy spectrometer (display only, with one operable joint).",
            category="metallurgy_welding",
            module="archetypes.static_items4", cls="MetalSpectrometerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="heat_treatment_oven_display",
            description="A metal heat treatment oven (display only, with one operable joint).",
            category="metallurgy_welding",
            module="archetypes.static_items4", cls="HeatTreatmentOvenTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="welding_positioner_display",
            description="A welding positioner turntable (display only, with one operable joint).",
            category="metallurgy_welding",
            module="archetypes.static_items4", cls="WeldingPositionerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="impact_tester_charpy_display",
            description="A Charpy impact tester (display only, with one operable joint).",
            category="metallurgy_welding",
            module="archetypes.static_items4", cls="ImpactTesterCharpyTask", robot="ur5e", camera="table_cam_front",
        ),
        # --- protocol-to-task generated (see private/technical-log.md) ---
        CatalogEntry(
            name="hplc_injector_plunger",
            description="push the HPLC autosampler's injector plunger down to load the next vial",
            category="analytical_chemistry",
            module="archetypes.protocol_generated", cls="HplcInjectorPlungerTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="supernatant_transfer_display",
            description='transfer supernatant from one tube to another with a micropipette',
            category="molecular_biology",
            module="archetypes.protocol_generated", cls="SupernatantTransferTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="spectrophotometer_reading_display",
            description='read the absorbance value from a benchtop spectrophotometer',
            category="analytical_chemistry",
            module="archetypes.protocol_generated", cls="SpectrophotometerReadingTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="microplate_stacker_eject",
            description="press the microplate stacker's eject button housing down to release a plate",
            category="general_lab",
            module="archetypes.protocol_generated", cls="MicroplateStackerEjectTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="supernatant_decant_display",
            description='decant clarified supernatant into a fresh vial without disturbing the pellet',
            category="molecular_biology",
            module="archetypes.protocol_generated", cls="SupernatantDecantTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="freezer_rack_access_display",
            description='open a freezer door to access a cryovial rack shelf',
            category="general_lab",
            module="archetypes.protocol_generated", cls="FreezerRackAccessTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="balance_draft_shield_display",
            description='lift the draft shield lid of an analytical balance before weighing a sample',
            category="general_lab",
            module="archetypes.protocol_generated", cls="BalanceDraftShieldTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="peristaltic_pump_dial_display",
            description="turn the peristaltic pump's flow-rate dial to the target setting",
            category="general_lab",
            module="archetypes.protocol_generated", cls="PeristalticPumpDialTask", robot="ur5e", camera="table_cam_front",
        ),
        CatalogEntry(
            name="filter_cartridge_housing",
            description='pull the filter cartridge housing up off the vacuum manifold',
            category="chemistry",
            module="archetypes.protocol_generated", cls="FilterCartridgeHousingTask", robot="ur5e", camera="table_cam_front",
        ),
    ]
}


def catalog_prompt_listing() -> str:
    """Renders the catalog as a compact listing suitable for an LLM prompt."""
    lines = []
    for entry in CATALOG.values():
        lines.append(f"- {entry.name} [{entry.category}]: {entry.description}")
    return "\n".join(lines)
