"""Second batch of visual-only lab-equipment scenes, same pattern as
static_items.py -- see archetypes/static_display.py's docstring for the
scope tradeoff."""
from archetypes.static_display import StaticDisplaySpec, make_static_task

BuretteStandTask, BuretteStandExpert = make_static_task(StaticDisplaySpec(
    name="burette_stand_display",
    scene_file="mani_burette_stand.xml",
    prompt="a burette on a stand for titration",
))

RotaryEvaporatorTask, RotaryEvaporatorExpert = make_static_task(StaticDisplaySpec(
    name="rotary_evaporator_display",
    scene_file="mani_rotary_evaporator.xml",
    prompt="a rotary evaporator with condenser and flasks",
))

PhMeterTask, PhMeterExpert = make_static_task(StaticDisplaySpec(
    name="ph_meter_display",
    scene_file="mani_ph_meter.xml",
    prompt="a benchtop ph meter with probe",
))

VacuumFiltrationSetupTask, VacuumFiltrationSetupExpert = make_static_task(StaticDisplaySpec(
    name="vacuum_filtration_setup_display",
    scene_file="mani_vacuum_filtration_setup.xml",
    prompt="a buchner funnel and vacuum filtration flask",
))

DesiccatorTask, DesiccatorExpert = make_static_task(StaticDisplaySpec(
    name="desiccator_display",
    scene_file="mani_desiccator.xml",
    prompt="a desiccator for moisture-sensitive samples",
))

MuffleFurnaceTask, MuffleFurnaceExpert = make_static_task(StaticDisplaySpec(
    name="muffle_furnace_display",
    scene_file="mani_muffle_furnace.xml",
    prompt="a muffle furnace for high-temperature heating",
))

UltrasonicCleanerTask, UltrasonicCleanerExpert = make_static_task(StaticDisplaySpec(
    name="ultrasonic_cleaner_display",
    scene_file="mani_ultrasonic_cleaner.xml",
    prompt="an ultrasonic cleaning bath",
))

DistillationApparatusTask, DistillationApparatusExpert = make_static_task(StaticDisplaySpec(
    name="distillation_apparatus_display",
    scene_file="mani_distillation_apparatus.xml",
    prompt="a distillation setup with boiling flask, condenser, and receiver",
))

BunsenBurnerTask, BunsenBurnerExpert = make_static_task(StaticDisplaySpec(
    name="bunsen_burner_display",
    scene_file="mani_bunsen_burner.xml",
    prompt="a bunsen burner",
))

UvVisSpectrophotometerTask, UvVisSpectrophotometerExpert = make_static_task(StaticDisplaySpec(
    name="uv_vis_spectrophotometer_display",
    scene_file="mani_uv_vis_spectrophotometer.xml",
    prompt="a uv-vis spectrophotometer",
))

CapsuleFillingMachineTask, CapsuleFillingMachineExpert = make_static_task(StaticDisplaySpec(
    name="capsule_filling_machine_display",
    scene_file="mani_capsule_filling_machine.xml",
    prompt="a capsule filling machine",
))

BlisterPackSealerTask, BlisterPackSealerExpert = make_static_task(StaticDisplaySpec(
    name="blister_pack_sealer_display",
    scene_file="mani_blister_pack_sealer.xml",
    prompt="a blister pack sealing machine",
))

TabletCoatingPanTask, TabletCoatingPanExpert = make_static_task(StaticDisplaySpec(
    name="tablet_coating_pan_display",
    scene_file="mani_tablet_coating_pan.xml",
    prompt="a rotating tablet coating pan",
))

FreezeDryerTask, FreezeDryerExpert = make_static_task(StaticDisplaySpec(
    name="freeze_dryer_display",
    scene_file="mani_freeze_dryer.xml",
    prompt="a freeze dryer (lyophilizer)",
))

DissolutionTesterTask, DissolutionTesterExpert = make_static_task(StaticDisplaySpec(
    name="dissolution_tester_display",
    scene_file="mani_dissolution_tester.xml",
    prompt="a dissolution tester with sample vessels",
))

HardnessTesterTask, HardnessTesterExpert = make_static_task(StaticDisplaySpec(
    name="hardness_tester_display",
    scene_file="mani_hardness_tester.xml",
    prompt="a tablet hardness tester",
))

VialFillingLineTask, VialFillingLineExpert = make_static_task(StaticDisplaySpec(
    name="vial_filling_line_display",
    scene_file="mani_vial_filling_line.xml",
    prompt="a vial filling line",
))

AmpouleSealerTask, AmpouleSealerExpert = make_static_task(StaticDisplaySpec(
    name="ampoule_sealer_display",
    scene_file="mani_ampoule_sealer.xml",
    prompt="an ampoule sealing station",
))

HplcSystemTask, HplcSystemExpert = make_static_task(StaticDisplaySpec(
    name="hplc_system_display",
    scene_file="mani_hplc_system.xml",
    prompt="an hplc (high-performance liquid chromatography) system",
))

CleanRoomPassBoxTask, CleanRoomPassBoxExpert = make_static_task(StaticDisplaySpec(
    name="clean_room_pass_box_display",
    scene_file="mani_clean_room_pass_box.xml",
    prompt="a clean room pass box",
))

ElectrodeCoatingMachineTask, ElectrodeCoatingMachineExpert = make_static_task(StaticDisplaySpec(
    name="electrode_coating_machine_display",
    scene_file="mani_electrode_coating_machine.xml",
    prompt="a roll-to-roll electrode coating machine",
))

SlurryMixerTask, SlurryMixerExpert = make_static_task(StaticDisplaySpec(
    name="slurry_mixer_display",
    scene_file="mani_slurry_mixer.xml",
    prompt="an electrode slurry mixer",
))

CalenderingPressTask, CalenderingPressExpert = make_static_task(StaticDisplaySpec(
    name="calendering_press_display",
    scene_file="mani_calendering_press.xml",
    prompt="a calendering roll press for electrode film",
))

PouchCellSealerTask, PouchCellSealerExpert = make_static_task(StaticDisplaySpec(
    name="pouch_cell_sealer_display",
    scene_file="mani_pouch_cell_sealer.xml",
    prompt="a pouch cell heat sealer",
))

BatteryCyclerRackTask, BatteryCyclerRackExpert = make_static_task(StaticDisplaySpec(
    name="battery_cycler_rack_display",
    scene_file="mani_battery_cycler_rack.xml",
    prompt="a battery cycler test rack",
))

CellStackingMachineTask, CellStackingMachineExpert = make_static_task(StaticDisplaySpec(
    name="cell_stacking_machine_display",
    scene_file="mani_cell_stacking_machine.xml",
    prompt="a battery cell stacking machine",
))

TabWeldingStationTask, TabWeldingStationExpert = make_static_task(StaticDisplaySpec(
    name="tab_welding_station_display",
    scene_file="mani_tab_welding_station.xml",
    prompt="a battery tab welding station",
))

ElectrolyteFillingStationTask, ElectrolyteFillingStationExpert = make_static_task(StaticDisplaySpec(
    name="electrolyte_filling_station_display",
    scene_file="mani_electrolyte_filling_station.xml",
    prompt="an electrolyte filling station",
))

BatteryFormationChamberTask, BatteryFormationChamberExpert = make_static_task(StaticDisplaySpec(
    name="battery_formation_chamber_display",
    scene_file="mani_battery_formation_chamber.xml",
    prompt="a battery formation chamber",
))

CellDisassemblyStationTask, CellDisassemblyStationExpert = make_static_task(StaticDisplaySpec(
    name="cell_disassembly_station_display",
    scene_file="mani_cell_disassembly_station.xml",
    prompt="a battery cell disassembly station",
))

OscilloscopeTask, OscilloscopeExpert = make_static_task(StaticDisplaySpec(
    name="oscilloscope_display",
    scene_file="mani_oscilloscope.xml",
    prompt="a benchtop oscilloscope",
))

FunctionGeneratorTask, FunctionGeneratorExpert = make_static_task(StaticDisplaySpec(
    name="function_generator_display",
    scene_file="mani_function_generator.xml",
    prompt="a benchtop function generator",
))

OpticalBreadboardLaserTask, OpticalBreadboardLaserExpert = make_static_task(StaticDisplaySpec(
    name="optical_breadboard_laser_display",
    scene_file="mani_optical_breadboard_laser.xml",
    prompt="an optical breadboard with a laser source",
))

VacuumChamberTask, VacuumChamberExpert = make_static_task(StaticDisplaySpec(
    name="vacuum_chamber_display",
    scene_file="mani_vacuum_chamber.xml",
    prompt="a vacuum chamber with a viewport",
))

SolderingStationTask, SolderingStationExpert = make_static_task(StaticDisplaySpec(
    name="soldering_station_display",
    scene_file="mani_soldering_station.xml",
    prompt="a soldering station",
))

OpticalMicroscopeTask, OpticalMicroscopeExpert = make_static_task(StaticDisplaySpec(
    name="optical_microscope_display",
    scene_file="mani_optical_microscope.xml",
    prompt="an optical microscope",
))

WaterBathTask, WaterBathExpert = make_static_task(StaticDisplaySpec(
    name="water_bath_display",
    scene_file="mani_water_bath.xml",
    prompt="a laboratory water bath",
))

BiosafetyCabinetTask, BiosafetyCabinetExpert = make_static_task(StaticDisplaySpec(
    name="biosafety_cabinet_display",
    scene_file="mani_biosafety_cabinet.xml",
    prompt="a biosafety cabinet",
))

OrbitalShakerTask, OrbitalShakerExpert = make_static_task(StaticDisplaySpec(
    name="orbital_shaker_display",
    scene_file="mani_orbital_shaker.xml",
    prompt="an orbital shaker",
))

KarlFischerTitratorTask, KarlFischerTitratorExpert = make_static_task(StaticDisplaySpec(
    name="karl_fischer_titrator_display",
    scene_file="mani_karl_fischer_titrator.xml",
    prompt="a karl fischer titrator",
))
