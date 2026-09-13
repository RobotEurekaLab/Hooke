"""Third batch of visual-only lab-equipment scenes, same pattern as
static_items.py/static_items2.py -- see archetypes/static_display.py's
docstring for the scope tradeoff. Unlike batch 2, each item here was
built with its operable joint from the start (see private/technical-log.md
for why batch 2 needed a separate retrofit pass)."""
from archetypes.static_display import StaticDisplaySpec, make_static_task

TensileTestingMachineTask, TensileTestingMachineExpert = make_static_task(StaticDisplaySpec(
    name="tensile_testing_machine_display",
    scene_file="mani_tensile_testing_machine.xml",
    prompt="a tensile testing machine with a moving crosshead",
))

RockwellHardnessTesterTask, RockwellHardnessTesterExpert = make_static_task(StaticDisplaySpec(
    name="rockwell_hardness_tester_display",
    scene_file="mani_rockwell_hardness_tester.xml",
    prompt="a rockwell hardness tester",
))

SemChamberTask, SemChamberExpert = make_static_task(StaticDisplaySpec(
    name="sem_chamber_display",
    scene_file="mani_sem_chamber.xml",
    prompt="a scanning electron microscope sample chamber",
))

XrdDiffractometerTask, XrdDiffractometerExpert = make_static_task(StaticDisplaySpec(
    name="xrd_diffractometer_display",
    scene_file="mani_xrd_diffractometer.xml",
    prompt="an x-ray diffractometer with a goniometer arm",
))

TgaDscAnalyzerTask, TgaDscAnalyzerExpert = make_static_task(StaticDisplaySpec(
    name="tga_dsc_analyzer_display",
    scene_file="mani_tga_dsc_analyzer.xml",
    prompt="a tga/dsc thermal analyzer",
))

SputterCoaterTask, SputterCoaterExpert = make_static_task(StaticDisplaySpec(
    name="sputter_coater_display",
    scene_file="mani_sputter_coater.xml",
    prompt="a sputter coater with a bell jar",
))

HotPressTask, HotPressExpert = make_static_task(StaticDisplaySpec(
    name="hot_press_display",
    scene_file="mani_hot_press.xml",
    prompt="a hot press for material sample preparation",
))

SpinCoaterTask, SpinCoaterExpert = make_static_task(StaticDisplaySpec(
    name="spin_coater_display",
    scene_file="mani_spin_coater.xml",
    prompt="a spin coater for thin-film deposition",
))

WireBonderTask, WireBonderExpert = make_static_task(StaticDisplaySpec(
    name="wire_bonder_display",
    scene_file="mani_wire_bonder.xml",
    prompt="a wire bonder for chip packaging",
))

WaferDicingSawTask, WaferDicingSawExpert = make_static_task(StaticDisplaySpec(
    name="wafer_dicing_saw_display",
    scene_file="mani_wafer_dicing_saw.xml",
    prompt="a wafer dicing saw",
))

PlasmaEtcherTask, PlasmaEtcherExpert = make_static_task(StaticDisplaySpec(
    name="plasma_etcher_display",
    scene_file="mani_plasma_etcher.xml",
    prompt="a plasma etching chamber",
))

WaferProberTask, WaferProberExpert = make_static_task(StaticDisplaySpec(
    name="wafer_prober_display",
    scene_file="mani_wafer_prober.xml",
    prompt="a wafer probing station",
))

WaferInspectionStationTask, WaferInspectionStationExpert = make_static_task(StaticDisplaySpec(
    name="wafer_inspection_station_display",
    scene_file="mani_wafer_inspection_station.xml",
    prompt="a wafer inspection station",
))

PhotolithographyAlignerTask, PhotolithographyAlignerExpert = make_static_task(StaticDisplaySpec(
    name="photolithography_aligner_display",
    scene_file="mani_photolithography_aligner.xml",
    prompt="a photolithography mask aligner",
))

TextureAnalyzerTask, TextureAnalyzerExpert = make_static_task(StaticDisplaySpec(
    name="texture_analyzer_display",
    scene_file="mani_texture_analyzer.xml",
    prompt="a food texture analyzer",
))

ViscometerTask, ViscometerExpert = make_static_task(StaticDisplaySpec(
    name="viscometer_display",
    scene_file="mani_viscometer.xml",
    prompt="a rotational viscometer",
))

MoistureAnalyzerTask, MoistureAnalyzerExpert = make_static_task(StaticDisplaySpec(
    name="moisture_analyzer_display",
    scene_file="mani_moisture_analyzer.xml",
    prompt="a moisture analyzer",
))

HomogenizerTask, HomogenizerExpert = make_static_task(StaticDisplaySpec(
    name="homogenizer_display",
    scene_file="mani_homogenizer.xml",
    prompt="a laboratory homogenizer",
))

BrixRefractometerTask, BrixRefractometerExpert = make_static_task(StaticDisplaySpec(
    name="brix_refractometer_display",
    scene_file="mani_brix_refractometer.xml",
    prompt="a brix refractometer",
))

PasteurizerTask, PasteurizerExpert = make_static_task(StaticDisplaySpec(
    name="pasteurizer_display",
    scene_file="mani_pasteurizer.xml",
    prompt="a benchtop pasteurizer tank",
))

EnvironmentalChamberTask, EnvironmentalChamberExpert = make_static_task(StaticDisplaySpec(
    name="environmental_chamber_display",
    scene_file="mani_environmental_chamber.xml",
    prompt="an environmental test chamber",
))

HumidityChamberTask, HumidityChamberExpert = make_static_task(StaticDisplaySpec(
    name="humidity_chamber_display",
    scene_file="mani_humidity_chamber.xml",
    prompt="a humidity test chamber",
))

SaltSprayChamberTask, SaltSprayChamberExpert = make_static_task(StaticDisplaySpec(
    name="salt_spray_chamber_display",
    scene_file="mani_salt_spray_chamber.xml",
    prompt="a salt spray corrosion test chamber",
))

UvWeatheringChamberTask, UvWeatheringChamberExpert = make_static_task(StaticDisplaySpec(
    name="uv_weathering_chamber_display",
    scene_file="mani_uv_weathering_chamber.xml",
    prompt="a uv weathering test chamber",
))

ThermalShockChamberTask, ThermalShockChamberExpert = make_static_task(StaticDisplaySpec(
    name="thermal_shock_chamber_display",
    scene_file="mani_thermal_shock_chamber.xml",
    prompt="a thermal shock test chamber",
))

GelElectrophoresisRigTask, GelElectrophoresisRigExpert = make_static_task(StaticDisplaySpec(
    name="gel_electrophoresis_rig_display",
    scene_file="mani_gel_electrophoresis_rig.xml",
    prompt="a gel electrophoresis rig",
))

FlowCytometerTask, FlowCytometerExpert = make_static_task(StaticDisplaySpec(
    name="flow_cytometer_display",
    scene_file="mani_flow_cytometer.xml",
    prompt="a flow cytometer",
))

CellCultureIncubatorTask, CellCultureIncubatorExpert = make_static_task(StaticDisplaySpec(
    name="cell_culture_incubator_display",
    scene_file="mani_cell_culture_incubator.xml",
    prompt="a cell culture incubator",
))

ColonyCounterTask, ColonyCounterExpert = make_static_task(StaticDisplaySpec(
    name="colony_counter_display",
    scene_file="mani_colony_counter.xml",
    prompt="a colony counter",
))

MicroplateReaderTask, MicroplateReaderExpert = make_static_task(StaticDisplaySpec(
    name="microplate_reader_display",
    scene_file="mani_microplate_reader.xml",
    prompt="a microplate reader",
))

CryostatMicrotomeTask, CryostatMicrotomeExpert = make_static_task(StaticDisplaySpec(
    name="cryostat_microtome_display",
    scene_file="mani_cryostat_microtome.xml",
    prompt="a cryostat microtome",
))

SpectrofluorometerTask, SpectrofluorometerExpert = make_static_task(StaticDisplaySpec(
    name="spectrofluorometer_display",
    scene_file="mani_spectrofluorometer.xml",
    prompt="a spectrofluorometer",
))

Fdm3dPrinterTask, Fdm3dPrinterExpert = make_static_task(StaticDisplaySpec(
    name="fdm_3d_printer_display",
    scene_file="mani_fdm_3d_printer.xml",
    prompt="an fdm 3d printer",
))

LaserCutterTask, LaserCutterExpert = make_static_task(StaticDisplaySpec(
    name="laser_cutter_display",
    scene_file="mani_laser_cutter.xml",
    prompt="a laser cutter",
))

CncMillTask, CncMillExpert = make_static_task(StaticDisplaySpec(
    name="cnc_mill_display",
    scene_file="mani_cnc_mill.xml",
    prompt="a benchtop cnc mill",
))

ConveyorBeltTask, ConveyorBeltExpert = make_static_task(StaticDisplaySpec(
    name="conveyor_belt_display",
    scene_file="mani_conveyor_belt.xml",
    prompt="a conveyor belt",
))

PickAndPlaceMachineTask, PickAndPlaceMachineExpert = make_static_task(StaticDisplaySpec(
    name="pick_and_place_machine_display",
    scene_file="mani_pick_and_place_machine.xml",
    prompt="a pick-and-place machine",
))

InjectionMoldingMachineTask, InjectionMoldingMachineExpert = make_static_task(StaticDisplaySpec(
    name="injection_molding_machine_display",
    scene_file="mani_injection_molding_machine.xml",
    prompt="a benchtop injection molding machine",
))
VacuumOvenTask, VacuumOvenExpert = make_static_task(StaticDisplaySpec(
    name="vacuum_oven_display",
    scene_file="mani_vacuum_oven.xml",
    prompt="a vacuum drying oven",
))

VibrationTestShakerTask, VibrationTestShakerExpert = make_static_task(StaticDisplaySpec(
    name="vibration_test_shaker_display",
    scene_file="mani_vibration_test_shaker.xml",
    prompt="a vibration test shaker table",
))
