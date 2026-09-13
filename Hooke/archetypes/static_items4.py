"""Fourth batch of visual-only display scenes, same pattern as
static_items.py/2.py/3.py -- see archetypes/static_display.py's docstring
for the scope tradeoff. Seven new domains (nanotechnology, forensic
science, textile testing, petrochemical/refining, geology/mining,
microbiology, metallurgy/welding), pushing the catalog from 101 toward
the proposal's 150-200 target. Built with each item's operable joint in
from the start, same as batch 3."""
from archetypes.static_display import StaticDisplaySpec, make_static_task

# ===================== Nanotechnology =====================

AtomicForceMicroscopeTask, AtomicForceMicroscopeExpert = make_static_task(StaticDisplaySpec(
    name="atomic_force_microscope_display",
    scene_file="mani_atomic_force_microscope.xml",
    prompt="an atomic force microscope",
))

TemChamberTask, TemChamberExpert = make_static_task(StaticDisplaySpec(
    name="tem_chamber_display",
    scene_file="mani_tem_chamber.xml",
    prompt="a transmission electron microscope sample chamber",
))

NanoparticleSynthesizerTask, NanoparticleSynthesizerExpert = make_static_task(StaticDisplaySpec(
    name="nanoparticle_synthesizer_display",
    scene_file="mani_nanoparticle_synthesizer.xml",
    prompt="a nanoparticle synthesis reactor",
))

DipPenLithographyStageTask, DipPenLithographyStageExpert = make_static_task(StaticDisplaySpec(
    name="dip_pen_lithography_stage_display",
    scene_file="mani_dip_pen_lithography_stage.xml",
    prompt="a dip-pen nanolithography stage",
))

QuartzCrystalMicrobalanceTask, QuartzCrystalMicrobalanceExpert = make_static_task(StaticDisplaySpec(
    name="quartz_crystal_microbalance_display",
    scene_file="mani_quartz_crystal_microbalance.xml",
    prompt="a quartz crystal microbalance sensor",
))

LangmuirBlodgettTroughTask, LangmuirBlodgettTroughExpert = make_static_task(StaticDisplaySpec(
    name="langmuir_blodgett_trough_display",
    scene_file="mani_langmuir_blodgett_trough.xml",
    prompt="a langmuir-blodgett trough for thin-film deposition",
))

ElectrospinningSetupTask, ElectrospinningSetupExpert = make_static_task(StaticDisplaySpec(
    name="electrospinning_setup_display",
    scene_file="mani_electrospinning_setup.xml",
    prompt="an electrospinning setup for nanofiber production",
))

# ===================== Forensic science =====================

FingerprintFumingChamberTask, FingerprintFumingChamberExpert = make_static_task(StaticDisplaySpec(
    name="fingerprint_fuming_chamber_display",
    scene_file="mani_fingerprint_fuming_chamber.xml",
    prompt="a cyanoacrylate fingerprint fuming chamber",
))

AlternateLightSourceTask, AlternateLightSourceExpert = make_static_task(StaticDisplaySpec(
    name="alternate_light_source_display",
    scene_file="mani_alternate_light_source.xml",
    prompt="a forensic alternate light source",
))

GunshotResidueCollectorTask, GunshotResidueCollectorExpert = make_static_task(StaticDisplaySpec(
    name="gunshot_residue_collector_display",
    scene_file="mani_gunshot_residue_collector.xml",
    prompt="a gunshot residue collection station",
))

LuminolSpraySationTask, LuminolSpraySationExpert = make_static_task(StaticDisplaySpec(
    name="luminol_spray_station_display",
    scene_file="mani_luminol_spray_station.xml",
    prompt="a luminol spray station for blood trace detection",
))

EvidenceDryingCabinetTask, EvidenceDryingCabinetExpert = make_static_task(StaticDisplaySpec(
    name="evidence_drying_cabinet_display",
    scene_file="mani_evidence_drying_cabinet.xml",
    prompt="an evidence drying cabinet",
))

TraceEvidenceVacuumTask, TraceEvidenceVacuumExpert = make_static_task(StaticDisplaySpec(
    name="trace_evidence_vacuum_display",
    scene_file="mani_trace_evidence_vacuum.xml",
    prompt="a trace evidence vacuum collection unit",
))

DnaExtractionRobotTask, DnaExtractionRobotExpert = make_static_task(StaticDisplaySpec(
    name="dna_extraction_robot_display",
    scene_file="mani_dna_extraction_robot.xml",
    prompt="an automated forensic DNA extraction robot",
))

# ===================== Textile testing =====================

TensileFabricTesterTask, TensileFabricTesterExpert = make_static_task(StaticDisplaySpec(
    name="tensile_fabric_tester_display",
    scene_file="mani_tensile_fabric_tester.xml",
    prompt="a tensile tester for fabric samples",
))

MartindaleAbrasionTesterTask, MartindaleAbrasionTesterExpert = make_static_task(StaticDisplaySpec(
    name="martindale_abrasion_tester_display",
    scene_file="mani_martindale_abrasion_tester.xml",
    prompt="a martindale fabric abrasion tester",
))

ColorfastnessTesterTask, ColorfastnessTesterExpert = make_static_task(StaticDisplaySpec(
    name="colorfastness_tester_display",
    scene_file="mani_colorfastness_tester.xml",
    prompt="a textile colorfastness tester",
))

FabricFlammabilityTesterTask, FabricFlammabilityTesterExpert = make_static_task(StaticDisplaySpec(
    name="fabric_flammability_tester_display",
    scene_file="mani_fabric_flammability_tester.xml",
    prompt="a fabric flammability tester",
))

PillingTesterTask, PillingTesterExpert = make_static_task(StaticDisplaySpec(
    name="pilling_tester_display",
    scene_file="mani_pilling_tester.xml",
    prompt="a fabric pilling tester drum",
))

MoistureWickingTesterTask, MoistureWickingTesterExpert = make_static_task(StaticDisplaySpec(
    name="moisture_wicking_tester_display",
    scene_file="mani_moisture_wicking_tester.xml",
    prompt="a fabric moisture-wicking tester",
))

YarnTwistTesterTask, YarnTwistTesterExpert = make_static_task(StaticDisplaySpec(
    name="yarn_twist_tester_display",
    scene_file="mani_yarn_twist_tester.xml",
    prompt="a yarn twist tester",
))

# ===================== Petrochemical / refining =====================

DistillationColumnDisplayTask, DistillationColumnDisplayExpert = make_static_task(StaticDisplaySpec(
    name="distillation_column_display",
    scene_file="mani_distillation_column.xml",
    prompt="a benchtop-scale petrochemical distillation column",
))

CatalyticCrackerModelTask, CatalyticCrackerModelExpert = make_static_task(StaticDisplaySpec(
    name="catalytic_cracker_model_display",
    scene_file="mani_catalytic_cracker_model.xml",
    prompt="a bench-scale catalytic cracker model",
))

FlareStackMonitorTask, FlareStackMonitorExpert = make_static_task(StaticDisplaySpec(
    name="flare_stack_monitor_display",
    scene_file="mani_flare_stack_monitor.xml",
    prompt="a flare stack emissions monitor",
))

PipelineValveActuatorTask, PipelineValveActuatorExpert = make_static_task(StaticDisplaySpec(
    name="pipeline_valve_actuator_display",
    scene_file="mani_pipeline_valve_actuator.xml",
    prompt="a pipeline valve actuator",
))

OilWaterSeparatorTask, OilWaterSeparatorExpert = make_static_task(StaticDisplaySpec(
    name="oil_water_separator_display",
    scene_file="mani_oil_water_separator.xml",
    prompt="an oil-water separator",
))

ViscosityIndexTesterTask, ViscosityIndexTesterExpert = make_static_task(StaticDisplaySpec(
    name="viscosity_index_tester_display",
    scene_file="mani_viscosity_index_tester.xml",
    prompt="a petroleum viscosity index tester",
))

SulfurAnalyzerTask, SulfurAnalyzerExpert = make_static_task(StaticDisplaySpec(
    name="sulfur_analyzer_display",
    scene_file="mani_sulfur_analyzer.xml",
    prompt="a petroleum sulfur content analyzer",
))

# ===================== Geology / mining =====================

RockCrusherTask, RockCrusherExpert = make_static_task(StaticDisplaySpec(
    name="rock_crusher_display",
    scene_file="mani_rock_crusher.xml",
    prompt="a benchtop rock crusher",
))

CoreSampleSplitterTask, CoreSampleSplitterExpert = make_static_task(StaticDisplaySpec(
    name="core_sample_splitter_display",
    scene_file="mani_core_sample_splitter.xml",
    prompt="a geological core sample splitter",
))

SieveShakerTask, SieveShakerExpert = make_static_task(StaticDisplaySpec(
    name="sieve_shaker_display",
    scene_file="mani_sieve_shaker.xml",
    prompt="a sieve shaker for particle size analysis",
))

XrfRockAnalyzerTask, XrfRockAnalyzerExpert = make_static_task(StaticDisplaySpec(
    name="xrf_rock_analyzer_display",
    scene_file="mani_xrf_rock_analyzer.xml",
    prompt="an XRF rock sample analyzer",
))

DrillCoreLoggerTask, DrillCoreLoggerExpert = make_static_task(StaticDisplaySpec(
    name="drill_core_logger_display",
    scene_file="mani_drill_core_logger.xml",
    prompt="a drill core logging scanner",
))

FlotationCellTask, FlotationCellExpert = make_static_task(StaticDisplaySpec(
    name="flotation_cell_display",
    scene_file="mani_flotation_cell.xml",
    prompt="a mineral flotation cell",
))

SamplePulverizerTask, SamplePulverizerExpert = make_static_task(StaticDisplaySpec(
    name="sample_pulverizer_display",
    scene_file="mani_sample_pulverizer.xml",
    prompt="a geological sample pulverizer",
))

# ===================== Microbiology =====================

AnaerobicChamberTask, AnaerobicChamberExpert = make_static_task(StaticDisplaySpec(
    name="anaerobic_chamber_display",
    scene_file="mani_anaerobic_chamber.xml",
    prompt="an anaerobic culture chamber",
))

MicrobialIncubatorShakerTask, MicrobialIncubatorShakerExpert = make_static_task(StaticDisplaySpec(
    name="microbial_incubator_shaker_display",
    scene_file="mani_microbial_incubator_shaker.xml",
    prompt="a microbial incubator shaker",
))

ColonyPickRobotTask, ColonyPickRobotExpert = make_static_task(StaticDisplaySpec(
    name="colony_pick_robot_display",
    scene_file="mani_colony_pick_robot.xml",
    prompt="an automated colony-picking robot",
))

MaldiTofAnalyzerTask, MaldiTofAnalyzerExpert = make_static_task(StaticDisplaySpec(
    name="maldi_tof_analyzer_display",
    scene_file="mani_maldi_tof_analyzer.xml",
    prompt="a MALDI-TOF mass spectrometer for microbial identification",
))

BiofilmReactorTask, BiofilmReactorExpert = make_static_task(StaticDisplaySpec(
    name="biofilm_reactor_display",
    scene_file="mani_biofilm_reactor.xml",
    prompt="a biofilm reactor",
))

PetriDishStackerTask, PetriDishStackerExpert = make_static_task(StaticDisplaySpec(
    name="petri_dish_stacker_display",
    scene_file="mani_petri_dish_stacker.xml",
    prompt="a petri dish stacker",
))

GramStainStationTask, GramStainStationExpert = make_static_task(StaticDisplaySpec(
    name="gram_stain_station_display",
    scene_file="mani_gram_stain_station.xml",
    prompt="a gram stain station",
))

# ===================== Metallurgy / welding =====================

ArcWeldingStationTask, ArcWeldingStationExpert = make_static_task(StaticDisplaySpec(
    name="arc_welding_station_display",
    scene_file="mani_arc_welding_station.xml",
    prompt="an arc welding station",
))

MetallographyPolisherTask, MetallographyPolisherExpert = make_static_task(StaticDisplaySpec(
    name="metallography_polisher_display",
    scene_file="mani_metallography_polisher.xml",
    prompt="a metallography sample polisher",
))

InductionFurnaceTask, InductionFurnaceExpert = make_static_task(StaticDisplaySpec(
    name="induction_furnace_display",
    scene_file="mani_induction_furnace.xml",
    prompt="an induction furnace",
))

MetalSpectrometerTask, MetalSpectrometerExpert = make_static_task(StaticDisplaySpec(
    name="metal_spectrometer_display",
    scene_file="mani_metal_spectrometer.xml",
    prompt="a metal alloy spectrometer",
))

HeatTreatmentOvenTask, HeatTreatmentOvenExpert = make_static_task(StaticDisplaySpec(
    name="heat_treatment_oven_display",
    scene_file="mani_heat_treatment_oven.xml",
    prompt="a metal heat treatment oven",
))

WeldingPositionerTask, WeldingPositionerExpert = make_static_task(StaticDisplaySpec(
    name="welding_positioner_display",
    scene_file="mani_welding_positioner.xml",
    prompt="a welding positioner turntable",
))

ImpactTesterCharpyTask, ImpactTesterCharpyExpert = make_static_task(StaticDisplaySpec(
    name="impact_tester_charpy_display",
    scene_file="mani_impact_tester_charpy.xml",
    prompt="a Charpy impact tester",
))
