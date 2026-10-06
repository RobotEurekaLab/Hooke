"""Display-only scenes for the Real Lab Reconstruction pilot (QS 101-250,
see real_lab/ at the repo root). Each task here represents one piece of
real, named equipment confirmed by sourced web research for a specific
university lab -- simplified/representative geometry in the same
display-only style as static_items.py/static_items2.py, not a CAD-fidelity
reconstruction. See the matching real_lab/labs/<school_id>.md card for the
sourcing (room, equipment list, citations) behind each scene."""
from archetypes.static_display import StaticDisplaySpec, make_static_task

WesternUniversityPcrTask, WesternUniversityPcrExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_western_university_pcr_display",
    scene_file="mani_real_lab_western_university_pcr.xml",
    prompt=(
        "a real-time PCR system in the Molecular Genetics Unit, Room 357, "
        "Western Science Centre, Western University (simplified "
        "representation; see real_lab/labs/western-university.md)"
    ),
))

LancasterUniversityConfocalTask, LancasterUniversityConfocalExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_lancaster_university_confocal_display",
    scene_file="mani_real_lab_lancaster_university_confocal.xml",
    prompt=(
        "an inverted confocal microscope in the Advanced Light Microscopy "
        "Facility, Division of Biomedical and Life Sciences, Lancaster "
        "University (simplified representation; see "
        "real_lab/labs/lancaster-university.md)"
    ),
))

RiceUniversityMaskAlignerTask, RiceUniversityMaskAlignerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_rice_university_mask_aligner_display",
    scene_file="mani_real_lab_rice_university_mask_aligner.xml",
    prompt=(
        "a photolithography mask aligner chamber in the Rice "
        "Nanofabrication Facility, Space Science and Technology Building, "
        "Rice University (simplified representation; see "
        "real_lab/labs/rice-university.md)"
    ),
))

AarhusUniversitySpinningDiskTask, AarhusUniversitySpinningDiskExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_aarhus_university_spinning_disk_display",
    scene_file="mani_real_lab_aarhus_university_spinning_disk.xml",
    prompt=(
        "a spinning-disk confocal microscope with a rotating objective "
        "turret in the Bioimaging Core Facility, Skou Building 1116 Room "
        "256, Aarhus University (simplified representation; see "
        "real_lab/labs/aarhus-university.md)"
    ),
))

CurtinUniversitySemTask, CurtinUniversitySemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_curtin_university_sem_display",
    scene_file="mani_real_lab_curtin_university_sem.xml",
    prompt=(
        "an electron microscope with a specimen load-lock chamber in the "
        "John de Laeter Centre, Curtin University (simplified "
        "representation; see real_lab/labs/curtin-university.md)"
    ),
))

EmoryUniversitySequencerTask, EmoryUniversitySequencerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_emory_university_sequencer_display",
    scene_file="mani_real_lab_emory_university_sequencer.xml",
    prompt=(
        "a benchtop DNA sequencer with a slide-out flow-cell drawer in "
        "the Emory Integrated Genomics Core, Woodruff Memorial Research "
        "Building, Emory University (simplified representation; see "
        "real_lab/labs/emory-university.md)"
    ),
))

UnicampAfmTask, UnicampAfmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_unicamp_afm_display",
    scene_file="mani_real_lab_unicamp_afm.xml",
    prompt=(
        "an atomic force microscope with a vertical scan head in the "
        "LIMicro-IQ microscopy core facility, Room D106, Institute of "
        "Chemistry, Universidade Estadual de Campinas (simplified "
        "representation; see "
        "real_lab/labs/universidade-estadual-de-campinas-unicamp.md)"
    ),
))

LoughboroughUniversityNmrTask, LoughboroughUniversityNmrExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_loughborough_university_nmr_display",
    scene_file="mani_real_lab_loughborough_university_nmr.xml",
    prompt=(
        "a benchtop NMR spectrometer with a hinged sample port in the "
        "WPL.2.09 Chemistry Synthesis Laboratory, STEMLab building, "
        "Loughborough University (simplified representation; see "
        "real_lab/labs/loughborough-university.md)"
    ),
))

TwenteUniversitySputterTask, TwenteUniversitySputterExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_twente_university_sputter_display",
    scene_file="mani_real_lab_twente_university_sputter.xml",
    prompt=(
        "a thin-film deposition chamber with a hinged viewport hatch in "
        "the MESA+ Institute NanoLab cleanroom, University of Twente "
        "(simplified representation; see "
        "real_lab/labs/university-of-twente.md)"
    ),
))

NagoyaUniversityPalmTask, NagoyaUniversityPalmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_nagoya_university_palm_display",
    scene_file="mani_real_lab_nagoya_university_palm.xml",
    prompt=(
        "a PALM Combi laser microdissection and optical-tweezers "
        "cell-manipulation microscope with a swinging laser delivery arm "
        "in the ITbM Live Imaging Center, Nagoya University (simplified "
        "representation; see real_lab/labs/nagoya-university.md)"
    ),
))

KhalifaUniversityTemTask, KhalifaUniversityTemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_khalifa_university_tem_display",
    scene_file="mani_real_lab_khalifa_university_tem.xml",
    prompt=(
        "a transmission electron microscope with a hinged specimen "
        "airlock port in the Electron Microscopy Facility, Building L "
        "Room 1020, Khalifa University (simplified representation; see "
        "real_lab/labs/khalifa-university-of-science-and-technology.md)"
    ),
))

GenevaUniversitySlideScannerTask, GenevaUniversitySlideScannerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_geneva_university_slide_scanner_display",
    scene_file="mani_real_lab_geneva_university_slide_scanner.xml",
    prompt=(
        "an Olympus VS120-style slide scanner with a sliding slide tray "
        "in Room C06.1533.a, Bioimaging Core Facility, CMU Building C, "
        "University of Geneva (simplified representation; see "
        "real_lab/labs/university-of-geneva.md)"
    ),
))

UncLightsheetTask, UncLightsheetExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_unc_lightsheet_display",
    scene_file="mani_real_lab_unc_lightsheet.xml",
    prompt=(
        "a light-sheet fluorescence microscope with a translating sample "
        "mount in the Biology Microscopy Core, Genome Sciences Building "
        "Room 1152, University of North Carolina at Chapel Hill "
        "(simplified representation; see "
        "real_lab/labs/university-of-north-carolina-chapel-hill.md)"
    ),
))

GroningenUniversityCellSorterTask, GroningenUniversityCellSorterExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_groningen_university_cell_sorter_display",
    scene_file="mani_real_lab_groningen_university_cell_sorter.xml",
    prompt=(
        "a SORP BD FACSAria-style flow cytometry cell sorter with a "
        "swinging sample probe arm in the GBB Dedicated Research "
        "Facilities, Linnaeusborg, University of Groningen (simplified "
        "representation; see real_lab/labs/university-of-groningen.md)"
    ),
))

BostonUniversityWidefieldTask, BostonUniversityWidefieldExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_boston_university_widefield_display",
    scene_file="mani_real_lab_boston_university_widefield.xml",
    prompt=(
        "a Nikon-style deconvolution wide-field epifluorescence "
        "microscope with a rotating filter turret in the Cellular "
        "Imaging Core, Evans Biomedical Research Center Basement B15, "
        "Boston University (simplified representation; see "
        "real_lab/labs/boston-university.md)"
    ),
))

WaterlooUniversityGelImagerTask, WaterlooUniversityGelImagerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_waterloo_university_gel_imager_display",
    scene_file="mani_real_lab_waterloo_university_gel_imager.xml",
    prompt=(
        "a gel-imaging system with a hinged UV-transilluminator lid in "
        "the Molecular Biology Core Facility, Room B1-371, Biology 1 "
        "building, University of Waterloo (simplified representation; "
        "see real_lab/labs/university-of-waterloo.md)"
    ),
))

UtrechtUniversityFibsemTask, UtrechtUniversityFibsemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_utrecht_university_fibsem_display",
    scene_file="mani_real_lab_utrecht_university_fibsem.xml",
    prompt=(
        "a dual-beam FIB-SEM with a hinged specimen chamber door in the "
        "Cell Microscopy Core, Room H02.313, Utrecht University "
        "(simplified representation; see "
        "real_lab/labs/utrecht-university.md)"
    ),
))

UppsalaUniversityMultiphotonTask, UppsalaUniversityMultiphotonExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_uppsala_university_multiphoton_display",
    scene_file="mani_real_lab_uppsala_university_multiphoton.xml",
    prompt=(
        "a Leica DIVE-style multiphoton microscope with a rotating "
        "filter turret in the BioVis imaging core facility, Rudbeck "
        "Laboratory, Uppsala University (simplified representation; see "
        "real_lab/labs/uppsala-university.md)"
    ),
))

KfupmSemTask, KfupmSemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_kfupm_sem_display",
    scene_file="mani_real_lab_kfupm_sem.xml",
    prompt=(
        "a field-emission SEM with a swinging EDX detector arm in the "
        "Chemistry Department Microscopy Laboratory, Building 4 Room "
        "157, King Fahd University of Petroleum and Minerals "
        "(simplified representation; see "
        "real_lab/labs/king-fahd-university-of-petroleum-minerals.md)"
    ),
))

KitKarlsruheTemTask, KitKarlsruheTemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_kit_karlsruhe_tem_display",
    scene_file="mani_real_lab_kit_karlsruhe_tem.xml",
    prompt=(
        "a transmission electron microscope with a tilting goniometer "
        "specimen stage in the Laboratory for Electron Microscopy, "
        "Building 30.22 Room 228, Karlsruhe Institute of Technology "
        "(simplified representation; see "
        "real_lab/labs/kit-karlsruhe-institute-of-technology.md)"
    ),
))

SheffieldUniversityChromiumTask, SheffieldUniversityChromiumExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_sheffield_university_chromium_display",
    scene_file="mani_real_lab_sheffield_university_chromium.xml",
    prompt=(
        "a 10x Genomics Chromium Controller with a hinged chip-loading "
        "lid in the Multiomics Facility, Sheffield Institute for "
        "Translational Neuroscience, University of Sheffield "
        "(simplified representation; see "
        "real_lab/labs/the-university-of-sheffield.md)"
    ),
))

BaselUniversityLmdTask, BaselUniversityLmdExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_basel_university_lmd_display",
    scene_file="mani_real_lab_basel_university_lmd.xml",
    prompt=(
        "a Leica LMD7-style laser microdissection microscope with a "
        "rotating objective/laser-path turret in the Imaging Core "
        "Facility, Biozentrum, University of Basel (simplified "
        "representation; see real_lab/labs/university-of-basel.md)"
    ),
))

UkmPcrTask, UkmPcrExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_ukm_pcr_display",
    scene_file="mani_real_lab_ukm_pcr.xml",
    prompt=(
        "a gradient PCR thermocycler with a hinged heating-block lid in "
        "the Genomics Lab, INBIOSIS, Universiti Kebangsaan Malaysia "
        "(simplified representation; see "
        "real_lab/labs/universiti-kebangsaan-malaysia-ukm.md)"
    ),
))

EindhovenMicrofabTask, EindhovenMicrofabExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_eindhoven_microfab_display",
    scene_file="mani_real_lab_eindhoven_microfab.xml",
    prompt=(
        "a femtosecond-laser micromachining setup with a swinging optics "
        "arm in the TU/e Microfab Lab, Building 15 Gemini-Noord, "
        "Eindhoven University of Technology (simplified representation; "
        "see real_lab/labs/eindhoven-university-of-technology.md)"
    ),
))

LeidenUniversityCryoemTask, LeidenUniversityCryoemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_leiden_university_cryoem_display",
    scene_file="mani_real_lab_leiden_university_cryoem.xml",
    prompt=(
        "a Titan Krios-style cryo-electron microscope with a cryo-dewar "
        "and hinged specimen airlock in the NeCEN facility, Gorlaeus "
        "Laboratory, Leiden University (simplified representation; see "
        "real_lab/labs/leiden-university.md)"
    ),
))

QueenMarySequencerTask, QueenMarySequencerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_queen_mary_sequencer_display",
    scene_file="mani_real_lab_queen_mary_room.xml",
    prompt=(
        "a multi-capillary DNA sequencer with a hinged loading door in "
        "the Barts and the London Genome Centre, Queen Mary University "
        "of London (simplified representation; see "
        "real_lab/labs/queen-mary-university-of-london.md)"
    ),
))

GothenburgUniversityMaldiTask, GothenburgUniversityMaldiExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_gothenburg_university_maldi_display",
    scene_file="mani_real_lab_gothenburg_university_maldi.xml",
    prompt=(
        "a MALDI imaging mass spectrometer with a hinged sample-plate "
        "loading door in the Centre for Cellular Imaging, University of "
        "Gothenburg (simplified representation; see "
        "real_lab/labs/university-of-gothenburg.md)"
    ),
))

RmitUniversityFibsemTask, RmitUniversityFibsemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_rmit_university_fibsem_display",
    scene_file="mani_real_lab_rmit_university_fibsem.xml",
    prompt=(
        "a JEOL/FEI-style dual-beam FIB-SEM with a hinged specimen "
        "chamber door in the RMIT Microscopy and Microanalysis Facility, "
        "Building 14, RMIT University (simplified representation; see "
        "real_lab/labs/rmit-university.md)"
    ),
))

McmasterUniversitySequencerTask, McmasterUniversitySequencerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_mcmaster_university_sequencer_display",
    scene_file="mani_real_lab_mcmaster_university_sequencer.xml",
    prompt=(
        "a PacBio-style long-read DNA sequencer with a hinged loading "
        "door in the McMaster Genomics Facility, Room 3N4, Health "
        "Sciences Centre, McMaster University (simplified "
        "representation; see real_lab/labs/mcmaster-university.md)"
    ),
))

WustlHistologyTask, WustlHistologyExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_wustl_histology_display",
    scene_file="mani_real_lab_wustl_histology.xml",
    prompt=(
        "a wide-field microscope for histology and fluorescence imaging "
        "with a rotating filter turret in the Molecular Microbiology "
        "Imaging Facility, Room 10302, McDonnell Pediatric Research "
        "Building, Washington University in St. Louis (simplified "
        "representation; see "
        "real_lab/labs/washington-university-in-st-louis.md)"
    ),
))

RochesterUniversitySonicatorTask, RochesterUniversitySonicatorExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_rochester_university_sonicator_display",
    scene_file="mani_real_lab_rochester_university_sonicator.xml",
    prompt=(
        "a Covaris-style focused-ultrasonicator with a hinged water-bath "
        "lid in the Rochester Genomics Center, James P. Wilmot Cancer "
        "Institute, University of Rochester (simplified representation; "
        "see real_lab/labs/university-of-rochester.md)"
    ),
))

QueensKingstonMaldiTask, QueensKingstonMaldiExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_queens_kingston_maldi_display",
    scene_file="mani_real_lab_queens_kingston_maldi.xml",
    prompt=(
        "a Bruker AutoFlex-style MALDI-TOF mass spectrometer with a "
        "hinged sample-plate loading door in the Department of "
        "Chemistry shared analytical facilities, Chernoff Hall, Queen's "
        "University at Kingston (simplified representation; see "
        "real_lab/labs/queen-s-university-at-kingston.md)"
    ),
))

PadovaUniversityTemTask, PadovaUniversityTemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_padova_university_tem_display",
    scene_file="mani_real_lab_padova_university_tem.xml",
    prompt=(
        "a FEI Tecnai G2-style transmission electron microscope with a "
        "tilting goniometer specimen stage in the DiBio Imaging "
        "Facility, University of Padova (simplified representation; see "
        "real_lab/labs/universit-di-padova.md)"
    ),
))

DtuNanolabFibsemTask, DtuNanolabFibsemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_dtu_nanolab_fibsem_display",
    scene_file="mani_real_lab_dtu_nanolab_fibsem.xml",
    prompt=(
        "a dual-beam FIB-SEM with a hinged specimen chamber door in the "
        "DTU Nanolab cleanroom, Building 347, Technical University of "
        "Denmark (simplified representation; see "
        "real_lab/labs/technical-university-of-denmark.md)"
    ),
))

WuhanUniversityPlateReaderTask, WuhanUniversityPlateReaderExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_wuhan_university_platereader_display",
    scene_file="mani_real_lab_wuhan_university_platereader.xml",
    prompt=(
        "a microplate reader with a hinged sample-tray lid on the "
        "biochemical and molecular detection platform, Shared "
        "Instrument Platform, College of Life Sciences, Wuhan "
        "University (simplified representation; see "
        "real_lab/labs/wuhan-university.md)"
    ),
))

TohokuUniversityMriTask, TohokuUniversityMriExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_tohoku_university_mri_display",
    scene_file="mani_real_lab_tohoku_university_mri.xml",
    prompt=(
        "a 3.0T MRI scanner bore with a sliding patient table in the "
        "ToMMo biobank/genome-medicine facility, Seiryo campus, Tohoku "
        "University (simplified representation; see "
        "real_lab/labs/tohoku-university.md)"
    ),
))

IitbNmrTask, IitbNmrExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_iitb_nmr_display",
    scene_file="mani_real_lab_iitb_nmr.xml",
    prompt=(
        "a 500 MHz NMR spectrometer with a hinged sample-insertion port "
        "in the IOE facility, Room 212, Department of Chemistry, Indian "
        "Institute of Technology Bombay (simplified representation; see "
        "real_lab/labs/indian-institute-of-technology-bombay-iitb.md)"
    ),
))

ChulalongkornPotentiostatTask, ChulalongkornPotentiostatExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_chulalongkorn_potentiostat_display",
    scene_file="mani_real_lab_chulalongkorn_potentiostat.xml",
    prompt=(
        "a GAMRY-style benchtop potentiostat with a hinged electrode "
        "access lid in the Energetic Materials Research Laboratory, "
        "Chulalongkorn University (simplified representation; see "
        "real_lab/labs/chulalongkorn-university.md)"
    ),
))

KyushuUniversityLaserTask, KyushuUniversityLaserExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_kyushu_university_laser_display",
    scene_file="mani_real_lab_kyushu_university_laser.xml",
    prompt=(
        "an ultrafast laser spectroscopy bench with a swinging optics "
        "arm in the Laboratory of Spectrochemistry, Room B1009, Ito "
        "Campus, Kyushu University (simplified representation; see "
        "real_lab/labs/kyushu-university.md)"
    ),
))

VanderbiltUniversityEsemTask, VanderbiltUniversityEsemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_vanderbilt_university_esem_display",
    scene_file="mani_real_lab_vanderbilt_university_esem.xml",
    prompt=(
        "an environmental scanning electron microscope with a swinging "
        "EDX detector arm in the Cell Imaging Shared Resource, "
        "Vanderbilt University (simplified representation; see "
        "real_lab/labs/vanderbilt-university.md)"
    ),
))

TuBerlinFlowCytometerTask, TuBerlinFlowCytometerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_tu_berlin_flow_cytometer_display",
    scene_file="mani_real_lab_tu_berlin_flow_cytometer.xml",
    prompt=(
        "a benchtop flow cytometer with a hinged sample-tube access lid "
        "in the Chair of Environmental Microbiomics laboratory, Room "
        "BH 6-1, Technische Universitat Berlin (simplified "
        "representation; see "
        "real_lab/labs/technische-universit-t-berlin-tu-berlin.md)"
    ),
))

BonnLimesStedTask, BonnLimesStedExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_bonn_limes_sted_display",
    scene_file="mani_real_lab_bonn_limes_sted.xml",
    prompt=(
        "an Abberior easy3D STED super-resolution microscope with a "
        "rotating filter turret in the LIMES Technical Platforms, "
        "University of Bonn (simplified representation; see "
        "real_lab/labs/rheinische-friedrich-wilhelms-universit-t-bonn.md)"
    ),
))

FauErlangenCenemTask, FauErlangenCenemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_fau_erlangen_cenem_display",
    scene_file="mani_real_lab_fau_erlangen_cenem.xml",
    prompt=(
        "a dual-beam electron microscope with a hinged specimen chamber "
        "door in CENEM, the Center for Nanoanalysis and Electron "
        "Microscopy, Friedrich-Alexander-Universitat Erlangen-Nurnberg "
        "(simplified representation; see "
        "real_lab/labs/friedrich-alexander-universit-t-erlangen-n-rnberg.md)"
    ),
))

UlbLimifTask, UlbLimifExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_ulb_limif_display",
    scene_file="mani_real_lab_ulb_limif.xml",
    prompt=(
        "a Zeiss LSM780-style confocal/multiphoton microscope with a "
        "rotating filter turret in the Light Microscopy Facility "
        "(LiMiF), IRIBHM, Universite Libre de Bruxelles (simplified "
        "representation; see "
        "real_lab/labs/universite-libre-de-bruxelles.md)"
    ),
))

AsuLiquidHandlerTask, AsuLiquidHandlerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_asu_liquid_handler_display",
    scene_file="mani_real_lab_asu_liquid_handler.xml",
    prompt=(
        "a Beckman Biomek-style automated liquid-handling gantry robot "
        "in the Desert Southwest Genomics Center, Arizona State "
        "University (simplified representation; see "
        "real_lab/labs/arizona-state-university.md)"
    ),
))

StAndrewsOrbitrapTask, StAndrewsOrbitrapExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_st_andrews_orbitrap_display",
    scene_file="mani_real_lab_st_andrews_orbitrap.xml",
    prompt=(
        "a Thermo Orbitrap Exploris-style mass spectrometer with a "
        "hinged sample-plate loading door in the School of Chemistry "
        "analytical facilities, Purdie Building, University of St "
        "Andrews (simplified representation; see "
        "real_lab/labs/university-of-st-andrews.md)"
    ),
))

UcdConwayCytometerTask, UcdConwayCytometerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_ucd_conway_cytometer_display",
    scene_file="mani_real_lab_ucd_room.xml",
    prompt=(
        "a Beckman Coulter-style flow cytometer with a hinged sample "
        "access lid in the Conway Genomics and Imaging Core, University "
        "College Dublin (simplified representation; see "
        "real_lab/labs/university-college-dublin.md)"
    ),
))

StockholmUniversitySpsTask, StockholmUniversitySpsExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_stockholm_university_sps_display",
    scene_file="mani_real_lab_stockholm_university_sps.xml",
    prompt=(
        "a spark plasma sintering press with a moving upper ram in the "
        "Spark Plasma Sintering Facility, Arrhenius Laboratory, "
        "Stockholm University (simplified representation; see "
        "real_lab/labs/stockholm-university.md)"
    ),
))

PolimiFlowReactorTask, PolimiFlowReactorExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_polimi_flow_reactor_display",
    scene_file="mani_real_lab_polimi_flow_reactor.xml",
    prompt=(
        "a continuous-flow chemical reactor with a swinging valve lever "
        "in BiocatLab, Politecnico di Milano (simplified representation; "
        "see real_lab/labs/politecnico-di-milano.md)"
    ),
))

IitmBioincubatorFermenterTask, IitmBioincubatorFermenterExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_iitm_bioincubator_fermenter_display",
    scene_file="mani_real_lab_iitm_bioincubator_fermenter.xml",
    prompt=(
        "a benchtop fermenter with a hinged vessel lid in the IITM "
        "Bioincubator, IIT Madras Research Park (simplified "
        "representation; see "
        "real_lab/labs/indian-institute-of-technology-madras-iitm.md)"
    ),
))

WisconsinMadisonNovaseqTask, WisconsinMadisonNovaseqExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_wisconsin_madison_novaseq_display",
    scene_file="mani_real_lab_wisconsin_madison_room.xml",
    prompt=(
        "an Illumina NovaSeq X Plus-style sequencer with a hinged "
        "loading door in the UW Biotech Center DNA Sequencing Facility, "
        "Room 1250, 425 Henry Mall, University of Wisconsin-Madison "
        "(simplified representation; see "
        "real_lab/labs/university-of-wisconsin-madison.md)"
    ),
))

HelsinkiUniversitySequencerTask, HelsinkiUniversitySequencerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_helsinki_university_sequencer_display",
    scene_file="mani_real_lab_helsinki_university_sequencer.xml",
    prompt=(
        "a benchtop DNA sequencer with a hinged flow-cell loading lid in "
        "the DNA Sequencing and Genomics Laboratory (BIDGEN), Biocenter "
        "2, Viikki campus, University of Helsinki (simplified "
        "representation; see real_lab/labs/university-of-helsinki.md)"
    ),
))

NottinghamGenomicsTask, NottinghamGenomicsExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_nottingham_genomics_display",
    scene_file="mani_real_lab_nottingham_genomics.xml",
    prompt=(
        "a genomics array scanner with a hinged sample-tray lid in the "
        "Post-Genomic Technologies Facility, A Floor West Block, Queen's "
        "Medical Centre, University of Nottingham (simplified "
        "representation; see real_lab/labs/university-of-nottingham.md)"
    ),
))

OsloUniversityLivecellTask, OsloUniversityLivecellExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_oslo_university_livecell_display",
    scene_file="mani_real_lab_oslo_university_livecell.xml",
    prompt=(
        "a live-cell confocal imaging microscope with a rotating filter "
        "turret in the Advanced Light Microscopy core facility, "
        "Rikshospitalet, University of Oslo (simplified representation; "
        "see real_lab/labs/university-of-oslo.md)"
    ),
))

DartmouthCollegeDragonflyTask, DartmouthCollegeDragonflyExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_dartmouth_college_dragonfly_display",
    scene_file="mani_real_lab_dartmouth_college_dragonfly.xml",
    prompt=(
        "an Andor Dragonfly-style spinning-disk confocal microscope with "
        "a rotating filter turret in the Life Sciences Light Microscopy "
        "Facility, Class of 1978 Life Sciences Center, Dartmouth College "
        "(simplified representation; see "
        "real_lab/labs/dartmouth-college.md)"
    ),
))

CalgaryUniversityAfmTask, CalgaryUniversityAfmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_calgary_university_afm_display",
    scene_file="mani_real_lab_calgary_university_afm.xml",
    prompt=(
        "a Bruker NanoWizard IV atomic force microscope with a vertical "
        "scan head in the Microscopy and Imaging Facility (Charbonneau "
        "Microscopy Facility), University of Calgary (simplified "
        "representation; see real_lab/labs/university-of-calgary.md)"
    ),
))

AberdeenUniversityCelldiscovererTask, AberdeenUniversityCelldiscovererExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_aberdeen_university_celldiscoverer_display",
    scene_file="mani_real_lab_aberdeen_university_celldiscoverer.xml",
    prompt=(
        "a ZEISS Celldiscoverer 7 high-content imaging platform with a "
        "robotic plate-handling arm in the Microscopy and Histology Core "
        "Facility, Institute of Medical Sciences, University of Aberdeen "
        "(simplified representation; see "
        "real_lab/labs/university-of-aberdeen.md)"
    ),
))

TubingenUniversityEmTask, TubingenUniversityEmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_tubingen_university_em_display",
    scene_file="mani_real_lab_tubingen_university_em.xml",
    prompt=(
        "an electron microscope with a swinging EDX detector arm in the "
        "Tubingen Structural Microscopy (TSM) Core Facility, Campus "
        "Morgenstelle, Eberhard Karls Universitat Tubingen (simplified "
        "representation; see "
        "real_lab/labs/eberhard-karls-universit-t-t-bingen.md)"
    ),
))

BernUniversityLsmTask, BernUniversityLsmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_bern_university_lsm_display",
    scene_file="mani_real_lab_bern_university_lsm.xml",
    prompt=(
        "a Zeiss LSM710-style confocal laser scanning microscope with a "
        "rotating filter turret in the Microscopy Imaging Center (MIC), "
        "University of Bern (simplified representation; see "
        "real_lab/labs/university-of-bern.md)"
    ),
))

ErasmusRotterdamBiomicsTask, ErasmusRotterdamBiomicsExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_erasmus_rotterdam_biomics_display",
    scene_file="mani_real_lab_erasmus_rotterdam_biomics.xml",
    prompt=(
        "a benchtop genomics sequencer with a hinged loading lid in the "
        "Erasmus Center for Biomics, Department of Molecular Genetics, "
        "Erasmus MC, Erasmus University Rotterdam (simplified "
        "representation; see "
        "real_lab/labs/erasmus-university-rotterdam.md)"
    ),
))

AubNgsSequencerTask, AubNgsSequencerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_aub_ngs_sequencer_display",
    scene_file="mani_real_lab_aub_ngs_sequencer.xml",
    prompt=(
        "a next-generation sequencing platform with a hinged loading lid "
        "in the Genomic Profiling Program, Aida and Halim Daniel "
        "Academic and Clinical Center, American University of Beirut "
        "(simplified representation; see "
        "real_lab/labs/american-university-of-beirut-aub.md)"
    ),
))

SussexUniversityScanrTask, SussexUniversityScanrExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_sussex_university_scanr_display",
    scene_file="mani_real_lab_sussex_university_scanr.xml",
    prompt=(
        "an Olympus ScanR automated high-content screening platform with "
        "a robotic plate-handling arm in the Wolfson Centre for "
        "Biological Imaging, University of Sussex (simplified "
        "representation; see real_lab/labs/university-of-sussex.md)"
    ),
))

WellingtonUniversityVucelTask, WellingtonUniversityVucelExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_wellington_university_vucel_display",
    scene_file="mani_real_lab_wellington_university_vucel.xml",
    prompt=(
        "a flow-through seawater wet-lab bench with a swinging tap lever "
        "in the Victoria University Coastal Ecology Lab (VUCEL), "
        "Victoria University of Wellington (simplified representation; "
        "see real_lab/labs/victoria-university-of-wellington.md)"
    ),
))

ChileUniversityPcrTask, ChileUniversityPcrExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_chile_university_pcr_display",
    scene_file="mani_real_lab_chile_university_pcr.xml",
    prompt=(
        "a PCR thermocycler with a hinged heating-block lid in the "
        "Laboratorio de Biologia Molecular e Ingenieria Genetica, "
        "Universidad de Chile (simplified representation; see "
        "real_lab/labs/universidad-de-chile.md)"
    ),
))

CapeTownBscTask, CapeTownBscExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_cape_town_bsc_display",
    scene_file="mani_real_lab_cape_town_bsc.xml",
    prompt=(
        "a Class II biosafety cabinet with a sliding sash in the "
        "Institute of Infectious Disease and Molecular Medicine (IDM), "
        "University of Cape Town (simplified representation; see "
        "real_lab/labs/university-of-cape-town.md)"
    ),
))

GadjahMadaPromethionTask, GadjahMadaPromethionExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_gadjah_mada_promethion_display",
    scene_file="mani_real_lab_gadjah_mada_promethion.xml",
    prompt=(
        "an Oxford Nanopore PromethION 24-style sequencer with a hinged "
        "loading lid in the Integrated Genome Factory (IGF), Faculty of "
        "Biology, Gadjah Mada University (simplified representation; "
        "see real_lab/labs/gadjah-mada-university.md)"
    ),
))

DeakinUniversitySemTask, DeakinUniversitySemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_deakin_university_sem_display",
    scene_file="mani_real_lab_deakin_university_sem.xml",
    prompt=(
        "a JEOL IT 300-style scanning electron microscope with a "
        "swinging EDX detector arm in the Materials Science Labs, Deakin "
        "University (simplified representation; see "
        "real_lab/labs/deakin-university.md)"
    ),
))

IitkgpFesemTask, IitkgpFesemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_iitkgp_fesem_display",
    scene_file="mani_real_lab_iitkgp_fesem.xml",
    prompt=(
        "a ZEISS/JEOL-style field-emission SEM with a hinged specimen "
        "chamber door in the Central Research Facility (CRF), IIT "
        "Kharagpur (simplified representation; see "
        "real_lab/labs/indian-institute-of-technology-kharagpur-iit-kgp.md)"
    ),
))

DarmstadtAemStemTask, DarmstadtAemStemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_darmstadt_aem_stem_display",
    scene_file="mani_real_lab_darmstadt_aem_stem.xml",
    prompt=(
        "a JEOL ARM 200F aberration-corrected scanning transmission "
        "electron microscope with a tilting goniometer stage in the "
        "Advanced Electron Microscopy Division, Building L2|01 Room 52, "
        "Technical University of Darmstadt (simplified representation; "
        "see real_lab/labs/technical-university-of-darmstadt.md)"
    ),
))

PolitoUltrasonicTask, PolitoUltrasonicExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_polito_ultrasonic_display",
    scene_file="mani_real_lab_polito_ultrasonic.xml",
    prompt=(
        "a Vibracell VC-505 ultrasonic processor with a hinged lid in "
        "the Chemical Synthesis Laboratory, Technological Centre in "
        "Alessandria, Politecnico di Torino (simplified representation; "
        "see real_lab/labs/politecnico-di-torino.md)"
    ),
))

MasseyUniversityAbiTask, MasseyUniversityAbiExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_massey_university_abi_display",
    scene_file="mani_real_lab_massey_university_abi.xml",
    prompt=(
        "an ABI 3500xl-style capillary genetic analyzer with a hinged "
        "loading door in the Massey Genome Service, Room ScD3.15A, "
        "Turitea Campus, Massey University (simplified representation; "
        "see real_lab/labs/massey-university.md)"
    ),
))

ExeterUniversityEmTask, ExeterUniversityEmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_exeter_university_em_display",
    scene_file="mani_real_lab_exeter_university_em.xml",
    prompt=(
        "an electron microscope with a swinging EDX detector arm in the "
        "Bioimaging Centre, Geoffrey Pope Building, University of Exeter "
        "(simplified representation; see "
        "real_lab/labs/the-university-of-exeter.md)"
    ),
))

UscGenomicsCoreTask, UscGenomicsCoreExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_usc_genomics_core_display",
    scene_file="mani_real_lab_usc_room.xml",
    prompt=(
        "a benchtop genomics instrument with a hinged loading lid in the "
        "Molecular Genomics Core (MGC), Norris Research Tower, USC "
        "Health Sciences Campus, University of Southern California "
        "(simplified representation; see "
        "real_lab/labs/university-of-southern-california.md)"
    ),
))

UsmPfgeTask, UsmPfgeExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_usm_pfge_display",
    scene_file="mani_real_lab_usm_pfge.xml",
    prompt=(
        "a pulsed-field gel electrophoresis (PFGE) system with a hinged "
        "buffer-chamber lid in the INFORMM equipment facility, "
        "Universiti Sains Malaysia (simplified representation; see "
        "real_lab/labs/universiti-sains-malaysia-usm.md)"
    ),
))

MsuQuantstudioTask, MsuQuantstudioExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_msu_quantstudio_display",
    scene_file="mani_real_lab_msu_quantstudio.xml",
    prompt=(
        "an ABI QuantStudio 7 Flex real-time PCR system with a hinged "
        "sample-plate loading door in the RTSF Genomics Core, Michigan "
        "State University (simplified representation; see "
        "real_lab/labs/michigan-state-university.md)"
    ),
))

UamMadridStellarisTask, UamMadridStellarisExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_uam_madrid_stellaris_display",
    scene_file="mani_real_lab_uam_madrid_stellaris.xml",
    prompt=(
        "a Leica Stellaris 8 confocal microscope with FLIM/STED modules "
        "and a rotating filter turret in the SMOA Advanced Optical "
        "Microscopy Facility, Lab 310, CBM, Universidad Autonoma de "
        "Madrid (simplified representation; see "
        "real_lab/labs/universidad-aut-noma-de-madrid.md)"
    ),
))

MinnesotaUniversityCtrapTask, MinnesotaUniversityCtrapExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_minnesota_university_ctrap_display",
    scene_file="mani_real_lab_minnesota_university_ctrap.xml",
    prompt=(
        "a LUMICKS C-Trap optical-tweezers instrument with a sliding "
        "microfluidic chip stage in the University Imaging Centers, "
        "Jackson Hall, University of Minnesota Twin Cities (simplified "
        "representation; see "
        "real_lab/labs/university-of-minnesota-twin-cities.md)"
    ),
))

KingSaudGammacellTask, KingSaudGammacellExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_king_saud_gammacell_display",
    scene_file="mani_real_lab_king_saud_gammacell.xml",
    prompt=(
        "a Gammacell 220 sample irradiator with a sliding sample drawer "
        "in the Central Laboratory, College of Science, King Saud "
        "University (simplified representation; see "
        "real_lab/labs/king-saud-university.md)"
    ),
))

HamburgUniversityConfocalTask, HamburgUniversityConfocalExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_hamburg_university_confocal_display",
    scene_file="mani_real_lab_hamburg_university_confocal.xml",
    prompt=(
        "a Leica/Nikon/Zeiss-style confocal microscope with a rotating "
        "filter turret in the Technology Platform Light Microscopy, "
        "Universitat Hamburg (simplified representation; see "
        "real_lab/labs/universit-t-hamburg.md)"
    ),
))

UclouvainNmrTask, UclouvainNmrExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_uclouvain_nmr_display",
    scene_file="mani_real_lab_uclouvain_nmr.xml",
    prompt=(
        "a 600 MHz FT-NMR spectrometer with a hinged sample-insertion "
        "port in the NMR Platform, IMCN, Universite Catholique de "
        "Louvain (simplified representation; see "
        "real_lab/labs/universit-catholique-de-louvain-uclouvain.md)"
    ),
))

FreiburgUniversityStedTask, FreiburgUniversityStedExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_freiburg_university_sted_display",
    scene_file="mani_real_lab_freiburg_university_sted.xml",
    prompt=(
        "a Leica TCS SP8 STED 3x super-resolution microscope with a "
        "rotating filter turret in the Life Imaging Center (LIC), "
        "Hilde-Mangold-Haus, University of Freiburg (simplified "
        "representation; see "
        "real_lab/labs/albert-ludwigs-universitaet-freiburg.md)"
    ),
))

IiscAptTask, IiscAptExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_iisc_apt_display",
    scene_file="mani_real_lab_iisc_apt.xml",
    prompt=(
        "an atom probe tomography (APT) system with a tilting specimen "
        "stage in the Advanced Facility for Microscopy and Microanalysis "
        "(AFMM), Indian Institute of Science (simplified representation; "
        "see real_lab/labs/indian-institute-of-science.md)"
    ),
))

NckuJem1400Task, NckuJem1400Expert = make_static_task(StaticDisplaySpec(
    name="real_lab_ncku_jem1400_display",
    scene_file="mani_real_lab_ncku_jem1400.xml",
    prompt=(
        "a JEOL JEM-1400 transmission electron microscope with a "
        "tilting goniometer stage in the TEM service, Room 82-B139, "
        "Medical Building, National Cheng Kung University (simplified "
        "representation; see "
        "real_lab/labs/national-cheng-kung-university-ncku.md)"
    ),
))

NthuCmsTask, NthuCmsExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_nthu_cms_display",
    scene_file="mani_real_lab_nthu_cms.xml",
    prompt=(
        "a ZEISS LSM 800 confocal microscope with Airyscan and a "
        "rotating filter turret in the Confocal Microscope System (CMS) "
        "facility, Life Sciences Building II, National Tsing Hua "
        "University (simplified representation; see "
        "real_lab/labs/national-tsing-hua-university.md)"
    ),
))

NycuLsm900Task, NycuLsm900Expert = make_static_task(StaticDisplaySpec(
    name="real_lab_nycu_lsm900_display",
    scene_file="mani_real_lab_nycu_lsm900.xml",
    prompt=(
        "a Carl Zeiss LSM900 confocal microscope with Airyscan2 and a "
        "rotating filter turret in the Imaging Core Facility, Room 639, "
        "Library Building, Yang Ming Campus, National Yang Ming Chiao "
        "Tung University (simplified representation; see "
        "real_lab/labs/national-yang-ming-chiao-tung-university.md)"
    ),
))

QutCarfHimTask, QutCarfHimExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_qut_carf_him_display",
    scene_file="mani_real_lab_qut_carf_him.xml",
    prompt=(
        "a Zeiss Helios Helium Ion Microscope with a hinged specimen "
        "chamber door in the Central Analytical Research Facility "
        "(CARF), M Block, Queensland University of Technology "
        "(simplified representation; see "
        "real_lab/labs/queensland-university-of-technology-qut.md)"
    ),
))

TelAvivUniversityLsm510Task, TelAvivUniversityLsm510Expert = make_static_task(StaticDisplaySpec(
    name="real_lab_tel_aviv_university_lsm510_display",
    scene_file="mani_real_lab_tel_aviv_university_lsm510.xml",
    prompt=(
        "a Zeiss LSM 510-META confocal laser scanning microscope with a "
        "rotating filter turret in the Rosalie and Harold Rae Brown "
        "Cancer Research Core Facility, Tel Aviv University (simplified "
        "representation; see real_lab/labs/tel-aviv-university.md)"
    ),
))

OsuCmifLightsheetTask, OsuCmifLightsheetExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_osu_cmif_lightsheet_display",
    scene_file="mani_real_lab_osu_cmif_lightsheet.xml",
    prompt=(
        "a Miltenyi UltraMicroscope Blaze light-sheet microscope in "
        "Room 245A, Campus Microscopy and Imaging Facility (CMIF), "
        "Biomedical Research Tower, Ohio State University (simplified "
        "representation; see real_lab/labs/the-ohio-state-university.md)"
    ),
))

FloridaUniversityFlowcytometerTask, FloridaUniversityFlowcytometerExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_florida_university_flowcytometer_display",
    scene_file="mani_real_lab_florida_university_flowcytometer.xml",
    prompt=(
        "a BD Symphony S6-style flow cytometer/sorter with a hinged "
        "sample access lid in the Cytometry & Optical Microscopy Core, "
        "Cancer & Genetics Research Complex, University of Florida "
        "(simplified representation; see "
        "real_lab/labs/university-of-florida.md)"
    ),
))

MarylandUniversityJpkAfmTask, MarylandUniversityJpkAfmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_maryland_university_jpk_afm_display",
    scene_file="mani_real_lab_maryland_university_jpk_afm.xml",
    prompt=(
        "a JPK NanoWizard 4a atomic force microscope with a vertical "
        "scan head in the CMNS Imaging Incubator, Physical Sciences "
        "Complex, University of Maryland, College Park (simplified "
        "representation; see "
        "real_lab/labs/university-of-maryland-college-park.md)"
    ),
))

OtagoUniversityJsm6700fTask, OtagoUniversityJsm6700fExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_otago_university_jsm6700f_display",
    scene_file="mani_real_lab_otago_university_jsm6700f.xml",
    prompt=(
        "a JEOL JSM-6700F field-emission SEM with a cryo-preparation "
        "stage and a swinging EDX detector arm in the OMNI electron "
        "microscopy unit, Room B10, Lindo Ferguson Building, University "
        "of Otago (simplified representation; see "
        "real_lab/labs/university-of-otago.md)"
    ),
))

UcsbNanoliveTask, UcsbNanoliveExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_ucsb_nanolive_display",
    scene_file="mani_real_lab_ucsb_nanolive.xml",
    prompt=(
        "a Nanolive 3D Cell Explorer label-free live-cell tomography "
        "microscope with a rotating filter turret in the NRI-MCDB "
        "Microscopy Facility, Bio2 Building Room 5173B, UC Santa Barbara "
        "(simplified representation; see "
        "real_lab/labs/university-of-california-santa-barbara-ucsb.md)"
    ),
))

NewcastleAuOrbitrapTask, NewcastleAuOrbitrapExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_newcastle_au_orbitrap_display",
    scene_file="mani_real_lab_newcastle_au_orbitrap.xml",
    prompt=(
        "a Thermo Q Exactive Orbitrap mass spectrometer with a hinged "
        "sample-plate loading door in the Central Analytical Facilities "
        "(CAF), Life Sciences Building, University of Newcastle "
        "Australia (simplified representation; see "
        "real_lab/labs/the-university-of-newcastle-australia-uon.md)"
    ),
))

UniandesTescanTask, UniandesTescanExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_uniandes_tescan_display",
    scene_file="mani_real_lab_uniandes_tescan.xml",
    prompt=(
        "a Tescan Lyra 3 dual-beam FIB-SEM with a hinged specimen "
        "chamber door in the Centro de Microscopia (MicroCore), Lab "
        "B101-B102, Universidad de los Andes (simplified "
        "representation; see "
        "real_lab/labs/universidad-de-los-andes.md)"
    ),
))

WasedaUniversityJem2100fTask, WasedaUniversityJem2100fExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_waseda_university_jem2100f_display",
    scene_file="mani_real_lab_waseda_university_jem2100f.xml",
    prompt=(
        "a JEOL JEM-2100F field-emission scanning transmission electron "
        "microscope with a tilting goniometer stage in the Analytical "
        "Instrument Laboratory, Building 42-1 Room 212, Waseda "
        "University (simplified representation; see "
        "real_lab/labs/waseda-university.md)"
    ),
))

YorkUniversityElyra7Task, YorkUniversityElyra7Expert = make_static_task(StaticDisplaySpec(
    name="real_lab_york_university_elyra7_display",
    scene_file="mani_real_lab_york_university_elyra7.xml",
    prompt=(
        "a Carl Zeiss ELYRA 7 super-resolution SIM/PALM/STORM microscope "
        "with a rotating filter turret in the Technology Facility "
        "Imaging and Cytometry Laboratory, University of York "
        "(simplified representation; see "
        "real_lab/labs/university-of-york.md)"
    ),
))

KeioUniversityMultisemTask, KeioUniversityMultisemExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_keio_university_multisem_display",
    scene_file="mani_real_lab_keio_university_multisem.xml",
    prompt=(
        "a Zeiss MultiSEM 505 multi-beam scanning electron microscope "
        "with a swinging EDX detector arm in the Electron Microscope "
        "Center, Preventive Medicine Building, Keio University "
        "(simplified representation; see "
        "real_lab/labs/keio-university.md)"
    ),
))

OttawaUniversitySp5stedTask, OttawaUniversitySp5stedExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_ottawa_university_sp5sted_display",
    scene_file="mani_real_lab_ottawa_university_sp5sted.xml",
    prompt=(
        "a Leica TCS SP5-STED super-resolution confocal microscope with "
        "a rotating filter turret in the CBIA Core Facility, 451 Smyth "
        "Road, University of Ottawa (simplified representation; see "
        "real_lab/labs/university-of-ottawa.md)"
    ),
))

TuwienUstemNeoarmTask, TuwienUstemNeoarmExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_tuwien_ustem_neoarm_display",
    scene_file="mani_real_lab_tuwien_ustem_neoarm.xml",
    prompt=(
        "a JEOL NeoARM 200 aberration-corrected transmission electron "
        "microscope with a tilting goniometer stage in USTEM, Room "
        "057-02, Freihaus building, TU Wien (simplified representation; "
        "see real_lab/labs/technische-universit-t-wien.md)"
    ),
))

HbkuQbriImagestreamTask, HbkuQbriImagestreamExpert = make_static_task(StaticDisplaySpec(
    name="real_lab_hbku_qbri_imagestream_display",
    scene_file="mani_real_lab_hbku_qbri_imagestream.xml",
    prompt=(
        "an Amnis ImageStream MKII imaging flow cytometer with a hinged "
        "sample access lid in the QBRI Imaging and Flow Cytometry Core, "
        "Hamad Bin Khalifa University (simplified representation; see "
        "real_lab/labs/hamad-bin-khalifa-university.md)"
    ),
))
