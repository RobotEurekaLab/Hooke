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
    scene_file="mani_real_lab_queen_mary_sequencer.xml",
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
