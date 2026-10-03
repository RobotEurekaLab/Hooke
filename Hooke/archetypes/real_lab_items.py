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
