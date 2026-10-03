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
