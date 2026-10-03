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
