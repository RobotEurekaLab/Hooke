# Cell manipulation workstation

All five public microscopy experiments now operate on micrometre-scale simulated
cells. The original `push`, `pick_place` and `injection` identifiers remain valid,
so existing launch links select the updated experiments. Millimetre calibration
beads and the 100 nL chamber demonstration are retained as internal regression
fixtures, outside the public task catalogue.

| Experiment | Specimen and operation | Completion evidence |
| --- | --- | --- |
| Cell pushing | 36 µm spherical cell, 4 µm blunt probe tip | ≥50 ms probe contact, target error <3 µm, nominal compression ≤4.5 µm |
| Cell grasp and transfer | 36 µm cell, independently driven fine forceps | Bilateral contact ≥50 ms, lift ≥20 µm, maintained grasp, placement error <3 µm and release |
| Volume-controlled cell injection | 36 × 28 × 8 µm adherent cell | Membrane indentation and puncture, 0.5 pL delivery, withdrawal, pressure shutdown and volume conservation |
| Adherent-cell injection | Same injection model, separate retained task identifier | Existing intracellular-dose and membrane criteria |
| Suspended-cell holding and injection | 36 µm cell, vacuum holding and injection pipettes | Existing seal, holding, intracellular-dose and release criteria |

The field of view is 160 µm with a 20 µm scale bar. Every experiment offers the
same estimated grayscale phase-contrast view and a separate synthetic nuclear
fluorescence observation. Controllers locate the labelled nucleus from images;
sample truth is used by image formation and evaluation. Fine forceps contacts
and the blunt probe appear at their observed instrument positions. Focus follows
the nominal lift height during transfer to keep the cell in view.

## Contact and motion

MuJoCo or native Isaac/PhysX advances the electric instruments. Pushing and
gripping use their observed poses to calculate unilateral elastic contacts.
The suspended cell moves through an implicit overdamped force balance, with
Stokes drag at nominal viscosity 0.001 Pa·s, cell radius 18 µm and contact
stiffness 0.03 N/m. A bilateral grasp supplies tangential forces limited by
nominal friction; opening the forceps removes those contacts. Cell position is
never copied from a commanded tool pose, and no cell attachment weld is added.
Generalized reaction forces act on the instruments in both backends.

Cells are simulated specimens with original statistical textures. These models
do not resolve a soft membrane, bath flow, cell-cell collision, biological
survival or measured material properties. The 0.5 pL dose and compression
thresholds are demonstration parameters. Existing microscope references and CAD
licensing boundaries continue to apply; this change does not reconstruct a new
manufacturer microscope or forceps assembly.

## Run and record

See [deployment and versioned recordings](microscopy_live_backends.md) for the
instrument profile, per-account Isaac setup and media configuration. For example,
open `/microscopy?experiment=pick_place&backend=isaac` and run the experiment.
Requests carrying `target_nl` are rejected for these cellular tasks; use
`target_pl` for picolitre dose control. Changing operation rebuilds its scene
instead of reusing incompatible contact models.

New recordings are encoded from each backend's saved observations and published
as a separate release. Earlier videos, provenance reports and their hashes remain
available at their original paths. Encoder reports include specimen kind, nominal
diameters and volume unit, so a recording refresh can verify that all selected
videos contain cells rather than the older bead/chamber demonstration.
