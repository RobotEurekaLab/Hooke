"""Parametric rotor-capacity variants of the centrifuge Eppendorf 5430 asset.

The rotor's tube slots are a MuJoCo `<replicate count="30" euler="0 0
0.2094395">` block wrapped around a single slot definition (30 slots * 12
degrees = 360 degrees) in `model/instrument/centrifuge_eppendorf_5430.xml` --
the original AutoBio authors even left a commented-out reference to a
`centrifuge_10slot.gen.xml` in the scene file, suggesting this exact kind of
variant was anticipated but never built. Producing a different-capacity
rotor is therefore a pure MJCF edit (change `count` and recompute `euler` so
the slots stay evenly spaced): no mesh regeneration or external 3D generation
needed, since every replicated slot reuses the same collision/visual meshes.

This is deliberately the *cheap* end of Phase I's asset-scaling plan (see
`private/proposal.tex` Phase I and `private/technical-log.md`): reducing the
slot count (wider spacing) reuses the same per-slot collision geometry
safely. Increasing it beyond the original 30 is not attempted here, since
the collision meshes were modeled for 12-degree spacing and tighter spacing
risks self-intersection -- validate that separately before relying on it.

`instrument.py`'s `Centrifuge_Eppendorf_5430._reload` auto-detects
`slot00`, `slot01`, ... until a name doesn't resolve, so it (and everything
built on it) works unchanged against any of these variants; only the scene
file differs.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

MODEL_ROOT = Path(__file__).parent.parent / "model"
BASE_INSTRUMENT = MODEL_ROOT / "instrument" / "centrifuge_eppendorf_5430.xml"
BASE_SCENE = MODEL_ROOT / "scene" / "insert_centrifuge_5430.xml"

_REPLICATE_RE = re.compile(r'<replicate count="30" euler="0 0 0\.2094395">')
_INSTRUMENT_FILE_RE = re.compile(
    r'(<model name="centrifuge_eppendorf_5430" file=")\.\./instrument/centrifuge_eppendorf_5430\.xml(")'
)

MAX_SAFE_SLOTS = 30  # see module docstring: do not increase without re-checking collision geometry


def generate_rotor_variant(num_slots: int, *, overwrite: bool = False) -> Path:
    """Generates a `num_slots`-slot rotor variant (instrument + scene XML) and
    returns the path to the new scene file, suitable for
    `InsertCentrifuge5430.for_variant(scene_path, task_name)`."""
    if not (1 <= num_slots <= MAX_SAFE_SLOTS):
        raise ValueError(
            f"num_slots={num_slots} outside the range this generator has validated "
            f"collision geometry for (1..{MAX_SAFE_SLOTS}); see module docstring."
        )

    instrument_out = MODEL_ROOT / "instrument" / f"centrifuge_eppendorf_5430_{num_slots}slot.gen.xml"
    scene_out = MODEL_ROOT / "scene" / f"insert_centrifuge_5430_{num_slots}slot.gen.xml"

    if overwrite or not instrument_out.exists():
        text = BASE_INSTRUMENT.read_text()
        euler_z = 2 * math.pi / num_slots
        new_text, n = _REPLICATE_RE.subn(
            f'<replicate count="{num_slots}" euler="0 0 {euler_z:.7f}">', text
        )
        if n != 1:
            raise RuntimeError(
                f"Expected exactly one rotor <replicate> block in {BASE_INSTRUMENT}, found {n}. "
                "The base asset may have changed; update _REPLICATE_RE."
            )
        instrument_out.write_text(new_text)

    if overwrite or not scene_out.exists():
        text = BASE_SCENE.read_text()
        rel_path = f"../instrument/{instrument_out.name}"
        new_text, n = _INSTRUMENT_FILE_RE.subn(rf'\g<1>{rel_path}\g<2>', text)
        if n != 1:
            raise RuntimeError(
                f"Expected exactly one instrument <model file=...> reference in {BASE_SCENE}, found {n}. "
                "The base scene may have changed; update _INSTRUMENT_FILE_RE."
            )
        scene_out.write_text(new_text)

    return scene_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_slots", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for n in args.num_slots:
        path = generate_rotor_variant(n, overwrite=args.overwrite)
        print(f"{n} slots -> {path}")
