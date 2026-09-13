"""Phase I's "task synthesis from protocol text" pipeline (private/proposal.tex,
Phase I item 2): parse a real lab-protocol step into Hooke/AutoBio's
existing primitive taxonomy (transfer, conditioning, separation,
combination, measurement, preservation), instantiate it against the asset
library, and auto-generate the Task/Expert boilerplate that every prior
batch this project has hand-written or hand-templated per instrument.

Design, and why it's scoped the way it is
------------------------------------------
The proposal's own wording is "use an LLM to parse protocol text... and
auto-generate the Task subclass boilerplate... that is currently
hand-written per task." Read literally, that invites an LLM to emit
arbitrary Python. This pipeline deliberately does not do that: an LLM
emitting free-form MJCF/motion code reproduces every fragile-IK,
self-collision, and grip-slip failure this project spent real engineering
time diagnosing this session (see private/technical-log.md) with none of
the hard-won guardrails. Instead, the LLM's output is constrained to a
small structured schema (`ProtocolTaskSpec`) whose fields map onto:

- the *display-only* archetype (`static_display.py`) for anything that
  isn't confirmed to fit the one shape below -- real MJCF physics, a real
  operable joint, no scripted grasp attempted. This is the safe default,
  used for the large majority of real protocol steps (liquid transfers,
  measurements, anything involving a round or complex part) exactly as
  it has been for every display-only item in the catalog so far.
- the one *interactive* recipe validated this session
  (`push_pull_box.py`: grip a box-shaped part from directly above, drag it
  through a vertical slide joint's range) -- used only when the LLM's own
  parse explicitly claims the moving part is box-shaped and the joint is
  a vertical slide, i.e. only for protocol steps that plausibly reduce to
  "push/pull a control down/up."

This is the same "constrain the LLM to a small typed action space, keep a
deterministic executor underneath" shape as GenSim/RoboGen (cited in
private/proposal.tex) rather than an LLM writing simulator code freely.

Automatic validation (yield metric)
------------------------------------
Every generated task is run for `n_seeds` (default 20) and scored on:
  - stability: no non-finite qpos/qvel after `execute()`
  - solvability: `check()` returns True (trivially true for the
    display-only fallback, which has no failure condition; meaningfully
    tested for the interactive path)
`yield = passed / total` across a batch, reported the same way
private/proposal.tex's own Phase I evaluation protocol asks for.

No stored API key
------------------
Parsing calls out to an LLM using the caller's own API key, read once per
call and never persisted -- the same security model as
`webui/custom_gen.py`'s `generate_custom_asset`. When no key is available
(e.g. running this module standalone in this sandbox, which has none),
`parse_step_with_llm` cannot run; `demo_protocols.py`
(scratchpad-only, not committed) is where this session's own end-to-end
demonstration run substitutes hand-authored specs for the LLM call to
validate everything *downstream* of parsing -- see
private/technical-log.md for that run's results and an explicit note that
the parsing step itself was done by a human/LLM-in-the-loop, not a live
API call, for exactly that reason.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from task import MODEL_ROOT
from archetypes.geom_helpers import geom, moving_body, chamber_door, write_instrument_and_scene
from archetypes.push_pull_box import PushPullBoxSpec, make_task_classes as make_push_pull_classes
from archetypes.static_display import StaticDisplaySpec, make_static_task

PRIMITIVES = ("transfer", "conditioning", "separation", "combination", "measurement", "preservation")

MOVING_PART_SHAPES = ("box_vertical_slide", "door_hinge", "lid_hinge", "none")


class SpecValidationError(Exception):
    """Raised when a parsed (LLM or hand-authored) spec fails schema/taxonomy
    validation -- these count as yield failures, not crashes."""


@dataclasses.dataclass(frozen=True)
class ProtocolTaskSpec:
    name: str
    primitive: str
    domain: str
    description: str
    moving_part_shape: str = "none"
    # device_size / part_size are (hx, hy, hz) half-extents in meters.
    device_size: tuple[float, float, float] = (0.08, 0.08, 0.05)
    device_color: str = "0.75 0.78 0.8 1"
    part_size: tuple[float, float, float] = (0.02, 0.02, 0.02)
    part_pos: tuple[float, float, float] = (0.0, 0.0, 0.15)
    part_color: str = "0.2 0.2 0.2 1"
    joint_range: tuple[float, float] = (-0.05, 0.0)
    start_qpos: float = 0.0
    target_qpos: float = -0.05

    def __post_init__(self):
        if self.primitive not in PRIMITIVES:
            raise SpecValidationError(f"primitive {self.primitive!r} not in {PRIMITIVES}")
        if self.moving_part_shape not in MOVING_PART_SHAPES:
            raise SpecValidationError(f"moving_part_shape {self.moving_part_shape!r} not in {MOVING_PART_SHAPES}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise SpecValidationError(f"name {self.name!r} must be snake_case")
        lo, hi = self.joint_range
        if not (lo <= self.start_qpos <= hi and lo <= self.target_qpos <= hi):
            raise SpecValidationError("start_qpos/target_qpos must lie within joint_range")
        if self.start_qpos == self.target_qpos:
            raise SpecValidationError("start_qpos and target_qpos must differ")

    @property
    def interactive(self) -> bool:
        """Only the one recipe validated this session (see module
        docstring) is ever attempted as a real scripted-expert task --
        everything else is display-only, regardless of primitive."""
        return self.moving_part_shape == "box_vertical_slide"


# ---------------------------------------------------------------------------
# 1. LLM parsing (real API call, never stored key -- see module docstring)

PARSE_SYSTEM_PROMPT = f"""You convert one step of a real laboratory protocol into a strict JSON object \
describing a simulated benchtop task for a robot arm, following this exact schema:

{{
  "name": "snake_case_identifier",
  "primitive": one of {list(PRIMITIVES)},
  "domain": "short domain label, e.g. molecular_biology",
  "description": "one sentence describing the task, present tense imperative",
  "moving_part_shape": one of {list(MOVING_PART_SHAPES)},
  "device_size": [hx, hy, hz] half-extents in meters, benchtop scale (0.03-0.15 each),
  "part_size": [hx, hy, hz] half-extents in meters (only meaningful if moving_part_shape is not "none"),
  "part_pos": [x, y, z] position of the moving part relative to the device base, meters,
  "joint_range": [lo, hi] in meters (slide) or radians (hinge),
  "start_qpos": number within joint_range,
  "target_qpos": number within joint_range, different from start_qpos
}}

Rules:
- Use "box_vertical_slide" for moving_part_shape ONLY if the step describes pushing, pressing, \
sliding, or pulling a roughly box-shaped control (a drawer, tray, plunger, stage, button housing) \
straight up or down -- this is the only shape this pipeline can turn into a real robot-executed \
task. Use "door_hinge" or "lid_hinge" for a swinging door/lid (these become a real physical joint \
but the robot does not operate them yet). Use "none" for anything else (liquid transfers, using a \
handheld tool, reading an instrument, a round/cylindrical control, or anything not physically \
localized to one piece of equipment on the bench).
- Respond with ONLY the JSON object, no prose, no markdown fence.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text)


def parse_step_with_llm(step_text: str, api_key: str, model: str = "gpt-5") -> ProtocolTaskSpec:
    """Calls an LLM (OpenAI, same dependency `webui/custom_gen.py` already
    uses) to parse one protocol-step sentence into a `ProtocolTaskSpec`.
    `api_key` is used only for this call and never written to disk,
    logged, or held past it -- same security model as
    `webui/custom_gen.py.generate_custom_asset`."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": step_text},
        ],
    )
    data = _extract_json(response.choices[0].message.content)
    return spec_from_dict(data)


def spec_from_dict(data: dict[str, Any]) -> ProtocolTaskSpec:
    """Validates and converts a raw (LLM- or hand-authored) dict into a
    `ProtocolTaskSpec`, raising `SpecValidationError` on anything malformed
    -- a malformed parse is a yield failure, not a crash."""
    try:
        return ProtocolTaskSpec(
            name=data["name"],
            primitive=data["primitive"],
            domain=data.get("domain", "general_lab"),
            description=data["description"],
            moving_part_shape=data.get("moving_part_shape", "none"),
            device_size=tuple(data.get("device_size", (0.08, 0.08, 0.05))),
            part_size=tuple(data.get("part_size", (0.02, 0.02, 0.02))),
            part_pos=tuple(data.get("part_pos", (0.0, 0.0, 0.15))),
            joint_range=tuple(data.get("joint_range", (-0.05, 0.0))),
            start_qpos=data.get("start_qpos", 0.0),
            target_qpos=data.get("target_qpos", data.get("joint_range", (-0.05, 0.0))[0]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise SpecValidationError(f"malformed spec: {e}") from e


# ---------------------------------------------------------------------------
# 2. Deterministic code generation (no LLM involved past this point)

def _device_geoms(spec: ProtocolTaskSpec) -> list[str]:
    hx, hy, hz = spec.device_size
    return [geom("box", f"{hx} {hy} {hz}", f"0 0 {hz}", spec.device_color)]


def _moving_part(spec: ProtocolTaskSpec) -> str:
    hx, hy, hz = spec.part_size
    px, py, pz = spec.part_pos
    lo, hi = spec.joint_range
    if spec.moving_part_shape == "box_vertical_slide":
        return moving_body(
            "part", "slide", "0 0 1", f"{lo} {hi}", f"{px} {py} {pz}",
            [geom("box", f"{hx} {hy} {hz}", "0 0 0", spec.part_color, name="part_body")],
        )
    if spec.moving_part_shape == "door_hinge":
        return chamber_door(spec.device_size[0], spec.device_size[2])
    if spec.moving_part_shape == "lid_hinge":
        return moving_body(
            "lid", "hinge", "1 0 0", "0 1.6", f"0 {spec.device_size[1]} {2*spec.device_size[2]}",
            [geom("box", f"{hx} 0.005 {hz}", "0 0 0", "0.6 0.8 0.9 0.4")],
            damping="0.3",
        )
    return ""  # "none": no moving part at all, purely a static prop


def build_instrument_files(spec: ProtocolTaskSpec) -> tuple[str, str]:
    """Writes model/instrument/{spec.name}.xml and
    model/scene/mani_{spec.name}.xml, returns their paths."""
    static_geoms = _device_geoms(spec)
    moving = _moving_part(spec)
    if spec.moving_part_shape == "box_vertical_slide":
        # The grasp site the interactive archetype needs, on the part body.
        # `moving_body()` always ends with a literal "\n      </body>" --
        # a plain substring replace (not `.rstrip`, which strips characters,
        # not substrings, and would corrupt this) inserts the site just
        # before that closing tag.
        assert moving.endswith("\n      </body>")
        site_line = '\n        <site name="grasp_site" pos="0 0 0" size="0.005" rgba="1 0 0 1" group="4"/>'
        moving = moving[: -len("\n      </body>")] + site_line + "\n      </body>"
    return write_instrument_and_scene(spec.name, static_geoms, moving, MODEL_ROOT)


def build_task(spec: ProtocolTaskSpec) -> tuple[type, type]:
    """Generates the instrument/scene XML and returns (Task, Expert)
    classes -- the interactive `push_pull_box` archetype if the spec
    claims a box-on-a-vertical-slide part, `static_display` otherwise."""
    build_instrument_files(spec)
    if spec.interactive:
        pp_spec = PushPullBoxSpec(
            name=spec.name,
            scene_file=f"mani_{spec.name}.xml",
            instrument_prefix=f"/{spec.name}:",
            joint_name="part_joint",
            start_qpos=spec.start_qpos,
            target_qpos=spec.target_qpos,
            prompt_prefix=spec.description,
        )
        return make_push_pull_classes(pp_spec)
    display_spec = StaticDisplaySpec(
        name=f"{spec.name}_display",
        scene_file=f"mani_{spec.name}.xml",
        prompt=spec.description,
    )
    return make_static_task(display_spec)


# ---------------------------------------------------------------------------
# 3. Automatic validation / yield

@dataclasses.dataclass
class ValidationReport:
    name: str
    interactive: bool
    n_seeds: int
    stable_count: int
    solved_count: int

    @property
    def stable_frac(self) -> float:
        return self.stable_count / self.n_seeds

    @property
    def solved_frac(self) -> float:
        return self.solved_count / self.n_seeds

    @property
    def passed(self) -> bool:
        # Filtering criterion from private/proposal.tex Phase I item 4:
        # simulation stability + solvability by the expert layer.
        return self.stable_frac == 1.0 and self.solved_frac == 1.0


def validate_task(task_cls: type, n_seeds: int = 20) -> ValidationReport:
    stable = 0
    solved = 0
    for seed in range(n_seeds):
        spec_obj = task_cls.load()
        expert = task_cls.Expert(spec_obj)
        expert.reset(seed)
        try:
            expert.execute()
        except AssertionError:
            continue  # IK/reachability failure -- neither stable nor solved
        finite = np.all(np.isfinite(expert.data.qpos)) and np.all(np.isfinite(expert.data.qvel))
        if finite:
            stable += 1
            if expert.check():
                solved += 1
    return ValidationReport(
        name=task_cls.default_task,
        interactive=task_cls.__name__ != "StaticDisplayTask",
        n_seeds=n_seeds, stable_count=stable, solved_count=solved,
    )


# ---------------------------------------------------------------------------
# 4. Batch driver

@dataclasses.dataclass
class PipelineReport:
    total: int
    parsed: int
    passed: int
    entries: list[dict]

    @property
    def parse_yield(self) -> float:
        return self.parsed / self.total if self.total else 0.0

    @property
    def overall_yield(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_pipeline(
    protocol_steps: list[str],
    api_key: str | None = None,
    pre_parsed_specs: dict[str, dict] | None = None,
    n_seeds: int = 20,
) -> PipelineReport:
    """Runs the full parse -> generate -> validate pipeline over a batch of
    protocol-step sentences. If `api_key` is given, parses via
    `parse_step_with_llm`; otherwise `pre_parsed_specs` must map each step
    string to an already-parsed spec dict (used for this session's own
    demonstration run, where no API key was available -- see module
    docstring)."""
    entries = []
    parsed_count = 0
    passed_count = 0
    for step in protocol_steps:
        entry = {"step": step, "spec": None, "error": None, "report": None, "_spec_obj": None}
        try:
            if api_key:
                spec = parse_step_with_llm(step, api_key)
            else:
                raw = (pre_parsed_specs or {}).get(step)
                if raw is None:
                    raise SpecValidationError("no api_key and no pre-parsed spec provided for this step")
                spec = spec_from_dict(raw)
            parsed_count += 1
            entry["spec"] = dataclasses.asdict(spec)
            entry["_spec_obj"] = spec
            task_cls, expert_cls = build_task(spec)
            report = validate_task(task_cls, n_seeds=n_seeds)
            entry["report"] = dataclasses.asdict(report)
            if report.passed:
                passed_count += 1
        except SpecValidationError as e:
            entry["error"] = str(e)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        entries.append(entry)
    return PipelineReport(total=len(protocol_steps), parsed=parsed_count, passed=passed_count, entries=entries)


# ---------------------------------------------------------------------------
# 5. Promoting a passing spec into the real catalog

GENERATED_MODULE_PATH = Path(__file__).parent / "protocol_generated.py"
TASK_CATALOG_PATH = Path(__file__).parent / "task_catalog.py"
GENERATED_BATCH_MARKER = "        # --- protocol-to-task generated (see private/technical-log.md) ---"


def _class_name(spec: ProtocolTaskSpec) -> str:
    return "".join(part.capitalize() for part in spec.name.split("_")) + "Task"


def _ensure_generated_module_header():
    if not GENERATED_MODULE_PATH.exists():
        GENERATED_MODULE_PATH.write_text(
            '"""Task/Expert bindings auto-generated by '
            'archetypes/protocol_to_task.py\'s pipeline from parsed protocol '
            'steps that passed automatic validation (stability + '
            'solvability over N seeds) -- see private/technical-log.md for '
            'the run that produced these. Each binding just re-invokes '
            '`build_task` on the spec that already passed; regenerating the '
            'XML here is deterministic and idempotent, not re-validation."""\n'
            "from archetypes.protocol_to_task import ProtocolTaskSpec, build_task\n\n"
        )


def promote_passing_to_catalog(report: PipelineReport, category_default: str = "general_lab") -> list[str]:
    """For every entry in `report` that passed validation, appends a
    binding to `protocol_generated.py` and a `CatalogEntry` to
    `task_catalog.py`. Returns the list of catalog names added. Entries
    that failed validation are left out entirely -- this *is* the
    filtering step private/proposal.tex's Phase I asks for, not a
    formality."""
    _ensure_generated_module_header()
    added = []
    binding_lines = []
    catalog_lines = [GENERATED_BATCH_MARKER]
    for entry in report.entries:
        if entry["error"] or not entry["report"] or not entry["report"]["stable_count"] == entry["report"]["n_seeds"] or not entry["report"]["solved_count"] == entry["report"]["n_seeds"]:
            continue
        spec: ProtocolTaskSpec = entry["_spec_obj"]
        cls_name = _class_name(spec)
        var_name = spec.name.upper() + "_SPEC"
        binding_lines.append(f"{var_name} = ProtocolTaskSpec(**{dataclasses.asdict(spec)!r})")
        binding_lines.append(f"{cls_name}, {cls_name}Expert = build_task({var_name})")
        binding_lines.append("")
        catalog_name = spec.name if spec.interactive else f"{spec.name}_display"
        catalog_lines.append("        CatalogEntry(")
        catalog_lines.append(f'            name="{catalog_name}",')
        catalog_lines.append(f'            description={spec.description!r},')
        catalog_lines.append(f'            category="{spec.domain or category_default}",')
        catalog_lines.append(
            f'            module="archetypes.protocol_generated", cls="{cls_name}", '
            'robot="ur5e", camera="table_cam_front",'
        )
        catalog_lines.append("        ),")
        added.append(catalog_name)

    if not added:
        return []

    with GENERATED_MODULE_PATH.open("a") as f:
        f.write("\n".join(binding_lines) + "\n")

    catalog_src = TASK_CATALOG_PATH.read_text()
    close_marker = "    ]\n}"
    assert close_marker in catalog_src, "task_catalog.py's CATALOG list-closing marker not found as expected"
    insertion = "\n".join(catalog_lines) + "\n"
    catalog_src = catalog_src.replace(close_marker, insertion + close_marker, 1)
    TASK_CATALOG_PATH.write_text(catalog_src)
    return added
