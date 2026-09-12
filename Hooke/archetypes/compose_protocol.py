"""Phase I Step 4: parse a free-text lab protocol into a sequence of atomic
Hooke tasks (archetypes/task_catalog.py), using an LLM to do the
decomposition.

Deliberately narrow scope: the LLM is asked to *choose and order* tasks from
a fixed, pre-verified catalog, not to invent new low-level manipulation
geometry (grasp offsets, waypoints, etc.) -- that would require real
asset/CAD grounding we don't have a safe way to hand an LLM yet (see
private/TODO.md's generative-3D entry for the analogous, deliberately
deferred, harder problem). Output is a plain ordered list of catalog task
names, which is exactly the raw material Phase II's compositional benchmark
needs (see private/proposal.tex, Phase II).

Usage (OpenAI, default):
    export OPENAI_API_KEY=...
    python -m archetypes.compose_protocol "Take a tube from the rack, spin it \
        down in the 5430 centrifuge, then place a second tube in the slot \
        opposite it for balance."

Usage (Anthropic, pass --provider anthropic):
    export ANTHROPIC_API_KEY=...
    python -m archetypes.compose_protocol --provider anthropic "..."

Every step of parsing/validation below is unit-testable (and tested, see
archetypes/test_compose_protocol.py) without ever calling the actual API --
only the network call itself needs a real ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import re

from archetypes.task_catalog import CATALOG, catalog_prompt_listing

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {"openai": "gpt-6-astra", "anthropic": "claude-sonnet-5"}

SYSTEM_PROMPT = """You turn a free-text biology-lab protocol description into an ordered \
sequence of robot tasks, chosen ONLY from a fixed catalog of tasks a robot can actually \
perform. You must not invent tasks that are not in the catalog, and you must not merge or \
split catalog tasks -- pick the closest-matching ones and put them in the order implied by \
the protocol text.

Available tasks:
{catalog}

Respond with ONLY a JSON array, no prose, no markdown code fences. Each element must be an \
object with exactly one key, "task", whose value is one of the exact task names listed above. \
If a step in the protocol has no matching task in the catalog, omit that step rather than \
inventing a task name -- do not use a name that isn't in the list above.

Example response format:
[{{"task": "pickup_centrifuge_tube"}}, {{"task": "centrifuge_5430_close_lid"}}]"""


def build_prompt() -> str:
    return SYSTEM_PROMPT.format(catalog=catalog_prompt_listing())


class CompositionError(ValueError):
    pass


def parse_and_validate(response_text: str) -> list[dict]:
    """Extracts a JSON array from `response_text` and validates every "task"
    value against archetypes.task_catalog.CATALOG. Raises CompositionError
    (never silently drops/guesses) if the response isn't well-formed JSON or
    names a task outside the catalog -- a benchmark built on hallucinated
    task names is worse than one that fails loudly here."""
    text = response_text.strip()
    # Be tolerant of an accidental ```json ... ``` fence even though the
    # prompt asks the model not to use one.
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise CompositionError(f"Response was not valid JSON: {e}\n---\n{response_text}") from e

    if not isinstance(parsed, list):
        raise CompositionError(f"Expected a JSON array, got {type(parsed).__name__}: {parsed!r}")

    validated = []
    for i, step in enumerate(parsed):
        if not isinstance(step, dict) or "task" not in step:
            raise CompositionError(f"Step {i} is not an object with a 'task' key: {step!r}")
        task_name = step["task"]
        if task_name not in CATALOG:
            raise CompositionError(
                f"Step {i} names task '{task_name}', which is not in the catalog. "
                f"Known tasks: {sorted(CATALOG.keys())}"
            )
        validated.append({"task": task_name})

    if not validated:
        raise CompositionError("Response validated to an empty task sequence.")

    return validated


def call_llm(protocol_text: str, provider: str = DEFAULT_PROVIDER, model: str | None = None) -> str:
    """Imports the SDK lazily so parse_and_validate can be unit-tested
    without either package installed."""
    model = model or DEFAULT_MODELS[provider]
    if provider == "openai":
        import openai

        client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment
        response = client.responses.create(
            model=model,
            instructions=build_prompt(),
            input=protocol_text,
        )
        return response.output_text
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=build_prompt(),
            messages=[{"role": "user", "content": protocol_text}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
    else:
        raise ValueError(f"Unknown provider: {provider!r} (expected 'openai' or 'anthropic')")


def compose(protocol_text: str, provider: str = DEFAULT_PROVIDER, model: str | None = None) -> list[dict]:
    response_text = call_llm(protocol_text, provider=provider, model=model)
    return parse_and_validate(response_text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol_text")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=None, help=f"Defaults per-provider: {DEFAULT_MODELS}")
    parser.add_argument("--out", help="Optional path to save the validated task sequence as JSON")
    args = parser.parse_args()

    sequence = compose(args.protocol_text, provider=args.provider, model=args.model)
    print(json.dumps(sequence, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(sequence, f, indent=2)
        print(f"\nSaved to {args.out}")
