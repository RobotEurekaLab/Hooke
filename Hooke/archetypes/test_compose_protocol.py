"""Unit tests for archetypes.compose_protocol's parsing/validation logic,
using canned (fake) LLM responses -- no ANTHROPIC_API_KEY or network access
needed. The only untested path is the actual `call_llm` network call.

Invoke directly:

    python -m archetypes.test_compose_protocol
"""
from archetypes.compose_protocol import parse_and_validate, CompositionError

CASES_OK = [
    (
        '[{"task": "pickup_centrifuge_tube"}, {"task": "centrifuge_5430_close_lid"}]',
        ["pickup_centrifuge_tube", "centrifuge_5430_close_lid"],
    ),
    (
        # tolerate an accidental markdown code fence even though the prompt asks against it
        '```json\n[{"task": "thermal_cycler_open"}]\n```',
        ["thermal_cycler_open"],
    ),
    (
        '[{"task": "insert_centrifuge_5430"}]',
        ["insert_centrifuge_5430"],
    ),
]

CASES_REJECTED = [
    ("not json at all", "invalid JSON"),
    ('{"task": "pickup_centrifuge_tube"}', "not a JSON array"),
    ('[{"task": "levitate_the_tube"}]', "unknown task name"),
    ('[{"wrong_key": "pickup_centrifuge_tube"}]', "missing 'task' key"),
    ("[]", "empty sequence"),
]


def run():
    failures = []

    for response, expected_names in CASES_OK:
        try:
            result = parse_and_validate(response)
            names = [step["task"] for step in result]
            status = "OK" if names == expected_names else "MISMATCH"
            if status != "OK":
                failures.append(f"expected {expected_names}, got {names} for input {response!r}")
            print(f"[accept case] {status}: {response!r} -> {names}")
        except CompositionError as e:
            failures.append(f"expected acceptance but raised for {response!r}: {e}")
            print(f"[accept case] UNEXPECTED REJECTION: {response!r} -> {e}")

    for response, why in CASES_REJECTED:
        try:
            result = parse_and_validate(response)
            failures.append(f"expected rejection ({why}) but got {result} for input {response!r}")
            print(f"[reject case] UNEXPECTED ACCEPTANCE: {response!r} -> {result}")
        except CompositionError as e:
            print(f"[reject case] OK (correctly rejected, {why}): {response!r} -> {e}")

    assert not failures, "\n".join(failures)
    print("\nAll compose_protocol parsing/validation checks passed.")


if __name__ == "__main__":
    run()
