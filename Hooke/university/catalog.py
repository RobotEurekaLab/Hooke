"""Join ranked institutions, research leads and existing scene implementations.

Ranking membership, public-reference evidence and runnable scenes are separate
facts. Reading this module never imports a simulator or creates a renderer.
"""

from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import urlencode

DATA = Path(__file__).parent


def _read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError(f"Invalid university catalogue identifier: {value!r}")
    return value


def institutions():
    """Include ranked universities and explicitly classified legacy institutions."""
    ranking = _read("qs_world_2027.json")
    ranked = ranking["universities"]
    if len(ranked) != ranking["institution_count"]:
        raise ValueError("Ranking institution count does not match the roster")
    result = {}
    for row in ranked + _read("additional_institutions.json"):
        identifier = _identifier(row["id"])
        if identifier in result:
            raise ValueError(f"Duplicate institution: {identifier}")
        kind = row.get("kind", "university")
        rank = row.get("rank")
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError(f"Invalid QS rank: {identifier}")
        if kind not in {"university", "research_institute"}:
            raise ValueError(f"Invalid institution kind: {identifier}")
        result[identifier] = dict(
            row,
            kind=kind,
            in_scope=kind == "university" and rank is not None and rank <= 100,
            ranking_edition=ranking["edition"] if rank is not None else None,
        )
    return result


def affiliations():
    """Stable scene bindings; geometry and task implementations stay in real_labs."""
    known = institutions()
    rows = _read("scene_index.json")
    lab_definitions = {}
    for scene_id, row in rows.items():
        _identifier(scene_id)
        for key in ("institution_id", "subject_id", "lab_id"):
            _identifier(row[key])
        if row["institution_id"] not in known:
            raise ValueError(f"Unknown institution for scene: {scene_id}")
        lab_key = (row["institution_id"], row["lab_id"])
        identity = (row["subject_id"], row["lab_name"])
        if lab_key in lab_definitions and lab_definitions[lab_key] != identity:
            raise ValueError(f"Conflicting laboratory identity: {lab_key}")
        lab_definitions[lab_key] = identity
        row["institution"] = known[row["institution_id"]]
    return rows


def research_candidates():
    """Return leads without promoting them to runnable scenes."""
    rows = _read("research_candidates.json")["candidates"]
    known = institutions()
    ids = set()
    for row in rows:
        identifier = _identifier(row["id"])
        if identifier in ids or row["institution_id"] not in known:
            raise ValueError(
                f"Duplicate candidate or unknown institution: {identifier}"
            )
        ids.add(identifier)
    return rows


def catalogue():
    """English picker metadata and honest research/model coverage counts."""
    from real_labs.catalog import scenes

    definitions = scenes()
    bindings = affiliations()
    if set(definitions) != set(bindings):
        missing = sorted(set(definitions) - set(bindings))
        stale = sorted(set(bindings) - set(definitions))
        raise ValueError(
            f"Scene classification mismatch: missing={missing}, stale={stale}"
        )
    candidates = research_candidates()
    all_labs = []
    for scene_id, row in bindings.items():
        definition = definitions[scene_id]
        all_labs.append(
            dict(
                row,
                scene_id=scene_id,
                task_name="real_lab_" + scene_id,
                title=definition["title"],
                url="/real-labs?" + urlencode({"scene": scene_id}),
                layout_fidelity=definition["layout_fidelity"],
                model_stage="mechanical_prototype",
                source_urls=definition["source_urls"],
            )
        )
    all_labs.sort(key=lambda row: row["lab_name"].casefold())
    counts = Counter(row["institution_id"] for row in all_labs)
    leads = Counter(row["institution_id"] for row in candidates)
    universities = [
        dict(row, scene_count=counts[row["id"]], candidate_count=leads[row["id"]])
        for row in institutions().values()
        if row["kind"] == "university"
    ]
    universities.sort(key=lambda row: row["name"].casefold())
    ranking = _read("qs_world_2027.json")
    ranking.pop("universities")
    ranking["title"] = "QS World University Rankings"
    labs = [row for row in all_labs if row["institution"]["kind"] == "university"]
    return dict(
        ranking=ranking,
        universities=universities,
        labs=labs,
        other_institutions=[row for row in all_labs if row not in labs],
        coverage=dict(
            scope_institutions=sum(row["in_scope"] for row in universities),
            scope_institutions_with_scenes=sum(
                row["in_scope"] and row["scene_count"] > 0 for row in universities
            ),
            scope_institutions_with_candidates=sum(
                row["in_scope"] and row["candidate_count"] > 0 for row in universities
            ),
            university_scenes=len(labs),
            other_institution_scenes=len(all_labs) - len(labs),
            research_candidates=len(candidates),
        ),
    )
