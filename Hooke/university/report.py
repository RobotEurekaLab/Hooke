"""Generate a dated research and reconstruction coverage report from the catalogue."""

import argparse
from collections import Counter
import json
from pathlib import Path

from university.catalog import catalogue, institutions, research_candidates


def render_report():
    directory = catalogue()
    known = institutions()
    candidates = research_candidates()
    counts = directory["coverage"]
    in_scope = [row for row in candidates if known[row["institution_id"]]["in_scope"]]
    layouts = Counter(row["layout_evidence"] for row in candidates)
    reviewed = sum(row["verification_level"] == "image_reviewed" for row in candidates)
    built_candidates = [row for row in candidates if row.get("existing_scene_id")]
    scope_scenes = sum(row["institution"]["in_scope"] for row in directory["labs"])
    ranking = directory["ranking"]
    lines = [
        "# University laboratory research register",
        "",
        f"Reviewed: {ranking['checked_on']}. Generated from `Hooke/university/` metadata.",
        "",
        f"Scope: [{ranking['title']} {ranking['edition']}]({ranking['source_url']}), "
        f"published overall rank ≤100, including ties and corrections: **{counts['scope_institutions']} institutions**. "
        f"[QS correction notes]({ranking['correction_url']}).",
        "",
        "## What has actually been completed",
        "",
        f"- {len(candidates)} retained candidates, {len(in_scope)} within QS scope, across "
        f"{counts['scope_institutions_with_candidates']} in-scope universities.",
        f"- {reviewed} candidates have inspected images; {layouts['room']} support room layout "
        f"and {layouts['workstation']} support workstation geometry. A product-only image is not room evidence.",
        f"- {len(candidates) - reviewed} candidates have official-page/media-entry review only; linked tours were not fully viewed.",
        f"- {counts['university_scenes']} existing university scene prototypes, covering "
        f"{counts['scope_institutions_with_scenes']} in-scope universities; "
        f"{counts['other_institution_scenes']} additional research-institute scene.",
        f"- {scope_scenes} constructed scenes are within the pinned QS scope. "
        f"{len(built_candidates)} retained candidates are linked to constructed prototypes; "
        "the candidate register and the full scene catalogue have different coverage.",
        "- Candidate counts and school names are not completed-model counts. Room evidence does not establish measured dimensions, current layout, scientific validity or institutional endorsement.",
        "- No retained candidate has a verified media/derived-asset release authorization. Public viewing and academic intent do not establish redistribution permission.",
        "",
        "## Retained laboratories",
        "",
        "P1 means prioritize the next evidence/modeling step; it does not certify a finished reconstruction. "
        "`none` means spatial evidence remains unverified, even when an equipment inventory is useful.",
        "",
        "| University | Lab / official source | QS rank | Evidence | Priority | Existing scene | Model stage |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in candidates:
        institution = known[row["institution_id"]]
        lines.append(
            f"| {institution['name']} | [{row['lab_name']}]({row['official_url']}) | "
            f"{institution.get('rank_display', '—')} | {row['layout_evidence']} | "
            f"{row['reconstruction_priority']} | {row.get('existing_scene_id') or 'Not built'} | "
            f"{row.get('model_stage', 'candidate')} |"
        )
    lines += ["", "## Evidence and missing information", ""]
    for row in candidates:
        lines += [
            f"### {row['id']}",
            "",
            row["observed_evidence"],
            "",
            "**Identifiable equipment:** "
            + "; ".join(row["identifiable_equipment"])
            + ".",
            "",
            "**Remaining gap:** " + row["access_limitations"],
            "",
            f"[Official page]({row['official_url']}) · [Visual reference]({row['visual_evidence_url']})",
            "",
        ]
    lines += [
        "## School coverage and assignment queue",
        "",
        "Zero candidates means no retained lab in this research batch; it is not a claim that the university has no suitable laboratories. "
        "Institutions with existing scenes can still need source refreshes. See the collaboration guide for search and reconstruction criteria.",
        "",
        "| Institution ID | University | QS rank | Retained candidates | Existing scenes |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in directory["universities"]:
        if row["in_scope"]:
            lines.append(
                f"| `{row['id']}` | {row['name']} | {row['rank_display']} | {row['candidate_count']} | {row['scene_count']} |"
            )
    lines += [
        "",
        "## Constructed batches and next review gates",
        "",
        "The following batches identify actual scene implementations. A constructed prototype does not establish "
        "surveyed geometry, calibrated scientific behavior, institutional endorsement or media redistribution permission.",
        "",
        "| Batch | Additional scenes | Universities | Scene IDs |",
        "| --- | ---: | ---: | --- |",
    ]
    indexed = {row["scene_id"]: row for row in directory["labs"]}
    for path in sorted(Path(__file__).with_name("batches").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        identifiers = batch["scene_ids"]
        baseline = set(batch["baseline_scene_ids"])
        if any(
            identifier not in indexed or identifier in baseline
            for identifier in identifiers
        ):
            raise ValueError(
                f"Batch is not completely bound to additional scenes: {batch['id']}"
            )
        members = [indexed[identifier] for identifier in identifiers]
        if any(not row["institution"]["in_scope"] for row in members):
            raise ValueError(
                f"Batch contains an institution outside the pinned QS scope: {batch['id']}"
            )
        university_count = len({row["institution_id"] for row in members})
        lines.append(
            f"| `{batch['id']}` | {len(identifiers)} | {university_count} | "
            + ", ".join(f"`{identifier}`" for identifier in identifiers)
            + " |"
        )
    lines += [
        "",
        "1. **Spatial fidelity:** compare source views with the constructed layouts and obtain scale references. "
        "The [NTU SGSR prototype](university_ntu_sgsr_reconstruction.md) now contains partitions and two instruments; "
        "its floor-plan revision, dimensions and exact placements remain unresolved.",
        "2. **Equipment:** review each apparatus silhouette, supports, service access and declared moving mechanisms. "
        "Room evidence and manufacturer dimensions are tracked separately.",
        "3. **Controls and tasks:** review local CPU results and observable reset/failure behavior. "
        "Joint motion does not establish scientific processes, autonomous workflows, or validated Isaac dynamics.",
        "4. **Release review:** inspect generated overview/interior/detail images under ignored `temp/`, "
        "retain reference-image rights limitations, and obtain the user's approval before any GitHub submission.",
        "5. **Next candidates:** select from unbuilt entries with inspected spatial evidence, then obtain missing views "
        "and device information before claiming additional reconstructions.",
        "",
        "See [integration and contribution rules](university_labs.md).",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(), encoding="utf-8")


if __name__ == "__main__":
    main()
