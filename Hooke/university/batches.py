"""Validate expansion scope independently of research leads and render counts.

An expansion counts distinct laboratories, not camera angles, room variants,
or schools. A partial batch remains useful while its completion gate stays open.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from real_labs.catalog import scenes
from real_labs.export import current_scene_digest
from university.catalog import affiliations, research_candidates
from university.review import collect_records


def load_batch(path):
    batch = json.loads(Path(path).read_text())
    ids = batch["scene_ids"]
    baseline = batch["baseline_scene_ids"]
    if len(ids) != len(set(ids)) or len(baseline) != len(set(baseline)):
        raise ValueError("Duplicate scene in expansion or baseline")
    if set(ids) & set(baseline):
        raise ValueError("A baseline scene cannot count as an additional laboratory")
    if type(batch["required_new_labs"]) is not int or batch["required_new_labs"] < 1:
        raise ValueError("Expansion target must be a positive laboratory count")
    definitions, bindings = scenes(), affiliations()
    if set(ids + baseline) != set(definitions):
        raise ValueError("Expansion manifest must account for every registered scene")
    identities = {tuple(row) for row in batch["baseline_lab_identities"]}
    for identifier in ids:
        row = bindings[identifier]
        institution = row["institution"]
        if (
            not institution["in_scope"]
            or institution["ranking_edition"] != batch["ranking_edition"]
        ):
            raise ValueError(f"Laboratory is outside the ranked scope: {identifier}")
        identity = row["institution_id"], row["lab_id"]
        if identity in identities:
            raise ValueError(f"Laboratory counted more than once: {identity}")
        identities.add(identity)
    return batch


def progress(batch_path, package_root, visual_reviews=None):
    """Require current mechanical exports, rendered images and explicit visual QC."""
    batch = load_batch(batch_path)
    package_root = Path(package_root)
    bindings = affiliations()
    candidates = {}
    for row in research_candidates():
        if row.get("existing_scene_id"):
            candidates.setdefault(row["existing_scene_id"], []).append(row)
    reviews = {}
    if visual_reviews is not None and Path(visual_reviews).is_file():
        reviews = json.loads(Path(visual_reviews).read_text())["scenes"]
    records = []
    for identifier in batch["scene_ids"]:
        row = dict(
            id=identifier,
            evidence=False,
            mechanical=False,
            rendered=False,
            visually_reviewed=False,
            ready=False,
            pending=[],
        )
        evidence = candidates.get(identifier, [])
        row["evidence"] = bool(evidence) and all(
            lead.get("verification_level") == "image_reviewed"
            and lead.get("physical_site_verified")
            and lead.get("reviewed_visuals")
            and lead["institution_id"] == bindings[identifier]["institution_id"]
            for lead in evidence
        )
        directory = package_root / identifier
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            validation = json.loads((directory / "validation.json").read_text())
            digest = hashlib.sha256((directory / "scene.xml").read_bytes()).hexdigest()
            row["mechanical"] = bool(
                validation["passed"]
                and validation["control_count"] >= 2
                and digest
                == manifest["scene_sha256"]
                == validation.get("scene_sha256")
                == current_scene_digest(identifier)
            )
            render = collect_records(package_root, [identifier])[0]
            row["rendered"] = True
            review = reviews.get(identifier, {})
            reviewed_images = review.get("image_sha256", {})
            row["visually_reviewed"] = bool(
                review.get("passed") is True
                and review.get("reviewer")
                and review.get("observations")
                and review.get("source_urls")
                and review.get("scene_sha256") == digest
                and all(
                    reviewed_images.get(view) == render["images"][view]["sha256"]
                    for view in ("overview", "interior", "workstation")
                )
            )
        except (OSError, ValueError, KeyError) as error:
            row["pending"].append(str(error))
        for stage in ("evidence", "mechanical", "rendered", "visually_reviewed"):
            if not row[stage]:
                row["pending"].append(stage)
        row["ready"] = not row["pending"]
        records.append(row)
    counts = {
        stage: sum(row[stage] for row in records)
        for stage in (
            "evidence",
            "mechanical",
            "rendered",
            "visually_reviewed",
            "ready",
        )
    }
    return dict(
        batch=batch["id"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        required_new_labs=batch["required_new_labs"],
        baseline_scenes=len(batch["baseline_scene_ids"]),
        implemented_new_labs=len(records),
        remaining_to_implement=max(0, batch["required_new_labs"] - len(records)),
        counts=counts,
        remaining_new_labs=max(0, batch["required_new_labs"] - counts["ready"]),
        ready_for_user_review=counts["ready"] >= batch["required_new_labs"],
        publication="Mechanical prototypes; specialist and Isaac runtime validation pending",
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--visual-reviews", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = progress(args.batch, args.input, args.visual_reviews)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
