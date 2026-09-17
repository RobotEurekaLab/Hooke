"""Export actual episode media and public curves to an ignored local directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image
import numpy as np

from backends.run import write_json
from experiments.records import ScienceRecords
from experiments import OPERATIONS
from worlds.profiles import WORLDS

ROOT = Path(__file__).resolve().parents[2]


def verify_video(path, expected_frames):
    """Decode every frame and verify the container preserves the sample count."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,width,height,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    if int(probe["nb_read_frames"]) != expected_frames:
        raise ValueError("Video does not preserve every sampled episode image")
    return probe


def paired_inputs(rows):
    """Validate shared initial models without publishing any private parameters."""
    comparisons = []
    for (task, backend, seed), row in rows.items():
        counterpart = rows.get((task, "mujoco", seed))
        if backend != "isaac" or counterpart is None:
            continue
        a = Path(row["folder"]) / "source"
        b = Path(counterpart["folder"]) / "source"
        digest_a = hashlib.sha256((a / "scene.xml").read_bytes()).hexdigest()
        digest_b = hashlib.sha256((b / "scene.xml").read_bytes()).hexdigest()
        with np.load(a / "model.npz") as native, np.load(b / "model.npz") as source:
            different = sorted(set(native.files) ^ set(source.files))
            different.extend(
                key
                for key in set(native.files) & set(source.files)
                if not np.array_equal(native[key], source[key], equal_nan=True)
            )
            fields = len(native.files)
        passed = digest_a == digest_b and not different
        comparisons.append(
            dict(
                task=task,
                seed=seed,
                passed=passed,
                scene_xml_sha256=digest_a,
                compiled_fields_compared=fields,
                different_fields=sorted(different),
                comparison="exact_array_values_with_matching_internal_nan_placeholders",
            )
        )
        if not passed:
            raise ValueError(f"Paired initial inputs differ: {task}, seed {seed}")
    return sorted(comparisons, key=lambda row: (row["task"], row["seed"]))


def export_case(row, destination):
    folder = Path(row["folder"])
    result = json.loads((folder / "result.json").read_text())
    if not row["passed"] or result["status"] != "TASK_SUCCEEDED":
        raise ValueError("A failed episode cannot supply qualified gallery media")
    camera = next(
        c["id"] for c in result["cameras"] if c["name"] == "experiment_closeup"
    )
    images = folder / row["backend"] / f"camera_{camera}"
    frames = sorted(images.glob("*.png"))
    if len(frames) < 3:
        raise ValueError("Continuous episode images are missing")
    if [p.name for p in frames] != [f"{i:05d}.png" for i in range(len(frames))]:
        raise ValueError("Episode image sequence has gaps")
    if len(frames) != result["runtime"]["frames"]:
        raise ValueError("Saved images do not match the episode render count")
    fps = result["render_settings"]["frames_per_second"]
    phase = next(
        p
        for p in result["expert_phases"]
        if p["phase"] == "locked_and_robot_clear" and p["sample_id"] == "candidate"
    )
    index = min(round(phase["time_s"] * fps), len(frames) - 1)
    base = f"space-experiment-{row['world']}-{row['operation']}-{row['backend']}"
    image = destination / (base + ".png")
    temporary_image = destination / ("." + base + ".tmp.png")
    shutil.copyfile(frames[index], temporary_image)
    with Image.open(temporary_image) as source_image:
        pixels = np.asarray(source_image.convert("RGB"))
    if pixels.std() < 3 or np.mean(pixels.max(axis=2) > 25) < 0.05:
        raise ValueError("Gallery image is blank or black")
    video = destination / (base + ".mp4")
    temporary_video = destination / ("." + base + ".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(images / "%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_video),
        ],
        check=True,
    )
    probe = verify_video(temporary_video, len(frames))
    webm = destination / (base + ".webm")
    temporary_webm = destination / ("." + base + ".tmp.webm")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(images / "%05d.png"),
            "-c:v",
            "libvpx",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-threads",
            "2",
            "-crf",
            "10",
            "-b:v",
            "1M",
            "-pix_fmt",
            "yuv420p",
            str(temporary_webm),
        ],
        check=True,
    )
    webm_probe = verify_video(temporary_webm, len(frames))
    temporary_image.replace(image)
    temporary_video.replace(video)
    temporary_webm.replace(webm)
    store = ScienceRecords(folder)
    public = {
        identifier: store.read_measurement(identifier)
        for identifier in store.list_measurements()
    }
    evidence = dict(
        task=row["task"],
        backend=row["backend"],
        seed=row["seed"],
        simulation_s=result["simulation_s"],
        conclusion=result["space_experiment"]["conclusion"],
        measurements=public,
        handling=result["space_experiment"]["handling"],
        frame_sampling_fps=fps,
        frame_times_s=[i / fps for i in range(len(frames))],
        source_result_sha256=row["result_sha256"],
    )
    evidence_dir = destination / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    write_json(evidence_dir / (base + ".json"), evidence)
    return dict(
        world=row["world"],
        operation=row["operation"],
        backend=row["backend"],
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        webm_sha256=hashlib.sha256(webm.read_bytes()).hexdigest(),
        video_frames=len(frames),
        frame_sampling_fps=fps,
        simulated_clip_seconds=(len(frames) - 1) / fps,
        decoding="PASS",
        decoded_frames=int(probe["nb_read_frames"]),
        webm_decoding="PASS",
        webm_frames=int(webm_probe["nb_read_frames"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "temp/space_experiments/publication/summary.json",
        help="Summary path; defaults to ignored temp/space_experiments/publication/summary.json",
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=ROOT / "temp/space_experiments/publication/media",
        help="Media and evidence directory; defaults to ignored temp/space_experiments/publication/media",
    )
    args = parser.parse_args()
    rows = {}
    for path in args.summaries:
        report = json.loads(path.read_text())
        if report.get("complete") is not True:
            raise ValueError(f"Qualification is still running: {path}")
        for row in report["cases"]:
            rows[(row["task"], row["backend"], row["seed"])] = row
    cases = sorted(rows.values(), key=lambda r: (r["task"], r["backend"], r["seed"]))
    expected = {
        (f"space_{world}_{operation}", backend, 0)
        for world in WORLDS
        for operation in OPERATIONS
        for backend in ("mujoco", "isaac")
    }
    if not expected.issubset(rows):
        raise ValueError("Both backends must finish all nine rendered tasks")
    destination = args.media_dir
    destination.mkdir(parents=True, exist_ok=True)
    media = [export_case(row, destination) for row in cases if row["seed"] == 0]
    spectral_outcomes = []
    for row in cases:
        if row["operation"] == "spectral_measurement":
            result = json.loads((Path(row["folder"]) / "result.json").read_text())
            conclusion = result["space_experiment"].get("conclusion")
            if conclusion:
                spectral_outcomes.append(conclusion)
    summary = dict(
        schema_version=1,
        total=len(cases),
        passed=sum(r["passed"] for r in cases),
        cases=cases,
        paired_initial_inputs=paired_inputs(rows),
        media=media,
        complete=all(r["passed"] for r in cases),
        no_action_controls=dict(
            total=sum("no_action" in r for r in cases),
            false_successes=sum(
                r.get("no_action", {}).get("false_success", False) for r in cases
            ),
        ),
        seeds_per_backend={
            backend: sorted({r["seed"] for r in cases if r["backend"] == backend})
            for backend in ("mujoco", "isaac")
        },
        scientific_scope="spring_encoder_mass_and_analytic_test_spectra",
        metrics=dict(
            max_absolute_mass_error_kg=max(
                (r["metrics"].get("absolute_mass_error_kg", 0) for r in cases),
                default=0,
            ),
            max_relative_mass_error=max(
                (r["metrics"].get("relative_mass_error", 0) for r in cases), default=0
            ),
            max_simulation_s=max((r["simulation_s"] for r in cases), default=0),
            spectral_known_matches=sum(not c["abstained"] for c in spectral_outcomes),
            spectral_holdout_abstentions=sum(c["abstained"] for c in spectral_outcomes),
            max_spectral_repeat_rms=max(
                (c["repeat_rms"] for c in spectral_outcomes), default=0
            ),
        ),
        limitations=[
            "no_usgs_spectral_data_imported",
            "default_robot_scenes_use_procedural_terrain",
            "no_gas_seal_or_pressure_solver",
            "no_station_recoil",
            "no_llm_scientist_benchmark",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print(
        json.dumps(
            dict(total=summary["total"], passed=summary["passed"], media=len(media))
        )
    )
    raise SystemExit(0 if summary["complete"] else 1)


if __name__ == "__main__":
    main()
