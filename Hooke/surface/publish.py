"""Publish verified episode media to ignored local storage."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from backends.run import write_json
from experiments.publish import paired_inputs, verify_video
from surface.profiles import MISSIONS
from surface.artifacts import MEDIA, SUMMARY, TEAM_SUMMARY
from surface.evidence import control_passed, controls_held, expert_passed

CAMERAS = {
    "overview": "expedition_overview",
    "landing": "landing_site",
    "follow": "rover_follow",
    "sample": "sampling_closeup",
}
TEAM_CAMERAS = dict(CAMERAS, follow="team_follow", sample="team_closeup")


def playback(folder, result):
    """Public rover positions come from recorded physics, sampled at RGB times."""
    meta = json.loads((folder / "source/scene.json").read_text())
    with np.load(folder / "source/model.npz", allow_pickle=False) as model:
        rover = meta["names"]["body"].index("rover")
        qa = int(model["jnt_qposadr"][model["body_jntadr"][rover]])
        initial = model["reset_qpos"][qa : qa + 3].copy()
        human_qa = None
        if result["surface_mission"].get("scenario") == "team":
            body = meta["names"]["body"].index("/g1:pelvis")
            human_qa = int(model["jnt_qposadr"][model["body_jntadr"][body]])
            human_initial = model["reset_qpos"][human_qa : human_qa + 3].copy()
    with np.load(folder / "trajectory.npz", allow_pickle=False) as trajectory:
        times = trajectory["time"]
        positions = trajectory["qpos"][:, qa : qa + 3]
        distance = np.cumsum(
            np.linalg.norm(
                np.diff(np.vstack((initial, positions))[:, :2], axis=0), axis=1
            )
        )
        if human_qa is not None:
            human_positions = trajectory["qpos"][:, human_qa : human_qa + 3]
            human_distance = np.cumsum(
                np.linalg.norm(
                    np.diff(np.vstack((human_initial, human_positions))[:, :2], axis=0),
                    axis=1,
                )
            )
        fps = result["render_settings"]["frames_per_second"]
        frames = []
        for i in range(result["runtime"]["frames"]):
            time = i / fps
            index = max(
                0,
                min(
                    int(np.searchsorted(times, time, side="right")) - 1, len(times) - 1
                ),
            )
            phase = next(
                (
                    row["phase"]
                    for row in reversed(result["expert_phases"])
                    if row["time_s"] <= time + 1e-9
                ),
                "ready",
            )
            frames.append(
                dict(
                    time_s=time,
                    phase=phase,
                    rover_position_m=(initial if i == 0 else positions[index]).tolist(),
                    traveled_m=0.0 if i == 0 else float(distance[index]),
                )
            )
            if human_qa is not None:
                frames[-1].update(
                    humanoid_position_m=(
                        human_initial if i == 0 else human_positions[index]
                    ).tolist(),
                    humanoid_traveled_m=0.0 if i == 0 else float(human_distance[index]),
                )
    return frames


def export_case(folder, result, destination):
    if not expert_passed(result):
        raise ValueError(
            "Only a verified collection-and-return episode can supply mission media"
        )
    world = result["surface_mission"]["world"]
    backend = result["backend"]
    fps = result["render_settings"]["frames_per_second"]
    reports = []
    team = result["surface_mission"].get("scenario") == "team"
    for view, name in (TEAM_CAMERAS if team else CAMERAS).items():
        camera = next(c["id"] for c in result["cameras"] if c["name"] == name)
        images = folder / backend / f"camera_{camera}"
        frames = sorted(images.glob("*.png"))
        if len(frames) < 3 or len(frames) != result["runtime"]["frames"]:
            raise ValueError(f"Incomplete episode images: {images}")
        if [p.name for p in frames] != [f"{i:05d}.png" for i in range(len(frames))]:
            raise ValueError("Episode frame sequence contains a gap")
        phase = next(p for p in result["expert_phases"] if p["phase"] == "lift_rock")
        index = (
            0
            if view in ("overview", "landing")
            else min(round((phase["time_s"] + 2) * fps), len(frames) - 1)
        )
        stem = ("team-" if team else "") + f"{world}-{backend}-{view}"
        image = destination / f"{stem}.png"
        temporary_image = destination / f".{stem}.tmp.png"
        shutil.copyfile(frames[index], temporary_image)
        with Image.open(temporary_image) as source:
            pixels = np.asarray(source.convert("RGB"))
        if pixels.std() < 3 or np.mean(pixels.max(axis=2) > 25) < 0.05:
            raise ValueError("Refusing to publish a blank mission image")
        video = destination / f"{stem}.mp4"
        temporary_video = destination / f".{stem}.tmp.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-threads",
                "2",
                "-framerate",
                str(fps),
                "-i",
                str(images / "%05d.png"),
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary_video),
            ],
            check=True,
        )
        probe = verify_video(temporary_video, len(frames))
        webm = destination / f"{stem}.webm"
        temporary_webm = destination / f".{stem}.tmp.webm"
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
                "-threads",
                "2",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-crf",
                "10",
                "-b:v",
                "2M",
                "-pix_fmt",
                "yuv420p",
                str(temporary_webm),
            ],
            check=True,
        )
        verify_video(temporary_webm, len(frames))
        temporary_image.replace(image)
        temporary_video.replace(video)
        temporary_webm.replace(webm)
        reports.append(
            dict(
                world=world,
                backend=backend,
                view=view,
                frames=len(frames),
                sampling_fps=fps,
                width=int(probe["width"]),
                height=int(probe["height"]),
                representative_frame=index,
                image_time_s=index / fps,
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                webm_sha256=hashlib.sha256(webm.read_bytes()).hexdigest(),
                trajectory_sha256=hashlib.sha256(
                    (folder / "trajectory.npz").read_bytes()
                ).hexdigest(),
            )
        )
    return reports


def publish(folders, destination=MEDIA, summary=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cases, controls, media, rows, exports = [], [], [], {}, []
    experts = {}
    results = [
        (Path(folder), json.loads((Path(folder) / "result.json").read_text()))
        for folder in folders
    ]
    scenarios = {
        result.get("surface_mission", {}).get("scenario", "rover")
        for _, result in results
    }
    if len(scenarios) != 1:
        raise ValueError("Publish one mission scenario per gallery summary")
    scenario = scenarios.pop()
    if scenario not in ("rover", "team"):
        raise ValueError("Unknown mission scenario")
    summary = (
        Path(summary)
        if summary is not None
        else TEAM_SUMMARY if scenario == "team" else SUMMARY
    )
    for folder, result in results:
        world = result.get("surface_mission", {}).get("world")
        if world not in MISSIONS:
            raise ValueError("Input is not a registered surface world")
        task = (
            f"space_{world}_humanoid_rover"
            if scenario == "team"
            else MISSIONS[world].task_name
        )
        if result["task"] != task:
            raise ValueError("Input is not a registered surface sampling episode")
        if result["mode"] != "expert":
            continue
        key = result["task"], result["backend"], result["seed"]
        if key in rows:
            raise ValueError("Duplicate qualification case")
        row = dict(
            world=world,
            task=result["task"],
            backend=result["backend"],
            seed=result["seed"],
            status=result["status"],
            passed=expert_passed(result),
            simulation_s=result.get("simulation_s"),
            metrics=result["surface_mission"],
            folder=str(folder.resolve()),
        )
        rows[key] = row
        experts[key] = result
        case = {k: v for k, v in row.items() if k != "folder"}
        if "render_replay" in result:
            case["render_replay"] = result["render_replay"]
        cases.append(case)
        if row["passed"] and result["runtime"].get("frames", 0):
            exports.append((case, folder, result))
    for folder, result in results:
        if result["mode"] != "no_action":
            continue
        key = result["task"], result["backend"], result["seed"]
        expert = rows.get(key)
        duration = result.get("simulation_s", 0)
        matched = bool(
            expert
            and expert["passed"]
            and abs(expert["simulation_s"] - duration) < 0.0021
        )
        held = controls_held(folder)
        controls.append(
            dict(
                world=result["surface_mission"]["world"],
                backend=result["backend"],
                seed=result["seed"],
                status=result["status"],
                simulation_s=duration,
                duration_matched=matched,
                requested_horizon_s=result.get("requested_control_horizon_s", duration),
                terminated_early=result["status"] == "CONTROL_REJECTED_EARLY",
                control_rejection=result.get("control_rejection"),
                rejected_as_success=result.get("source_success") is False,
                reset_controls_held=held,
                passed=bool(expert and control_passed(experts[key], result, folder)),
            )
        )
    paired = paired_inputs(rows)
    for case, folder, result in exports:
        case["playback"] = playback(folder, result)
        media.extend(export_case(folder, result, destination))
    report = dict(
        cases=cases,
        no_action_controls=controls,
        paired_initial_inputs=paired,
        media=media,
        deterministic_layout=True,
        randomized_scene_coverage=False,
        parity_qualified=False,
        scope=(
            "Free-base G1 walk, contact confirmation, rover grasp, storage and joint return"
            if scenario == "team"
            else "Rigid-terrain wheel drive, contact grasp, physical storage and return; illustrative material properties"
        ),
        scenario=scenario,
    )
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--destination", type=Path, default=MEDIA)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    report = publish(args.folders, args.destination, args.summary)
    print(
        json.dumps(
            dict(
                cases=len(report["cases"]),
                passed=sum(r["passed"] for r in report["cases"]),
                media=len(report["media"]),
            )
        )
    )


if __name__ == "__main__":
    main()
