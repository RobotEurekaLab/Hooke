"""Refresh archived synthetic microscope panels without physics or GPU rendering.

The historical Sutter recordings did not save image_state.json. Their exact
interaction samples, motor feedback and exported static geometry recover the
optical inputs. This deliberately supports only archived slide-joint models;
unsupported mechanisms or incomplete evidence fail before rendering.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from microscopy.artifacts import write_image, write_json
from microscopy.cell_optics import CellMicroscope
from microscopy.recording import encode_episode
from microscopy.scene import ORIGIN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class ArchivedSlides:
    """Pure-array forward kinematics, loading no simulator or scene meshes."""

    def __init__(self, source):
        self.source = Path(source).resolve()
        scene = json.loads((self.source/"scene.json").read_text())
        expected = scene.get("archive_sha256", {}).get("model.npz")
        if expected is not None and digest(self.source/"model.npz") != expected:
            raise ValueError("Static model archive differs from its export manifest")
        self.names = scene["names"]
        fields = ("body_parentid", "body_pos", "body_quat", "jnt_type", "jnt_bodyid",
                  "jnt_axis", "jnt_qposadr", "qpos0", "geom_bodyid", "geom_pos",
                  "geom_quat", "geom_size", "site_bodyid", "site_pos")
        with np.load(self.source/"model.npz", allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in fields}
        if not np.all(self.arrays["jnt_type"] == 2):
            raise ValueError("Offline reconstruction supports only archived slide joints")
        self.body_rotations = np.asarray([rotation(q) for q in self.arrays["body_quat"]])
        self.joints = [[] for _ in self.arrays["body_parentid"]]
        for index, body in enumerate(self.arrays["jnt_bodyid"]):
            self.joints[body].append(index)

    def pose(self, encoders):
        missing = set(self.names["joint"])-set(encoders)
        if missing:
            raise ValueError(f"Missing actual joint feedback: {sorted(missing)}")
        a = self.arrays
        position = np.zeros_like(a["body_pos"], dtype=float)
        orientation = np.repeat(np.eye(3)[None], len(position), axis=0)
        for body in range(1, len(position)):
            parent = a["body_parentid"][body]
            orientation[body] = orientation[parent]@self.body_rotations[body]
            position[body] = position[parent]+orientation[parent]@a["body_pos"][body]
            for joint in self.joints[body]:
                distance = encoders[self.names["joint"][joint]]-a["qpos0"][a["jnt_qposadr"][joint]]
                position[body] += orientation[body]@a["jnt_axis"][joint]*distance
        return position, orientation

    def element(self, pose, kind, name):
        index = self.names[kind].index(name)
        body = self.arrays[kind+"_bodyid"][index]
        positions, rotations = pose
        position = positions[body]+rotations[body]@self.arrays[kind+"_pos"][index]
        return position, rotations[body]

    def geom(self, pose, name):
        index = self.names["geom"].index(name)
        position, body_rotation = self.element(pose, "geom", name)
        return (position, body_rotation@rotation(self.arrays["geom_quat"][index]),
                self.arrays["geom_size"][index])


def exact_rows(times, observations):
    if len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Interaction samples must have finite strictly increasing times")
    indices = np.searchsorted(times, observations)
    if np.any(indices >= len(times)) or not np.allclose(times[indices], observations, rtol=0, atol=1e-12):
        raise ValueError("Every saved observation must have an exact interaction sample")
    return indices


def reconstruct_episode(episode, source):
    """Return optical inputs and reconstruction evidence for each observation."""
    episode = Path(episode).resolve()
    model = ArchivedSlides(source)
    folders = sorted((episode/"observations").iterdir())
    states = [json.loads((folder/"state.json").read_text()) for folder in folders]
    interaction_hash = digest(episode/"interaction.npz")
    result = json.loads((episode/"result.json").read_text())
    expected_hash = result.get("interaction_trace", {}).get("sha256")
    if expected_hash is not None and interaction_hash != expected_hash:
        raise ValueError("Interaction archive differs from the episode report")
    with np.load(episode/"interaction.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    rows = exact_rows(trace["time_s"], [state["time_s"] for state in states])
    operation = states[0]["operation"]
    if states[0]["backend"] == "mujoco" and operation in ("push", "pick_place"):
        raise ValueError("Historical MuJoCo handling archives omit the previous-tick probe/jaw poses")
    scale = states[0]["sample_scale"]
    calibration = CellMicroscope().public_calibration()
    for name in ("field_width_m", "pixels", "empirical_defocus_scale_m", "wavelength_m", "numerical_aperture"):
        if states[0]["calibration"][name] != calibration[name]:
            raise ValueError(f"Archived microscope calibration differs: {name}")
    radii = np.asarray(scale["diameters_m"])/2
    geometry = scale["pipette_geometry"]["injector"]
    needle_back = np.asarray(geometry["tip_back"])
    # Preserve the observed point while the scalar membrane response relaxes.
    from microscopy.cell_deformation import needle_contact_offset, needle_contact_descriptor
    last_contact = None
    needle_contacts = {}
    selected_rows = set(rows.tolist())
    for row in range(len(trace["time_s"])):
        offset = needle_contact_offset(trace["tip_world_m"][row]-trace["contact_cell_world_m"][row],
                                       radii, needle_back)
        if offset is not None:
            last_contact = np.asarray(offset)
        if row in selected_rows:
            needle_contacts[row] = ([] if last_contact is None else [needle_contact_descriptor(
                last_contact, radii, needle_back, trace["indentation_m"][row])])
    reconstructed = []
    maximum_pose_error = 0.
    for folder, state, row in zip(folders, states, rows):
        if state["operation"] != operation or state["sample_scale"] != scale:
            raise ValueError("Episode geometry changed within the archived observations")
        pose = model.pose(state["encoders_m"])
        needle, _ = model.element(pose, "site", "injector_tcp")
        error = float(np.max(np.abs(needle-trace["tip_world_m"][row])))
        maximum_pose_error = max(maximum_pose_error, error)
        if error > 1e-11 and (state["backend"] == "isaac" or row == rows[0]):
            raise ValueError(f"Static archive does not reproduce actual needle pose: {error} m")
        centres = [(model.geom(pose, f"cell_{i}_shell")[0]-ORIGIN).tolist() for i in range(6)]
        if operation in ("injection", "cell_injection"):
            # MuJoCo's saved qpos is one integration tick newer than the sites
            # actually used by mechanics and optics. Recover stage translation
            # from the recorded adherent target instead of the newer encoder.
            stage_correction = trace["cell_world_m"][row]-model.element(pose, "site", "cell_center")[0]
            centres = (np.asarray(centres)+stage_correction).tolist()
        centres[0] = (trace["cell_world_m"][row]-ORIGIN).tolist()
        focus = state["encoders_m"]["focus"]
        image_state = dict(
            time_s=state["time_s"], focus_m=focus, centres_m=centres,
            tip_m=(trace["tip_world_m"][row]-ORIGIN).tolist(),
            needle_back=needle_back.tolist(), needle_geometry=geometry,
            indentation_m=float(trace["indentation_m"][row]),
            membrane_contacts=needle_contacts[row],
            punctured=bool(trace["punctured"][row]), withdrawn=bool(trace["withdrawn"][row]),
            delivered_pl=float(trace["delivered_m3"][row]*1e15),
            force_nn=float(trace["force_n"][row]*1e9),
            actual_pressure_pa=float(trace["actual_pressure_pa"][row]),
            cell_radii_m=radii.tolist(),
            nuclear_radii_m=(np.asarray(scale["nuclear_diameters_m"])/2).tolist(),
            optical_reference_height_m=float(model.arrays["site_pos"][model.names["site"].index("cell_center"), 2]))
        if operation == "suction_injection":
            holder_geometry = scale["pipette_geometry"]["holder"]
            holder, _ = model.element(pose, "site", "holder_tcp")
            if state["backend"] == "isaac" and not np.allclose(holder, trace["holding_tip_world_m"][row], rtol=0, atol=1e-11):
                raise ValueError("Static archive does not reproduce actual holder pose")
            image_state.update(holder_tip_m=(trace["holding_tip_world_m"][row]-ORIGIN).tolist(),
                               holder_back=holder_geometry["tip_back"], holder_geometry=holder_geometry,
                               holding_sealed=bool(trace["holding_sealed"][row]),
                               holding_pressure_pa=float(trace["holding_pressure_pa"][row]))
        if operation in ("push", "pick_place"):
            handling_contacts(model, pose, trace, row, image_state, operation)
        reconstructed.append((folder, state, image_state))
    evidence = dict(source_episode=str(episode), static_model_source=str(model.source),
                    interaction_sha256=interaction_hash,
                    static_model_sha256=digest(model.source/"model.npz"),
                    static_scene_sha256=digest(model.source/"scene.json"),
                    exact_time_matches=len(rows), maximum_encoder_to_recorded_tip_difference_m=maximum_pose_error,
                    tool_pose_source="Actual recorded interaction sites; MuJoCo qpos may be one integration tick newer",
                    physics_steps=0, gpu_rendering=False,
                    reconstruction="Recorded actual joint feedback, interaction samples and static slide-joint forward kinematics")
    return reconstructed, evidence


def handling_contacts(model, pose, trace, row, state, operation):
    """Recover original unilateral contact geometry and per-frame compression."""
    before = trace["contact_cell_world_m"][row]
    after = trace["cell_world_m"][row]
    radius = state["cell_radii_m"][0]
    contacts = []
    from microscopy.cell_deformation import handling_contact_descriptor
    if operation == "push":
        tip, _, size = model.geom(pose, "probe_tip")
        probe_radius = float(size[0])
        state.update(probe_tip_m=(tip-ORIGIN).tolist(), probe_radius_m=probe_radius)
        delta = before-tip
        length = np.linalg.norm(delta)
        if 0 < length < radius+probe_radius:
            normal = delta/length
            equilibrium = tip+normal*(radius+probe_radius)
            depth = max(0., float((equilibrium-after)@normal))
            contacts.append(handling_contact_descriptor(normal, state["cell_radii_m"], depth))
    else:
        jaws = [model.geom(pose, f"jaw_{side}_pad") for side in ("a", "b")]
        state.update(jaw_centres_m=[(position-ORIGIN).tolist() for position, _, _ in jaws],
                     jaw_halfsize_m=jaws[0][2].tolist())
        for (centre, orientation, halfsize), sign in zip(jaws, (1, -1)):
            normal = -sign*orientation[:, 1]
            local = orientation.T@(before-centre)
            face = centre+normal*halfsize[1]
            gap = float((before-face)@normal)
            if (0 <= gap < radius and abs(local[0]) < radius+halfsize[0]
                    and abs(local[2]) < radius+halfsize[2]):
                equilibrium = face+normal*radius
                depth = max(0., float((equilibrium-after)@normal))
                contacts.append(handling_contact_descriptor(normal, state["cell_radii_m"], depth))
    state["membrane_contacts"] = contacts
    state["indentation_m"] = max((contact["indentation_m"] for contact in contacts), default=0.)


def recovered_episode(episode, recovered):
    """Accept a separately verified deterministic CPU replay of missing poses."""
    episode, recovered = Path(episode).resolve(), Path(recovered).resolve()
    evidence_path = recovered/"evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if (Path(evidence["source_episode"]).resolve() != episode
            or evidence.get("exact_trace_match") is not True
            or evidence.get("exact_encoder_match") is not True):
        raise ValueError("Recovered poses require verified exact trace and encoder equality")
    interaction_hash = digest(episode/"interaction.npz")
    expected = json.loads((episode/"result.json").read_text())["interaction_trace"]["sha256"]
    if interaction_hash != expected or evidence["source_interaction_sha256"] != interaction_hash:
        raise ValueError("Recovered poses refer to a different interaction archive")
    folders = sorted((episode/"observations").iterdir())
    states = [json.loads((folder/"state.json").read_text()) for folder in folders]
    reconstructed = []
    with np.load(episode/"interaction.npz", allow_pickle=False) as trace:
        rows = exact_rows(trace["time_s"], [state["time_s"] for state in states])
        for folder, state, row in zip(folders, states, rows):
            image_path = recovered/"observations"/folder.name/"image_state.json"
            if digest(image_path) != evidence["image_state_sha256"][folder.name]:
                raise ValueError("Recovered optical state differs from its replay evidence")
            image_state = json.loads(image_path.read_text())
            if image_state["time_s"] != state["time_s"]:
                raise ValueError("Recovered optical state time differs from archived observation")
            for key, recorded in (("tip_m", trace["tip_world_m"][row]-ORIGIN),
                                  ("centres_m", trace["cell_world_m"][row]-ORIGIN)):
                value = image_state[key][0] if key == "centres_m" else image_state[key]
                if not np.array_equal(value, recorded):
                    raise ValueError(f"Recovered optical {key} differs from archived interaction")
            if image_state["focus_m"] != state["encoders_m"]["focus"]:
                raise ValueError("Recovered optical focus differs from archived encoder")
            reconstructed.append((folder, state, image_state))
    evidence.update(recovered_evidence_file=str(evidence_path),
                    recovered_evidence_sha256=digest(evidence_path),
                    exact_time_matches=len(reconstructed), gpu_rendering=False,
                    reconstruction="Deterministic CPU replay verified against every archived interaction column and observation encoder")
    return reconstructed, evidence


def refresh_episode(recording, source, output, *, encode=True, validate_only=False,
                    recovered_image_states=None):
    """Write only new microscope images; original world panels remain read-only."""
    recording, output = Path(recording), Path(output)
    provenance = json.loads(recording.read_text())
    if recovered_image_states is None:
        observations, evidence = reconstruct_episode(provenance["episode"], source)
    else:
        observations, evidence = recovered_episode(provenance["episode"], recovered_image_states)
    if len(observations) != len(provenance["observation_hashes"]):
        raise ValueError("Recording provenance observation count differs")
    for (folder, _, _), hashes in zip(observations, provenance["observation_hashes"]):
        for name, expected in hashes.items():
            if digest(folder/name) != expected:
                raise ValueError(f"Original observation changed: {folder/name}")
    evidence.update(original_recording_provenance=str(recording.resolve()),
                    original_provenance_sha256=digest(recording),
                    original_video_sha256=provenance["sha256"])
    if validate_only:
        return evidence
    microscope = CellMicroscope()
    baseline = next((item for item in observations if item[2]["indentation_m"] == 0), None)
    if baseline is None:
        raise ValueError("No undeformed observation available to validate optical reconstruction")
    baseline_folder, _, baseline_state = baseline
    baseline_channels = ["phase_contrast"]
    if (baseline_folder/"microscope_fluorescence.png").is_file():
        baseline_channels.append("fluorescence")
    for channel in baseline_channels:
        name = "microscope.png" if channel == "phase_contrast" else "microscope_fluorescence.png"
        with Image.open(baseline_folder/name) as original:
            if not np.array_equal(np.asarray(microscope.render(baseline_state, channel=channel)), np.asarray(original)):
                raise ValueError(f"Undeformed baseline differs from archive: {baseline_folder/name}")
    evidence.update(undeformed_baseline=baseline_folder.name, baseline_pixel_identical=True,
                    microscope_observations_regenerated=0, microscope_observations_reused=0)
    output.mkdir(parents=True, exist_ok=False)
    for folder, _, image_state in observations:
        destination = output/"observations"/folder.name
        destination.mkdir(parents=True)
        for original in folder.iterdir():
            if original.name not in ("microscope.png", "microscope_fluorescence.png", "image_state.json"):
                (destination/original.name).symlink_to(original.resolve())
        write_json(destination/"image_state.json", image_state)
        if image_state["indentation_m"] == 0:
            for name in ("microscope.png", "microscope_fluorescence.png"):
                if (folder/name).is_file():
                    (destination/name).symlink_to((folder/name).resolve())
            evidence["microscope_observations_reused"] += 1
        else:
            write_image(destination/"microscope.png", microscope.render(image_state))
            if (folder/"microscope_fluorescence.png").is_file():
                write_image(destination/"microscope_fluorescence.png", microscope.render(image_state, channel="fluorescence"))
            evidence["microscope_observations_regenerated"] += 1
    write_json(output/"refresh.json", evidence)
    if encode:
        for extension in ("mp4", "webm"):
            encode_episode(output, output/f"demo.{extension}", provenance["fps"])
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True, help="Original demo.mp4.json provenance")
    parser.add_argument("--source", type=Path, required=True, help="Matching exported source/ geometry directory")
    parser.add_argument("--output", type=Path, required=True, help="New episode directory under ignored temp/")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--recovered-image-states", type=Path,
                        help="Separately verified exact CPU replay containing evidence.json and image states")
    args = parser.parse_args()
    print(json.dumps(refresh_episode(args.recording, args.source, args.output,
                                    encode=not args.skip_encode, validate_only=args.validate_only,
                                    recovered_image_states=args.recovered_image_states)))


if __name__ == "__main__":
    main()
