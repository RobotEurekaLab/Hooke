"""Select a complete recording release while retaining historical files."""

import argparse
import hashlib
import json
from pathlib import Path

from microscopy import BACKENDS, OPERATIONS
from microscopy.artifacts import write_json

FORMATS = ("mp4", "webm")


def recording_directory(media, backend):
    """Return the published backend directory and its browser-cache revision."""
    if backend not in BACKENDS:
        raise ValueError("Unknown microscopy backend")
    media = Path(media).resolve()
    index = media/"recordings"/"current.json"
    if not index.is_file():
        return (media if backend == "mujoco" else media/"isaac"), "legacy"
    release = json.loads(index.read_text())
    if (not isinstance(release, dict) or release.get("schema_version") != 1
            or not isinstance(release.get("directory"), str)
            or not isinstance(release.get("revision"), str) or not release["revision"]):
        raise ValueError("Invalid microscopy recording release")
    relative = Path(release["directory"])
    directory = (media/relative/backend).resolve()
    if relative.is_absolute() or not directory.is_relative_to(media):
        raise ValueError("Recording release must stay inside the media directory")
    if not directory.is_dir():
        raise ValueError("Published microscopy recording directory is unavailable")
    return directory, release["revision"]


def publish_recordings(media, release):
    """Verify every encoded file before atomically selecting the new release."""
    media, release = Path(media).resolve(), Path(release).resolve()
    relative = release.relative_to(media)
    hashes = {}
    for backend in BACKENDS:
        for operation in OPERATIONS:
            for extension in FORMATS:
                path = release/backend/operation/f"demo.{extension}"
                report = json.loads(path.with_name(path.name+".json").read_text())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if (report.get("backend"), report.get("operation"), report.get("sha256")) != (
                        backend, operation, digest):
                    raise ValueError(f"Recording provenance mismatch: {path}")
                hashes[str(path.relative_to(release))] = digest
    manifest = dict(schema_version=1, directory=str(relative), revision=release.name,
                    video_sha256=hashes)
    index = media/"recordings"/"current.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    write_json(index, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish_recordings(args.media, args.release)))


if __name__ == "__main__":
    main()
