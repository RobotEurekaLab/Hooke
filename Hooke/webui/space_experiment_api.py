"""Experiment gallery and a public measurement interface for science clients."""

from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file

from experiments import OPERATIONS
from experiments.records import ScienceRecords
from worlds.profiles import WORLDS
from webui.backend_api import ROOT, get_job, read_json

bp = Blueprint("space_experiments", __name__)


@bp.get("/space-experiments")
def page():
    return send_file(Path(__file__).with_name("static") / "space-experiments.html")


@bp.get("/api/space-experiments")
def catalog():
    return jsonify(
        worlds=[
            dict(
                name=p.name,
                label=p.label,
                gravity_m_s2=p.gravity_m_s2,
                pressure_pa=p.workspace.pressure_pa,
            )
            for p in WORLDS.values()
        ],
        operations=list(OPERATIONS),
        qualification=read_json(
            ROOT / "docs/validation/space_experiments_summary.json"
        ),
        capabilities=ScienceRecords.capabilities(),
    )


@bp.get("/api/space-experiments/<world>/<operation>/<backend>/<media>")
def media(world, operation, backend, media):
    if (
        world not in WORLDS
        or operation not in OPERATIONS
        or backend not in ("isaac", "mujoco")
        or media not in ("image", "video")
    ):
        abort(404)
    suffix = ".png" if media == "image" else ".mp4"
    path = (
        ROOT / "docs/assets" / f"space-experiment-{world}-{operation}-{backend}{suffix}"
    )
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/png" if media == "image" else "video/mp4")


@bp.get("/api/science/capabilities")
def capabilities():
    return jsonify(ScienceRecords.capabilities())


@bp.get("/api/space-experiments/<world>/<operation>/<backend>/evidence")
def evidence(world, operation, backend):
    if (
        world not in WORLDS
        or operation not in OPERATIONS
        or backend not in ("isaac", "mujoco")
    ):
        abort(404)
    path = (
        ROOT
        / "docs/validation"
        / f"space-experiment-{world}-{operation}-{backend}.json"
    )
    if not path.is_file():
        abort(404)
    return jsonify(read_json(path))


@bp.get("/api/science/jobs/<identifier>/measurements")
def measurements(identifier):
    return jsonify(
        measurement_ids=ScienceRecords(
            get_job(identifier)["output"]
        ).list_measurements()
    )


@bp.get("/api/science/jobs/<identifier>/measurements/<measurement_id>")
def measurement(identifier, measurement_id):
    try:
        return jsonify(
            ScienceRecords(get_job(identifier)["output"]).read_measurement(
                measurement_id
            )
        )
    except (ValueError, KeyError):
        abort(404)


@bp.post("/api/science/jobs/<identifier>/analyze")
def analyze(identifier):
    records = ScienceRecords(get_job(identifier)["output"])
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) != {"method", "measurement_ids"}:
        return jsonify(error="Only method and measurement_ids are accepted"), 400
    try:
        return jsonify(records.analyze(body["method"], body["measurement_ids"]))
    except (ValueError, KeyError, TypeError) as error:
        return jsonify(error=str(error)), 400
