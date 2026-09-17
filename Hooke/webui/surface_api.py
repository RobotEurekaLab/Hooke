"""Outdoor sampling gallery; generated media stays outside the repository."""

from pathlib import Path

from flask import Blueprint, abort, jsonify, send_file

from surface.profiles import MISSIONS
from surface.artifacts import MEDIA, SUMMARY, TEAM_SUMMARY, GAIT_POLICY
from webui.backend_api import read_json

bp = Blueprint("surface_missions", __name__)
VIEWS = {
    "overview": "全景",
    "landing": "着陆区",
    "follow": "跟随机器人",
    "sample": "取样特写",
}
FORMATS = {
    "image": ("png", "image/png"),
    "video": ("mp4", "video/mp4"),
    "webm": ("webm", "video/webm"),
}


@bp.get("/surface-missions")
def page():
    return send_file(Path(__file__).with_name("static") / "surface-missions.html")


@bp.get("/api/surface-missions")
def catalog():
    missions = []
    report = read_json(SUMMARY)
    team_report = read_json(TEAM_SUMMARY)
    for world, mission in MISSIONS.items():
        available = {
            backend: {
                view: {
                    kind: (MEDIA / f"{world}-{backend}-{view}.{suffix}").is_file()
                    for kind, (suffix, _) in FORMATS.items()
                }
                for view in VIEWS
            }
            for backend in ("mujoco", "isaac")
        }
        missions.append(
            dict(
                world=world,
                label=mission.label,
                task=mission.task_name,
                gravity_m_s2=mission.environment.gravity_m_s2,
                pressure_pa=mission.environment.exterior.pressure_pa,
                extent_m=2 * mission.terrain_half_size_m,
                home=list(mission.home),
                collection_stop=list(mission.collection_stop),
                sample_position=list(mission.sample_position),
                media=available,
                qualification=[
                    row for row in report.get("cases", []) if row.get("world") == world
                ],
                team=dict(
                    task=f"space_{world}_humanoid_rover",
                    label=(
                        "月面人形机器人协作采样"
                        if world == "lunar"
                        else "火星人形机器人协作采样"
                    ),
                    policy_available=GAIT_POLICY.is_file(),
                    qualification=[
                        row
                        for row in team_report.get("cases", [])
                        if row.get("world") == world
                    ],
                    media={
                        backend: {
                            view: {
                                kind: (
                                    MEDIA / f"team-{world}-{backend}-{view}.{suffix}"
                                ).is_file()
                                for kind, (suffix, _) in FORMATS.items()
                            }
                            for view in VIEWS
                        }
                        for backend in ("mujoco", "isaac")
                    },
                ),
            )
        )
    return jsonify(
        missions=missions,
        views=VIEWS,
        limitations=[
            "人形协作：G1 行走并触压确认按钮，采样车负责抓取与收纳岩样",
            "使用刚性地形和状态反馈导航；未模拟松软土壤、扬尘和视觉自主导航",
            "火星岩样复用 Apollo 外形作为示意样本，未代表实际火星岩石成分",
        ],
    )


@bp.get("/api/surface-missions/<world>/<backend>/<view>/<kind>")
@bp.get(
    "/api/surface-missions/team/<world>/<backend>/<view>/<kind>",
    defaults={"team": True},
)
def media(world, backend, view, kind, team=False):
    if (
        world not in MISSIONS
        or backend not in ("mujoco", "isaac")
        or view not in VIEWS
        or kind not in FORMATS
    ):
        abort(404)
    suffix, mimetype = FORMATS[kind]
    path = MEDIA / (("team-" if team else "") + f"{world}-{backend}-{view}.{suffix}")
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=mimetype, max_age=0, conditional=True)
