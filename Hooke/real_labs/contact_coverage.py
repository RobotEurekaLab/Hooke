"""Read-only zero-contact screening for visible, directly actuated rigid groups.

A weld group is MuJoCo's compiled rigid assembly. Fixed child bodies belong to
it; a descendant with its own joint does not. Having a collision proxy somewhere
in that group is only enough to avoid a zero-shape warning, never evidence that
all visible surfaces are covered or that any particular collision is detected.
"""

from collections import defaultdict

import mujoco
import numpy as np

ZERO_CONTACT_ISSUE = "moving_visible_group_without_contact_geometry"
TRANSMISSION_ISSUE = "moving_contact_coverage_transmission_requires_review"


def _visible_shapes(model):
    """Match the existing offline snapshot export's geometry visibility rule.

    This is eligibility for rendering, not a camera visibility/occlusion test.
    MuJoCo's default geom RGBA inherits the material in the snapshot exporter.
    """
    visible = np.zeros(model.ngeom, dtype=bool)
    for geom in range(model.ngeom):
        rgba = model.geom_rgba[geom]
        material = int(model.geom_matid[geom])
        if material >= 0 and np.array_equal(rgba, [0.5, 0.5, 0.5, 1.0]):
            rgba = model.mat_rgba[material]
        visible[geom] = model.geom_group[geom] < 3 and rgba[3] > 0
    return visible


def screen_contact_coverage(model):
    """Report actuated visible weld groups with no configured contact shapes.

    Enabled here means a nonzero contact mask or explicit contact-pair
    membership. This does not certify a compatible counterpart, pair filtering,
    globally enabled physics, shape correctness, or complete surface coverage.
    The model is inspected in place without allocating data or changing arrays.
    Unsupported actuator transmissions are explicitly left for review.
    """
    visible = _visible_shapes(model)
    explicit = set(int(g) for g in model.pair_geom1) | set(
        int(g) for g in model.pair_geom2
    )
    enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    if explicit:
        enabled[list(explicit)] = True
    actuated = defaultdict(list)
    unexamined = []
    direct = {mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT}
    for actuator in range(model.nu):
        transmission = int(model.actuator_trntype[actuator])
        item = dict(
            id=actuator,
            name=model.actuator(actuator).name or f"actuator_{actuator}",
            transmission=mujoco.mjtTrn(transmission).name,
        )
        if transmission not in direct:
            unexamined.append(item)
            continue
        joint = int(model.actuator_trnid[actuator, 0])
        body = int(model.jnt_bodyid[joint])
        weld = int(model.body_weldid[body])
        if weld == 0:
            # No stationary-world geom may provide coverage for a moving group.
            unexamined.append(dict(item, reason="joint_bound_to_world_weld_group"))
            continue
        actuated[weld].append(
            dict(
                item,
                joint=model.joint(joint).name or f"joint_{joint}",
                joint_id=joint,
                joint_body=model.body(body).name or f"body_{body}",
            )
        )
    groups = []
    for weld, drives in sorted(actuated.items()):
        body_ids = np.flatnonzero(model.body_weldid == weld)
        geom_ids = np.flatnonzero(model.body_weldid[model.geom_bodyid] == weld)
        visible_ids = geom_ids[visible[geom_ids]]
        enabled_ids = geom_ids[enabled[geom_ids]]
        disabled_ids = geom_ids[~enabled[geom_ids]]
        missing = len(visible_ids) > 0 and len(enabled_ids) == 0
        groups.append(
            dict(
                weld_body_id=weld,
                weld_body=model.body(weld).name or f"body_{weld}",
                body_ids=body_ids.tolist(),
                bodies=[model.body(int(b)).name or f"body_{b}" for b in body_ids],
                actuators=drives,
                geometry_count=len(geom_ids),
                visible_count=len(visible_ids),
                contact_enabled_count=len(enabled_ids),
                contact_disabled_count=len(disabled_ids),
                visible_contact_disabled_count=int(
                    np.count_nonzero(visible[geom_ids] & ~enabled[geom_ids])
                ),
                hidden_contact_enabled_count=int(
                    np.count_nonzero(~visible[geom_ids] & enabled[geom_ids])
                ),
                visible_geoms=[
                    model.geom(int(g)).name or f"geom_{g}" for g in visible_ids
                ],
                contact_enabled_geoms=[
                    model.geom(int(g)).name or f"geom_{g}" for g in enabled_ids
                ],
                explicit_pair_geoms=[
                    model.geom(int(g)).name or f"geom_{g}"
                    for g in geom_ids
                    if g in explicit
                ],
                issue=ZERO_CONTACT_ISSUE if missing else None,
                status=(
                    "requires_contact_geometry_review"
                    if missing
                    else (
                        "no_visible_geometry"
                        if not len(visible_ids)
                        else "has_contact_shapes_unverified_coverage"
                    )
                ),
            )
        )
    missing_groups = [g for g in groups if g["issue"]]
    issues = ([ZERO_CONTACT_ISSUE] if missing_groups else []) + (
        [TRANSMISSION_ISSUE] if unexamined else []
    )
    return dict(
        status="requires_review" if issues else "no_zero_contact_group_found",
        issues=issues,
        actuated_weld_group_count=len(groups),
        visible_actuated_weld_group_count=sum(g["visible_count"] > 0 for g in groups),
        zero_contact_visible_group_count=len(missing_groups),
        counts={
            key: sum(g[key] for g in groups)
            for key in (
                "geometry_count",
                "visible_count",
                "contact_enabled_count",
                "contact_disabled_count",
                "visible_contact_disabled_count",
                "hidden_contact_enabled_count",
            )
        },
        groups=groups,
        unexamined_actuators=unexamined,
        global_physics=dict(
            contact_disabled=bool(
                int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            ),
            constraint_disabled=bool(
                int(model.opt.disableflags)
                & int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            ),
        ),
        scope={
            "grouping": "Compiled body_weldid of directly joint-actuated rigid assemblies; fixed-world and child-joint geometry do not contribute to the parent's counts.",
            "visibility": "Snapshot-export eligibility: geom group below 3 and effective RGBA alpha above zero; not camera visibility.",
            "enabled": "Nonzero contact masks or explicit contact-pair membership, irrespective of visual group. Global flags are recorded separately.",
            "limitations": "Counts do not establish complete or spatially correct collision coverage, compatible collision partners, pair-filter behavior, contact quality, safe travel, or scientific validity. No AABB or surface-overlap certification is performed.",
            "transmissions": "Only joint and joint-in-parent actuator transmissions are classified; others remain explicit review items.",
        },
    )
