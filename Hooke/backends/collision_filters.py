"""Compact collision exclusions without expanding uniform body pairs."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class CollisionFilterPlan:
    body_geoms: dict
    body_pairs: list
    geom_pairs: list


def collision_filter_plan(model, colliders, explicit_pairs):
    """Preserve masks, welded/adjacent bodies, exclusions and explicit pairs.

    Fully excluded source-body pairs use collision groups. Only partially
    excluded pairs need individual geom targets, avoiding millions of static
    geom relations in large scenes. Collections contain exact collider prims,
    so nested source bodies cannot accidentally inherit a parent's filter.
    """
    by_body = defaultdict(list)
    for geom in sorted(colliders):
        by_body[int(model['geom_bodyid'][geom])].append(geom)
    bodies = sorted(by_body)
    geoms = {body: np.asarray(by_body[body], dtype=int) for body in bodies}
    exceptions = defaultdict(list)
    for first, second in explicit_pairs:
        a, b = int(model['geom_bodyid'][first]), int(model['geom_bodyid'][second])
        if a > b:
            a, b, first, second = b, a, second, first
        exceptions[a, b].append((first, second))
    excludes = set(map(int, model['exclude_signature']))
    masks = {body: np.unique(np.column_stack((model['geom_contype'][ids],
                                             model['geom_conaffinity'][ids])), axis=0)
             for body, ids in geoms.items()}
    body_pairs, geom_pairs = [], []
    for index, a in enumerate(bodies):
        wa = int(model['body_weldid'][a])
        pa = int(model['body_weldid'][model['body_parentid'][wa]])
        for b in bodies[index + 1:]:
            wb = int(model['body_weldid'][b])
            pb = int(model['body_weldid'][model['body_parentid'][wb]])
            excluded = (wa == wb or (wa != 0 and wb != 0 and (wa == pb or wb == pa))
                        or ((a << 16) + b) in excludes)
            pairs = exceptions.get((a, b), ())
            ma, mb = masks[a], masks[b]
            compatible = ((ma[:, 0, None] & mb[:, 1]) != 0) | ((ma[:, 1, None] & mb[:, 0]) != 0)
            if not excluded and compatible.all():
                continue
            if (excluded or not compatible.any()) and not pairs:
                body_pairs.append((a, b))
                continue
            ga, gb = geoms[a], geoms[b]
            allowed = np.zeros((len(ga), len(gb)), dtype=bool) if excluded else (
                ((model['geom_contype'][ga, None] & model['geom_conaffinity'][gb]) != 0)
                | ((model['geom_conaffinity'][ga, None] & model['geom_contype'][gb]) != 0))
            if pairs:
                ia, ib = {int(g): i for i, g in enumerate(ga)}, {int(g): i for i, g in enumerate(gb)}
                for first, second in pairs:
                    allowed[ia[first], ib[second]] = True
            if not allowed.any():
                body_pairs.append((a, b))
            else:
                geom_pairs.extend((int(ga[i]), int(gb[j])) for i, j in zip(*np.where(~allowed)))
    return CollisionFilterPlan(dict(by_body), body_pairs, geom_pairs)
