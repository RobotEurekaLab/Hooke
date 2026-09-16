"""Source collision participation, including mask-independent contact pairs."""


def collision_participants(model):
    pairs={tuple(sorted((int(a),int(b)))) for a,b in zip(model['pair_geom1'],model['pair_geom2'])}
    colliders={i for i,(kind,affinity) in enumerate(zip(model['geom_contype'],model['geom_conaffinity'])) if kind or affinity}
    colliders.update(geom for pair in pairs for geom in pair)
    return colliders,pairs
