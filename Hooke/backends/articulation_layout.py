"""Bound native articulations while retaining independent world-fixed branches."""

MAX_ARTICULATION_LINKS = 64


def articulation_layout(model, rigid):
    """Prune immovable anchors only when their shared articulation is oversized.

    A fixed source body with world weld ID zero cannot move. Its geometry may
    remain static while its independently attached moving branches each get a
    world anchor. Moving links are never removed. Equality/tendon constraints
    require a dedicated partition analysis and prevent this conservative split.
    """
    rigid = set(rigid)

    def groups(bodies):
        parents = {}
        for body in bodies:
            parent = int(model['body_parentid'][body])
            while parent and parent not in bodies:
                parent = int(model['body_parentid'][parent])
            parents[body] = parent
        trees = {}
        for body in sorted(bodies):
            root = body
            while parents[root]:
                root = parents[root]
            trees.setdefault(root, []).append(body)
        result = []
        for root, members in trees.items():
            moving = [body for body in members if model['body_jntnum'][body]]
            if not moving:
                continue  # The runtime removes the articulation API here.
            root_joint = int(model['body_jntadr'][root])
            free = (model['body_jntnum'][root] > 0
                    and model['jnt_type'][root_joint] == 0)
            # Every authored rigid body is a link, including fixed supports.
            # Compound joints add auxiliary links; a scalar root also needs
            # the exporter's synthetic world anchor. Never assume fixed-link
            # merging will happen before native articulation allocation.
            links = sum(max(1, int(model['body_jntnum'][body])) for body in members)
            links += int(model['body_jntnum'][root] > 0 and not free)
            result.append({'root_body': root, 'members': members, 'link_count': links})
        return result

    original = groups(rigid)
    oversized = [tree for tree in original if tree['link_count'] > MAX_ARTICULATION_LINKS]
    removed = set()
    if oversized:
        if len(model['eq_type']) or len(model['tendon_adr']):
            raise NotImplementedError(
                'Oversized native articulation has equality or tendon constraints; '
                'independent branch partitioning is not qualified. Use MuJoCo or '
                'provide a validated articulation partition.')
        for tree in oversized:
            fixed = {body for body in tree['members']
                     if model['body_weldid'][body] == 0 and model['body_jntnum'][body] == 0}
            if not fixed:
                raise NotImplementedError(
                    f"Connected native articulation at body {tree['root_body']} needs "
                    f"{tree['link_count']} links, exceeding the conservative "
                    f"{MAX_ARTICULATION_LINKS}-link exporter limit; moving links cannot be split safely.")
            removed.update(fixed)
        rigid -= removed
    final = groups(rigid)
    remaining = [tree for tree in final if tree['link_count'] > MAX_ARTICULATION_LINKS]
    if remaining:
        raise NotImplementedError(
            'Moving branches still exceed the conservative native articulation link limit: '
            + ', '.join(f"body {tree['root_body']} ({tree['link_count']} links)" for tree in remaining))
    # Independent branches may share one immovable world anchor without
    # introducing a dynamic connection. Pack only branches from split trees;
    # leave every originally bounded/coupled articulation unchanged.
    split_members = {body for tree in oversized for body in tree['members']}
    packed, shared_anchors = [], []
    current = None
    for tree in final:
        root = tree['root_body']
        eligible = (root in split_members and model['body_jntnum'][root] == 1
                    and model['jnt_type'][int(model['body_jntadr'][root])] != 0)
        if not eligible:
            packed.append(tree)
            continue
        moving_links = tree['link_count'] - 1
        if current is None or current['link_count'] + moving_links > MAX_ARTICULATION_LINKS:
            current = {'root_body': root, 'members': [], 'link_count': 1, 'branch_roots': []}
            packed.append(current)
            shared_anchors.append(current['branch_roots'])
        current['members'].extend(tree['members'])
        current['link_count'] += moving_links
        current['branch_roots'].append(root)
    return rigid, {
        'link_limit': MAX_ARTICULATION_LINKS,
        'removed_fixed_rigid_bodies': sorted(removed),
        'original_articulations': original,
        'source_branches': final,
        'articulations': packed,
        'shared_world_anchor_roots': shared_anchors,
        'scope': 'Conservative exporter bound; static world support partitioning only, no moving-link split.',
    }
