"""Select declared native integration settings from equipment capabilities."""
import math


def _sample_transfer_options(equipment, model, bound, owned):
    """Resolve an authored rigid-sample interface without scene-specific names."""
    interface = equipment.get('sample_transfer_interface')
    if interface is None:
        return {}
    if not isinstance(interface, dict) or interface.get('kind') != 'rigid_sample_staging_v1':
        raise ValueError('Unsupported sample transfer interface')
    sample = bound('geom', interface.get('sample_geom'))
    seat = bound('geom', interface.get('stage_seat_geom'))
    if sample == seat or owned(int(model.geom_bodyid[sample])):
        raise ValueError('Staging requires a separate sample outside the rover articulation')
    import mujoco
    body, free_sample = int(model.geom_bodyid[sample]), False
    while body:
        start = int(model.body_jntadr[body])
        free_sample |= any(model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
                           for joint in range(start, start + int(model.body_jntnum[body])))
        body = int(model.body_parentid[body])
    if not free_sample:
        raise ValueError('Staging requires a freely moving rigid sample')
    body = int(model.geom_bodyid[seat])
    while body:
        if model.body_jntnum[body] or model.body_mocapid[body] >= 0:
            raise ValueError('Staging contact profile requires a static seat')
        body = int(model.body_parentid[body])
    from backends.collision_rules import collision_participants
    from backends.collision_filters import collision_filter_plan
    fields = ('pair_geom1', 'pair_geom2', 'geom_contype', 'geom_conaffinity',
              'geom_bodyid', 'exclude_signature', 'body_weldid', 'body_parentid')
    arrays = {name: getattr(model, name) for name in fields}
    colliders, pairs = collision_participants(arrays)
    if sample not in colliders or seat not in colliders:
        raise ValueError('Sample transfer surfaces must have active colliders')
    # Use the converter's exact exclusions and explicit-pair priority for this
    # interface; unrelated explicit pairs must not enable the selected surfaces.
    selected = {sample, seat}
    plan = collision_filter_plan(arrays, selected,
                                 {pair for pair in pairs if set(pair) <= selected})
    if plan.body_pairs or plan.geom_pairs:
        raise ValueError('Sample transfer surfaces must permit mutual contact')
    from backends.physics_options import validate_contact_offset_overrides
    overrides = validate_contact_offset_overrides({
        interface['stage_seat_geom']: interface.get('native_contact_offset_m')})
    return {'contact_offset_overrides_m': overrides}


def _scissor_options(mechanisms, model):
    if not mechanisms:
        return {}
    if model is None:
        raise ValueError('A scissor native profile requires compiled equality ownership checks')
    owned = set()
    for mechanism in mechanisms:
        bindings = mechanism.get('equalities', {})
        names = [bindings.get(key) for key in ('lower_scissor', 'upper_scissor', 'deck_slot')]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError('Scissor native profile requires every CONNECT equality binding')
        owned.update(names)
    connects = {model.equality(index).name for index, kind in enumerate(model.eq_type) if kind == 0}
    if connects != owned:
        raise ValueError('Scissor native profile cannot harden unowned or missing CONNECT equalities: '
                         f'unowned={sorted(connects-owned)}, missing={sorted(owned-connects)}')
    return {'native_substeps': 4, 'soft_connect_constraints': False}


def _rover_options(equipment, model):
    if (equipment.get('kind') != 'rover' or equipment.get('params', {}).get('free_base') is not True
            or not equipment.get('arm_interfaces')):
        return {}
    if model is None:
        raise ValueError('A precision rover profile requires compiled ownership checks')
    import mujoco

    def bound(kind, name):
        if not isinstance(name, str) or not name:
            raise ValueError(f'Precision rover profile needs a named {kind} binding')
        try:
            return int(getattr(model, kind)(name).id)
        except KeyError as error:
            raise ValueError(f'Precision rover {kind} binding is missing: {name}') from error

    base = bound('body', equipment.get('body'))
    free = bound('joint', equipment.get('freejoint'))
    if model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[free] != base:
        raise ValueError('Declared rover base does not own its free joint')

    def owned(body):
        while body and body != base:
            body = int(model.body_parentid[body])
        return body == base

    joints = equipment.get('joints', {})
    wheels = [name for name in joints if name.startswith('wheel_')]
    if len(wheels) != 6:
        raise ValueError('Precision rover profile requires six explicitly driven wheels')
    joint_ids, actuator_ids = set(), set()
    for name in wheels:
        joint = bound('joint', joints[name])
        actuator = bound('actuator', equipment.get('actuators', {}).get(name))
        if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                or not owned(int(model.jnt_bodyid[joint]))
                or equipment.get('actuator_modes', {}).get(name) != 'velocity'
                or model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[actuator, 0] != joint):
            raise ValueError(f'Invalid rover wheel drive binding: {name}')
        joint_ids.add(joint)
        actuator_ids.add(actuator)
    if len(joint_ids) != 6 or len(actuator_ids) != 6:
        raise ValueError('Precision rover wheel bindings must be distinct')
    pairs = 0
    for interface in equipment['arm_interfaces'].values():
        for pair in interface.get('engineered_pair_classes', []):
            clearance = pair.get('minimum_clearance_m')
            if (isinstance(clearance, bool) or not isinstance(clearance, (int, float))
                    or not math.isfinite(clearance) or clearance <= 0):
                raise ValueError('Rover arm interface needs a finite positive declared clearance')
            groups = [pair.get('fixed_or_proximal'), pair.get('moving_or_distal')]
            if not all(isinstance(group, list) and group for group in groups):
                raise ValueError('Rover arm interface must declare both exact surface groups')
            for group in groups:
                for name in group:
                    geom = bound('geom', name)
                    if not owned(int(model.geom_bodyid[geom])):
                        raise ValueError(f'Arm interface surface escapes declared rover: {name}')
            pairs += 1
    if not pairs:
        raise ValueError('Precision rover profile requires explicit arm interfaces')
    # Guarded rover operations consume per-actuator applied effort. Implicit
    # joint loads cannot supply it. Reject ambiguous transmissions before
    # selecting explicit affine actuation for this entire native scene.
    from backends.effort_observation import explicit_affine_joint_map
    fields = ('actuator_trntype', 'actuator_dyntype', 'actuator_gaintype',
              'actuator_biastype', 'actuator_gear', 'actuator_trnid', 'jnt_type')
    explicit_affine_joint_map({name: getattr(model, name) for name in fields})
    # The converter option applies to ALL scene collision cylinders, including
    # equipment outside this rover. Integration settings select a candidate
    # numerical model; each platform still requires dynamic qualification.
    return {'compound_cylinder_colliders': True, 'contact_offset_m': 5e-5,
            'sdf_contact_offset_m': 5e-5, 'solver_type': 'PGS', 'simulation_threads': 4,
            'explicit_affine_actuators': True, 'native_substeps': 4,
            **_sample_transfer_options(equipment, model, bound, owned)}


def native_physics_options_for_manifest(manifest, model=None):
    """Compose validated equipment profiles without institution or scene IDs.

    Hard CONNECT closure is scene-wide and requires complete scissor ownership.
    The rover profile selects scene-wide compound collision cylinders; unrelated
    cylinders are included and must be covered by the export's native validation.
    """
    equipment = manifest.get('equipment', [])
    mechanisms = [mechanism for item in equipment
                  for mechanism in item.get('mechanisms', {}).values()
                  if mechanism.get('kind') == 'scissor_jack_v1']
    profiles = [_scissor_options(mechanisms, model)]
    profiles.extend(_rover_options(item, model) for item in equipment)
    options = {}
    for profile in profiles:
        for name, value in profile.items():
            if name == 'contact_offset_overrides_m':
                merged = dict(options.get(name, {}))
                for geom, offset in value.items():
                    if geom in merged and merged[geom] != offset:
                        raise ValueError(f'Conflicting native contact offset for {geom}')
                    merged[geom] = offset
                options[name] = merged
                continue
            if name in options and (type(options[name]) is not type(value) or options[name] != value):
                raise ValueError(f'Conflicting native equipment profile option: {name}')
            options[name] = value
    return options
