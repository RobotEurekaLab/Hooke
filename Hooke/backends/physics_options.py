"""Validate optional archived native settings without importing a simulator."""
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real


_INTEGERS = {
    'native_substeps': (1, 32),
    'solver_position_iterations': (1, 255),
    'solver_velocity_iterations': (1, 255),
    'simulation_threads': (1, 64),
}
_BOOLEANS = {
    'explicit_affine_actuators',
    'external_forces_every_iteration', 'soft_connect_constraints',
    'enable_stabilization',
    'approximate_cylinders', 'compound_cylinder_colliders',
}
_POSITIVE = {'contact_offset_m', 'sdf_contact_offset_m'}
_NONNEGATIVE = {'compliant_contact_scale'}
_FRACTIONS = {'compliant_impedance_fraction', 'connect_impedance_fraction'}
_ALLOWED = set(_INTEGERS) | _BOOLEANS | _POSITIVE | _NONNEGATIVE | _FRACTIONS | {'solver_type', 'world_origin_m', 'contact_offset_overrides_m'}


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f'{name} must be a finite real number') from error
    if not isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return value


def validate_contact_offset_overrides(value):
    """Normalize optional source-geometry contact skins without changing rest surfaces."""
    if not isinstance(value, Mapping):
        raise ValueError('Contact offset overrides must map source geometry names to metres')
    result = {}
    for name, offset in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Contact offset override geometry names must be nonempty strings')
        offset = _finite_real(offset, f'Contact offset for {name}')
        if offset <= 0:
            raise ValueError(f'Contact offset for {name} must be positive')
        result[name] = offset
    return result


def validate_physics_options(options):
    """Return normalized supplied options; never add defaults or mutate input."""
    if not isinstance(options, Mapping):
        raise ValueError('Native physics options must be a mapping')
    if any(not isinstance(key, str) for key in options):
        raise ValueError('Native physics option names must be strings')
    unknown = set(options) - _ALLOWED
    if unknown:
        raise ValueError(f'Unknown native physics options: {sorted(unknown)}')
    result = {}
    for name, value in options.items():
        if name in _INTEGERS:
            lower, upper = _INTEGERS[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer from {lower} to {upper}')
            result[name] = int(value)
        elif name in _BOOLEANS:
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a boolean')
            result[name] = value
        elif name == 'contact_offset_overrides_m':
            result[name] = validate_contact_offset_overrides(value)
        elif name == 'solver_type':
            if value not in ('TGS', 'PGS'):
                raise ValueError('solver_type must be TGS or PGS')
            result[name] = value
        elif name == 'world_origin_m':
            if isinstance(value, (str, bytes, Mapping, set, frozenset)):
                raise ValueError('world_origin_m must contain three finite coordinates')
            try:
                coordinates = list(value)
            except TypeError as error:
                raise ValueError('world_origin_m must contain three finite coordinates') from error
            if len(coordinates) != 3:
                raise ValueError('world_origin_m must contain three finite coordinates')
            result[name] = [_finite_real(item, name) for item in coordinates]
        else:
            value = _finite_real(value, name)
            if (name in _POSITIVE and value <= 0
                    or name in _NONNEGATIVE and value < 0
                    or name in _FRACTIONS and not 0 <= value <= 1):
                raise ValueError(f'{name} is outside its supported range')
            result[name] = value
    return result


def merge_native_physics_options(metadata, overrides=None):
    """Explicit caller choices override validated archive profile options."""
    if not isinstance(metadata, Mapping):
        raise ValueError('Source metadata must be a mapping')
    archived = validate_physics_options(metadata.get('native_physics_options', {}))
    supplied = validate_physics_options({} if overrides is None else overrides)
    return {**archived, **supplied}
