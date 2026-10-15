"""Resolve optional named contact skins while retaining source margins and rest surfaces."""

from math import isfinite

from backends.physics_options import validate_contact_offset_overrides


def validate_offset_pair(contact, rest):
    """Contact generation must begin outside the unchanged resting surface."""
    if not isfinite(contact) or not isfinite(rest) or contact <= 0 or rest >= contact:
        raise ValueError('Contact offset must be finite, positive, and greater than rest offset')
    return float(contact), float(rest)


def resolve_contact_offset_overrides(overrides, names, colliders, margins):
    """Reject unknown, ambiguous or inactive names before writing any USD shape."""
    result = {}
    for name, value in validate_contact_offset_overrides(overrides).items():
        matches = [index for index, candidate in enumerate(names) if candidate == name]
        if len(matches) != 1:
            raise ValueError(f'Contact offset override requires one source geometry named {name!r}')
        index = matches[0]
        if index not in colliders:
            raise ValueError(f'Contact offset override geometry {name!r} has no active collider')
        offset = max(value, float(margins[index]))
        validate_offset_pair(offset, 0.0)
        result[index] = offset
    return result
