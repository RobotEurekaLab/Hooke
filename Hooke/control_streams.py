"""Advance independent control streams against one shared physics clock."""
from collections.abc import Callable, Iterator, Mapping
from numbers import Integral

import numpy as np

ControlStream = Iterator[Mapping[int, float]]


def run_control_streams(control: np.ndarray, step: Callable[[], None], *streams: ControlStream):
    """Merge one command from each live stream, then advance physics once.

    An empty command holds the previous target. Streams may observe the state
    from the preceding physics tick; they never advance physics themselves.
    """
    active = list(streams)
    owners = {}
    try:
        while active:
            commands = {}
            remaining = []
            for stream in active:
                try:
                    command = next(stream)
                except StopIteration:
                    continue
                if commands.keys() & command.keys():
                    raise ValueError('Concurrent streams command the same actuator')
                for actuator in command:
                    if owners.setdefault(actuator, id(stream)) != id(stream):
                        raise ValueError('Concurrent streams command the same actuator')
                commands.update(command)
                remaining.append(stream)
            active = remaining
            if active:
                for actuator, value in commands.items():
                    if (not isinstance(actuator, Integral) or isinstance(actuator, bool)
                            or not 0 <= actuator < len(control) or not np.isfinite(value)):
                        raise ValueError('Invalid actuator command')
                for actuator, value in commands.items():
                    control[actuator] = value
                step()
    finally:
        for stream in streams:
            close = getattr(stream, 'close', None)
            if close is not None:
                close()
