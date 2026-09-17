"""Bounded primitive messages over a private, trusted local worker socket.

Binary framing retains float bits and avoids decimal contact serialization.
Marshal version 4 is shared by the pinned Python 3.10 and 3.12 runtimes. It is
used only for our own worker, never for uploaded files or remote connections.
"""

import json
import marshal
import struct

MAX_MESSAGE = 16 * 1024 * 1024


def write_message(stream, value, transport="binary"):
    if transport == "binary":
        payload = marshal.dumps(value, 4)
        prefix = struct.pack("!I", len(payload))
    elif transport == "json":
        payload = (json.dumps(value, allow_nan=False) + "\n").encode()
        prefix = b""
    else:
        raise ValueError("Unknown worker transport")
    if len(payload) > MAX_MESSAGE:
        raise ValueError("Worker message exceeds size limit")
    stream.write(prefix + payload)
    stream.flush()


def read_message(stream, transport="binary"):
    if transport == "json":
        payload = stream.readline(MAX_MESSAGE + 1)
        if not payload:
            return None
        if len(payload) > MAX_MESSAGE or not payload.endswith(b"\n"):
            raise ValueError("Oversized or interrupted worker message")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Worker message must be a dictionary")
        return value
    if transport != "binary":
        raise ValueError("Unknown worker transport")
    prefix = stream.read(4)
    if not prefix:
        return None
    if len(prefix) != 4:
        raise EOFError("Interrupted worker frame header")
    (size,) = struct.unpack("!I", prefix)
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid worker frame size")
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("Interrupted worker frame payload")
    value = marshal.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Worker message must be a dictionary")
    return value
