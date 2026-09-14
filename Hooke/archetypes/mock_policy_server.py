"""A dummy policy server implementing the exact same websocket wire
protocol as `openpi.serving.websocket_policy_server.WebsocketPolicyServer`
(see `archetypes/policy_client.py`'s module docstring), so
`archetypes/policy_rollout.py`'s harness can be validated end-to-end --
observation rendering, the websocket round trip, action-chunk unpacking,
applying actions to `data.ctrl` -- without any trained checkpoint, GPU,
or the openpi environment at all.

This is a real, useful testing tool, not a stand-in for anything that
should eventually be deleted: it's what proves the *harness* is correct
before ever pointing it at a real served policy, and gives a fast local
loop for iterating on the harness without waiting on model inference.

Usage:
    python -m archetypes.mock_policy_server --port 8000 --action_dim 7 --action_horizon 10

Then point `PolicyRolloutExpert` at host="localhost", port=8000.
"""
from __future__ import annotations

import argparse
import asyncio
import logging

import msgpack
import numpy as np
import websockets.asyncio.server as _server

from archetypes.policy_client import _pack_array, _unpack_array

logger = logging.getLogger(__name__)


def _make_dummy_action(obs: dict, action_dim: int, action_horizon: int) -> np.ndarray:
    """Holds the current joint state steady (zero delta on the arm, gripper
    unchanged) -- a deliberately inert policy. The point of this server is
    to exercise the wire protocol and the harness's own control loop, not
    to produce a policy that does anything useful; a rollout against this
    server should show the arm staying essentially still, not moving
    toward any task goal."""
    state = obs.get("state")
    if state is not None and len(state) >= action_dim:
        base = np.asarray(state[:action_dim], dtype=np.float32)
    else:
        base = np.zeros(action_dim, dtype=np.float32)
    return np.tile(base, (action_horizon, 1))


class MockPolicy:
    def __init__(self, action_dim: int, action_horizon: int):
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def infer(self, obs: dict) -> dict:
        return {"actions": _make_dummy_action(obs, self.action_dim, self.action_horizon)}


async def _handler(websocket, policy: MockPolicy):
    packer = msgpack.Packer(default=_pack_array)
    await websocket.send(packer.pack({"mock": True, "action_dim": policy.action_dim}))
    while True:
        try:
            obs = msgpack.unpackb(await websocket.recv(), object_hook=_unpack_array)
            action = policy.infer(obs)
            await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            logger.info("client disconnected")
            break


async def _run(host: str, port: int, action_dim: int, action_horizon: int):
    policy = MockPolicy(action_dim, action_horizon)
    async with _server.serve(lambda ws: _handler(ws, policy), host, port, compression=None, max_size=None) as server:
        logger.info(f"mock policy server listening on {host}:{port}")
        await server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_horizon", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(_run(args.host, args.port, args.action_dim, args.action_horizon))
