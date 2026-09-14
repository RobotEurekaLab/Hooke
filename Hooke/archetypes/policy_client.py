"""A minimal client for talking to an openpi policy-inference server over
websocket, for driving Hooke's own MuJoCo rollouts with a served π0/π0.5
policy instead of a scripted expert.

Deliberately vendored (not `pip install openpi-client`) rather than
depending on the real package: `openpi-client`'s own `pyproject.toml`
pins `numpy>=1.22.4,<2.0.0`, and this project's `autobio` conda env
already runs numpy 2.4.6 (required by the rest of this codebase --
mujoco, jax). Installing the real package risked a resolver-driven
numpy downgrade that could break mujoco/jax across the whole
environment; not worth the risk for ~100 lines of wire-protocol code.
This file reproduces `websocket_client_policy.py` + `msgpack_numpy.py` +
`base_policy.py` from openpi's `packages/openpi-client`
(Apache License 2.0, https://github.com/Physical-Intelligence/openpi)
closely enough to stay wire-compatible with a real
`openpi.serving.websocket_policy_server.WebsocketPolicyServer` -- the
actual serialization format (msgpack with a numpy-array extension type)
and RPC shape (connect -> receive metadata -> send obs dict -> receive
action dict, repeat) are unchanged from upstream, only re-typed/trimmed
to this file's own use and given a couple of Hooke-specific docstrings.

This module makes no assumption about what's actually running behind
the websocket -- a real trained checkpoint, a base (non-fine-tuned) π0.5
checkpoint, or `mock_policy_server.py`'s dummy server used to validate
the rollout harness itself without any model or GPU at all.
"""
from __future__ import annotations

import functools
import logging
import time

import msgpack
import mujoco
import numpy as np
import websockets.sync.client
from PIL import Image


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class WebsocketClientPolicy:
    """Talks to a served policy over websocket: connect, read one metadata
    message, then `infer(obs) -> action` round trips for as long as the
    connection is open. See module docstring for provenance."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None,
                 connect_timeout: float | None = None):
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server(connect_timeout)

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float | None):
        logging.info(f"Waiting for server at {self._uri}...")
        start = time.time()
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                if timeout is not None and time.time() - start > timeout:
                    raise TimeoutError(f"No policy server responded at {self._uri} within {timeout}s") from None
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._ws.close()


class ActionChunkBroker:
    """Wraps a policy to return one action at a time from its returned
    chunk, only re-inferring once the chunk is exhausted -- vendored from
    `openpi_client.action_chunk_broker` (see module docstring), simplified
    to slice every ndarray-valued top-level key without needing the
    `dm-tree` dependency (this project only ever has one such key,
    `"actions"`, so a manual slice is equivalent and avoids adding a
    package purely for a generic tree-map)."""

    def __init__(self, policy: WebsocketClientPolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step = 0
        self._last_result: dict | None = None

    def infer(self, obs: dict) -> dict:
        if self._last_result is None:
            self._last_result = self._policy.infer(obs)
            self._cur_step = 0
        result = {
            k: (v[self._cur_step, ...] if isinstance(v, np.ndarray) else v)
            for k, v in self._last_result.items()
        }
        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_result = None
        return result

    def reset(self) -> None:
        self._policy.reset()
        self._last_result = None
        self._cur_step = 0


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Vendored from `openpi_client.image_tools` -- converts a float image
    (e.g. MuJoCo's own render output) to uint8, which is what a served
    policy expects and is far cheaper to send over the wire."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Vendored from `openpi_client.image_tools` -- resize-without-distortion
    (letterboxed with zero padding) to the model's expected input size."""
    if image.shape[:2] == (height, width):
        return image
    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)
    resized = np.array(Image.fromarray(image).resize((resized_width, resized_height), resample=Image.BILINEAR))
    padded = np.zeros((height, width, image.shape[2]), dtype=resized.dtype)
    pad_h = max(0, (height - resized_height) // 2)
    pad_w = max(0, (width - resized_width) // 2)
    padded[pad_h:pad_h + resized_height, pad_w:pad_w + resized_width] = resized
    return padded


class CameraRenderer:
    """Renders a task's named cameras into the `(C, H, W)` uint8, resized-
    and-padded format a served UR5 policy expects -- shared by
    `policy_rollout.py` (live inference) and `demo_export.py` (offline
    dataset export) so the two paths can't silently drift into rendering
    observations differently from each other, which would make exported
    demonstrations a poor match for what the policy sees at eval time."""

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self._cache: dict[int, mujoco.Renderer] = {}

    def _renderer_for(self, task) -> mujoco.Renderer:
        key = id(task.model)
        if key not in self._cache:
            self._cache[key] = mujoco.Renderer(task.model, 224, 224)
        return self._cache[key]

    def render(self, task, camera_name: str | None) -> np.ndarray:
        renderer = self._renderer_for(task)
        if camera_name:
            renderer.update_scene(task.data, camera=camera_name)
        else:
            renderer.update_scene(task.data)
        img = renderer.render()
        img = convert_to_uint8(img)
        img = resize_with_pad(img, self.img_size, self.img_size)
        return np.transpose(img, (2, 0, 1))  # HWC -> CHW, matching openpi's own convention

    def get_observation(self, task) -> dict:
        """`state`/`base_rgb`/`wrist_rgb`/`prompt`, matching
        `openpi/examples/ur5/README.md`'s `UR5Inputs` schema."""
        arm = task.arm
        joints = task.data.qpos[arm.jnt_span].copy()
        gripper = np.array([task.data.qpos[arm.gripper_jnt_adr]])
        cams = task.task_info["camera_mapping"]
        return {
            "state": np.concatenate([joints, gripper]).astype(np.float32),
            "base_rgb": self.render(task, cams.get("image")),
            "wrist_rgb": self.render(task, cams.get("wrist_image")),
            "prompt": task.task_info.get("prefix", task.task),
        }
