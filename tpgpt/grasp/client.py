"""ZMQ client for a GraspGen-X inference server.

GraspGen-X cannot run in this process. It needs Python 3.11 with
``diffusers==0.11.1`` and ``huggingface-hub==0.25.2``, while TPGPT runs on
Python 3.10 with MuJoCo and robosuite; the pins are mutually exclusive. The
model is also 1.6 GB and takes ~3 s to load, so reloading it per call would
dominate the runtime. Both problems are solved the same way: GraspGen-X runs as
a long-lived server in its own conda environment and we talk to it over ZMQ.

The wire protocol is re-implemented here rather than imported from
``graspgenx.serving.zmq_client``. That module is itself torch-free, but
importing it executes ``graspgenx/serving/__init__.py``, which pulls in
``zmq_server`` -> ``grasp_server`` -> torch. This file needs only ``pyzmq``,
``msgpack``, ``msgpack-numpy`` and ``numpy``.

Adapted from the equivalent client in the sibling project at
``/home/ishita/task_embod_aware_grasp/6dof_GraspMAS/GraspMAS/graspgen/client.py``.
Protocol reference: ``GraspGenX/client-server/README.md``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

DEFAULT_HOST = os.environ.get("GRASPGEN_SERVER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("GRASPGEN_SERVER_PORT", "5556"))

#: Server-side planners.
#:
#: ``diffusion`` is the learned sampler alone. ``graspmoe`` adds an analytic
#: oriented-bounding-box branch and scores both with the same discriminator; it
#: returns more candidates but is known to include poses approaching from the
#: unobserved back of the object, which is a matter for the filtering
#: discussion rather than something to paper over here.
PLANNERS = ("diffusion", "graspmoe")

#: CPU inference measured at 4-12 s for 50-200 samples, so the default timeout
#: is generous. A short timeout here looks exactly like a dead server.
DEFAULT_TIMEOUT_MS = 180_000


class GraspGenUnavailable(RuntimeError):
    """The server is unreachable, timed out, or returned an error.

    One exception type so callers can degrade to "no grasps" rather than crash
    on a transient socket problem.
    """


class GraspGenClient:
    """Synchronous REQ client. One socket, reconnected after a timeout.

    A ZMQ REQ socket that has timed out is stuck in a bad state and must be
    closed rather than reused, which is why :meth:`_request` drops the socket
    on failure.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._metadata: Optional[dict] = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if self._socket is not None:
            return
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.address)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "GraspGenClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- request
    def _request(self, payload: dict) -> dict:
        self.connect()
        try:
            self._socket.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._socket.recv()
        except zmq.Again as exc:
            self.close()
            raise GraspGenUnavailable(
                f"No reply from {self.address} within {self.timeout_ms} ms. "
                "Is the GraspGen-X server running? See tpgpt.grasp.server."
            ) from exc
        except zmq.ZMQError as exc:
            self.close()
            raise GraspGenUnavailable(f"ZMQ error talking to {self.address}: {exc}") from exc

        try:
            response = msgpack.unpackb(raw, raw=False)
        except Exception as exc:  # pragma: no cover - malformed server reply
            self.close()
            raise GraspGenUnavailable(f"Malformed reply from {self.address}: {exc}") from exc

        if not isinstance(response, dict):
            raise GraspGenUnavailable(f"Expected a dict reply, got {type(response)}")
        if response.get("error"):
            raise GraspGenUnavailable(f"Server error: {response['error']}")
        return response

    # -------------------------------------------------------------- actions
    def health(self) -> dict:
        """Server liveness. Raises :class:`GraspGenUnavailable` if unreachable."""
        return self._request({"action": "health"})

    def is_alive(self) -> bool:
        """Non-raising liveness check, for skipping tests."""
        try:
            return self.health().get("status") == "ok"
        except GraspGenUnavailable:
            return False

    @property
    def metadata(self) -> dict:
        """Server metadata, cached. Includes the grippers it has loaded."""
        if self._metadata is None:
            self._metadata = self._request({"action": "metadata"})
        return self._metadata

    def infer(
        self,
        point_cloud: np.ndarray,
        gripper_name: str | None = None,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        planner: str = "diffusion",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate grasps for one object.

        Args:
            point_cloud: ``(N, 3)`` object points **in metres**. Grasps are
                returned in this cloud's frame, so pass a world-frame cloud to
                get world-frame grasps.
            gripper_name: A GraspGen-X gripper, e.g. ``"franka_panda"``. ``None``
                uses the server's default.
            num_grasps: Samples drawn before thresholding.
            grasp_threshold: Discriminator cutoff; ``-1.0`` disables it and
                falls back to top-k.
            topk_num_grasps: Cap on returned grasps; ``-1`` keeps everything.
            planner: One of :data:`PLANNERS`.

        Returns:
            ``(poses, scores)`` of shape ``(K, 4, 4)`` and ``(K,)``. The poses
            are anchored at the **gripper base** with ``+Z`` the approach axis
            and ``+X`` the closing direction. They are **not sorted** -- the
            server concatenates per-iteration and per-branch results.
        """
        cloud = np.ascontiguousarray(np.asarray(point_cloud, dtype=np.float32))
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3); got {cloud.shape}")
        if len(cloud) == 0:
            raise ValueError("point_cloud is empty")
        if not np.isfinite(cloud).all():
            raise ValueError("point_cloud contains non-finite values")
        if planner not in PLANNERS:
            raise ValueError(f"planner must be one of {PLANNERS}; got {planner!r}")

        payload = {
            "action": "infer",
            "point_cloud": cloud,
            "num_grasps": int(num_grasps),
            "grasp_threshold": float(grasp_threshold),
            "topk_num_grasps": int(topk_num_grasps),
            "planner": planner,
        }
        if gripper_name is not None:
            payload["gripper_name"] = str(gripper_name)

        response = self._request(payload)
        poses = np.asarray(response.get("grasps"), dtype=np.float64)
        scores = np.asarray(response.get("confidences"), dtype=np.float64)
        if poses.size == 0:
            return np.zeros((0, 4, 4)), np.zeros(0)
        poses = poses.reshape(-1, 4, 4)
        if len(poses) != len(scores):
            raise GraspGenUnavailable(
                f"Server returned {len(poses)} grasps but {len(scores)} scores"
            )
        return poses, scores
