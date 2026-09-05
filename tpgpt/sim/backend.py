"""Simulation backend: environment construction, seeding and rendering.

Centralises the things that are easy to get wrong once and then never notice:

* **Seeding happens before reset**, via robosuite's own ``seed`` argument, so the
  object placement and goal slot are actually reproducible. The prototype called
  ``np.random.seed`` *after* ``env.reset()``, which had no effect on either.
* **The GL backend is pinned to EGL.** ``osmesa`` cannot bind on this machine.
* **Frames are not accumulated in memory.** A long episode of RGB frames will
  exhaust the available RAM, so :class:`FrameWriter` streams them to disk.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

DEFAULT_CONTROL_FREQ = 20
DEFAULT_CAMERA = "frontview"
DEFAULT_RENDER_SIZE = 256


def configure_gl(backend: str = "egl") -> None:
    """Pin MuJoCo's rendering backend. Call before importing renderers."""
    os.environ.setdefault("MUJOCO_GL", backend)


def make_reshelving_env(
    embodiment: str = "panda",
    control_freq: int = DEFAULT_CONTROL_FREQ,
    seed: int | None = 0,
    impedance_control: bool = True,
    offscreen: bool = False,
    render_size: int = DEFAULT_RENDER_SIZE,
    camera: str = DEFAULT_CAMERA,
    **env_kwargs,
):
    """Build the reshelving environment (paper Sec. V-A).

    Args:
        embodiment: Key into :data:`tpgpt.sim.embodiments.EMBODIMENTS`.
        control_freq: Control rate in Hz. Also the rate label timestamps assume.
        seed: Environment seed, applied before the first reset.
        impedance_control: Drive the arm by joint torque so the Cartesian
            impedance controller can apply full stiffness matrices. When
            ``False`` the default OSC pose controller is used instead.
        offscreen: Enable offscreen rendering and camera observations.
        render_size: Square render resolution. Kept small by default; memory on
            this machine is tight.
        camera: Camera name for offscreen rendering.

    Returns:
        A :class:`~tpgpt.sim.scenes.reshelving.Reshelving` environment.
    """
    configure_gl()
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.controllers.cartesian_impedance import make_torque_controller_config
    from tpgpt.sim.embodiments import get_embodiment
    from tpgpt.sim.scenes.reshelving import Reshelving

    emb = get_embodiment(embodiment)
    config = load_composite_controller_config(controller="BASIC", robot=emb.robot)
    if impedance_control:
        config = make_torque_controller_config(config)

    kwargs = dict(
        controller_configs=config,
        control_freq=control_freq,
        seed=seed,
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        use_camera_obs=offscreen,
        camera_names=camera if offscreen else None,
        camera_heights=render_size,
        camera_widths=render_size,
    )
    kwargs.update(emb.as_make_kwargs())
    kwargs.update(env_kwargs)
    return Reshelving(**kwargs)


@contextmanager
def managed_env(*args, **kwargs):
    """Environment context manager that always closes.

    MuJoCo contexts are not reclaimed promptly on garbage collection, so an
    unclosed environment is a real leak on a memory-constrained machine.
    """
    env = make_reshelving_env(*args, **kwargs)
    try:
        yield env
    finally:
        env.close()


class FrameWriter:
    """Stream rendered frames to an mp4 instead of holding them in memory."""

    def __init__(self, path: str | Path, fps: int = DEFAULT_CONTROL_FREQ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self._writer = None
        self.n_frames = 0

    def append(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        if self._writer is None:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(self.path, fps=self.fps, macro_block_size=1)
        # robosuite returns camera images bottom-up.
        self._writer.append_data(np.ascontiguousarray(frame[::-1]))
        self.n_frames += 1

    def close(self) -> Path | None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            return self.path
        return None

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def observation_frame(obs: dict, camera: str = DEFAULT_CAMERA) -> np.ndarray | None:
    """Pull a camera image out of an observation dict, if one is present."""
    return obs.get(f"{camera}_image")
