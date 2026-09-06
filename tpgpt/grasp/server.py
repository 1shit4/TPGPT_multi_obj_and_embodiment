"""Talking to the external GraspGen-X server.

The server lives in the sibling project and runs in its own conda environment.
This module only checks on it and, if asked, starts it -- it is never started
implicitly, because a test that silently spawns a 2.6 GB process is a test that
will one day exhaust this machine's memory while someone is looking elsewhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from tpgpt.grasp.client import DEFAULT_HOST, DEFAULT_PORT, GraspGenClient

#: Sibling project holding the GraspGen-X checkout, weights and launch script.
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get(
        "GRASPGENX_PROJECT_ROOT", "/home/ishita/task_embod_aware_grasp/6dof_GraspMAS"
    )
)


def server_available(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Whether a server is reachable. Used to skip tests, never to fail them."""
    client = GraspGenClient(host=host, port=port, timeout_ms=3000)
    try:
        return client.is_alive()
    finally:
        client.close()


def server_status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """One-paragraph description of the server, for logs and CLI output."""
    client = GraspGenClient(host=host, port=port, timeout_ms=5000)
    try:
        if not client.is_alive():
            return (
                f"GraspGen-X server not reachable at tcp://{host}:{port}.\n"
                f"Start it with: {launch_command()}"
            )
        meta = client.metadata
        return (
            f"GraspGen-X server at tcp://{host}:{port}\n"
            f"  default gripper : {meta.get('default_gripper')}\n"
            f"  loaded grippers : {meta.get('loaded_grippers')}\n"
            f"  precision       : {meta.get('precision')}\n"
            f"  actions         : {meta.get('actions')}"
        )
    finally:
        client.close()


def launch_command(project_root: Path | None = None) -> str:
    """The shell command that starts the server."""
    root = project_root or DEFAULT_PROJECT_ROOT
    return f"{root / 'scripts' / 'run_server.sh'} --daemon"


def start_server(project_root: Path | None = None, wait_s: float = 120.0) -> bool:
    """Start the server as a daemon and wait for it to answer.

    Opt-in only. Loading 1.6 GB of weights takes a few seconds and the process
    then holds ~2.6 GB resident.

    Returns:
        True once the server responds; False if it never did.
    """
    import time

    root = project_root or DEFAULT_PROJECT_ROOT
    script = root / "scripts" / "run_server.sh"
    if not script.is_file():
        raise FileNotFoundError(
            f"no GraspGen-X launch script at {script}. Set GRASPGENX_PROJECT_ROOT "
            "to the checkout that provides it."
        )
    if shutil.which("bash") is None:  # pragma: no cover - defensive
        raise RuntimeError("bash is required to launch the GraspGen-X server")

    subprocess.run(["bash", str(script), "--daemon"], check=True, cwd=str(root))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if server_available():
            return True
        time.sleep(2.0)
    return False


if __name__ == "__main__":  # pragma: no cover - operational helper
    print(server_status())
