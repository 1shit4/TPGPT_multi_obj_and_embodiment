#!/usr/bin/env bash
# GraspGen-X grasp server for tpgpt_planner_node, on the Thor's GPU.
#
#   run_graspgenx_server.sh            # foreground
#   run_graspgenx_server.sh --daemon   # background, log in ~/.ros/graspgenx_server.log
#   run_graspgenx_server.sh --stop
#   run_graspgenx_server.sh --status
#
# One-time setup (already done on this machine):
#   python3 -m venv ~/venvs/graspgenx
#   ~/venvs/graspgenx/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#   ~/venvs/graspgenx/bin/pip install "numpy==1.26.4" omegaconf hydra-core "trimesh==4.5.3" h5py \
#       scikit-learn scipy imageio webdataset "yourdfpy==0.0.56" "diffusers==0.11.1" \
#       "huggingface-hub==0.25.2" "timm==1.0.15" addict pyyaml tqdm tensordict matplotlib \
#       pyzmq msgpack msgpack-numpy "setuptools<78" torch-geometric
#   ~/venvs/graspgenx/bin/pip install --no-deps -e $GRASP_ROOT/GraspGenX
#   checkpoints  -> $GRASP_ROOT/assets/graspgenx_checkpoints/release/{gen,dis}  (HF adithyamurali/GraspGenXModel)
#   grippers     -> $GRASP_ROOT/assets/gripper_descriptions                    (HF dataset adithyamurali/gripper_descriptions, git lfs pull)

set -euo pipefail

GRASP_ROOT="${GRASP_ROOT:-$HOME/thor_ws/src/ycb_point_cloud/6DOF_MultiEmbod_Grasp}"
VENV="${GRASPGENX_VENV:-$HOME/venvs/graspgenx}"
PORT="${GRASPGEN_SERVER_PORT:-5556}"
HOST="${GRASPGEN_SERVER_HOST:-127.0.0.1}"
DEVICE="${GRASPGENX_DEVICE:-cuda}"
LOG="${HOME}/.ros/graspgenx_server.log"
SERVER_PY="${GRASP_ROOT}/GraspGenX/client-server/graspgenx_server.py"
PATTERN="graspgenx_server.py.*--port ${PORT}"

stop_server() {
  if pkill -f "${PATTERN}" 2>/dev/null; then
    echo "Stopping GraspGen-X server on port ${PORT} ..."
    for _ in $(seq 1 20); do pgrep -f "${PATTERN}" >/dev/null || break; sleep 0.5; done
    pkill -9 -f "${PATTERN}" 2>/dev/null || true
  fi
}

case "${1:-}" in
  --stop) stop_server; exit 0 ;;
  --status)
    if pgrep -f "${PATTERN}" >/dev/null; then echo "running (pid $(pgrep -f "${PATTERN}" | tr '\n' ' '))";
    else echo "not running"; fi
    exit 0 ;;
esac

for required in "${VENV}/bin/python" "${SERVER_PY}" \
    "${GRASP_ROOT}/assets/graspgenx_checkpoints/release/gen" \
    "${GRASP_ROOT}/assets/gripper_descriptions/gripper_descriptions/assets/x_grippers/franka_panda/config.json"; do
  [[ -e "${required}" ]] || { echo "missing: ${required} (see the setup notes in $0)" >&2; exit 1; }
done

stop_server   # never two servers on one port

# A sourced ROS workspace puts Jazzy's site-packages on PYTHONPATH; keep them out
# of the GraspGen-X venv (different numpy/torch expectations).
CMD=(env -u PYTHONPATH
     GRASPGENX_GRIPPER_CFG_DIR="${GRASP_ROOT}/assets/gripper_descriptions"
     GRASPGENX_CHECKPOINT_DIR="${GRASP_ROOT}/assets/graspgenx_checkpoints"
     "${VENV}/bin/python" "${SERVER_PY}"
     --config "${GRASP_ROOT}/assets/graspgenx_checkpoints/release"
     --assets_dir "${GRASP_ROOT}/GraspGenX/assets"
     --default_gripper franka_panda
     --host "${HOST}" --port "${PORT}" --device "${DEVICE}")

if [[ "${1:-}" == "--daemon" ]]; then
  mkdir -p "$(dirname "${LOG}")"
  nohup "${CMD[@]}" >"${LOG}" 2>&1 &
  echo "GraspGen-X server starting on tcp://${HOST}:${PORT} (${DEVICE}), log: ${LOG}"
  for _ in $(seq 1 60); do
    grep -q "listening on" "${LOG}" 2>/dev/null && { echo "ready"; exit 0; }
    pgrep -f "${PATTERN}" >/dev/null || { echo "server exited; see ${LOG}" >&2; tail -20 "${LOG}" >&2; exit 1; }
    sleep 1
  done
  echo "still loading after 60 s; check ${LOG}"
else
  exec "${CMD[@]}"
fi
