#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --upgrade uv
fi

# A partially created .venv causes uv to abort before it can repair anything.
if [ -d .venv ] && { [ ! -f .venv/pyvenv.cfg ] || [ ! -x .venv/bin/python ]; }; then
  rm -rf .venv
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MODEL="${MODEL:-k2-fsa/OmniVoice}"
RUNNER="${RUNNER:-hybrid}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"

uv sync --frozen --python "$PYTHON_VERSION"

cmd=(
  .venv/bin/python
  -m
  omnivoice_triton.cli.api_server
  --model "$MODEL"
  --runner "$RUNNER"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --ip "$HOST"
  --port "$PORT"
)

if [ "${NO_ASR:-1}" = "1" ]; then
  cmd+=(--no-asr)
fi

if [ "${ALLOW_LAZY_ASR:-0}" = "1" ]; then
  cmd+=(--allow-lazy-asr)
fi

if [ "${SAGE_ATTENTION:-0}" = "1" ]; then
  cmd+=(--sage-attention)
fi

if [ "${FULL_TRITON_PATCH:-0}" = "1" ]; then
  cmd+=(--full-triton-patch)
fi

if [ -n "${ROOT_PATH:-}" ]; then
  cmd+=(--root-path "$ROOT_PATH")
fi

if [ -n "${SAVE_DIR:-}" ]; then
  cmd+=(--save-dir "$SAVE_DIR")
fi

if [ -n "${BATCH_COLLECT_MS:-}" ]; then
  cmd+=(--batch-collect-ms "$BATCH_COLLECT_MS")
fi

if [ -n "${MAX_BATCH_REQUESTS:-}" ]; then
  cmd+=(--max-batch-requests "$MAX_BATCH_REQUESTS")
fi

if [ -n "${MAX_PENDING_REQUESTS:-}" ]; then
  cmd+=(--max-pending-requests "$MAX_PENDING_REQUESTS")
fi

if [ -n "${MAX_PENDING_GPU_TASKS:-}" ]; then
  cmd+=(--max-pending-gpu-tasks "$MAX_PENDING_GPU_TASKS")
fi

if [ -n "${MAX_BATCH_TARGET_TOKENS:-}" ]; then
  cmd+=(--max-batch-target-tokens "$MAX_BATCH_TARGET_TOKENS")
fi

if [ -n "${MAX_BATCH_CONDITIONING_TOKENS:-}" ]; then
  cmd+=(--max-batch-conditioning-tokens "$MAX_BATCH_CONDITIONING_TOKENS")
fi

if [ -n "${MAX_BATCH_PADDING_RATIO:-}" ]; then
  cmd+=(--max-batch-padding-ratio "$MAX_BATCH_PADDING_RATIO")
fi

if [ -n "${MAX_CUDA_GRAPHS:-}" ]; then
  cmd+=(--max-cuda-graphs "$MAX_CUDA_GRAPHS")
fi

if [ -n "${PREWARM_CLONE_BATCH_SIZES:-}" ]; then
  cmd+=(--prewarm-clone-batch-sizes "$PREWARM_CLONE_BATCH_SIZES")
fi

if [ -n "${PREWARM_CLONE_SEQUENCE_LENGTHS:-}" ]; then
  cmd+=(--prewarm-clone-sequence-lengths "$PREWARM_CLONE_SEQUENCE_LENGTHS")
fi

if [ "$#" -gt 0 ]; then
  cmd+=("$@")
fi

exec "${cmd[@]}"
