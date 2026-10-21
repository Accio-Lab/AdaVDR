#!/usr/bin/env bash
# Start one foreground server. Never stop another process.
set -euo pipefail
: "${MODEL:?Set MODEL to your local Hugging Face model directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_VIDEO_FETCH_TIMEOUT="${VLLM_VIDEO_FETCH_TIMEOUT:-1200}"
extra_args=()
if [[ -n "${GDN_PREFILL_BACKEND-triton}" ]]; then
  extra_args+=(--gdn-prefill-backend "${GDN_PREFILL_BACKEND-triton}")
fi
exec "${SERVE_PYTHON:-python}" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name "${MODEL_NAME:-eval-model}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
  --max-num-seqs 8 --max-num-batched-tokens 32768 \
  --limit-mm-per-prompt '{"video":20,"image":20}' \
  --media-io-kwargs '{"video":{"num_frames":256,"fps":2}}' \
  --allowed-local-media-path "${MEDIA_ROOT:-$PWD}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --trust-remote-code --enable-prefix-caching --enforce-eager \
  --compilation-config '{"mode":0}' "${extra_args[@]}" "$@"
