#!/usr/bin/env bash
# Evaluate an existing model service; never manage inference processes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    exec "${EVAL_PYTHON:-python}" -m vdr_eval.runner "$@"
  fi
done
: "${INPUT_FILE:?Set INPUT_FILE to QA annotations (.parquet, .jsonl, or .json)}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE to a new predictions.jsonl path}"
export EVAL_MEDIA_FILE_URI=1
export SEARCH_TOP_K="${SEARCH_TOP_K:-6}"
export VISIT_PAGE_CACHE_SUMMARY="${VISIT_PAGE_CACHE_SUMMARY:-1}"
video_args=()
model_args=()
if [[ -n "${VIDEO_DIR:-}" ]]; then
  video_args+=(--video-dir "$VIDEO_DIR")
fi
if [[ -n "${MODEL_NAME:-}" ]]; then
  model_args+=(--model-name "$MODEL_NAME")
fi
if [[ -n "${API_BASE:-}" ]]; then
  model_args+=(--api-base "$API_BASE")
fi
exec "${EVAL_PYTHON:-python}" -m vdr_eval.runner \
  --input "$INPUT_FILE" "${video_args[@]}" \
  --protocol "${PROTOCOL_FILE:-$ROOT/protocols/protocol.jsonl}" \
  "${model_args[@]}" \
  --output-file "$OUTPUT_FILE" --max-turns 20 --concurrency "${CONCURRENCY:-1}" \
  --seed 3407 --max-new-tokens 6144 --temperature 0.7 --top-p 1.0 --top-k -1 \
  --request-timeout 600 --request-retries 5 \
  --fps 2 --min-frames 4 --max-frames 256 \
  --min-pixels 32768 --max-pixels 65536 --total-pixels 16777216 "$@"
