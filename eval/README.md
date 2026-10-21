# Video Research Evaluation

Multi-turn video QA evaluation with grounding, image search, web search, and page summaries.

```text
scripts/          # Evaluation, model serving, and judging entrypoints
src/vdr_eval/     # Runner, tools, judge, and configuration loader
configs/          # Blank service configuration template
protocols/        # System prompt and tool schemas
```

## Setup

```bash
pip install -r requirements.txt
```

Install `ffmpeg` and `ffprobe` for video tools. Model serving requires a separate GPU environment with vLLM 0.27.1.

Copy the configuration template outside the public code directory:

```bash
mkdir -p ../.local
cp -n configs/services.example.yaml ../.local/services.yaml
chmod 600 ../.local/services.yaml
```

Fill in your inference, search, OSS, page-summary, and judge services. Real addresses and API keys belong only in this private YAML, which is ignored by Git. Environment variables override YAML values.

The default configuration path is `../.local/services.yaml`. To use another file:

```bash
export VDR_SERVICES_CONFIG=/path/to/private/services.yaml
```

## Evaluate

Input supports Parquet, JSONL, or a JSON array. Each row needs `uid` (or `id`), `question`, `answer`, and either a local `video_path` or a `video_id` matching a file in `VIDEO_DIR`. Videos are not bundled; the model server must also be able to read them.

```bash
INPUT_FILE=/path/to/benchmark.parquet \
VIDEO_DIR=/path/to/videos \
OUTPUT_FILE=/path/to/results/predictions.jsonl \
bash scripts/eval.sh
```

Use `bash scripts/eval.sh --resume` to continue an existing output. Defaults: temperature 0.7, top-p 1.0, top-k -1, thinking disabled, 6,144 tool-loop generation tokens, up to 20 tool turns, and 256 video frames at 2 fps. A final-answer fallback is enabled by default.

To start your own inference service:

```bash
MODEL=/path/to/model MODEL_NAME=eval-model \
MEDIA_ROOT=/path/to/videos CUDA_VISIBLE_DEVICES=0 \
bash scripts/serve.sh
```

Set `inference.model` in the YAML to the same served model name. The evaluation entrypoint only sends requests; it does not manage model processes.

## Judge

```bash
bash scripts/judge.sh \
  --input-dir /path/to/results \
  --output-dir /path/to/judgments \
  --expected-samples 250
```

The input directory should contain `predictions*.jsonl` without duplicate merged/sharded copies. Semantic accuracy is saved in the judge output's `summary.json`; prediction-level `correct` uses exact string matching.

Run `bash scripts/eval.sh --help` or `bash scripts/judge.sh --help` for options. Code provenance is recorded in `SOURCE_PROVENANCE.json`.
