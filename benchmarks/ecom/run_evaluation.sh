#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${KWAIMIND_ECOM_ROOT:-${ROOT_DIR}/../KwaiMind-Ecom-Bench}"
OUTPUT_ROOT="${ROOT_DIR}/outputs/ecom"
THREADS=8
MODEL="gemini-2.5-pro"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for manifest in "$DATA_ROOT"/data/*.json; do
  task="$(basename "$manifest" .json)"
  python "$ROOT_DIR/benchmarks/ecom/evaluate.py" \
    --task-id "$task" \
    --manifest "$manifest" \
    --data-root "$DATA_ROOT" \
    --result-dir "$OUTPUT_ROOT/$task" \
    --judge-jsonl "$DATA_ROOT/judge/task_judge_prompts.jsonl" \
    --output "$OUTPUT_ROOT/scores/$task.json" \
    --threads "$THREADS" \
    --model "$MODEL"
done

python "$ROOT_DIR/benchmarks/ecom/summarize.py" --scores-dir "$OUTPUT_ROOT/scores" --output "$OUTPUT_ROOT/summary"
