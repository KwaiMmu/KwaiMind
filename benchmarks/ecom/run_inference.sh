#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT=""
DATA_ROOT="${KWAIMIND_ECOM_ROOT:-${ROOT_DIR}/../KwaiMind-Ecom-Bench}"
OUTPUT_ROOT="${ROOT_DIR}/outputs/ecom"
RANK=0
WORLD_SIZE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --rank) RANK="$2"; shift 2 ;;
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required" >&2
  exit 2
fi

MANIFEST_ARGS=()
for manifest in "$DATA_ROOT"/data/*.json; do
  MANIFEST_ARGS+=(--manifest "$manifest")
done

if [[ ${#MANIFEST_ARGS[@]} -eq 0 ]]; then
  echo "No manifests found under $DATA_ROOT/data" >&2
  exit 2
fi

# One Python process handles every manifest so the ~39 GiB checkpoint is loaded once
# instead of once per task. Each manifest writes into $OUTPUT_ROOT/<task>/.
python "$ROOT_DIR/examples/infer.py" \
  --checkpoint "$CHECKPOINT" \
  "${MANIFEST_ARGS[@]}" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --rank "$RANK" \
  --world-size "$WORLD_SIZE"
