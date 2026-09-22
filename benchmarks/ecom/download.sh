#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${KWAIMIND_ECOM_ROOT:-${ROOT_DIR}/../KwaiMind-Ecom-Bench}"
REPO_ID="laziji402/Kwai-Ecom-Bench"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$DATA_ROOT"
huggingface-cli download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$DATA_ROOT" \
  --include 'README.md' \
  --include 'DATA_LICENSE' \
  --include 'data/*' \
  --include 'images/*' \
  --include 'judge/*'

[[ -d "$DATA_ROOT/data" ]] || { echo "Missing data directory: $DATA_ROOT/data" >&2; exit 1; }
[[ -d "$DATA_ROOT/images" ]] || { echo "Missing images directory: $DATA_ROOT/images" >&2; exit 1; }
[[ -f "$DATA_ROOT/judge/task_judge_prompts.jsonl" ]] || { echo "Missing judge prompts" >&2; exit 1; }
printf 'E-com Bench downloaded to %s\n' "$DATA_ROOT"
