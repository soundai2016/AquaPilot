#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
CHECKPOINT="${1:-}"
if [[ "$CHECKPOINT" == "--help" || "$CHECKPOINT" == "-h" ]]; then
  echo "Usage: $0 WATERMARK_CHECKPOINT [OUTPUT_DIR] [fine-tuning options]"
  exit 0
fi
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: $0 WATERMARK_CHECKPOINT [OUTPUT_DIR] [fine-tuning options]" >&2
  exit 2
fi
shift
OUTPUT=results/acommset/few_shot
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT="$1"
  shift
fi
"${PYTHON:-python3}" scripts/run_acommset_fewshot.py \
  --checkpoint "$CHECKPOINT" \
  --config configs/acommset_few_shot.yaml \
  --output "$OUTPUT" "$@"
