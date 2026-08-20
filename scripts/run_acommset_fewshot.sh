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
OUTPUT="${2:-results/acommset/few_shot}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: $0 WATERMARK_CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi
if [[ ! -d data/prepared/acommset_few_shot/train ]]; then
  echo "Prepared ACommSet few-shot data not found. Run prepare_acommset.py --split-mode continuous first." >&2
  exit 2
fi
python scripts/run_acommset_fewshot.py \
  --checkpoint "$CHECKPOINT" \
  --config configs/acommset_few_shot.yaml \
  --output "$OUTPUT"
