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
OUTPUT="${2:-results/acommset/zero_shot}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: $0 WATERMARK_CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi
if [[ ! -d data/prepared/acommset_zero_shot/test ]]; then
  echo "Prepared ACommSet zero-shot test data not found. Run prepare_acommset.py --split-mode all-test first." >&2
  exit 2
fi
python -m aquapilot.cli evaluate \
  --config configs/acommset_zero_shot.yaml \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT"
