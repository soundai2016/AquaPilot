#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
if [[ ! -f data/prepared/watermark/folds.json ]]; then
  echo "Prepared Watermark LOEO folds not found. Run scripts/download_watermark.py and scripts/prepare_watermark_loeo.py first." >&2
  exit 2
fi
python scripts/run_watermark_loeo.py "$@"
