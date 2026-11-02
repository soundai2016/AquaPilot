#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Forward explicit CLI options unchanged. With no arguments, use the benchmark settings.
if [[ $# -eq 0 ]]; then
  set -- --config configs/watermark.yaml --ranks 0 1 2 4 8 --seeds 17 37 57
fi
"${PYTHON:-python3}" -m aquapilot.cli rank-search "$@"
