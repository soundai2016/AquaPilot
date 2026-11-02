#!/usr/bin/env bash
set -euo pipefail

AQUAPILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AQUAPILOT_ROOT"
export PYTHONPATH="$AQUAPILOT_ROOT/src:$AQUAPILOT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
AQUAPILOT_PYTHON="${PYTHON:-python}"

exec "$AQUAPILOT_PYTHON" -u scripts/run_watermark_loeo.py "$@"
