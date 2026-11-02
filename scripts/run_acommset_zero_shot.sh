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
  echo "Usage: $0 WATERMARK_CHECKPOINT [OUTPUT_DIR] [evaluate options]"
  exit 0
fi
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: $0 WATERMARK_CHECKPOINT [OUTPUT_DIR] [evaluate options]" >&2
  exit 2
fi
shift
OUTPUT=results/acommset/zero_shot
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT="$1"
  shift
fi
"${PYTHON:-python3}" -m aquapilot.cli evaluate \
  --config configs/acommset_zero_shot.yaml \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" "$@"
