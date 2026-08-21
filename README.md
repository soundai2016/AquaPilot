# AquaPilot

Code for **AquaPilot: Uncertainty-Guided Adaptive Pilot Placement for Underwater Acoustic OFDM** (WUWNet 2026 short paper).

Repository: https://github.com/soundai2016/AquaPilot

## Setup

```bash
python -m pip install -e .
pytest -q
bash scripts/run_smoke.sh
```

## Watermark: paper matrix

WatermarkV1 is available from https://www.ffi.no/en/research/watermark.

```bash
python scripts/download_watermark.py
python scripts/prepare_watermark_loeo.py \
  --input data/raw/watermark/WatermarkV1 \
  --output-root data/prepared/watermark

bash scripts/run_watermark.sh \
  --pilot-ratios 0.0625 0.125 0.25 \
  --seeds 17 37 57 \
  --skip-existing

python scripts/summarize_watermark.py
```

The reported matrix contains 4 held-out environments x 3 pilot fractions x 3 seeds = 36 configurations. Each configuration evaluates `comb`, `random`, `uncertainty` (top-B), and `active` (A-optimal). The main setup uses 128 subcarriers, history 16, hidden size 64, rank 4, at most 45 epochs, observation-noise fraction 0.03, covariance length scale 5, and preferred pilot spacing 3.

`closed_*_coverage90` is post-update component-wise 90% interval coverage. It is not a direct calibration metric for the pre-selection prior.

## ACommSet: external transfer path

ACommSet contains OFDM channel measurements from at-sea experiments. Access is managed by the dataset authors at https://github.com/cpslab-rutgers/ACommSet and requires their release agreement. Keep the approved raw archive under `data/raw/acommset/`; do not redistribute it.

Inspect one file first because release layouts and CSI variable names can differ:

```bash
python scripts/inspect_mat.py data/raw/acommset/<file>.mat
```

Prepare zero-shot evaluation data:

```bash
python scripts/prepare_acommset.py \
  --input data/raw/acommset \
  --output data/prepared/acommset_zero_shot \
  --key <CSI_KEY> --time-axis <T_AXIS> --frequency-axis <F_AXIS> \
  --split-mode all-test --n-subcarriers 128

bash scripts/run_acommset_zero_shot.sh <WATERMARK_CHECKPOINT>
```

Prepare chronological 10% adaptation data:

```bash
python scripts/prepare_acommset.py \
  --input data/raw/acommset \
  --output data/prepared/acommset_few_shot \
  --key <CSI_KEY> --time-axis <T_AXIS> --frequency-axis <F_AXIS> \
  --split-mode continuous --train-ratio 0.10 --val-ratio 0.10 \
  --guard-frames 16 --n-subcarriers 128

bash scripts/run_acommset_fewshot.sh <WATERMARK_CHECKPOINT>
```

ACommSet is supported for external zero-shot/few-shot checks but is not part of the 36-run Watermark table in the paper.
