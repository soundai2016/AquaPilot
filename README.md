# AquaPilot

## Setup

```bash
cd /data/AquaPilot
conda activate aqua
python -m pip install -e .
```

## Data

Place the original Watermark data under `data/raw/`. Keep the original files unchanged. Prepared folds are written to `data/prepared/`.

```bash
cd /data/AquaPilot
bash run.sh --prepare
```

## Train and evaluate

Run the four leave-one-environment-out folds at the 32/128 pilot budget with seeds 17, 37, and 57:

```bash
cd /data/AquaPilot
bash run.sh --quarter-budget --device cuda --background --skip-existing
```

Training and evaluation artifacts are stored under `results/watermark_quarter/`.

## Export results

```bash
cd /data/AquaPilot
python scripts/export_quarter.py \
  --results results/watermark_quarter/aquapilot \
  --axes results/axes.json
```

The export produces `results/AquaPilot_quarter_results.zip`, including the paired comparison CSV, run manifest, and report.
