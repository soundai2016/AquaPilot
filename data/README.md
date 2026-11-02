# Data

## Watermark

The training input is `data/prepared/watermark_inband/`. Its `folds.json` defines four held-out environments. Each `fold_test_*` directory contains `index.json` and one NPZ file per recording in `train/`, `val/` or `test/`. Channel frequency responses have shape `[snapshots, 128]`, stored as complex `csi` or as `csi_real` and `csi_imag`.

| Test environment | Training groups (records) | Validation group (records) | Test records |
| --- | --- | --- | ---: |
| NOF | BCH + KAU (36) | NCS (60) | 60 |
| NCS | NOF + KAU (92) | BCH (4) | 60 |
| BCH | NOF + NCS (120) | KAU (32) | 4 |
| KAU | BCH + NCS (64) | NOF (60) | 32 |

KAU1 and KAU2 contribute 16 records each and stay in the same split. The 624 NPZ files are four fold copies of 156 source records.

Source: [FFI Watermark](https://www.ffi.no/en/research/watermark). The rows of `h` are consecutive soundings; its columns are delay taps. The 60 NCS1 matrices have shape `[1024, 510]`. The converter uses 128 uniformly spaced tones on the inner 80% of the published probe band. The nominal probe bands are 10–18 kHz for NOF/NCS, 32.5–37.5 kHz for BCH and 4–8 kHz for KAU; the corresponding tone spacings are 50, 31.25 and 25 Hz.

From the project root, run `bash run.sh --prepare`. It reads the original MAT files in `data/raw/watermark/WatermarkV1/Watermark/input/channels/`, creates `data/prepared/watermark_inband/`, and independently checks the complex responses at the 128 physical frequencies. It writes the verification record to `results/watermark_inband/axes.json`. If the prepared directory already exists, its files are checked in place. The previous `data/prepared/watermark/` and `results/watermark/` are left in place. No preparation is needed to export an existing run.

The check binds the prepared split files to the results by their ordered fingerprints. After 16 initialization soundings per record, all four test sets together have 85,008 scored snapshots per seed and budget: 14,520 NOF, 60,480 NCS, 2,264 BCH and 7,744 KAU.

