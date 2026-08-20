# Data

Raw datasets are not bundled.

```text
data/
  raw/
    watermark/WatermarkV1/
    acommset/
  prepared/
    watermark/fold_test_NOF/
    watermark/fold_test_NCS/
    watermark/fold_test_BCH/
    watermark/fold_test_KAU/
    acommset_zero_shot/
    acommset_few_shot/
```

WatermarkV1: https://www.ffi.no/en/research/watermark

ACommSet access: https://github.com/cpslab-rutgers/ACommSet (release agreement required).

Use `scripts/prepare_watermark_loeo.py` for Watermark. For ACommSet, inspect an example file with `scripts/inspect_mat.py`, then run `scripts/prepare_acommset.py` with the correct CSI key and time/frequency axes for the approved release.
