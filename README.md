# AquaPilot

**Uncertainty-Guided Adaptive Pilot Placement for Underwater Acoustic OFDM**  

AquaPilot studies adaptive pilot placement for underwater acoustic OFDM under a fixed pilot budget. The receiver predicts the complex channel frequency response from 16 previous soundings and uses the forecast to select pilot subcarriers. Evaluation uses replay of measured channels from the Watermark dataset.

## Pilot placement

Four strategies are evaluated at the same pilot count:

- **Comb:** rotating, regularly spaced pilots.
- **Random:** randomly selected pilot subcarriers.
- **Top-B:** subcarriers ranked by predicted pre-acquisition uncertainty.
- **A-optimal:** greedy pilot selection using an A-optimality criterion.

Within each experimental configuration, the strategies share a trained checkpoint and initial observations. Their subsequent observation histories differ according to the pilots selected.

## Experimental protocol

The Watermark experiment includes four held-out environments, three pilot budgets (8, 16, and 32 of 128 subcarriers), and three random seeds (17, 37, and 57): **36 training configurations** in total. All four placement strategies are evaluated for each configuration.

The original Watermark recordings are available from the [Norwegian Defence Research Establishment (FFI)](https://www.ffi.no/en/research/watermark). See [data/README.md](data/README.md) for data layout, preparation, and source information. Original recordings and trained checkpoints are not included in the release package.

## Installation and reproduction

```bash
git clone https://github.com/soundai2016/AquaPilot.git
cd AquaPilot
python -m pip install -e .

# Prepare and validate the Watermark data.
bash run.sh --prepare
bash run.sh --check-data --device cuda

# Inspect the experiment matrix, or launch training.
bash run.sh --dry-run
bash run.sh --device cuda --background --export-tables
```

Preparation writes to `data/prepared/watermark_inband/`. Experiment outputs and logs are stored in `results/watermark_inband/`. The training command requires a CUDA-enabled environment.

After training finishes, export the results:

```bash
python scripts/export_results.py
```

The exporter writes `results/AquaPilot_analysis.zip` and checks the available experiment evidence. Only complete, verified in-band experiments should be used for formal reporting; earlier and partial runs remain diagnostic.

## Evaluation and scope

- **Channel NMSE (dB):** mean of per-snapshot log ratios of squared complex estimation error to reference channel energy.
- **Channel NRMSE (%):** normalized channel-estimation error, **not** demodulated-symbol EVM.
- **Paired comparison:** a negative Top-B minus Comb difference favors Top-B.
- **Uncertainty ranking:** within-sounding Spearman correlation between predicted pre-acquisition variance and squared forecast error across subcarriers.

These evaluations concern **channel estimation**. Packet delivery and feedback delay are not measured by this protocol.

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## License

Released under the [MIT License](LICENSE).
