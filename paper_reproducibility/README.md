# Paper reproducibility package

This folder contains the public, anonymized material needed to reproduce the
classification statistics reported in the manuscript from derived epoch-level
features. It does not contain raw XDF recordings, webcam video, eye-tracker
logs, absolute LSL timestamps, local paths, or private acquisition logs.

## Contents

- `data/features_visual_alpha_anonymized.csv`: visual cue-locked alpha features
  by trial/window.
- `data/features_auditory_erp_anonymized.csv`: auditory ERP features by
  event, restricted to the 250-600 ms window used in the final table.
- `results/table3_audit_final.csv`: frozen final audit table with BA, bootstrap
  CI and permutation p-values.
- `results/table3_compact_final.csv`: compact version of the same final
  results.
- `scripts/compute_table3_ba_ci_p.py`: recomputes BA, bootstrap CI and
  permutation p-values from the anonymized features.
- `scripts/extract_visual_alpha_features_from_xdf.py`: public extraction
  reference for visual alpha features from an XDF plus an anonymized cue table.
- `scripts/extract_auditory_erp_features_from_xdf.py`: public extraction
  reference for auditory ERP features from an XDF plus an anonymized event
  table.
- `scripts/make_reproducibility_figures.py`: regenerates the figures in this
  folder from the public CSVs.
- `scripts/exploratory_p300_xdawn_from_epochs.py`: optional exploratory
  xDAWN/OAS/Tangent pipeline from locally derived epoch arrays. It is not
  required to reproduce the final table.
- `requirements_xdawn_optional.txt`: optional dependencies for the xDAWN
  sensitivity script.
- `figures/`: code-generated reproducibility figures.

## Recompute final statistics

The final manuscript run used 10,000 permutations and 5,000 bootstrap
resamples. It can take substantial time on a laptop.

```powershell
python paper_reproducibility\scripts\compute_table3_ba_ci_p.py `
  --data-dir paper_reproducibility\data `
  --out-dir paper_reproducibility\results\recomputed `
  --permutations 10000 `
  --bootstrap 5000 `
  --seed 42
```

For a quick smoke test of the code path:

```powershell
python paper_reproducibility\scripts\compute_table3_ba_ci_p.py `
  --data-dir paper_reproducibility\data `
  --out-dir paper_reproducibility\results\smoke_test `
  --permutations 5 `
  --bootstrap 10 `
  --seed 42
```

The smoke-test p-values and intervals are not scientifically meaningful; they
only verify that the pipeline runs.

## Regenerate figures

```powershell
python paper_reproducibility\scripts\make_reproducibility_figures.py
```

## Notes on raw-feature extraction

The extraction scripts document the signal-processing choices used before the
public feature tables:

- visual alpha: 8-12 Hz bandpass, Hilbert power, baseline -0.8 to -0.2 s, main
  cue window 0.5 to 1.2 s;
- auditory ERP: 50 Hz notch, 0.1-15 Hz bandpass, baseline -0.2 to 0 s, main ERP
  window 250 to 600 ms, peak-to-peak rejection threshold 250 uV.

Raw recordings are private and therefore not included. A full raw-data rerun
requires locally available XDF files and anonymized event tables with the
schemas described in the extraction scripts.

## Optional xDAWN/OAS/Tangent sensitivity

The optional xDAWN script requires time-resolved epochs and MNE-Python. It is
documented separately in `docs/EXPLORATORY_P300_XDAWN.md` because those epochs
are derived from private recordings and are not part of the public release.
