# Optional exploratory xDAWN/OAS/Tangent analysis

This repository keeps the manuscript's main reproducibility path separate from
the optional P300-oriented sensitivity analysis.

The final public table can be recomputed from anonymized epoch-level feature
CSVs and does not require MNE-Python. The exploratory xDAWN analysis requires
time-resolved epochs, so the raw recordings or private derived epoch arrays are
needed locally. Those files are not distributed in this repository.

## What the optional pipeline tests

The optional script `paper_reproducibility/scripts/exploratory_p300_xdawn_from_epochs.py`
accepts an NPZ file with:

- `epochs`: array `(n_epochs, n_channels, n_times)`;
- `times`: time in seconds around each auditory event;
- `labels`: binary labels, 0/1;
- `groups`: optional block or run identifiers;
- `channel_names`: optional channel labels.

It compares:

- tabular window features plus balanced logistic regression;
- xDAWN plus flattened epoch features plus balanced logistic regression;
- xDAWN plus OAS covariance, tangent-space features and balanced logistic
  regression.

If `groups` are provided, the recommended validation is leave-one-group-out,
where each fold leaves out a full block/run. Permutations are performed within
groups so that the number of events per block is preserved.

## Install optional dependencies

```powershell
python -m pip install -r paper_reproducibility\requirements_xdawn_optional.txt
```

## Example command

```powershell
python paper_reproducibility\scripts\exploratory_p300_xdawn_from_epochs.py `
  --epochs-npz outputs\local_private_epochs\audio2_principal_epochs.npz `
  --out-dir outputs\xdawn_audio2_principal `
  --validation leave_one_group_out `
  --permutations 1000 `
  --seed 42
```

This analysis is exploratory. It should not replace the main table unless the
manuscript is explicitly reframed around that pipeline and the corresponding
epoch data can be shared or audited under an appropriate privacy agreement.
