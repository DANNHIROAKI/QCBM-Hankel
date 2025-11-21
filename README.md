# QCBM-Hankel

This repository contains a standalone implementation of the expanded Experiment 1
plan: verifying how MPS bond dimension controls the Hankel rank of amplitude
and probability Hankel matrices.

## Running Experiment 1

The script is dependency-free and performs the full protocol:
* random MPS with per-site left normalisation;
* structured rank-1 (§7.2A) and rank-3 (§7.2B) controls;
* skip configurations where prefix/suffix counts would cap the rank below
  \(D^2\);
* record raw ranks plus normalised ratios and full-rank frequencies.

```bash
python experiments/exp1_mps_rank.py \
  --random-samples 10 \
  --structured-samples 2 \
  --seed 42 \
  --tol 1e-8 \
  --output experiments/exp1_results.csv
```

Key CLI options:

- `--random-samples`: random MPS per `(alphabet, length, bond_dim)` combination.
- `--structured-samples`: samples for each structured control (rank-1 and
  rank-3) per configuration.
- `--eta`: `eta` parameter for structured examples (defaults to `0.2`).
- `--c-param`: `c` parameter for the rank-3 construction (defaults to `0.6`).
- `--tol`: relative tolerance used in Gram–Schmidt rank computations.
- `--seed`: RNG seed for reproducibility.
- `--output`: CSV path for saving per-instance ranks.

After completion the script prints mean rank ratios and full-rank frequencies
for random MPS, and writes all per-instance metrics to the CSV file.
