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

## Running Experiment 2

Experiment 2 (expanded) follows the full truncation protocol in the plan:
random brickwork Haar circuits produce base states, TT-SVD truncations cap the
bond dimension, and the script tracks tail energies, state and Hankel errors,
and effective ranks at theory-driven tolerances.

```bash
python experiments/exp2_truncation.py \
  --lengths 8,10,12 \
  --bond-max 8 \
  --bases 5 \
  --epsilons 1e-12,1e-10,1e-8,1e-6 \
  --depth-scale 1.0 \
  --seed 0 \
  --output experiments/exp2_results.csv
```

Key outputs per `(length, base_index, D_eff)`:

- Tail energy sum and norm, plus theoretical bound \(\Delta_{\text{th}}=2\sqrt{\sum_t\epsilon_t^2}\);
- State error \(\delta_\psi=\|\psi-\tilde\psi\|_2\);
- Hankel Frobenius and spectral differences using a fixed cut at \(t_*=\lfloor L/2\rfloor\);
- Effective ranks at user tolerances and at \(\varepsilon\in\{\Delta_{\text{th}},\,\|H-\tilde H\|_2\}\),
  together with success flags for \(\operatorname{rank}_\varepsilon(H)\le D_\text{eff}^2\).
