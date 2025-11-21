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

## Running Experiment 3

Experiment 3 evaluates length-dependent error growth under spectrally bounded
gate noise and contrasts raw QCBM dynamics with a row-substochastic projection
that enforces the contractive setting of Theorems 5.5/8.1.

```bash
python experiments/exp3_noise_growth.py \
  --lengths 5,10,15,20 \
  --bond-dim 4 \
  --bases 20 \
  --epsilons 0.001,0.003,0.01 \
  --max-prefixes 256 \
  --seed 0 \
  --output experiments/exp3_results.csv
```

Protocol highlights:
- Clean left-canonical MPS sampled per length with fixed bond dimension.
- Per-gate complex Gaussian perturbations rescaled to a chosen spectral norm
  `epsilon`.
- Hankel matrices built from sampled prefix/suffix sets (capped by
  `--max-prefixes`) using the exact probabilities of the selected sequences.
- Projection path: `B = A ⊗ conj(A)` → elementwise absolute value → per-row
  rescaling across symbols so that `Σσ Bσ 1 ≤ 1`, then surrogate probabilities
  from a fixed `(i, f)` pair.

The CSV reports spectral/Frobenius differences for both the raw and projected
models, indexed by `(length, base_index, epsilon)` along with the actual prefix
and suffix sample sizes.

## Running Experiment 4

Experiment 4 instantiates the analytic hard examples from §7.2A/§7.2B (rank-1
and rank-3 diagonal MPS), tracks Hankel singular values across lengths, and
simulates a Hankel spectral-learning pipeline (rank-truncated SVD + whitening +
masked transitions) on finite samples to expose how conditioning impacts sample
complexity.

```bash
python experiments/exp4_condition_numbers.py \
  --lengths 6,10,14 \
  --eta 0.6 \
  --c-param 0.5 \
  --sample-sizes 200,1000,5000 \
  --trials 5 \
  --seed 0 \
  --output experiments/exp4_results.csv
```

Outputs per `(model, length, sample_size, trial)` include the leading three
singular values (with `log_sigma3` for exponential-decay fits), spectral and
Frobenius errors of empirical Hankel estimates, the smallest singular value of
the empirical Hankel used for whitening, and downstream spectral-learning
metrics: total-variation and pointwise max error of the WFA-style reconstruction
over all length-`L` strings. The contrast between the rank-1 and rank-3 cases
highlights the `N \gtrsim \mu/\sigma_{\min}^2` dependence from §7.3.

## Running Experiment 5

Experiment 5 implements the end-to-end Hankel spectral-learning pipeline from
Chs. 9–11, sweeping sample sizes and sequence lengths while comparing raw
spectral recovery against a row-substochastic projection that enforces the
contractive regime (`G_L(κ)≈L`). The expanded version supports multiple ground
truths (low-rank MPS, higher-entropy MPS, and a contractive WFA), records
coherence/`gamma`/`kappa_B`, and reports scaled errors `gamma/F(L)` to match the
theoretical bound.

```bash
python experiments/exp5_sample_complexity.py \
  --lengths 8,10 \
  --sample-sizes 1000,3000,10000 \
  --models low,high,contractive \
  --bond-low 2 \
  --bond-high 4 \
  --wfa-dim 3 \
  --rank 4 \
  --trials 5 \
  --seed 0 \
  --output experiments/exp5_results.csv
```

Per configuration `(model, length, sample_size, trial)` the script records true
rank, `gamma`, coherence `mu`, `kappa_B`, Hankel deviation `delta_hankel`,
empirical `sigma_min`, and downstream errors (TV and max) for both the raw WFA
and the contractive projection. Scaled errors (`scaled_error_* = error *
gamma / F(L)`) and the prefix/suffix/evaluation sizes are included to facilitate
the 5a/5b analyses (error vs. `N`, error vs. `L`) under both geometric and
contractive regimes.
