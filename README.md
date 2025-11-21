# QCBM-Hankel

## Running Experiment 1

The script is dependency-free and performs the full protocol:
* random MPS with per-site left normalisation;
* structured rank-1 (§7.2A) and rank-3 (§7.2B) controls;
* skip configurations where prefix/suffix counts would cap the rank below
  \(D^2\);
* record raw ranks plus normalised ratios and full-rank frequencies.

```bash
python experiments/exp1_mps_rank.py --random-samples 10 --structured-samples 2 --seed 42 --tol 1e-8 --output experiments/exp1_results.csv
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
python experiments/exp2_truncation.py --lengths 8,10,12 --bond-max 8 --bases 5 --epsilons 1e-12,1e-10,1e-8,1e-6 --depth-scale 1.0 --seed 0 --output experiments/exp2_results.csv
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
python experiments/exp3_noise_growth.py --lengths 5,10 --bond-dim 4 --bases 3 --epsilons 0.001,0.003,0.01 --max-prefixes 64 --seed 0 --output experiments/exp3_results.csv
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
python experiments/exp4_condition_numbers.py --lengths 6,10,14 --eta 0.6 --c-param 0.5 --sample-sizes 200,1000,5000 --trials 5 --seed 0 --output experiments/exp4_results.csv
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
python experiments/exp5_sample_complexity.py --lengths 8,10 --sample-sizes 1000,3000,10000 --models low,high,contractive --bond-low 2 --bond-high 4 --wfa-dim 3 --rank 4 --trials 5 --seed 0 --output experiments/exp5_results.csv
```

Per configuration `(model, length, sample_size, trial)` the script records true
rank, `gamma`, coherence `mu`, `kappa_B`, Hankel deviation `delta_hankel`,
empirical `sigma_min`, and downstream errors (TV and max) for both the raw WFA
and the contractive projection. Scaled errors (`scaled_error_* = error *
gamma / F(L)`) and the prefix/suffix/evaluation sizes are included to facilitate
the 5a/5b analyses (error vs. `N`, error vs. `L`) under both geometric and
contractive regimes.

## Running Experiment 6

Experiment 6 tests the multi-view stacking result from §8.3 by comparing
single-view Hankel conditioning to the jointly stacked Hankel across multiple
cut “views” of the same ground-truth MPS (all views share the same length).
The script reports true and empirical smallest singular values for each view
and for the joint stacked matrix, coherences, Hankel deviations, and the
resulting pointwise/TV reconstruction errors for both single-view and
multi-view whitening.

```bash
python experiments/exp6_multi_view.py --lengths 6,6,6 --cuts 2,3,4 --sample-sizes 500,2000 --bond-dim 3 --max-prefixes 128 --max-suffixes 128 --rank 4 --trials 3 --seed 0 --output experiments/exp6_results.csv
```

Key metrics per row:

- `mode=single_true`: per-view true `sigma_r`, coherence `mu`, and rank;
- `mode=single_emp`: empirical `sigma_r`, Hankel deviation, and max/TV errors
  for each view at a given sample size;
- `mode=joint_true/emp`: smallest singular value of the stacked joint Hankel
  (built from all views), its deviation from the true joint matrix, and the
  aggregated multi-view prediction errors (averaged over the available cuts).

These outputs allow a direct check of Theorem 8.7: the joint `sigma_r` should
exceed the single-view values, and the empirical joint conditioning and
pointwise errors stabilize with the same samples used per view.

## Running Experiment 7

Experiment 7 instantiates the mixed-state/POVM Hankel-rank test from §12.1,
following the expanded brief:

- MPDO core: shallow noisy-channel MPDO with bond dimension `chi_rho` (default
  {2,4}) and configurable depth.
- POVMs: either separable projective (`chi_M=1`) or shallow correlated
  networks (`chi_M>1`) with configurable depth/bond and noise.
- Lengths `L` in {6,8,10} (default) with mid-cut Hankel built from
  `P=Σ^{t*}` and `S=Σ^{L-t*}` where `t*=floor(L/2)`. Larger cuts ensure
  `|P|,|S| >= chi_rho*chi_M` so the rank cap is visible.
- Metrics: numerical rank with relative threshold, effective rank
  (`effective_rank`), relative effective ranks at `1e-2`/`1e-3`, and
  `rank_cap=min(|P|,|S|, chi_rho*chi_M)` plus a `within_cap` indicator.

```bash
python experiments/exp7_mpdo_povm.py --lengths 6,8,10 --chi-rho 2,4 --chi-m 1,2,4 --trials 30 --d 2 --mpdo-depth 3 --meas-depth 2 --noise 0.1 --tol-abs 1e-12 --tol-rel 1e-10 --seed 0 --output experiments/exp7_results.csv
```

CSV columns include the configuration, measured Hankel rank, bound
(`rank_upper`) and dimension cap (`rank_cap`), `within_cap` flag, effective
ranks (`effective_rank`, `r_eff_1e2`, `r_eff_1e3`), and thresholded singular
value summaries (`sv_max`, `sv_min_thresholded`, `threshold`).

## Running Experiment 8

Experiment 8 now follows the expanded brief: a Part-A warm-up on a single
probability amplitude and a Part-B Hankel test on small random MPS/QCBM models,
both contrasting classical frequency estimates with QAE-style estimates whose
variance shrinks as \(1/K^2\):

```bash
python experiments/exp8_qae_scaling.py --lengths 6,8 --d 2 --bond-dim 3 --sample-size 5000 --qae-rounds 1,2,4,8,16,32 --trials 20 --amplitude-probs 0.15,0.35 --amplitude-shots 1000 --amplitude-trials 500 --noise-scale 1.0 --bias-scale 0.0 --seed 0 --output experiments/exp8_results.csv
```

What gets logged:

- **Part A (single amplitude):** mean absolute and squared errors for classical
  vs. QAE-inspired estimates of fixed probabilities `p_true` (column `section`
  = `amplitude`). Errors versus `K` should trace the \(1/K\) slope.
- **Part B (Hankel):** true probabilities come from a random complex MPS with
  bond dimension `--bond-dim`; the mid-cut Hankel is compared against classical
  and QAE-style estimates. Each QAE entry noise has variance
  `noise_scale * p(1-p) / (N K^2)` plus an optional bias `bias_scale/K`. The CSV
  records spectral and Frobenius errors per `(length, trial, K)` with
  `section=hankel`.

Plotting `error_spec` or `error_fro` against `K` on log–log axes exhibits the
expected \(1/K\) decay of the leading term from Theorem 12.3, while the
amplitude warm-up provides a direct single-parameter illustration of the same
scaling.
