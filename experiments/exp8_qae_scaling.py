#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 8: QAE-style acceleration of Hankel estimation (Theorem 12.3).

This script follows the strengthened experimental brief:

- Part A (single-amplitude scaling):
  Fix one or more probabilities p, compare classical frequency estimation to a
  QAE-inspired estimator whose variance shrinks as ~ p(1-p) / (shots * K^2),
  with optional bias ~ bias_scale / K. We log mean absolute error and RMSE
  versus the "inner" query count K.

- Part B (full Hankel scaling on small MPS/QCBM-style models):
  Build a small random MPS distribution, enumerate the exact probabilities,
  construct a mid-cut Hankel H, and compare:
    * classical Hankel estimate from N sampled sequences; and
    * a QAE-style Hankel estimate where each probability p(x) is perturbed
      by zero-mean Gaussian noise with variance ~ p(1-p) / (N * K^2),
      plus optional bias ~ bias_scale / K.
  We record spectral and Frobenius errors versus K.

The QAE noise model here is deliberately synthetic:
it encodes the ideal variance reduction (1/K^2) and an optional 1/K bias
without implementing full quantum amplitude estimation. This isolates the
"entry-wise variance -> Hankel spectral error" scaling that Theorem 12.3
is about, without depending on a quantum simulator.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # Optional: used for binomial sampling & SVD when present.
    import numpy as np
except Exception:  # pragma: no cover - numpy may be absent.
    np = None

SequenceType = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Basic linear algebra utilities
# ---------------------------------------------------------------------------


def frobenius_norm(mat: Sequence[Sequence[float]]) -> float:
    """Return the Frobenius norm ||mat||_F."""
    return math.sqrt(sum(x * x for row in mat for x in row))


def matvec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    """Matrix-vector product for small dense matrices."""
    out: List[float] = []
    for row in mat:
        out.append(sum(a * b for a, b in zip(row, vec)))
    return out


def spectral_norm(mat: Sequence[Sequence[float]], tol: float = 1e-12) -> float:
    """Return the spectral norm (largest singular value) of a real matrix.

    If NumPy is available we use SVD; otherwise we fall back to a simple
    power iteration on A^T A. The latter is sufficient for the small Hankel
    matrices used here.
    """
    if np is not None:
        arr = np.asarray(mat, dtype=float)
        if arr.size == 0:
            return 0.0
        # Largest singular value.
        return float(np.linalg.svd(arr, compute_uv=False)[0])

    # Fallback: power iteration on A^T A.
    if not mat:
        return 0.0
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    if rows == 0 or cols == 0:
        return 0.0

    # Build A^T A explicitly (small-dimensional case).
    ata: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(cols)]
    for i in range(cols):
        for j in range(cols):
            ata[i][j] = sum(mat[r][i] * mat[r][j] for r in range(rows))

    # Random initial vector.
    v = [random.random() + tol for _ in range(cols)]
    nrm = math.sqrt(sum(x * x for x in v))
    if nrm == 0.0:
        return 0.0
    v = [x / nrm for x in v]

    # Power iteration.
    for _ in range(200):
        Av = matvec(ata, v)
        nrm = math.sqrt(sum(x * x for x in Av))
        if nrm < tol:
            break
        v = [x / nrm for x in Av]

    Av = matvec(ata, v)
    eigval = sum(v[i] * Av[i] for i in range(cols))
    return math.sqrt(max(eigval, 0.0))


# ---------------------------------------------------------------------------
# Distribution & Hankel helpers
# ---------------------------------------------------------------------------


def generate_sequences(length: int, alphabet: Sequence[int]) -> List[SequenceType]:
    """Enumerate all sequences of given length over the given alphabet."""
    return list(itertools.product(alphabet, repeat=length))


def renormalize(prob_map: Mapping[SequenceType, float]) -> Dict[SequenceType, float]:
    """Renormalise a probability map; if total=0, return all zeros."""
    total = float(sum(prob_map.values()))
    if total <= 0.0:
        return {k: 0.0 for k in prob_map}
    return {k: v / total for k, v in prob_map.items()}


def sample_random_mps_distribution(
    length: int,
    d: int,
    bond_dim: int,
) -> Dict[SequenceType, float]:
    """Generate a small random MPS-style distribution over strings of given length.

    Construction (deliberately simple and not canonical-form aware):

      - Draw complex Gaussian matrices A[sigma] ∈ C^{D×D} for each symbol.
      - Draw complex Gaussian boundary vectors alpha, beta ∈ C^D.
      - Define amplitudes psi(x) = alpha^T (Π_t A[x_t]) beta.
      - Set p(x) = |psi(x)|^2 and normalise over Σ^L.

    If the random draw collapses to total mass ~0, we fall back to the
    uniform distribution over Σ^L.
    """
    alphabet = list(range(d))

    # Random complex matrices A[sigma].
    mats = {
        sigma: [
            [random.gauss(0.0, 1.0) + 1j * random.gauss(0.0, 1.0) for _ in range(bond_dim)]
            for _ in range(bond_dim)
        ]
        for sigma in alphabet
    }
    alpha = [random.gauss(0.0, 1.0) + 1j * random.gauss(0.0, 1.0) for _ in range(bond_dim)]
    beta = [random.gauss(0.0, 1.0) + 1j * random.gauss(0.0, 1.0) for _ in range(bond_dim)]

    def amplitude(seq: SequenceType) -> complex:
        vec = alpha[:]
        for symbol in seq:
            mat = mats[symbol]
            # vec_new[j] = sum_k vec[k] * mat[k][j]
            vec = [
                sum(vec[k] * mat[k][j] for k in range(bond_dim))
                for j in range(bond_dim)
            ]
        return sum(vec[j] * beta[j] for j in range(bond_dim))

    prob_map: Dict[SequenceType, float] = {}
    total = 0.0
    for seq in generate_sequences(length, alphabet):
        amp = amplitude(seq)
        p = (amp.real * amp.real) + (amp.imag * amp.imag)
        prob_map[seq] = p
        total += p

    if total <= 0.0:
        # Degenerate case: fall back to uniform distribution.
        uniform = 1.0 / (d ** length)
        return {seq: uniform for seq in generate_sequences(length, alphabet)}

    return {seq: val / total for seq, val in prob_map.items()}


def build_hankel(
    prob_map: Mapping[SequenceType, float],
    length: int,
    d: int,
) -> List[List[float]]:
    """Construct the mid-cut Hankel H(u,v)=p(uv) with t* = floor(L/2)."""
    t_star = length // 2
    prefixes = generate_sequences(t_star, range(d))
    suffixes = generate_sequences(length - t_star, range(d))

    hankel: List[List[float]] = []
    for u in prefixes:
        row: List[float] = []
        for v in suffixes:
            row.append(float(prob_map[u + v]))
        hankel.append(row)
    return hankel


def sample_classical(
    prob_map: Mapping[SequenceType, float],
    sample_size: int,
) -> Dict[SequenceType, float]:
    """Draw i.i.d. samples from prob_map and return empirical frequencies."""
    sequences = list(prob_map.keys())
    probs = [float(prob_map[seq]) for seq in sequences]

    # Build cumulative CDF for inverse transform sampling.
    cumulative: List[float] = []
    total = 0.0
    for p in probs:
        total += p
        cumulative.append(total)

    counts = {seq: 0 for seq in sequences}
    for _ in range(sample_size):
        r = random.random() * total
        for seq, cdf in zip(sequences, cumulative):
            if r <= cdf:
                counts[seq] += 1
                break

    return {seq: counts[seq] / sample_size for seq in sequences}


# ---------------------------------------------------------------------------
# QAE-style noise models
# ---------------------------------------------------------------------------


def qae_amplitude_estimate(
    p_true: float,
    shots: int,
    K: int,
    noise_scale: float,
    bias_scale: float,
) -> float:
    """Single-amplitude QAE-inspired estimator (Part A).

    Model:
        p_hat = p_true + bias_scale / K + Normal(0, sigma^2),
        sigma^2 = noise_scale * max(p_true(1-p_true), 1e-12) / (shots * K^2).

    The result is clipped to [0,1].
    """
    base_var = max(p_true * (1.0 - p_true), 1e-12)
    variance = noise_scale * base_var / max(shots * K * K, 1)
    std = math.sqrt(variance)
    bias = bias_scale / max(K, 1)
    noisy = random.gauss(p_true + bias, std)
    return min(1.0, max(0.0, noisy))


def qae_distribution_estimate(
    prob_map: Mapping[SequenceType, float],
    sample_size: int,
    K: int,
    noise_scale: float,
    bias_scale: float,
) -> Dict[SequenceType, float]:
    """QAE-style estimator for a full discrete distribution (Part B).

    For each x, model:
        p_hat(x) = p(x) + bias_scale / K + Normal(0, sigma_x^2),
        sigma_x^2 = noise_scale * max(p(x)(1-p(x)), 1e-12) / (sample_size * K^2).

    After perturbation, the distribution is renormalised to sum to 1.
    This corresponds to an idealised QAE variance ~ 1/(N K^2) with optional
    bias term ~ 1/K.
    """
    estimates: Dict[SequenceType, float] = {}
    for seq, p_true in prob_map.items():
        base_var = max(float(p_true) * (1.0 - float(p_true)), 1e-12)
        variance = noise_scale * base_var / max(sample_size * K * K, 1)
        std = math.sqrt(variance)
        bias = bias_scale / max(K, 1)
        noisy = random.gauss(float(p_true) + bias, std)
        # Clip to [0,1] before renormalisation.
        estimates[seq] = min(1.0, max(0.0, noisy))

    return renormalize(estimates)


# ---------------------------------------------------------------------------
# Experiment logic
# ---------------------------------------------------------------------------


def parse_comma_ints(text: str) -> List[int]:
    """Parse '1,2,4' style strings into a list of ints."""
    return [int(x) for x in text.split(",") if x.strip()]


def run_amplitude_part(
    p_values: Sequence[float],
    classical_shots: int,
    trials: int,
    qae_rounds: Sequence[int],
    noise_scale: float,
    bias_scale: float,
) -> List[Dict[str, object]]:
    """Part A: single-amplitude scaling vs. K.

    For each p_true and each trial:
      - Draw a classical frequency estimate from classical_shots Bernoulli(p_true).
      - For each K in qae_rounds, draw a QAE-style estimate via qae_amplitude_estimate.

    Returns a list of rows ready to be written to CSV.
    """
    rows: List[Dict[str, object]] = []

    for p_true in p_values:
        for _ in range(trials):
            # Classical baseline: frequency estimate from Bernoulli trials.
            if np is not None:
                classical_counts = int(np.random.binomial(classical_shots, p_true))
            else:
                classical_counts = sum(
                    1 for _ in range(classical_shots) if random.random() < p_true
                )
            p_classical = classical_counts / classical_shots

            rows.append(
                {
                    "section": "amplitude",
                    "p_true": p_true,
                    "method": "classical",
                    "K": 0,
                    "abs_error": abs(p_classical - p_true),
                    "sq_error": (p_classical - p_true) ** 2,
                    "length": None,
                    "d": None,
                    "bond_dim": None,
                    "trial": None,
                    "sample_size": classical_shots,
                    "error_spec": None,
                    "error_fro": None,
                }
            )

            # QAE-style estimates for each K.
            for K in qae_rounds:
                p_qae = qae_amplitude_estimate(
                    p_true=p_true,
                    shots=classical_shots,
                    K=K,
                    noise_scale=noise_scale,
                    bias_scale=bias_scale,
                )
                rows.append(
                    {
                        "section": "amplitude",
                        "p_true": p_true,
                        "method": "qae",
                        "K": K,
                        "abs_error": abs(p_qae - p_true),
                        "sq_error": (p_qae - p_true) ** 2,
                        "length": None,
                        "d": None,
                        "bond_dim": None,
                        "trial": None,
                        "sample_size": classical_shots,
                        "error_spec": None,
                        "error_fro": None,
                    }
                )

    return rows


def run_single_hankel_trial(
    length: int,
    d: int,
    bond_dim: int,
    sample_size: int,
    qae_rounds: Sequence[int],
    noise_scale: float,
    bias_scale: float,
) -> Tuple[Tuple[float, float], Dict[int, Tuple[float, float]]]:
    """One Part B trial: random MPS model + Hankel errors.

    Returns:
      - classical_err = (spec_norm_diff, frob_norm_diff) for the classical estimator;
      - qae_errs = {K: (spec_norm_diff, frob_norm_diff)} for each K in qae_rounds.
    """
    # Ground-truth distribution and Hankel.
    prob_map = sample_random_mps_distribution(length=length, d=d, bond_dim=bond_dim)
    hankel_true = build_hankel(prob_map, length=length, d=d)

    def hankel_errors(est_map: Mapping[SequenceType, float]) -> Tuple[float, float]:
        """Compute spectral and Frobenius errors between H(est_map) and H_true."""
        hankel_est = build_hankel(est_map, length=length, d=d)
        diff = [
            [hankel_est[i][j] - hankel_true[i][j] for j in range(len(hankel_true[0]))]
            for i in range(len(hankel_true))
        ]
        return spectral_norm(diff), frobenius_norm(diff)

    # Classical Hankel from N sampled sequences.
    classical_map = sample_classical(prob_map, sample_size=sample_size)
    classical_err = hankel_errors(classical_map)

    # QAE-style Hankel for each K: add synthetic QAE noise to the true distribution.
    qae_errs: Dict[int, Tuple[float, float]] = {}
    for K in qae_rounds:
        qae_map = qae_distribution_estimate(
            prob_map=prob_map,
            sample_size=sample_size,
            K=K,
            noise_scale=noise_scale,
            bias_scale=bias_scale,
        )
        qae_errs[K] = hankel_errors(qae_map)

    return classical_err, qae_errs


def run_hankel_part(
    lengths: Sequence[int],
    d: int,
    bond_dim: int,
    sample_size: int,
    trials: int,
    qae_rounds: Sequence[int],
    noise_scale: float,
    bias_scale: float,
) -> List[Dict[str, object]]:
    """Part B: full Hankel scaling on random MPS models."""
    rows: List[Dict[str, object]] = []

    for length in lengths:
        for trial_idx in range(trials):
            classical_err, qae_errs = run_single_hankel_trial(
                length=length,
                d=d,
                bond_dim=bond_dim,
                sample_size=sample_size,
                qae_rounds=qae_rounds,
                noise_scale=noise_scale,
                bias_scale=bias_scale,
            )

            # Classical baseline row (K=0).
            rows.append(
                {
                    "section": "hankel",
                    "length": length,
                    "d": d,
                    "bond_dim": bond_dim,
                    "trial": trial_idx,
                    "sample_size": sample_size,
                    "method": "classical",
                    "K": 0,
                    "error_spec": classical_err[0],
                    "error_fro": classical_err[1],
                    "p_true": None,
                    "abs_error": None,
                    "sq_error": None,
                }
            )

            # QAE-style rows for each K.
            for K in qae_rounds:
                spec_err, fro_err = qae_errs[K]
                rows.append(
                    {
                        "section": "hankel",
                        "length": length,
                        "d": d,
                        "bond_dim": bond_dim,
                        "trial": trial_idx,
                        "sample_size": sample_size,
                        "method": "qae",
                        "K": K,
                        "error_spec": spec_err,
                        "error_fro": fro_err,
                        "p_true": None,
                        "abs_error": None,
                        "sq_error": None,
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 8: QAE-style scaling for Hankel estimation (Theorem 12.3)"
    )
    # Hankel (Part B) configuration.
    parser.add_argument(
        "--lengths",
        type=str,
        default="6,8",
        help="Comma-separated sequence lengths for Hankel experiments (e.g. '6,8').",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=2,
        help="Alphabet size for the MPS/QCBM model (default: 2).",
    )
    parser.add_argument(
        "--bond-dim",
        type=int,
        default=3,
        help="Bond dimension for random MPS distributions (default: 3).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Classical outer sample size N for Hankel baselines (default: 5000).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of random Hankel trials per length (default: 20).",
    )
    parser.add_argument(
        "--qae-rounds",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated QAE query counts K (default: '1,2,4,8,16,32').",
    )

    # Amplitude (Part A) configuration.
    parser.add_argument(
        "--amplitude-probs",
        type=str,
        default="0.15,0.35",
        help="Comma-separated probabilities for Part-A warm-up, e.g. '0.15,0.35'.",
    )
    parser.add_argument(
        "--amplitude-shots",
        type=int,
        default=1000,
        help="Classical shots per amplitude in Part A (default: 1000).",
    )
    parser.add_argument(
        "--amplitude-trials",
        type=int,
        default=500,
        help="Number of Part-A trials per probability (default: 500).",
    )

    # QAE noise model hyperparameters.
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Variance prefactor for QAE noise model (default: 1.0).",
    )
    parser.add_argument(
        "--bias-scale",
        type=float,
        default=0.0,
        help="Optional bias term scaled as bias_scale/K (default: 0.0).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp8_results.csv",
        help="CSV output path combining Part A and Part B (default: experiments/exp8_results.csv).",
    )

    args = parser.parse_args()

    # Set seeds for reproducibility.
    random.seed(args.seed)
    if np is not None:
        np.random.seed(args.seed)

    lengths = parse_comma_ints(args.lengths)
    qae_rounds = parse_comma_ints(args.qae_rounds)
    p_values = [float(x) for x in args.amplitude_probs.split(",") if x.strip()]

    all_rows: List[Dict[str, object]] = []

    # ----------------------
    # Part A: amplitude warm-up
    # ----------------------
    all_rows.extend(
        run_amplitude_part(
            p_values=p_values,
            classical_shots=args.amplitude_shots,
            trials=args.amplitude_trials,
            qae_rounds=qae_rounds,
            noise_scale=args.noise_scale,
            bias_scale=args.bias_scale,
        )
    )

    # ----------------------
    # Part B: full Hankel scaling
    # ----------------------
    all_rows.extend(
        run_hankel_part(
            lengths=lengths,
            d=args.d,
            bond_dim=args.bond_dim,
            sample_size=args.sample_size,
            trials=args.trials,
            qae_rounds=qae_rounds,
            noise_scale=args.noise_scale,
            bias_scale=args.bias_scale,
        )
    )

    # CSV schema: share a unified header for both Part A & Part B.
    fieldnames = [
        "section",      # 'amplitude' or 'hankel'
        "length",       # sequence length (Part B); None for Part A
        "d",            # alphabet size (Part B); None for Part A
        "bond_dim",     # MPS bond dimension (Part B); None for Part A
        "trial",        # Hankel trial index (Part B); None for Part A
        "sample_size",  # N for Hankel, shots for amplitude
        "method",       # 'classical' or 'qae'
        "K",            # QAE query count; 0 for classical baselines
        "error_spec",   # Hankel spectral error (Part B); None for Part A
        "error_fro",    # Hankel Frobenius error (Part B); None for Part A
        "p_true",       # true probability (Part A); None for Part B
        "abs_error",    # |p_hat - p_true| (Part A); None for Part B
        "sq_error",     # (p_hat - p_true)^2 (Part A); None for Part B
    ]

    # Write combined CSV.
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[Experiment 8] Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()



"""
python experiments/exp8_qae_scaling.py --lengths 6,8 --d 2 --bond-dim 3 --sample-size 5000 --qae-rounds 1,2,4,8,16,32 --trials 20 --amplitude-probs 0.15,0.35 --amplitude-shots 1000 --amplitude-trials 500 --noise-scale 1.0 --bias-scale 0.0 --seed 0 --output experiments/exp8_results.csv
"""