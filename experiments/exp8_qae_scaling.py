"""Experiment 8: QAE-style acceleration of Hankel estimation (Theorem 12.3).

This expanded script implements both parts of the experimental brief:

* **Part A (single-amplitude scaling):** fix one or more probabilities ``p``
  and compare classical frequency estimation against a QAE-inspired estimator
  whose noise shrinks as ``1/K``. We log mean absolute error and RMSE versus the
  query count ``K``.
* **Part B (full Hankel scaling):** build a small random MPS/QCBM-style model,
  enumerate the induced distribution, construct the mid-cut Hankel, and compare
  classical Hankel estimation to a QAE-style estimator with per-entry variance
  ``~ p(1-p)/(N*K^2)``. We report spectral and Frobenius errors versus ``K``.

The noise model is deliberately lightweight: it captures the ``1/K`` variance
reduction and optional bias terms without requiring a full quantum simulator.
NumPy is optional; a power-iteration fallback is provided for the spectral norm.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # Optional: used for spectral norms when present.
    import numpy as np
except Exception:  # pragma: no cover - numpy may be absent.
    np = None

SequenceType = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Small linear algebra utilities
# ---------------------------------------------------------------------------


def frobenius_norm(mat: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(x * x for row in mat for x in row))


def matvec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    out: List[float] = []
    for row in mat:
        out.append(sum(a * b for a, b in zip(row, vec)))
    return out


def spectral_norm(mat: Sequence[Sequence[float]], tol: float = 1e-12) -> float:
    if np is not None:
        arr = np.asarray(mat, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.linalg.svd(arr, compute_uv=False)[0])

    # Power iteration on A^T A to extract the largest singular value.
    if not mat:
        return 0.0
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    if rows == 0 or cols == 0:
        return 0.0
    ata: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(cols)]
    for i in range(cols):
        for j in range(cols):
            ata[i][j] = sum(mat[r][i] * mat[r][j] for r in range(rows))
    v = [random.random() + tol for _ in range(cols)]
    # Normalise v
    nrm = math.sqrt(sum(x * x for x in v))
    if nrm == 0:
        return 0.0
    v = [x / nrm for x in v]
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
# Distribution and Hankel helpers
# ---------------------------------------------------------------------------


def generate_sequences(length: int, alphabet: Sequence[int]) -> List[SequenceType]:
    return list(itertools.product(alphabet, repeat=length))


def sample_distribution(sequences: Sequence[SequenceType]) -> Dict[SequenceType, float]:
    weights = [random.random() + 1e-9 for _ in sequences]
    total = sum(weights)
    probs = [w / total for w in weights]
    return {seq: p for seq, p in zip(sequences, probs)}


def sample_random_mps_distribution(
    length: int, d: int, bond_dim: int
) -> Dict[SequenceType, float]:
    """Generate a small random MPS distribution over strings of length ``length``.

    The construction is intentionally simple: draw complex random matrices
    ``A[sigma]`` with shape (D, D), random boundary vectors, compute amplitudes
    ``alpha^T prod A[x_t] beta``, and take squared magnitude to form
    probabilities. The result is normalised over all ``d^length`` strings.
    """

    alphabet = list(range(d))
    mats = {
        sigma: [
            [random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)]
            for _ in range(bond_dim)
        ]
        for sigma in alphabet
    }
    alpha = [random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)]
    beta = [random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)]

    def amplitude(seq: SequenceType) -> complex:
        vec = alpha[:]
        for symbol in seq:
            mat = mats[symbol]
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
    if total == 0:
        # Fall back to a uniform distribution if the random draw collapses.
        uniform = 1.0 / (d ** length)
        return {seq: uniform for seq in generate_sequences(length, alphabet)}
    return {seq: val / total for seq, val in prob_map.items()}


def renormalize(prob_map: Dict[SequenceType, float]) -> Dict[SequenceType, float]:
    total = sum(prob_map.values())
    if total == 0:
        return {k: 0.0 for k in prob_map}
    return {k: v / total for k, v in prob_map.items()}


def build_hankel(prob_map: Dict[SequenceType, float], length: int, d: int) -> List[List[float]]:
    t_star = length // 2
    prefixes = generate_sequences(t_star, range(d))
    suffixes = generate_sequences(length - t_star, range(d))
    hankel: List[List[float]] = []
    for u in prefixes:
        row: List[float] = []
        for v in suffixes:
            row.append(prob_map[u + v])
        hankel.append(row)
    return hankel


def sample_classical(prob_map: Dict[SequenceType, float], sample_size: int) -> Dict[SequenceType, float]:
    sequences = list(prob_map.keys())
    cumulative: List[float] = []
    total = 0.0
    for seq in sequences:
        total += prob_map[seq]
        cumulative.append(total)
    counts = {seq: 0 for seq in sequences}
    for _ in range(sample_size):
        r = random.random()
        for seq, cdf in zip(sequences, cumulative):
            if r <= cdf:
                counts[seq] += 1
                break
    return {seq: counts[seq] / sample_size for seq in sequences}


def qae_estimate(
    prob_map: Dict[SequenceType, float],
    sample_size: int,
    K: int,
    noise_scale: float,
    bias_scale: float,
) -> Dict[SequenceType, float]:
    """Simulate a QAE-style estimator with ``1/K`` variance reduction and bias."""

    estimates: Dict[SequenceType, float] = {}
    for seq, p_true in prob_map.items():
        variance = noise_scale * max(p_true * (1 - p_true), 1e-12) / (sample_size * K * K)
        std = math.sqrt(variance)
        bias = bias_scale / max(K, 1)
        noisy = random.gauss(p_true + bias, std)
        estimates[seq] = min(1.0, max(0.0, noisy))
    return renormalize(estimates)


def qae_amplitude_estimate(
    p_true: float, shots: int, K: int, noise_scale: float, bias_scale: float
) -> float:
    """Single-amplitude QAE-inspired estimator used in Part A."""

    variance = noise_scale * max(p_true * (1 - p_true), 1e-12) / (shots * K * K)
    std = math.sqrt(variance)
    bias = bias_scale / max(K, 1)
    noisy = random.gauss(p_true + bias, std)
    return min(1.0, max(0.0, noisy))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def parse_comma_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def run_hankel_trial(
    length: int,
    d: int,
    bond_dim: int,
    sample_size: int,
    qae_rounds: Sequence[int],
    noise_scale: float,
    bias_scale: float,
):
    """One Part-B trial: random MPS distribution + Hankel errors."""

    prob_map = sample_random_mps_distribution(length, d, bond_dim)
    hankel_true = build_hankel(prob_map, length, d)

    def hankel_errors(est_map: Dict[SequenceType, float]):
        hankel_est = build_hankel(est_map, length, d)
        diff = [
            [hankel_est[i][j] - hankel_true[i][j] for j in range(len(hankel_true[0]))]
            for i in range(len(hankel_true))
        ]
        return spectral_norm(diff), frobenius_norm(diff)

    classical_map = sample_classical(prob_map, sample_size)
    classical_err = hankel_errors(classical_map)
    qae_errs = {
        K: hankel_errors(qae_estimate(prob_map, sample_size, K, noise_scale, bias_scale))
        for K in qae_rounds
    }
    return classical_err, qae_errs


def run_amplitude_trials(
    p_values: Sequence[float],
    classical_shots: int,
    trials: int,
    qae_rounds: Sequence[int],
    noise_scale: float,
    bias_scale: float,
):
    """Part-A warm-up: single-amplitude scaling vs K."""

    rows: List[Dict[str, object]] = []
    for p_true in p_values:
        for _ in range(trials):
            # Classical baseline
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
                }
            )

            for K in qae_rounds:
                p_qae = qae_amplitude_estimate(p_true, classical_shots, K, noise_scale, bias_scale)
                rows.append(
                    {
                        "section": "amplitude",
                        "p_true": p_true,
                        "method": "qae",
                        "K": K,
                        "abs_error": abs(p_qae - p_true),
                        "sq_error": (p_qae - p_true) ** 2,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Experiment 8: QAE scaling for Hankel estimation")
    parser.add_argument("--lengths", type=str, default="6,8", help="Comma-separated lengths for Hankel experiments")
    parser.add_argument("--d", type=int, default=2, help="Alphabet size (default: 2)")
    parser.add_argument("--bond-dim", type=int, default=3, help="Bond dimension for random MPS distributions")
    parser.add_argument("--sample-size", type=int, default=5000, help="Classical outer sample size N")
    parser.add_argument("--qae-rounds", type=str, default="1,2,4,8,16,32", help="Comma-separated QAE query counts K")
    parser.add_argument("--trials", type=int, default=20, help="Number of random Hankel trials per length")
    parser.add_argument("--amplitude-probs", type=str, default="0.15,0.35", help="Comma-separated probabilities for Part-A warm-up")
    parser.add_argument("--amplitude-shots", type=int, default=1000, help="Classical shots for single-amplitude estimates")
    parser.add_argument("--amplitude-trials", type=int, default=500, help="Number of Part-A trials per probability")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Variance prefactor for QAE noise model")
    parser.add_argument("--bias-scale", type=float, default=0.0, help="Optional bias term scaled as bias_scale/K")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp8_results.csv",
        help="CSV output path combining Part A and Part B",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    if np is not None:
        np.random.seed(args.seed)
    lengths = parse_comma_list(args.lengths)
    qae_rounds = parse_comma_list(args.qae_rounds)
    p_values = [float(x) for x in args.amplitude_probs.split(",") if x]

    rows: List[Dict[str, object]] = []

    # Part A: single-amplitude scaling warm-up.
    rows.extend(
        run_amplitude_trials(
            p_values=p_values,
            classical_shots=args.amplitude_shots,
            trials=args.amplitude_trials,
            qae_rounds=qae_rounds,
            noise_scale=args.noise_scale,
            bias_scale=args.bias_scale,
        )
    )

    # Part B: full Hankel scaling with random MPS distributions.
    for length in lengths:
        for trial in range(args.trials):
            classical_err, qae_errs = run_hankel_trial(
                length=length,
                d=args.d,
                bond_dim=args.bond_dim,
                sample_size=args.sample_size,
                qae_rounds=qae_rounds,
                noise_scale=args.noise_scale,
                bias_scale=args.bias_scale,
            )
            rows.append(
                {
                    "section": "hankel",
                    "length": length,
                    "d": args.d,
                    "bond_dim": args.bond_dim,
                    "trial": trial,
                    "sample_size": args.sample_size,
                    "method": "classical",
                    "K": 0,
                    "error_spec": classical_err[0],
                    "error_fro": classical_err[1],
                }
            )
            for K in qae_rounds:
                spec, frob = qae_errs[K]
                rows.append(
                    {
                        "section": "hankel",
                        "length": length,
                        "d": args.d,
                        "bond_dim": args.bond_dim,
                        "trial": trial,
                        "sample_size": args.sample_size,
                        "method": "qae",
                        "K": K,
                        "error_spec": spec,
                        "error_fro": frob,
                    }
                )

    fieldnames = [
        "section",
        "length",
        "d",
        "bond_dim",
        "trial",
        "sample_size",
        "method",
        "K",
        "error_spec",
        "error_fro",
        "p_true",
        "abs_error",
        "sq_error",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
