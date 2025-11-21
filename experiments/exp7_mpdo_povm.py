#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 7 (Part A): Mixed-state MPDO / POVM Hankel ranks (Theorem 12.1).

This script implements the "Liouville positive-chain surrogate" described in the
strengthened Experiment 7 plan:

- MPDO core: shallow noisy-channel chains with bond dimension chi_rho;
- POVMs: either separable projective (chi_M=1) or shallow correlated
  row-stochastic networks (chi_M>1);
- lengths L in {6, 8, 10} (configurable), mid-cut t* = floor(L/2); Hankel
  matrices use P = Σ^{t*} and S = Σ^{L-t*};
- exact enumeration of p(x) over all d^L strings, Hankel construction, SVD,
  numerical rank with abs/rel threshold, and effective ranks at η∈{1e-2,1e-3}.

For every configuration we verify
    rank(H_p) <= min(|P|, |S|, chi_rho * chi_M)
and track how often the rank approaches the cap (and the continuous/effective
ranks). The implementation is dependency-light: only `math`, `random`, `csv`,
and optional `numpy` for SVD.

This is the "Part A" surrogate experiment; a possible "Part B" using genuine
MPDO+POVM tensor networks can be implemented separately if desired.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # Optional; used only for singular values.
    import numpy as np
except Exception:  # pragma: no cover - numpy may be absent.
    np = None


# ---------------------------------------------------------------------------
# Basic float-only linear algebra helpers
# ---------------------------------------------------------------------------


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(u, v))


def norm(u: Sequence[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(dot(u, u))


def matvec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    """Matrix-vector product."""
    out: List[float] = []
    for row in mat:
        out.append(sum(a * b for a, b in zip(row, vec)))
    return out


def kron_vec(u: Sequence[float], v: Sequence[float]) -> List[float]:
    """Kronecker product of two vectors."""
    return [a * b for a in u for b in v]


def kron_mat(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    """Kronecker product of two matrices."""
    rows_a = len(a)
    cols_a = len(a[0]) if rows_a > 0 else 0
    rows_b = len(b)
    cols_b = len(b[0]) if rows_b > 0 else 0
    out_rows = rows_a * rows_b
    out_cols = cols_a * cols_b
    out: List[List[float]] = [[0.0 for _ in range(out_cols)] for _ in range(out_rows)]
    for i in range(rows_a):
        for j in range(cols_a):
            aij = a[i][j]
            if aij == 0.0:
                continue
            for k in range(rows_b):
                row_idx = i * rows_b + k
                row_b = b[k]
                out_row = out[row_idx]
                for l in range(cols_b):
                    out_row[j * cols_b + l] = aij * row_b[l]
    return out


def gram_schmidt_rank(rows: Sequence[Sequence[float]], tol: float) -> int:
    """Estimate rank via Gram–Schmidt (used only for fallback SVD)."""
    basis: List[List[float]] = []
    for row in rows:
        vec = list(row)
        for b in basis:
            proj = dot(vec, b)
            vec = [x - proj * y for x, y in zip(vec, b)]
        nrm = norm(vec)
        if nrm > tol:
            basis.append([x / nrm for x in vec])
    return len(basis)


def singular_values(mat: Sequence[Sequence[float]], tol_abs: float) -> List[float]:
    """Compute singular values of a real matrix.

    - If numpy is available, use np.linalg.svd.
    - Otherwise, fall back to power iteration on A^T A with simple deflation,
      capping the number of singular values by a Gram–Schmidt rank estimate.
    """
    if not mat:
        return []

    if np is not None:
        arr = np.asarray(mat, dtype=float)
        sv = np.linalg.svd(arr, compute_uv=False)
        return sv.tolist()

    # Fallback: eigenvalues of A^T A via power iteration.
    cols = len(mat[0])
    rank_gs = gram_schmidt_rank(mat, tol_abs)
    ata: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(cols)]
    for row in mat:
        for i in range(cols):
            ri = row[i]
            if ri == 0.0:
                continue
            for j in range(cols):
                ata[i][j] += ri * row[j]

    def power_iter(A: List[List[float]], max_iters: int = 200) -> Tuple[float, List[float]]:
        v = [random.random() + 1e-6 for _ in range(len(A))]
        v = [x / norm(v) for x in v]
        for _ in range(max_iters):
            Av = matvec(A, v)
            nrm = norm(Av)
            if nrm == 0.0:
                break
            v = [x / nrm for x in Av]
        Av = matvec(A, v)
        eigval = dot(v, Av)
        return eigval, v

    sv: List[float] = []
    residual = [row[:] for row in ata]
    target = min(rank_gs, cols)
    for _ in range(target):
        eig, vec = power_iter(residual)
        if eig <= tol_abs * tol_abs:
            break
        sv.append(math.sqrt(max(eig, 0.0)))
        # Deflate: A <- A - eig * v v^T
        for i in range(cols):
            ri = residual[i]
            vi = vec[i]
            if vi == 0.0:
                continue
            for j in range(cols):
                ri[j] -= eig * vi * vec[j]

    sv.sort(reverse=True)
    return sv


# ---------------------------------------------------------------------------
# Random positive chains for MPDO core and POVM network (Liouville surrogate)
# ---------------------------------------------------------------------------


def random_row_stochastic_matrix(size: int) -> List[List[float]]:
    """Random strictly positive row-stochastic matrix of shape (size, size)."""
    mat: List[List[float]] = []
    for _ in range(size):
        row = [random.random() + 1e-6 for _ in range(size)]
        s = sum(row)
        mat.append([x / s for x in row])
    return mat


def random_distribution(size: int) -> List[float]:
    """Random strictly positive probability vector of given length."""
    vec = [random.random() + 1e-6 for _ in range(size)]
    s = sum(vec)
    return [x / s for x in vec]


def normalize_vector(vec: Sequence[float]) -> List[float]:
    """Normalise a vector to sum 1, used for alpha propagation."""
    s = sum(vec)
    if s == 0.0:
        return list(vec)
    return [x / s for x in vec]


def normalize_matrix_rows(mat: Sequence[Sequence[float]]) -> List[List[float]]:
    """Normalise each row to sum 1 (row-stochastic)."""
    out: List[List[float]] = []
    for row in mat:
        s = sum(row)
        if s == 0.0:
            out.append(list(row))
        else:
            out.append([x / s for x in row])
    return out


def mix_row_stochastic(base: List[List[float]], noise: float) -> List[List[float]]:
    """Blend a row-stochastic matrix with random jitter (remain row-stochastic)."""
    jitter = random_row_stochastic_matrix(len(base))
    mixed = [
        [(1.0 - noise) * a + noise * b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(base, jitter)
    ]
    return normalize_matrix_rows(mixed)


def build_chain(d: int, dim: int) -> Tuple[List[float], Dict[int, List[List[float]]]]:
    """Base random row-stochastic chain for each symbol."""
    alpha = random_distribution(dim)
    transitions: Dict[int, List[List[float]]] = {}
    for sym in range(d):
        transitions[sym] = random_row_stochastic_matrix(dim)
    return alpha, transitions


def mpdo_core(
    d: int, chi_rho: int, depth: int, noise: float
) -> Tuple[List[float], Dict[int, List[List[float]]], List[float]]:
    """Construct a shallow noisy-channel MPDO core with bond chi_rho.

    This is a Liouville surrogate: a nonnegative WFA of dimension chi_rho
    built from a few layers of row-stochastic mixing.
    """
    alpha, transitions = build_chain(d, chi_rho)
    # Fold 'depth - 1' extra noisy layers into the transitions.
    for _ in range(max(depth - 1, 0)):
        for sym in range(d):
            transitions[sym] = mix_row_stochastic(transitions[sym], noise)
        alpha = matvec(transitions[0], alpha)
        alpha = normalize_vector(alpha)
    beta = [1.0 for _ in range(chi_rho)]
    return alpha, transitions, beta


def povm_network(
    d: int, chi_m: int, depth: int, noise: float
) -> Tuple[List[float], Dict[int, List[List[float]]], List[float], str]:
    """Construct separable or shallow correlated POVM networks (Liouville surrogate).

    - chi_m == 1: "separable" site-wise measurement (Liouville bond 1).
    - chi_m > 1: "network" shallow correlated chain with bond chi_m.
    """
    if chi_m == 1:
        meas_alpha = [1.0]
        meas_beta = [1.0]
        meas_syms = {sym: [[1.0]] for sym in range(d)}
        return meas_alpha, meas_syms, meas_beta, "separable"

    meas_alpha, meas_syms = build_chain(d, chi_m)
    for _ in range(max(depth - 1, 0)):
        for sym in range(d):
            meas_syms[sym] = mix_row_stochastic(meas_syms[sym], noise)
        meas_alpha = matvec(meas_syms[0], meas_alpha)
        meas_alpha = normalize_vector(meas_alpha)
    meas_beta = [1.0 for _ in range(chi_m)]
    return meas_alpha, meas_syms, meas_beta, "network"


# ---------------------------------------------------------------------------
# Sequence enumeration and Hankel construction
# ---------------------------------------------------------------------------


def generate_words(length: int, d: int) -> List[List[int]]:
    """Enumerate all words of given length over alphabet {0,...,d-1}."""
    total = d ** length
    words: List[List[int]] = []
    for val in range(total):
        word = [0 for _ in range(length)]
        tmp = val
        for i in range(length - 1, -1, -1):
            word[i] = tmp % d
            tmp //= d
        words.append(word)
    return words


def probability_map(
    alpha_rho: Sequence[float],
    A_syms: Dict[int, List[List[float]]],
    beta_rho: Sequence[float],
    meas_alpha: Sequence[float],
    meas_syms: Dict[int, List[List[float]]],
    meas_beta: Sequence[float],
    length: int,
    d: int,
) -> Dict[Tuple[int, ...], float]:
    """Enumerate p(x) for all words x of given length using the Liouville surrogate."""
    init = kron_vec(alpha_rho, meas_alpha)
    terminal = kron_vec(beta_rho, meas_beta)

    # Precompute joint transitions T_sym = A_sym ⊗ M_sym.
    joint_syms: Dict[int, List[List[float]]] = {}
    for sym in range(d):
        joint_syms[sym] = kron_mat(A_syms[sym], meas_syms[sym])

    seq_prob: Dict[Tuple[int, ...], float] = {}
    total_weight = 0.0
    for word in generate_words(length, d):
        vec = list(init)
        for sym in word:
            step = joint_syms[sym]
            vec = matvec(step, vec)
        prob = dot(vec, terminal)
        seq_prob[tuple(word)] = prob
        total_weight += prob

    if total_weight > 0.0:
        inv_total = 1.0 / total_weight
        for key in seq_prob:
            seq_prob[key] *= inv_total

    return seq_prob


def build_hankel(
    seq_prob: Dict[Tuple[int, ...], float],
    prefixes: Sequence[Sequence[int]],
    suffixes: Sequence[Sequence[int]],
) -> List[List[float]]:
    """Build Hankel matrix H_p(P,S) with entries H(u,v) = p(uv)."""
    rows = len(prefixes)
    cols = len(suffixes)
    mat: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i, p in enumerate(prefixes):
        p_list = list(p)
        for j, s in enumerate(suffixes):
            seq = tuple(p_list + list(s))
            mat[i][j] = seq_prob.get(seq, 0.0)
    return mat


def effective_rank_from_singular_values(sv: Sequence[float]) -> float:
    """Continuous effective rank: (sum σ_i)^2 / sum σ_i^2."""
    if not sv:
        return 0.0
    s1 = sum(sv)
    s2 = sum(x * x for x in sv)
    if s2 == 0.0:
        return 0.0
    return (s1 * s1) / s2


def effective_rank_thresholded(sv: Sequence[float], eta: float) -> int:
    """Soft rank: number of σ_i with σ_i / σ_1 >= eta."""
    if not sv:
        return 0
    sigma1 = sv[0]
    if sigma1 == 0.0:
        return 0
    return sum(1 for x in sv if x / sigma1 >= eta)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_experiment(
    lengths: Sequence[int],
    chi_rhos: Sequence[int],
    chi_ms: Sequence[int],
    trials: int,
    tol_abs: float,
    tol_rel: float,
    d: int,
    mpdo_depth: int,
    meas_depth: int,
    noise: float,
    seed: int,
    output: str,
) -> None:
    """Core loop over configurations; writes all records into a CSV file."""
    random.seed(seed)
    records: List[Dict[str, float]] = []

    for L in lengths:
        t_star = L // 2
        prefixes = generate_words(t_star, d)
        suffixes = generate_words(L - t_star, d)
        prefix_ct = len(prefixes)
        suffix_ct = len(suffixes)

        for chi_rho in chi_rhos:
            for chi_m in chi_ms:
                rank_upper = chi_rho * chi_m
                rank_cap = min(prefix_ct, suffix_ct, rank_upper)

                for trial in range(trials):
                    # Build MPDO core and POVM network surrogates.
                    alpha_rho, A_syms, beta_rho = mpdo_core(d, chi_rho, mpdo_depth, noise)
                    meas_alpha, meas_syms, meas_beta, meas_type = povm_network(
                        d, chi_m, meas_depth, noise
                    )

                    # Enumerate probabilities and build Hankel.
                    seq_prob = probability_map(
                        alpha_rho,
                        A_syms,
                        beta_rho,
                        meas_alpha,
                        meas_syms,
                        meas_beta,
                        L,
                        d,
                    )
                    hankel = build_hankel(seq_prob, prefixes, suffixes)

                    # Singular values and rank/effective-rank metrics.
                    sv = singular_values(hankel, tol_abs)
                    sv_sorted = sorted(sv, reverse=True)
                    sv_max = sv_sorted[0] if sv_sorted else 0.0
                    threshold = max(tol_abs, tol_rel * sv_max) if sv_sorted else 0.0
                    rank = sum(1 for s in sv_sorted if s >= threshold)
                    sv_min = min((s for s in sv_sorted if s >= threshold), default=0.0)

                    eff_rank = effective_rank_from_singular_values(sv_sorted)
                    r_eff_1e2 = effective_rank_thresholded(sv_sorted, 1e-2)
                    r_eff_1e3 = effective_rank_thresholded(sv_sorted, 1e-3)

                    records.append(
                        {
                            "length": L,
                            "t_star": t_star,
                            "chi_rho": chi_rho,
                            "chi_M": chi_m,
                            "measurement": meas_type,
                            "trial": trial,
                            "hankel_rank": rank,
                            "rank_upper": rank_upper,
                            "rank_cap": rank_cap,
                            "within_cap": int(rank <= rank_cap + 1e-9),
                            "effective_rank": eff_rank,
                            "r_eff_1e2": r_eff_1e2,
                            "r_eff_1e3": r_eff_1e3,
                            "sv_max": sv_max,
                            "sv_min_thresholded": sv_min,
                            "threshold": threshold,
                            "prefixes": prefix_ct,
                            "suffixes": suffix_ct,
                            "d": d,
                        }
                    )

    fieldnames = [
        "length",
        "t_star",
        "chi_rho",
        "chi_M",
        "measurement",
        "trial",
        "hankel_rank",
        "rank_upper",
        "rank_cap",
        "within_cap",
        "effective_rank",
        "r_eff_1e2",
        "r_eff_1e3",
        "sv_max",
        "sv_min_thresholded",
        "threshold",
        "prefixes",
        "suffixes",
        "d",
    ]

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_int_list(arg: str) -> List[int]:
    """Parse comma-separated integers like '2,4,8'."""
    return [int(x) for x in arg.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 7 (Part A): MPDO/POVM Hankel ranks (Liouville surrogate)"
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="6,8,10",
        help="Comma-separated sequence lengths, e.g. '6,8,10'",
    )
    parser.add_argument(
        "--chi-rho",
        type=str,
        default="2,4",
        help="Comma-separated MPDO bond dimensions chi_rho",
    )
    parser.add_argument(
        "--chi-m",
        type=str,
        default="1,2,4",
        help="Comma-separated POVM bond dimensions chi_M",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Trials per configuration",
    )
    parser.add_argument(
        "--tol-abs",
        type=float,
        default=1e-12,
        help="Absolute tolerance for numerical rank threshold",
    )
    parser.add_argument(
        "--tol-rel",
        type=float,
        default=1e-8,
        help="Relative tolerance (times sigma_1) for numerical rank (recommended ~1e-8)",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=2,
        help="Local alphabet / physical dimension",
    )
    parser.add_argument(
        "--mpdo-depth",
        type=int,
        default=3,
        help="Depth of the noisy-channel MPDO constructor",
    )
    parser.add_argument(
        "--meas-depth",
        type=int,
        default=2,
        help="Depth of the correlated POVM network",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Noise strength when mixing stochastic layers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp7_results.csv",
        help="CSV output path",
    )

    args = parser.parse_args()

    lengths = parse_int_list(args.lengths)
    chi_rhos = parse_int_list(args.chi_rho)
    chi_ms = parse_int_list(args.chi_m)

    run_experiment(
        lengths=lengths,
        chi_rhos=chi_rhos,
        chi_ms=chi_ms,
        trials=args.trials,
        tol_abs=args.tol_abs,
        tol_rel=args.tol_rel,
        d=args.d,
        mpdo_depth=args.mpdo_depth,
        meas_depth=args.meas_depth,
        noise=args.noise,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()



"""
python experiments/exp7_mpdo_povm.py --lengths 6,8,10 --chi-rho 2,4 --chi-m 1,2,4 --trials 30 --d 2 --mpdo-depth 3 --meas-depth 2 --noise 0.1 --tol-abs 1e-12 --tol-rel 1e-8 --seed 0 --output experiments/exp7_results.csv
"""