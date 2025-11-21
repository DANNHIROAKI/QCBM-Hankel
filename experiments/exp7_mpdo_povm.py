"""Experiment 7: Mixed-state MPDO / POVM Hankel ranks (Theorem 12.1).

This expanded version follows the detailed experiment brief:

* random MPDOs with bond dimension :math:`\chi_\rho \in \{2,4\}` built
  from shallow noisy channels (a practical stand-in for thermal/finite-T MPOs);
* two POVM families: separable projective measurements (:math:`\chi_M=1`) and
  shallow correlated POVMs realised by a brickwall pre-measurement network
  (:math:`\chi_M>1`);
* sequence lengths :math:`L\in\{6,8,10\}` (configurable) with cut
  :math:`t_\* = \lfloor L/2 \rfloor`; Hankel matrices use
  :math:`P = \Sigma^{t_\*}` and :math:`S = \Sigma^{L-t_\*}`;
* exhaustive enumeration of :math:`p(x)=\mathrm{Tr}(M_x \rho_0)`, Hankel
  construction, SVD, numerical rank (relative threshold), and effective ranks
  at :math:`\eta \in \{10^{-2},10^{-3}\}`.

For every configuration we verify
:math:`\operatorname{rank}(H_p) \le \min(|P|,|S|, \chi_\rho\chi_M)` and track
how often the rank approaches :math:`\chi_\rho\chi_M` (or the dimensional cap
when :math:`|P|`/`|S|` is smaller). The implementation remains dependency-light
(`math`, `random`, optional `numpy` for SVD).
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
# Small linear algebra helpers (float-only)
# ---------------------------------------------------------------------------


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def norm(u: Sequence[float]) -> float:
    return math.sqrt(dot(u, u))


def normalize(u: Sequence[float]) -> List[float]:
    total = sum(u)
    if total == 0:
        return list(u)
    return [x / total for x in u]


def matvec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    out: List[float] = []
    for row in mat:
        out.append(sum(a * b for a, b in zip(row, vec)))
    return out


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(b[0]) if b else 0
    mid = len(b)
    out: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(mid):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def kron_vec(u: Sequence[float], v: Sequence[float]) -> List[float]:
    return [a * b for a in u for b in v]


def kron_mat(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    rows_a = len(a)
    cols_a = len(a[0]) if a else 0
    rows_b = len(b)
    cols_b = len(b[0]) if b else 0
    out: List[List[float]] = [[0.0 for _ in range(cols_a * cols_b)] for _ in range(rows_a * rows_b)]
    for i in range(rows_a):
        for j in range(cols_a):
            aij = a[i][j]
            if aij == 0:
                continue
            for k in range(rows_b):
                for l in range(cols_b):
                    out[i * rows_b + k][j * cols_b + l] = aij * b[k][l]
    return out


def gram_schmidt_rank(rows: Sequence[Sequence[float]], tol: float) -> int:
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


def singular_values(mat: Sequence[Sequence[float]], tol: float) -> List[float]:
    if np is not None:
        arr = np.asarray(mat, dtype=float)
        sv = np.linalg.svd(arr, compute_uv=False)
        return sv.tolist()

    # Fallback: power-iteration eigen-decomposition of A^T A to extract
    # singular values without external dependencies. Matrices are small in this
    # experiment (d^{t_star} by d^{L-t_star}), so a simple deflation scheme is
    # sufficient. We cap the number of extracted values using a Gram–Schmidt
    # rank to avoid over-counting numerical noise.
    if not mat:
        return []

    cols = len(mat[0])
    rank_gs = gram_schmidt_rank(mat, tol)
    ata: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(cols)]
    for i in range(cols):
        for j in range(cols):
            ata[i][j] = sum(row[i] * row[j] for row in mat)

    def power_iter(A: List[List[float]], max_iters: int = 200) -> Tuple[float, List[float]]:
        v = [random.random() + 1e-6 for _ in range(len(A))]
        v = [x / norm(v) for x in v]
        for _ in range(max_iters):
            Av = matvec(A, v)
            nrm = norm(Av)
            if nrm == 0:
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
        if eig <= tol * tol:
            break
        sv.append(math.sqrt(max(eig, 0.0)))
        # Deflate: A <- A - eig * v v^T
        for i in range(cols):
            for j in range(cols):
                residual[i][j] -= eig * vec[i] * vec[j]

    return sv


# ---------------------------------------------------------------------------
# Random positive chains for MPDO core and POVM network
# ---------------------------------------------------------------------------


def random_row_stochastic_matrix(size: int) -> List[List[float]]:
    mat: List[List[float]] = []
    for _ in range(size):
        row = [random.random() + 1e-6 for _ in range(size)]
        total = sum(row)
        mat.append([x / total for x in row])
    return mat


def random_distribution(size: int) -> List[float]:
    vec = [random.random() + 1e-6 for _ in range(size)]
    total = sum(vec)
    return [x / total for x in vec]


def build_chain(d: int, dim: int) -> Tuple[List[float], Dict[int, List[List[float]]]]:
    alpha = random_distribution(dim)
    transitions: Dict[int, List[List[float]]] = {}
    for sym in range(d):
        transitions[sym] = random_row_stochastic_matrix(dim)
    return alpha, transitions


def mix_row_stochastic(base: List[List[float]], noise: float) -> List[List[float]]:
    """Blend a stochastic matrix with a random jitter (keeps rows normalised)."""

    jitter = random_row_stochastic_matrix(len(base))
    return normalize_matrix([[ (1 - noise) * a + noise * b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(base, jitter)])


def normalize_matrix(mat: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for row in mat:
        total = sum(row)
        out.append([x / total if total else x for x in row])
    return out


def mpdo_chain(d: int, chi_rho: int, depth: int, noise: float) -> Tuple[List[float], Dict[int, List[List[float]]], List[float]]:
    """Construct a shallow noisy-channel MPDO core with bond ``chi_rho``.

    The channel depth emulates a few layers of nearest-neighbour noise; we fold
    them into a single row-stochastic family for each symbol to stay lightweight
    while keeping a clear bond-dimension control.
    """

    alpha, transitions = build_chain(d, chi_rho)
    for _ in range(max(depth - 1, 0)):
        for sym in range(d):
            transitions[sym] = mix_row_stochastic(transitions[sym], noise)
        alpha = matvec(transitions[0], alpha)
        alpha = normalize(alpha)
    beta = [1.0 for _ in range(chi_rho)]
    return alpha, transitions, beta


def measurement_network(d: int, chi_m: int, depth: int, noise: float) -> Tuple[List[float], Dict[int, List[List[float]]], List[float], str]:
    """Construct separable or shallow correlated POVM networks.

    * chi_m == 1: site-wise projective measurement (Liouville bond 1).
    * chi_m > 1: brickwall-style shallow network approximated via a few noisy
      stochastic layers with target bond ``chi_m``.
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
        meas_alpha = normalize(meas_alpha)
    meas_beta = [1.0 for _ in range(chi_m)]
    return meas_alpha, meas_syms, meas_beta, "network"


# ---------------------------------------------------------------------------
# Sequence enumeration and Hankel construction
# ---------------------------------------------------------------------------


def generate_words(length: int, d: int) -> List[List[int]]:
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


def prob_map(
    alpha_rho: Sequence[float],
    A_syms: Dict[int, List[List[float]]],
    beta_rho: Sequence[float],
    meas_alpha: Sequence[float],
    meas_syms: Dict[int, List[List[float]]],
    meas_beta: Sequence[float],
    length: int,
    d: int,
) -> Dict[Tuple[int, ...], float]:
    init = kron_vec(alpha_rho, meas_alpha)
    terminal = kron_vec(beta_rho, meas_beta)
    seq_map: Dict[Tuple[int, ...], float] = {}
    total = 0.0
    for word in generate_words(length, d):
        vec = list(init)
        for sym in word:
            step = kron_mat(A_syms[sym], meas_syms[sym])
            vec = matvec(step, vec)
        prob = dot(vec, terminal)
        seq_map[tuple(word)] = prob
        total += prob
    if total > 0:
        for k in list(seq_map.keys()):
            seq_map[k] /= total
    return seq_map


def build_hankel(seq_prob: Dict[Tuple[int, ...], float], prefixes: Sequence[Sequence[int]], suffixes: Sequence[Sequence[int]]) -> List[List[float]]:
    rows = len(prefixes)
    cols = len(suffixes)
    mat: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i, p in enumerate(prefixes):
        for j, s in enumerate(suffixes):
            seq = tuple(p + list(s))
            mat[i][j] = seq_prob.get(seq, 0.0)
    return mat


def effective_rank_from_singular_values(sv: Sequence[float]) -> float:
    if not sv:
        return 0.0
    s1 = sum(sv)
    s2 = sum(x * x for x in sv)
    if s2 == 0:
        return 0.0
    return (s1 * s1) / s2


def effective_rank_thresholded(sv: Sequence[float], eta: float) -> int:
    if not sv:
        return 0
    if sv[0] == 0:
        return 0
    return sum(1 for x in sv if x / sv[0] >= eta)


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
                upper_bound = chi_rho * chi_m
                dim_cap = min(prefix_ct, suffix_ct, upper_bound)
                for trial in range(trials):
                    alpha_rho, A_syms, beta_rho = mpdo_chain(d, chi_rho, mpdo_depth, noise)
                    meas_alpha, meas_syms, meas_beta, meas_type = measurement_network(d, chi_m, meas_depth, noise)

                    seq_prob = prob_map(alpha_rho, A_syms, beta_rho, meas_alpha, meas_syms, meas_beta, L, d)
                    hankel = build_hankel(seq_prob, prefixes, suffixes)
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
                            "rank_upper": upper_bound,
                            "rank_cap": dim_cap,
                            "within_cap": int(rank <= dim_cap + 1e-9),
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
        for row in records:
            writer.writerow(row)


def parse_int_list(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 7: MPDO/POVM Hankel ranks")
    parser.add_argument("--lengths", type=str, default="6,8,10", help="Comma-separated sequence lengths")
    parser.add_argument("--chi-rho", type=str, default="2,4", help="Comma-separated MPDO bond dimensions")
    parser.add_argument("--chi-m", type=str, default="1,2,4", help="Comma-separated measurement bond dimensions")
    parser.add_argument("--trials", type=int, default=30, help="Trials per configuration")
    parser.add_argument("--tol-abs", type=float, default=1e-12, help="Absolute tolerance for numerical rank")
    parser.add_argument("--tol-rel", type=float, default=1e-10, help="Relative tolerance (times sigma_1) for numerical rank")
    parser.add_argument("--d", type=int, default=2, help="Local alphabet / physical dimension")
    parser.add_argument("--mpdo-depth", type=int, default=3, help="Depth of the noisy-channel MPDO constructor")
    parser.add_argument("--meas-depth", type=int, default=2, help="Depth of the correlated POVM network")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise strength when mixing stochastic layers")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--output", type=str, default="experiments/exp7_results.csv", help="CSV output path")
    args = parser.parse_args()

    lengths = parse_int_list(args.lengths)
    chi_rhos = parse_int_list(args.chi_rho)
    chi_ms = parse_int_list(args.chi_m)

    run_experiment(
        lengths,
        chi_rhos,
        chi_ms,
        args.trials,
        args.tol_abs,
        args.tol_rel,
        args.d,
        args.mpdo_depth,
        args.meas_depth,
        args.noise,
        args.seed,
        args.output,
    )


if __name__ == "__main__":
    main()
