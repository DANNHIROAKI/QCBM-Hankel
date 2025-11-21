"""Experiment 5: Hankel spectral learning vs sample size and length.

This version follows the expanded plan:

* multiple ground-truth sources (low-rank MPS, higher-entropy MPS, and an
  optional contractive WFA model);
* fixed-cut Hankel construction with capped prefix/suffix sets and recorded
  coherence/\(\gamma\)/\(\kappa_B\);
* rank-truncated SVD + whitening to build a WFA estimator; optional
  row-substochastic projection to enforce the contractive regime;
* end-to-end errors (TV / max) alongside Hankel deviation and normalised
  scalings \(\gamma/F(L)\) to check the \(N^{-1/2}\) law and the contrast
  between geometric vs linear length dependence.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Small linear algebra helpers (dependency-free)
# ---------------------------------------------------------------------------


def dot(u: Sequence[complex], v: Sequence[complex]) -> complex:
    return sum(a.conjugate() * b for a, b in zip(u, v))


def norm(u: Sequence[complex]) -> float:
    return math.sqrt(sum(abs(x) ** 2 for x in u))


def normalize(u: Sequence[complex]) -> List[complex]:
    n = norm(u)
    if n == 0:
        return list(u)
    return [x / n for x in u]


def matvec(mat: Sequence[Sequence[complex]], vec: Sequence[complex]) -> List[complex]:
    out: List[complex] = []
    for row in mat:
        out.append(sum(a * b for a, b in zip(row, vec)))
    return out


def matmul(a: Sequence[Sequence[complex]], b: Sequence[Sequence[complex]]) -> List[List[complex]]:
    rows = len(a)
    cols = len(b[0]) if b else 0
    mid = len(b)
    out: List[List[complex]] = [[0.0j for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(mid):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def conj_transpose(mat: Sequence[Sequence[complex]]) -> List[List[complex]]:
    rows = len(mat)
    cols = len(mat[0]) if mat else 0
    return [[mat[i][j].conjugate() for i in range(rows)] for j in range(cols)]


def identity(n: int) -> List[List[complex]]:
    return [[1.0 + 0.0j if i == j else 0.0j for j in range(n)] for i in range(n)]


def mat_add(a: Sequence[Sequence[complex]], b: Sequence[Sequence[complex]]) -> List[List[complex]]:
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def mat_scale(a: Sequence[Sequence[complex]], s: complex) -> List[List[complex]]:
    return [[s * x for x in row] for row in a]


def frobenius_norm(mat: Sequence[Sequence[complex]]) -> float:
    return math.sqrt(sum(abs(x) ** 2 for row in mat for x in row))


def spectral_norm(mat: Sequence[Sequence[complex]], iters: int = 30) -> float:
    if not mat or not mat[0]:
        return 0.0
    n = len(mat[0])
    v = normalize([random.random() + 1j * random.random() for _ in range(n)])
    for _ in range(iters):
        w = matvec(mat, v)
        vt = matvec(conj_transpose(mat), w)
        v = normalize(vt)
    w = matvec(mat, v)
    return norm(w)


def matrix_power_inverse_sqrt(mat: Sequence[Sequence[complex]], iters: int = 8) -> List[List[complex]]:
    n = len(mat)
    trace = sum(mat[i][i].real for i in range(n))
    scale = trace / n if trace != 0 else 1.0
    A = mat_scale(mat, 1.0 / scale)
    Y = [row[:] for row in A]
    Z = identity(n)
    for _ in range(iters):
        ZY = matmul(Z, Y)
        mid = mat_add(identity(n), identity(n))
        mid = mat_add(mid, identity(n))  # 3I
        mid = mat_add(mid, mat_scale(ZY, -1.0))  # 3I - ZY
        mid = mat_scale(mid, 0.5)
        Y = matmul(Y, mid)
        Z = matmul(mid, Z)
    return mat_scale(Z, 1.0 / math.sqrt(scale))


# ---------------------------------------------------------------------------
# MPS generation and amplitudes
# ---------------------------------------------------------------------------


def random_left_canonical_mps(length: int, bond_dim: int, d: int) -> Tuple[List[List[List[complex]]], List[complex], List[complex]]:
    cores: List[List[List[complex]]] = []
    for _ in range(length):
        raw = []
        for _ in range(d):
            mat = [[random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)] for _ in range(bond_dim)]
            raw.append(mat)
        M = [[0.0j for _ in range(bond_dim)] for _ in range(bond_dim)]
        for mat in raw:
            Mt = matmul(mat, conj_transpose(mat))
            M = mat_add(M, Mt)
        inv_sqrt = matrix_power_inverse_sqrt(M)
        site = [matmul(inv_sqrt, mat) for mat in raw]
        cores.append(site)
    alpha = normalize([random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)])
    beta = normalize([random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)])
    return cores, alpha, beta


def amplitude_for_sequence(seq: Sequence[int], cores: Sequence[Sequence[Sequence[complex]]], alpha: Sequence[complex], beta: Sequence[complex]) -> complex:
    vec = list(alpha)
    for site, sym in enumerate(seq):
        vec = matvec(cores[site][sym], vec)
    return dot(vec, beta)


# ---------------------------------------------------------------------------
# Hankel helpers
# ---------------------------------------------------------------------------


def int_to_word(val: int, length: int, d: int) -> List[int]:
    word = [0 for _ in range(length)]
    for i in range(length - 1, -1, -1):
        word[i] = val % d
        val //= d
    return word


def all_words(length: int, d: int) -> List[List[int]]:
    return [int_to_word(v, length, d) for v in range(d ** length)]


def sample_words(length: int, d: int, max_words: int) -> List[List[int]]:
    total = d ** length
    if total <= max_words:
        return all_words(length, d)
    chosen = set()
    while len(chosen) < max_words:
        chosen.add(random.randrange(total))
    return [int_to_word(v, length, d) for v in chosen]


def probability_map(cores: Sequence[Sequence[Sequence[complex]]], alpha: Sequence[complex], beta: Sequence[complex], length: int, d: int, support: Sequence[Sequence[int]]) -> Dict[Tuple[int, ...], float]:
    probs: Dict[Tuple[int, ...], float] = {}
    total = 0.0
    for word in support:
        amp = amplitude_for_sequence(word, cores, alpha, beta)
        p = abs(amp) ** 2
        probs[tuple(word)] = p
        total += p
    if total > 0:
        for k in list(probs.keys()):
            probs[k] /= total
    return probs


def enumerate_prefix_suffix(length: int, d: int, max_prefixes: int, max_suffixes: int) -> Tuple[List[List[int]], List[List[int]]]:
    t_star = length // 2
    prefixes = sample_words(t_star, d, max_prefixes)
    suffixes = sample_words(length - t_star, d, max_suffixes)
    # ensure empty prefix/suffix exist
    empty_prefix = [0 for _ in range(t_star)]
    empty_suffix = [0 for _ in range(length - t_star)]
    if empty_prefix not in prefixes:
        prefixes.insert(0, empty_prefix)
    if empty_suffix not in suffixes:
        suffixes.insert(0, empty_suffix)
    return prefixes, suffixes


def hankel_from_prob_map(prob: Dict[Tuple[int, ...], float], prefixes: Sequence[Sequence[int]], suffixes: Sequence[Sequence[int]]) -> List[List[float]]:
    hankel: List[List[float]] = []
    for pre in prefixes:
        row: List[float] = []
        for suf in suffixes:
            word = tuple(list(pre) + list(suf))
            row.append(prob.get(word, 0.0))
        hankel.append(row)
    return hankel


def hankel_column_masks(suffixes: Sequence[Sequence[int]], d: int) -> List[List[bool]]:
    masks: List[List[bool]] = []
    for sigma in range(d):
        mask = []
        for suf in suffixes:
            mask.append(len(suf) > 0 and suf[0] == sigma)
        masks.append(mask)
    return masks


def masked_sub_hankel(hankel: Sequence[Sequence[float]], mask: Sequence[bool]) -> List[List[float]]:
    rows = len(hankel)
    cols = len(hankel[0]) if hankel else 0
    idxs = [j for j, flag in enumerate(mask) if flag]
    sub: List[List[float]] = [[0.0 for _ in idxs] for _ in range(rows)]
    for i in range(rows):
        for k, j in enumerate(idxs):
            sub[i][k] = hankel[i][j]
    return sub


def empirical_hankel_samples(
    samples: Sequence[Sequence[int]],
    prefixes: Sequence[Sequence[int]],
    suffixes: Sequence[Sequence[int]],
    d: int,
) -> Tuple[List[List[float]], List[List[List[float]]]]:
    hankel = [[0.0 for _ in suffixes] for _ in prefixes]
    masks = hankel_column_masks(suffixes, d)
    hankel_masks = [[[0.0 for _ in suffixes] for _ in prefixes] for _ in range(d)]
    valid = 0.0
    for seq in samples:
        pre = seq[: len(prefixes[0])]
        suf = seq[len(prefixes[0]) :]
        try:
            i = prefixes.index(pre)
            j = suffixes.index(suf)
        except ValueError:
            continue
        valid += 1.0
        hankel[i][j] += 1.0
        for sigma in range(d):
            if masks[sigma][j]:
                hankel_masks[sigma][i][j] += 1.0
    n = float(valid)
    if n > 0:
        for i in range(len(prefixes)):
            for j in range(len(suffixes)):
                hankel[i][j] /= n
                for sigma in range(d):
                    hankel_masks[sigma][i][j] /= n
    sub_blocks = [masked_sub_hankel(mask, masks[sigma]) for sigma, mask in enumerate(hankel_masks)]
    return hankel, sub_blocks


# ---------------------------------------------------------------------------
# Spectral learning utilities
# ---------------------------------------------------------------------------


def truncated_svd_rank(hankel: Sequence[Sequence[float]], rank: int) -> Tuple[List[List[complex]], List[float], List[List[complex]]]:
    # power iteration eig on H H^T
    m = len(hankel)
    n = len(hankel[0]) if hankel else 0
    rank = max(1, min(rank, m, n))
    # build Gram HH^T
    gram: List[List[float]] = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            gram[i][j] = sum(hankel[i][k] * hankel[j][k] for k in range(n))
    U: List[List[complex]] = [[0.0j for _ in range(rank)] for _ in range(m)]
    S: List[float] = [0.0 for _ in range(rank)]
    for r in range(rank):
        v = normalize([random.random() for _ in range(m)])
        for _ in range(40):
            w = [sum(gram[i][j] * v[j] for j in range(m)) for i in range(m)]
            v = normalize(w)
        sigma_sq = sum(v[i] * sum(gram[i][j] * v[j] for j in range(m)) for i in range(m))
        sigma = math.sqrt(max(sigma_sq, 0.0))
        S[r] = sigma
        for i in range(m):
            U[i][r] = v[i]
        # deflate the Gram matrix
        for i in range(m):
            for j in range(m):
                gram[i][j] -= sigma_sq * (v[i] * v[j])
    # compute V via H^T U S^{-1}
    V: List[List[complex]] = [[0.0j for _ in range(rank)] for _ in range(n)]
    for k in range(n):
        for r in range(rank):
            if S[r] > 0:
                V[k][r] = sum(hankel[i][k] * U[i][r] for i in range(m)) / S[r]
    return U, S, V


def spectral_wfa_from_hankel(hankel: Sequence[Sequence[float]], hankel_blocks: Sequence[Sequence[Sequence[float]]], rank: int, prefix_zero_idx: int, suffix_zero_idx: int) -> Tuple[List[List[List[float]]], List[float], List[float], float]:
    U, S, V = truncated_svd_rank(hankel, rank)
    eps = 1e-12
    inv_sqrt = [1.0 / math.sqrt(s + eps) for s in S]
    W_L: List[List[float]] = [[0.0 for _ in range(rank)] for _ in range(len(hankel))]
    W_R: List[List[float]] = [[0.0 for _ in range(rank)] for _ in range(len(hankel[0]))]
    for i in range(len(hankel)):
        for r in range(rank):
            W_L[i][r] = (U[i][r].real) * inv_sqrt[r]
    for j in range(len(hankel[0])):
        for r in range(rank):
            W_R[j][r] = (V[j][r].real) * inv_sqrt[r]

    B_list: List[List[List[float]]] = []
    sigma_min = min(S) if S else 0.0
    for block in hankel_blocks:
        # block already masked; pad back to full column width
        mapped: List[List[float]] = [[0.0 for _ in range(len(W_R))] for _ in range(len(W_L))]
        for i in range(min(len(block), len(mapped))):
            for k, val in enumerate(block[i]):
                mapped[i][k] = val

        # temp = W_L^T * mapped  (r x n)
        temp = [[0.0 for _ in range(len(mapped[0]))] for _ in range(rank)]
        for r in range(rank):
            for j in range(len(mapped[0])):
                temp[r][j] = sum(W_L[i][r] * mapped[i][j] for i in range(len(W_L)))

        # B = temp * W_R  (r x r)
        B = [[0.0 for _ in range(rank)] for _ in range(rank)]
        for r in range(rank):
            for c in range(rank):
                B[r][c] = sum(temp[r][j] * W_R[j][c] for j in range(len(W_R)))
        B_list.append(B)

    init_vec = [W_L[prefix_zero_idx][r] for r in range(rank)]
    final_vec = [W_R[suffix_zero_idx][r] for r in range(rank)]
    return B_list, init_vec, final_vec, sigma_min


def apply_sequence(seq: Sequence[int], B_list: Sequence[Sequence[Sequence[float]]], init_vec: Sequence[float], final_vec: Sequence[float]) -> float:
    state = list(init_vec)
    for sym in seq:
        B = B_list[sym]
        state = [sum(B[i][j] * state[j] for j in range(len(state))) for i in range(len(state))]
    return sum(state[i] * final_vec[i] for i in range(len(state)))


def project_row_substochastic(B_list: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
    projected: List[List[List[float]]] = []
    for B in B_list:
        rows = len(B)
        cols = len(B[0]) if B else 0
        out = [[abs(B[i][j]) for j in range(cols)] for i in range(rows)]
        for i in range(rows):
            s = sum(out[i])
            if s > 1.0:
                out[i] = [x / s for x in out[i]]
        projected.append(out)
    return projected


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def sample_sequences(prob: Dict[Tuple[int, ...], float], n: int) -> List[List[int]]:
    keys = list(prob.keys())
    cdf: List[float] = []
    running = 0.0
    for k in keys:
        running += prob[k]
        cdf.append(running)
    samples: List[List[int]] = []
    for _ in range(n):
        r = random.random() * running
        idx = 0
        while idx < len(cdf) and cdf[idx] < r:
            idx += 1
        idx = min(idx, len(keys) - 1)
        samples.append(list(keys[idx]))
    return samples


def evaluate_errors(
    B_list: Sequence[Sequence[Sequence[float]]],
    init_vec: Sequence[float],
    final_vec: Sequence[float],
    prob_true: Dict[Tuple[int, ...], float],
    eval_words: Sequence[Sequence[int]],
) -> Tuple[float, float]:
    max_err = 0.0
    tv_err = 0.0
    for word in eval_words:
        p_true = prob_true.get(tuple(word), 0.0)
        p_hat = apply_sequence(word, B_list, init_vec, final_vec)
        p_hat = max(p_hat, 0.0)
        max_err = max(max_err, abs(p_true - p_hat))
        tv_err += abs(p_true - p_hat)
    tv_err *= 0.5
    return max_err, tv_err


def compute_coherence(hankel: Sequence[Sequence[float]]) -> float:
    if not hankel:
        return 0.0
    row_sums = [sum(row) for row in hankel]
    col_sums = [sum(hankel[i][j] for i in range(len(hankel))) for j in range(len(hankel[0]))]
    return max(max(row_sums), max(col_sums))


def numerical_rank(singular_values: Sequence[float], tol: float) -> int:
    if not singular_values:
        return 0
    top = singular_values[0]
    return sum(1 for s in singular_values if s >= tol * top)


def wfa_ground_truth(d: int, dim: int) -> Tuple[List[List[List[float]]], List[float], List[float]]:
    B_list: List[List[List[float]]] = []
    for _ in range(d):
        mat: List[List[float]] = []
        for _ in range(dim):
            row = [random.random() for _ in range(dim)]
            s = sum(row)
            row = [x / s for x in row] if s > 0 else row
            mat.append(row)
        B_list.append(mat)
    init_vec = [0.0 for _ in range(dim)]
    init_vec[0] = 1.0
    final_vec = [1.0 / dim for _ in range(dim)]
    return B_list, init_vec, final_vec


def probability_from_wfa(
    B_list: Sequence[Sequence[Sequence[float]]],
    init_vec: Sequence[float],
    final_vec: Sequence[float],
    support: Sequence[Sequence[int]],
) -> Dict[Tuple[int, ...], float]:
    probs: Dict[Tuple[int, ...], float] = {}
    total = 0.0
    for word in support:
        val = apply_sequence(word, B_list, init_vec, final_vec)
        val = max(val, 0.0)
        probs[tuple(word)] = val
        total += val
    if total > 0:
        for k in list(probs.keys()):
            probs[k] /= total
    return probs


def kappa_from_mps(cores: Sequence[Sequence[Sequence[complex]]]) -> float:
    max_kappa = 0.0
    for site in cores:
        for mat in site:
            max_kappa = max(max_kappa, spectral_norm(mat) ** 2)
    return max_kappa


def run_configuration(
    model: str,
    length: int,
    d: int,
    bond_low: int,
    bond_high: int,
    wfa_dim: int,
    sample_size: int,
    eval_cap: int,
    max_prefixes: int,
    max_suffixes: int,
    rank_cap: int,
    tol: float,
) -> Dict[str, float]:
    if model == "contractive":
        B_true, i_true, f_true = wfa_ground_truth(d, wfa_dim)
        support = all_words(length, d)
        prob_true = probability_from_wfa(B_true, i_true, f_true, support)
        kappa_B = max(spectral_norm(B) for B in B_true) if B_true else 0.0
    else:
        bond_dim = bond_low if model == "low" else bond_high
        cores, alpha, beta = random_left_canonical_mps(length, bond_dim, d)
        support = all_words(length, d)
        prob_true = probability_map(cores, alpha, beta, length, d, support)
        kappa_B = kappa_from_mps(cores)

    prefixes, suffixes = enumerate_prefix_suffix(length, d, max_prefixes, max_suffixes)
    hankel_true = hankel_from_prob_map(prob_true, prefixes, suffixes)
    masks = hankel_column_masks(suffixes, d)
    hankel_blocks_true = [masked_sub_hankel(hankel_true, masks[sigma]) for sigma in range(d)]
    if hankel_true and hankel_true[0]:
        _, S_full, _ = truncated_svd_rank(hankel_true, min(len(hankel_true), len(hankel_true[0])))
    else:
        S_full = []
    true_rank = numerical_rank(S_full, tol)
    gamma = S_full[true_rank - 1] if true_rank > 0 else 0.0
    mu = compute_coherence(hankel_true)

    samples = sample_sequences(prob_true, sample_size)
    emp_hankel, emp_blocks = empirical_hankel_samples(samples, prefixes, suffixes, d)
    delta = spectral_norm([[hankel_true[i][j] - emp_hankel[i][j] for j in range(len(suffixes))] for i in range(len(prefixes))])

    zero_prefix = prefixes.index([0 for _ in range(length // 2)]) if [0 for _ in range(length // 2)] in prefixes else 0
    zero_suffix = suffixes.index([0 for _ in range(length - length // 2)]) if [0 for _ in range(length - length // 2)] in suffixes else 0

    use_rank = max(1, min(rank_cap, true_rank if true_rank > 0 else rank_cap))
    B_hat, i_hat, f_hat, sigma_min_emp = spectral_wfa_from_hankel(emp_hankel, emp_blocks, use_rank, zero_prefix, zero_suffix)
    eval_set = support if len(support) <= eval_cap else sample_words(length, d, eval_cap)
    max_err_raw, tv_err_raw = evaluate_errors(B_hat, i_hat, f_hat, prob_true, eval_set)

    B_proj = project_row_substochastic(B_hat)
    max_err_proj, tv_err_proj = evaluate_errors(B_proj, i_hat, f_hat, prob_true, eval_set)

    F_raw = length * (kappa_B ** max(length - 1, 0))
    F_proj = float(length)
    scaled_raw = (max_err_raw * gamma / F_raw) if F_raw > 0 and gamma > 0 else 0.0
    scaled_proj = (max_err_proj * gamma / F_proj) if F_proj > 0 and gamma > 0 else 0.0

    return {
        "model": model,
        "length": length,
        "sample_size": sample_size,
        "rank_true": true_rank,
        "rank_used": use_rank,
        "delta_hankel": delta,
        "sigma_min_true": gamma,
        "sigma_min_emp": sigma_min_emp,
        "mu": mu,
        "kappa_B": kappa_B,
        "F_raw": F_raw,
        "max_error_raw": max_err_raw,
        "tv_error_raw": tv_err_raw,
        "max_error_proj": max_err_proj,
        "tv_error_proj": tv_err_proj,
        "scaled_error_raw": scaled_raw,
        "scaled_error_proj": scaled_proj,
        "prefixes": len(prefixes),
        "suffixes": len(suffixes),
        "eval_words": len(eval_set),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_comma_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 5: Hankel spectral learning vs sample complexity")
    parser.add_argument("--lengths", type=str, default="8,10", help="Comma-separated sequence lengths")
    parser.add_argument("--sample-sizes", type=str, default="1000,3000,10000", help="Comma-separated sample sizes")
    parser.add_argument("--models", type=str, default="low,high", help="Comma-separated models: low,high,contractive")
    parser.add_argument("--bond-low", type=int, default=2, help="Bond dimension for low-rank MPS")
    parser.add_argument("--bond-high", type=int, default=4, help="Bond dimension for higher-entropy MPS")
    parser.add_argument("--wfa-dim", type=int, default=3, help="State dimension for contractive WFA ground truth")
    parser.add_argument("--rank", type=int, default=4, help="Truncated rank for spectral learning")
    parser.add_argument("--alphabet", type=int, default=2, help="Alphabet size")
    parser.add_argument("--trials", type=int, default=5, help="Trials per configuration")
    parser.add_argument("--eval-cap", type=int, default=4096, help="Max evaluation words (all if smaller)")
    parser.add_argument("--max-prefixes", type=int, default=512, help="Cap on prefixes")
    parser.add_argument("--max-suffixes", type=int, default=512, help="Cap on suffixes")
    parser.add_argument("--tol", type=float, default=1e-10, help="Relative tolerance for numerical rank")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="experiments/exp5_results.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    lengths = parse_comma_ints(args.lengths)
    sample_sizes = parse_comma_ints(args.sample_sizes)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows: List[Dict[str, float]] = []
    for model in models:
        for length in lengths:
            for n in sample_sizes:
                for t in range(args.trials):
                    res = run_configuration(
                        model=model,
                        length=length,
                        d=args.alphabet,
                        bond_low=args.bond_low,
                        bond_high=args.bond_high,
                        wfa_dim=args.wfa_dim,
                        sample_size=n,
                        eval_cap=args.eval_cap,
                        max_prefixes=args.max_prefixes,
                        max_suffixes=args.max_suffixes,
                        rank_cap=args.rank,
                        tol=args.tol,
                    )
                    res["trial"] = t + 1
                    rows.append(res)

    fieldnames = [
        "model",
        "length",
        "sample_size",
        "trial",
        "rank_true",
        "rank_used",
        "delta_hankel",
        "sigma_min_true",
        "sigma_min_emp",
        "mu",
        "kappa_B",
        "F_raw",
        "max_error_raw",
        "tv_error_raw",
        "max_error_proj",
        "tv_error_proj",
        "scaled_error_raw",
        "scaled_error_proj",
        "prefixes",
        "suffixes",
        "eval_words",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
