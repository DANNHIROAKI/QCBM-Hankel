#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 5: Hankel spectral learning vs sample size and length.

This version is aligned with the strengthened plan:

- multiple ground-truth sources (low-rank MPS, higher-entropy MPS, and an
  optional contractive WFA model);
- fixed mid-cut Hankel construction with capped prefix/suffix sets and recorded
  coherence / gamma / kappa_B;
- rank-truncated SVD + whitening to build a WFA estimator;
- correct construction of B_sigma using column index sets J_sigma;
- optional row-substochastic projection to enforce a contractive regime;
- end-to-end errors (TV / max) alongside Hankel deviation and normalised
  scalings gamma/F(L) to check the N^{-1/2} law and the contrast between
  geometric vs linear length dependence.

Outputs are written to a CSV file; no visualisation is performed here.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Linear algebra helpers (dependency-free)
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


def spectral_norm(mat: Sequence[Sequence[complex]], iters: int = 30) -> float:
    """Power iteration for spectral norm."""
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
    """
    Newton–Schulz iteration for inverse sqrt of a positive definite matrix.
    Used to enforce left-canonical normalisation for MPS sites.
    """
    n = len(mat)
    if n == 0:
        return []
    trace = sum(mat[i][i].real for i in range(n))
    scale = trace / n if trace != 0 else 1.0
    A = mat_scale(mat, 1.0 / scale)
    Y = [row[:] for row in A]
    Z = identity(n)
    for _ in range(iters):
        ZY = matmul(Z, Y)
        # mid = 3I - ZY
        threeI = mat_add(identity(n), mat_add(identity(n), identity(n)))
        mid = mat_add(threeI, mat_scale(ZY, -1.0))
        mid = mat_scale(mid, 0.5)
        Y = matmul(Y, mid)
        Z = matmul(mid, Z)
    return mat_scale(Z, 1.0 / math.sqrt(scale))


# ---------------------------------------------------------------------------
# MPS generation and amplitudes
# ---------------------------------------------------------------------------


def random_left_canonical_mps(
    length: int, bond_dim: int, d: int
) -> Tuple[List[List[List[complex]]], List[complex], List[complex]]:
    """
    Random left-canonical MPS cores and boundary vectors (alpha, beta).
    cores[site][symbol] is a bond_dim x bond_dim matrix.
    """
    cores: List[List[List[complex]]] = []
    for _ in range(length):
        raw: List[List[List[complex]]] = []
        for _ in range(d):
            mat_site = [
                [random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)]
                for _ in range(bond_dim)
            ]
            raw.append(mat_site)
        # sum A A^† over symbols
        M = [[0.0j for _ in range(bond_dim)] for _ in range(bond_dim)]
        for mat_site in raw:
            Mt = matmul(mat_site, conj_transpose(mat_site))
            M = mat_add(M, Mt)
        inv_sqrt = matrix_power_inverse_sqrt(M)
        site = [matmul(inv_sqrt, mat_site) for mat_site in raw]
        cores.append(site)
    alpha = normalize([random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)])
    beta = normalize([random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)])
    return cores, alpha, beta


def amplitude_for_sequence(
    seq: Sequence[int],
    cores: Sequence[Sequence[Sequence[complex]]],
    alpha: Sequence[complex],
    beta: Sequence[complex],
) -> complex:
    vec = list(alpha)
    for site, sym in enumerate(seq):
        vec = matvec(cores[site][sym], vec)
    return dot(vec, beta)


# ---------------------------------------------------------------------------
# Word helpers
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


# ---------------------------------------------------------------------------
# Ground truth probabilities (MPS / WFA)
# ---------------------------------------------------------------------------


def probability_map(
    cores: Sequence[Sequence[Sequence[complex]]],
    alpha: Sequence[complex],
    beta: Sequence[complex],
    length: int,
    d: int,
    support: Sequence[Sequence[int]],
) -> Dict[Tuple[int, ...], float]:
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


def apply_sequence(
    seq: Sequence[int],
    B_list: Sequence[Sequence[Sequence[float]]],
    init_vec: Sequence[float],
    final_vec: Sequence[float],
) -> float:
    state = list(init_vec)
    for sym in seq:
        B = B_list[sym]
        new_state = [0.0 for _ in range(len(state))]
        for i in range(len(B)):
            s_val = 0.0
            for j in range(len(state)):
                s_val += B[i][j] * state[j]
            new_state[i] = s_val
        state = new_state
    val = 0.0
    for i in range(len(state)):
        val += state[i] * final_vec[i]
    return val


def wfa_ground_truth(d: int, dim: int) -> Tuple[List[List[List[float]]], List[float], List[float]]:
    """
    Simple nonnegative row-stochastic WFA:
    - B_sigma: each row sums to 1;
    - init_vec = e_1; final_vec = uniform.
    """
    B_list: List[List[List[float]]] = []
    for _ in range(d):
        mat: List[List[float]] = []
        for _ in range(dim):
            row = [max(random.random(), 1e-12) for _ in range(dim)]
            s = sum(row)
            row = [x / s for x in row]
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


# ---------------------------------------------------------------------------
# Hankel construction
# ---------------------------------------------------------------------------


def enumerate_prefix_suffix(
    length: int, d: int, max_prefixes: int, max_suffixes: int
) -> Tuple[List[List[int]], List[List[int]]]:
    t_star = length // 2
    prefixes = sample_words(t_star, d, max_prefixes)
    suffixes = sample_words(length - t_star, d, max_suffixes)
    # ensure all-zero prefix/suffix exist
    empty_prefix = [0 for _ in range(t_star)]
    empty_suffix = [0 for _ in range(length - t_star)]
    if empty_prefix not in prefixes:
        prefixes.insert(0, empty_prefix)
    if empty_suffix not in suffixes:
        suffixes.insert(0, empty_suffix)
    return prefixes, suffixes


def hankel_from_prob_map(
    prob: Dict[Tuple[int, ...], float],
    prefixes: Sequence[Sequence[int]],
    suffixes: Sequence[Sequence[int]],
) -> List[List[float]]:
    hankel: List[List[float]] = []
    for pre in prefixes:
        row: List[float] = []
        for suf in suffixes:
            word = tuple(list(pre) + list(suf))
            row.append(prob.get(word, 0.0))
        hankel.append(row)
    return hankel


def hankel_column_masks(suffixes: Sequence[Sequence[int]], d: int) -> List[List[bool]]:
    """
    masks[sigma][j] = True iff suffix j starts with symbol sigma.
    """
    masks: List[List[bool]] = []
    for sigma in range(d):
        mask = []
        for suf in suffixes:
            mask.append(len(suf) > 0 and suf[0] == sigma)
        masks.append(mask)
    return masks


def compute_coherence(hankel: Sequence[Sequence[float]]) -> float:
    if not hankel:
        return 0.0
    row_sums = [sum(row) for row in hankel]
    col_sums = [sum(hankel[i][j] for i in range(len(hankel))) for j in range(len(hankel[0]))]
    return max(max(row_sums), max(col_sums))


# ---------------------------------------------------------------------------
# Empirical Hankel from samples
# ---------------------------------------------------------------------------


def empirical_hankel(
    samples: Sequence[Sequence[int]],
    prefixes: Sequence[Sequence[int]],
    suffixes: Sequence[Sequence[int]],
) -> Tuple[List[List[float]], int]:
    """
    Build empirical Hankel from samples by counting prefix/suffix pairs
    restricted to the given sets.
    """
    m = len(prefixes)
    n = len(suffixes)
    H = [[0.0 for _ in range(n)] for _ in range(m)]
    if m == 0 or n == 0:
        return H, 0
    t_star = len(prefixes[0])

    prefix_index: Dict[Tuple[int, ...], int] = {tuple(p): i for i, p in enumerate(prefixes)}
    suffix_index: Dict[Tuple[int, ...], int] = {tuple(s): j for j, s in enumerate(suffixes)}

    valid = 0
    for seq in samples:
        pre = tuple(seq[:t_star])
        suf = tuple(seq[t_star:])
        i = prefix_index.get(pre, -1)
        j = suffix_index.get(suf, -1)
        if i < 0 or j < 0:
            continue
        H[i][j] += 1.0
        valid += 1

    if valid > 0:
        inv = 1.0 / float(valid)
        for i in range(m):
            for j in range(n):
                H[i][j] *= inv
    return H, valid


# ---------------------------------------------------------------------------
# SVD and spectral utilities
# ---------------------------------------------------------------------------


def truncated_svd_rank(
    hankel: Sequence[Sequence[float]], rank: int
) -> Tuple[List[List[complex]], List[float], List[List[complex]]]:
    """
    Simple truncated SVD via power-iteration on HH^T.
    Returns U (m x r), S (r), V (n x r) with hankel ≈ U diag(S) V^T.
    """
    m = len(hankel)
    n = len(hankel[0]) if hankel else 0
    if m == 0 or n == 0:
        return [], [], []

    rank = max(1, min(rank, m, n))

    # Build Gram = H H^T
    gram: List[List[float]] = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            s_val = 0.0
            row_i = hankel[i]
            row_j = hankel[j]
            for k in range(n):
                s_val += row_i[k] * row_j[k]
            gram[i][j] = s_val

    U: List[List[complex]] = [[0.0j for _ in range(rank)] for _ in range(m)]
    S: List[float] = [0.0 for _ in range(rank)]

    for r in range(rank):
        v = normalize([random.random() for _ in range(m)])
        for _ in range(40):
            w = [0.0 for _ in range(m)]
            for i in range(m):
                s_val = 0.0
                row_i = gram[i]
                for j in range(m):
                    s_val += row_i[j] * v[j]
                w[i] = s_val
            v = normalize(w)
        # Rayleigh quotient v^T Gram v = sigma^2
        sigma_sq = 0.0
        for i in range(m):
            inner = 0.0
            for j in range(m):
                inner += gram[i][j] * v[j]
            sigma_sq += v[i] * inner
        sigma_sq = float(sigma_sq)
        sigma = math.sqrt(max(sigma_sq, 0.0))
        S[r] = sigma
        for i in range(m):
            U[i][r] = v[i]
        # Deflate Gram: Gram <- Gram - sigma^2 v v^T
        for i in range(m):
            for j in range(m):
                gram[i][j] -= sigma_sq * (v[i] * v[j])

    # Compute V via H^T U S^{-1}
    V: List[List[complex]] = [[0.0j for _ in range(rank)] for _ in range(n)]
    for k in range(n):
        for r in range(rank):
            if S[r] > 0:
                s_val = 0.0
                for i in range(m):
                    s_val += hankel[i][k] * U[i][r]
                V[k][r] = s_val / S[r]
    return U, S, V


def numerical_rank(singular_values: Sequence[float], tol: float) -> int:
    if not singular_values:
        return 0
    top = singular_values[0]
    if top <= 0:
        return 0
    return sum(1 for s in singular_values if s >= tol * top)


def spectral_wfa_from_hankel(
    hankel: Sequence[Sequence[float]],
    masks: Sequence[Sequence[bool]],
    rank: int,
) -> Tuple[List[List[List[float]]], List[List[float]], List[List[float]], float]:
    """
    Given empirical Hankel Ĥ and column masks for each sigma, build:
    - whiteners W_L (r x m), W_R (n x r)
    - transition operators B_sigma = W_L H_sigma W_R, with correct column indices.
    """
    m = len(hankel)
    n = len(hankel[0]) if hankel else 0
    if m == 0 or n == 0:
        return [], [], [], 0.0

    U, S, V = truncated_svd_rank(hankel, rank)
    r = len(S)
    if r == 0:
        return [], [], [], 0.0

    eps = 1e-12
    inv_sqrt = [1.0 / math.sqrt(s + eps) for s in S]

    # W_L: r x m (rows: latent dim, cols: prefix index)
    W_L: List[List[float]] = [[0.0 for _ in range(m)] for _ in range(r)]
    for i in range(m):
        for a in range(r):
            W_L[a][i] = (U[i][a].real) * inv_sqrt[a]

    # W_R: n x r (rows: suffix index, cols: latent dim)
    W_R: List[List[float]] = [[0.0 for _ in range(r)] for _ in range(n)]
    for j in range(n):
        for a in range(r):
            W_R[j][a] = (V[j][a].real) * inv_sqrt[a]

    # temp = W_L @ Ĥ  (r x n)
    temp: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(r)]
    for a in range(r):
        for j in range(n):
            s_val = 0.0
            for i in range(m):
                s_val += W_L[a][i] * hankel[i][j]
            temp[a][j] = s_val

    B_list: List[List[List[float]]] = []
    for sigma, mask in enumerate(masks):
        indices = [j for j, flag in enumerate(mask) if flag]
        B_sigma = [[0.0 for _ in range(r)] for _ in range(r)]
        if indices:
            for a in range(r):
                for b in range(r):
                    s_val = 0.0
                    for j in indices:
                        s_val += temp[a][j] * W_R[j][b]
                    B_sigma[a][b] = s_val
        B_list.append(B_sigma)

    sigma_min_emp = min(S) if S else 0.0
    return B_list, W_L, W_R, sigma_min_emp


# ---------------------------------------------------------------------------
# Sampling and evaluation
# ---------------------------------------------------------------------------


def sample_sequences(prob: Dict[Tuple[int, ...], float], n: int) -> List[List[int]]:
    keys = list(prob.keys())
    cdf: List[float] = []
    running = 0.0
    for k in keys:
        running += prob[k]
        cdf.append(running)
    samples: List[List[int]] = []
    if running == 0.0:
        return [[0] for _ in range(n)]
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
    """
    Compute pointwise max and TV error on eval_words.
    We normalise the learned probabilities over eval_words.
    """
    vals_true: List[float] = []
    vals_hat: List[float] = []

    for word in eval_words:
        key = tuple(word)
        p_true = prob_true.get(key, 0.0)
        p_hat = apply_sequence(word, B_list, init_vec, final_vec)
        if p_hat < 0.0:
            p_hat = 0.0
        vals_true.append(p_true)
        vals_hat.append(p_hat)

    total_hat = sum(vals_hat)
    if total_hat > 0.0:
        vals_hat = [v / total_hat for v in vals_hat]

    max_err = 0.0
    tv_err = 0.0
    for pt, ph in zip(vals_true, vals_hat):
        diff = abs(pt - ph)
        max_err = max(max_err, diff)
        tv_err += diff
    tv_err *= 0.5
    return max_err, tv_err


# ---------------------------------------------------------------------------
# Kappa from MPS (Liouville proxy)
# ---------------------------------------------------------------------------


def kappa_from_mps(cores: Sequence[Sequence[Sequence[complex]]]) -> float:
    """
    Proxy for kappa_B: max over sites/symbols of ||A||_2^2.
    """
    max_kappa = 0.0
    for site in cores:
        for mat_site in site:
            max_kappa = max(max_kappa, spectral_norm(mat_site) ** 2)
    return max_kappa


# ---------------------------------------------------------------------------
# One configuration run
# ---------------------------------------------------------------------------


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
    # Ground truth distribution
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

    # Hankel construction (true)
    prefixes, suffixes = enumerate_prefix_suffix(length, d, max_prefixes, max_suffixes)
    hankel_true = hankel_from_prob_map(prob_true, prefixes, suffixes)
    masks = hankel_column_masks(suffixes, d)

    if hankel_true and hankel_true[0]:
        _, S_full, _ = truncated_svd_rank(
            hankel_true, min(len(hankel_true), len(hankel_true[0]))
        )
    else:
        S_full = []
    rank_true = numerical_rank(S_full, tol)
    gamma = S_full[rank_true - 1] if rank_true > 0 else 0.0
    mu = compute_coherence(hankel_true)

    # Empirical Hankel from samples
    samples = sample_sequences(prob_true, sample_size)
    emp_hankel, effective_samples = empirical_hankel(samples, prefixes, suffixes)
    # Spectral norm of Hankel error
    diff = [
        [hankel_true[i][j] - emp_hankel[i][j] for j in range(len(suffixes))]
        for i in range(len(prefixes))
    ]
    delta_hankel = spectral_norm(diff)

    # Indices for "all-zero" prefix/suffix proxies
    t_star = length // 2
    zero_prefix_word = tuple([0 for _ in range(t_star)])
    zero_suffix_word = tuple([0 for _ in range(length - t_star)])
    prefix_index = {tuple(p): idx for idx, p in enumerate(prefixes)}
    suffix_index = {tuple(s): idx for idx, s in enumerate(suffixes)}
    prefix_zero_idx = prefix_index.get(zero_prefix_word, 0)
    suffix_zero_idx = suffix_index.get(zero_suffix_word, 0)

    # Rank used (capped)
    rank_used = max(1, min(rank_cap, rank_true if rank_true > 0 else rank_cap))

    # Spectral WFA from empirical Hankel
    B_hat, W_L, W_R, sigma_min_emp = spectral_wfa_from_hankel(emp_hankel, masks, rank_used)

    # Build boundary vectors in whitened coordinates:
    # i_hat^T = Ĥ(u0,:) W_R,  f_hat = W_L Ĥ(:,v0)
    m = len(prefixes)
    n = len(suffixes)
    r = rank_used

    if n > 0 and m > 0 and r > 0 and B_hat:
        row0 = emp_hankel[prefix_zero_idx]
        init_vec = [0.0 for _ in range(r)]
        for a in range(r):
            s_val = 0.0
            for j in range(n):
                s_val += row0[j] * W_R[j][a]
            init_vec[a] = s_val

        col0 = [emp_hankel[i][suffix_zero_idx] for i in range(m)]
        final_vec = [0.0 for _ in range(r)]
        for a in range(r):
            s_val = 0.0
            for i in range(m):
                s_val += W_L[a][i] * col0[i]
            final_vec[a] = s_val
    else:
        init_vec = [0.0 for _ in range(rank_used)]
        final_vec = [0.0 for _ in range(rank_used)]

    # Evaluation set
    if len(support) <= eval_cap:
        eval_set = support
    else:
        eval_set = sample_words(length, d, eval_cap)

    max_error_raw, tv_error_raw = evaluate_errors(B_hat, init_vec, final_vec, prob_true, eval_set)

    # Contractive projection
    B_proj = project_row_substochastic(B_hat)
    max_error_proj, tv_error_proj = evaluate_errors(B_proj, init_vec, final_vec, prob_true, eval_set)

    # Sum of B_hat vs identity as a sanity check: ||Σ B_hat - I||_2
    sum_B = [[0.0 for _ in range(r)] for _ in range(r)]
    for B in B_hat:
        for i in range(r):
            for j in range(r):
                sum_B[i][j] += B[i][j]
    I_r = [[1.0 if i == j else 0.0 for j in range(r)] for i in range(r)]
    diff_sumB = [
        [sum_B[i][j] - I_r[i][j] for j in range(r)]
        for i in range(r)
    ]
    sumB_norm_raw = spectral_norm(diff_sumB) if r > 0 else 0.0

    # Length factors
    if kappa_B > 0.0:
        F_raw = float(length) * (kappa_B ** max(length - 1, 0))
    else:
        F_raw = float(length)
    F_proj = float(length)

    scaled_raw = max_error_raw * gamma / F_raw if (F_raw > 0 and gamma > 0) else 0.0
    scaled_proj = max_error_proj * gamma / F_proj if (F_proj > 0 and gamma > 0) else 0.0

    return {
        "model": model,
        "length": length,
        "sample_size": sample_size,
        "rank_true": rank_true,
        "rank_used": rank_used,
        "delta_hankel": delta_hankel,
        "sigma_min_true": gamma,
        "sigma_min_emp": sigma_min_emp,
        "mu": mu,
        "kappa_B": kappa_B,
        "F_raw": F_raw,
        "F_proj": F_proj,
        "max_error_raw": max_error_raw,
        "tv_error_raw": tv_error_raw,
        "max_error_proj": max_error_proj,
        "tv_error_proj": tv_error_proj,
        "scaled_error_raw": scaled_raw,
        "scaled_error_proj": scaled_proj,
        "prefixes": len(prefixes),
        "suffixes": len(suffixes),
        "eval_words": len(eval_set),
        "effective_samples": effective_samples,
        "sumB_norm_raw": sumB_norm_raw,
    }


# ---------------------------------------------------------------------------
# Row-substochastic projection (contractive cone)
# ---------------------------------------------------------------------------


def project_row_substochastic(
    B_list: Sequence[Sequence[Sequence[float]]]
) -> List[List[List[float]]]:
    """
    Project each B_sigma to a nonnegative row-substochastic matrix:
    - take absolute value elementwise;
    - if a row sums to >1, rescale that row to sum to 1.
    This guarantees ||B_sigma||_∞ <= 1, hence non-expansion in ℓ_∞.
    """
    projected: List[List[List[float]]] = []
    for B in B_list:
        rows = len(B)
        cols = len(B[0]) if B else 0
        out = [[abs(B[i][j]) for j in range(cols)] for i in range(rows)]
        for i in range(rows):
            s = sum(out[i])
            if s > 1.0:
                inv = 1.0 / s
                out[i] = [x * inv for x in out[i]]
        projected.append(out)
    return projected


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_comma_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Hankel spectral learning vs sample complexity"
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="8,10",
        help="Comma-separated sequence lengths",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="1000,3000,10000",
        help="Comma-separated sample sizes",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="low,high",
        help="Comma-separated models: low,high,contractive",
    )
    parser.add_argument(
        "--bond-low",
        type=int,
        default=2,
        help="Bond dimension for low-rank MPS",
    )
    parser.add_argument(
        "--bond-high",
        type=int,
        default=4,
        help="Bond dimension for higher-entropy MPS",
    )
    parser.add_argument(
        "--wfa-dim",
        type=int,
        default=3,
        help="State dimension for contractive WFA ground truth",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Truncated rank cap for spectral learning",
    )
    parser.add_argument(
        "--alphabet",
        type=int,
        default=2,
        help="Alphabet size",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Trials per configuration",
    )
    parser.add_argument(
        "--eval-cap",
        type=int,
        default=4096,
        help="Max evaluation words (all if smaller)",
    )
    parser.add_argument(
        "--max-prefixes",
        type=int,
        default=512,
        help="Cap on prefixes",
    )
    parser.add_argument(
        "--max-suffixes",
        type=int,
        default=512,
        help="Cap on suffixes",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Relative tolerance for numerical rank",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp5_results.csv",
        help="Output CSV path",
    )
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
        "F_proj",
        "max_error_raw",
        "tv_error_raw",
        "max_error_proj",
        "tv_error_proj",
        "scaled_error_raw",
        "scaled_error_proj",
        "prefixes",
        "suffixes",
        "eval_words",
        "effective_samples",
        "sumB_norm_raw",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()




"""
python experiments/exp5_sample_complexity.py --lengths 8,10 --sample-sizes 1000,3000,10000 --models low,high,contractive --bond-low 2 --bond-high 4 --wfa-dim 3 --rank 4 --trials 5 --seed 0 --output experiments/exp5_results.csv
"""