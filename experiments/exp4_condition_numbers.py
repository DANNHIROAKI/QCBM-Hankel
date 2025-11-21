#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 4 (epsilon-extended): Condition numbers and hard examples (rank-deficient MPS).

This script implements the *strengthened* Experiment 4 design aligned with the
updated outline:

  Part A (structural):
    - Instantiate the rank-1 (7.2A) and rank-3 (7.2B) diagonal MPS;
    - Build mid-cut Hankel matrices H_mid^{(L)} for several lengths L;
    - Compute top-3 singular values of H_mid^{(L)} and record sigma_3(L),
      log sigma_3(L), numerical rank, coherence, and condition number;
    - This validates that the rank-3 "hard" example is rank-3 but severely
      ill-conditioned, while the rank-1 example is well-conditioned.

  Part B (learning / sample complexity):
    - From H_mid^{(L)} and the (L+1)-length distribution, construct an
      epsilon-extended Hankel H_ext and blocks H_ext_sigma by adding
      an explicit ε-row/ε-column using prefix/suffix marginals.
    - Use a standard Hankel spectral-learning pipeline:

        H_ext ≈ U Σ V^T     (rank-r truncated SVD)
        P  = U Σ^{1/2},     S = Σ^{1/2} V^T
        P^+ = Σ^{-1/2} U^T, S^+ = V Σ^{-1/2}
        B_σ = P^+ H_ext_σ S^+
        α^T = h_{ε,*} S^+   (ε-row of H_ext)
        β   = P^+ h_{*,ε}   (ε-column of H_ext)

    - First, run the spectral pipeline on the *true* epsilon-extended Hankels
      (sample_size = 0) to check the baseline (TV error should be << 1e-6 for
      both rank-1 and rank-3).
    - Then, for each sample size N, estimate empirical mid-cut Hankels
      H_mid_hat, H_mid_hat_σ from i.i.d. samples, extend them with ε, run the
      spectral pipeline, and evaluate pointwise / TV errors between the
      reconstructed distribution and the true length-L distribution.

All results are written to a single CSV, one row per (model, length, sample_size, trial).

Usage example:
    python exp4_condition_numbers.py \\
        --lengths 6,10,14 \\
        --eta 0.6 --c-param 0.5 \\
        --sample-sizes 200,1000,5000 \\
        --trials 5 \\
        --seed 0 \\
        --output experiments/exp4_results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Basic linear algebra (pure Python, float only)
# ---------------------------------------------------------------------------


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def norm(u: Sequence[float]) -> float:
    return math.sqrt(dot(u, u))


def normalize(u: Sequence[float]) -> List[float]:
    n = norm(u)
    if n == 0.0:
        return list(u)
    return [x / n for x in u]


def matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    """Matrix-vector product: (m x n) * (n,) -> (m,)."""
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def transpose(a: Sequence[Sequence[float]]) -> List[List[float]]:
    if not a:
        return []
    rows = len(a)
    cols = len(a[0])
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    """Matrix-matrix product: (m x k) * (k x n) -> (m x n)."""
    if not a or not b:
        return []
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        out_i = out[i]
        row = a[i]
        for t in range(k):
            aik = row[t]
            if aik == 0.0:
                continue
            bt = b[t]
            for j in range(n):
                out_i[j] += aik * bt[j]
    return out


def outer(u: Sequence[float], v: Sequence[float]) -> List[List[float]]:
    return [[a * b for b in v] for a in u]


def frobenius_norm(a: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(x * x for row in a for x in row))


def spectral_norm(a: Sequence[Sequence[float]], iters: int = 60) -> float:
    """
    Approximate spectral norm ||A||_2 via power iteration on A^T A.
    Good enough for the small matrices in this experiment.
    """
    if not a or not a[0]:
        return 0.0
    m = len(a)
    n = len(a[0])
    v = normalize([random.random() for _ in range(n)])
    at = transpose(a)
    for _ in range(iters):
        w = matvec(a, v)
        v = normalize(matvec(at, w))
    Av = matvec(a, v)
    return norm(Av)


def power_eigenpair(mat: Sequence[Sequence[float]], iters: int = 80) -> Tuple[float, List[float]]:
    """
    Approximate dominant eigenpair of a symmetric matrix via power iteration.
    """
    if not mat:
        return 0.0, []
    n = len(mat)
    v = normalize([random.random() for _ in range(n)])
    for _ in range(iters):
        v = normalize(matvec(mat, v))
    mv = matvec(mat, v)
    lam = dot(v, mv)
    return lam, v


def truncated_svd(
    mat: Sequence[Sequence[float]],
    rank: int,
) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Approximate rank-r SVD of mat (m x n) using Gram deflation on H H^T.

    Returns:
        U : (m x r)
        S : [σ1,...,σr]  (non-increasing)
        V : (n x r)
    """
    if not mat or rank <= 0:
        return [], [], []
    m = len(mat)
    n = len(mat[0])
    if m == 0 or n == 0:
        return [], [], []

    # Gram matrix G = H H^T
    gram = matmul(mat, transpose(mat))  # (m x m)
    gram_work = [row[:] for row in gram]

    U_cols: List[List[float]] = []  # each is length m
    sigmas: List[float] = []

    for _ in range(rank):
        lam, vec = power_eigenpair(gram_work)
        if lam <= 0.0:
            break
        sigma = math.sqrt(max(lam, 0.0))
        sigmas.append(sigma)
        U_cols.append(vec)
        # Deflate G <- G - lam * v v^T
        vvT = outer(vec, vec)
        for i in range(m):
            gi = gram_work[i]
            vi = vvT[i]
            for j in range(m):
                gi[j] -= lam * vi[j]

    r = len(sigmas)
    if r == 0:
        return [], [], []

    # Build V using V_k = (1/σ_k) H^T u_k
    Ht = transpose(mat)
    V_cols: List[List[float]] = []
    for k in range(r):
        sigma = sigmas[k]
        u_k = U_cols[k]
        Hv = matvec(Ht, u_k)
        if sigma == 0.0:
            V_cols.append([0.0 for _ in range(n)])
        else:
            V_cols.append([x / sigma for x in Hv])

    # Convert column lists into row-major U (m x r), V (n x r)
    U = [[U_cols[k][i] for k in range(r)] for i in range(m)]
    V = [[V_cols[k][i] for k in range(r)] for i in range(n)]
    return U, sigmas, V


def top_k_singular_values(mat: Sequence[Sequence[float]], k: int) -> List[float]:
    U, sigmas, V = truncated_svd(mat, k)
    sigmas = sigmas[:k]
    if len(sigmas) < k:
        sigmas += [0.0 for _ in range(k - len(sigmas))]
    return sigmas


# ---------------------------------------------------------------------------
# Combinatorics on words
# ---------------------------------------------------------------------------


def int_to_word(val: int, length: int, d: int) -> List[int]:
    """Convert 0 <= val < d^length into a base-d word of given length."""
    word = [0 for _ in range(length)]
    for i in range(length - 1, -1, -1):
        word[i] = val % d
        val //= d
    return word


def all_words(length: int, d: int) -> List[List[int]]:
    """Enumerate all words of length 'length' over alphabet {0,...,d-1}."""
    return [int_to_word(val, length, d) for val in range(d**length)]


# ---------------------------------------------------------------------------
# Rank-1 and rank-3 diagonal MPS instances (7.2A / 7.2B)
# ---------------------------------------------------------------------------


def build_rank1_mps(
    max_length: int,
    eta: float,
) -> Tuple[List[List[List[float]]], List[float], List[float]]:
    """
    Rank-1 construction (7.2A):

        A(0) = diag(1, eta)
        A(1) = diag(eta, 1)
        α = β = [1, 0]^T

    We build 'max_length' sites so we can evaluate sequences up to that length.
    """
    A0 = [[1.0, 0.0], [0.0, eta]]
    A1 = [[eta, 0.0], [0.0, 1.0]]
    cores = [[A0, A1] for _ in range(max_length)]
    alpha = [1.0, 0.0]
    beta = [1.0, 0.0]
    return cores, alpha, beta


def build_rank3_mps(
    max_length: int,
    eta: float,
    c_param: float,
) -> Tuple[List[List[List[float]]], List[float], List[float]]:
    """
    Rank-3 construction (7.2B):

        A(0), A(1) as in rank-1,
        α = β = [1, c]^T.
    """
    A0 = [[1.0, 0.0], [0.0, eta]]
    A1 = [[eta, 0.0], [0.0, 1.0]]
    cores = [[A0, A1] for _ in range(max_length)]
    alpha = [1.0, c_param]
    beta = [1.0, c_param]
    return cores, alpha, beta


def amplitude_for_sequence(
    seq: Sequence[int],
    cores: Sequence[Sequence[Sequence[float]]],
    alpha: Sequence[float],
    beta: Sequence[float],
) -> float:
    """Compute amplitude α^T A(x1)...A(xL) β for a given word."""
    vec = list(alpha)
    for site, sym in enumerate(seq):
        A = cores[site][sym]
        vec = matvec(A, vec)
    return dot(vec, beta)


def probability_map_for_length(
    cores: Sequence[Sequence[Sequence[float]]],
    alpha: Sequence[float],
    beta: Sequence[float],
    length: int,
    d: int,
) -> Dict[Tuple[int, ...], float]:
    """
    Enumerate and return normalised probabilities p(x) over Σ^length
    from the given MPS (using the first 'length' cores).
    """
    probs: Dict[Tuple[int, ...], float] = {}
    total = 0.0
    for word in all_words(length, d):
        amp = amplitude_for_sequence(word, cores, alpha, beta)
        p = amp * amp
        key = tuple(word)
        probs[key] = p
        total += p
    if total > 0.0:
        inv_total = 1.0 / total
        for key in probs:
            probs[key] *= inv_total
    return probs


# ---------------------------------------------------------------------------
# Hankel construction (mid-cut and epsilon-extended)
# ---------------------------------------------------------------------------


def midcut_hankel_from_prob_map(
    prob_map: Dict[Tuple[int, ...], float],
    length: int,
    d: int,
) -> Tuple[List[List[float]], List[List[int]], List[List[int]]]:
    """
    Build mid-cut Hankel for length L:

        t_* = floor(L/2)
        P = Σ^{t_*}, S = Σ^{L-t_*}
        H_mid(u,v) = p(uv).

    This is used for Part A (spectral structure) and as the "core" block
    from which we build the epsilon-extended Hankel for Part B.
    """
    t_star = length // 2
    prefixes = all_words(t_star, d)
    suffixes = all_words(length - t_star, d)
    m = len(prefixes)
    n = len(suffixes)
    hankel: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(m)]
    for i, u in enumerate(prefixes):
        for j, v in enumerate(suffixes):
            seq = tuple(u + v)
            hankel[i][j] = prob_map.get(seq, 0.0)
    return hankel, prefixes, suffixes


def hankel_blocks_mid_from_prob_map(
    prob_map_Lplus: Dict[Tuple[int, ...], float],
    prefixes: List[List[int]],
    suffixes: List[List[int]],
    d: int,
) -> List[List[List[float]]]:
    """
    Build mid-cut blocks H_mid_sigma(u,v) = p(u σ v) using the (L+1)-length
    distribution, where u in P = Σ^{t_*}, v in S = Σ^{L-t_*}, σ in {0,...,d-1}.
    """
    m = len(prefixes)
    n = len(suffixes)
    H_blocks: List[List[List[float]]] = []
    for sigma in range(d):
        block = [[0.0 for _ in range(n)] for _ in range(m)]
        for i, u in enumerate(prefixes):
            for j, v in enumerate(suffixes):
                seq = tuple(u + [sigma] + v)
                block[i][j] = prob_map_Lplus.get(seq, 0.0)
        H_blocks.append(block)
    return H_blocks


def extend_hankel_with_epsilon(
    H_mid: List[List[float]],
) -> List[List[float]]:
    """
    Construct an epsilon-extended Hankel H_ext from a mid-cut block H_mid.

    Let H_mid be (m x n) with entries H_mid(i,j) = p(u_i v_j). We build
    H_ext of size (m+1 x n+1) with indices:

        row 0  -> ε        (empty prefix)
        row i>0 -> u_{i-1} (prefix of length t_*)
        col 0  -> ε        (empty suffix)
        col j>0 -> v_{j-1} (suffix of length L-t_*)

    Entries:
        H_ext[0][0]    = 1
        H_ext[0][j+1]  = sum_i H_mid[i][j]   (p(v_j)  : suffix marginals)
        H_ext[i+1][0]  = sum_j H_mid[i][j]   (p(u_i)  : prefix marginals)
        H_ext[i+1][j+1]= H_mid[i][j]         (p(u_i v_j))

    This matches the outline's intent of having ε-row/ε-column defined by
    prefix/suffix marginals, and the interior block equal to the mid-cut Hankel.
    """
    if not H_mid or not H_mid[0]:
        return [[1.0]]
    m = len(H_mid)
    n = len(H_mid[0])
    H_ext: List[List[float]] = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]

    # interior block: copy H_mid
    for i in range(m):
        for j in range(n):
            H_ext[i + 1][j + 1] = H_mid[i][j]

    # epsilon row: suffix marginals
    for j in range(n):
        col_sum = 0.0
        for i in range(m):
            col_sum += H_mid[i][j]
        H_ext[0][j + 1] = col_sum

    # epsilon column: prefix marginals
    for i in range(m):
        row_sum = 0.0
        for j in range(n):
            row_sum += H_mid[i][j]
        H_ext[i + 1][0] = row_sum

    # epsilon-epsilon entry: 1 (normalisation)
    H_ext[0][0] = 1.0
    return H_ext


def extend_hankel_blocks_with_epsilon(
    H_blocks_mid: List[List[List[float]]],
) -> List[List[List[float]]]:
    """
    Given mid-cut blocks H_mid_sigma of shape (m x n) for each σ, build
    epsilon-extended blocks H_ext_sigma of shape (m+1 x n+1) by the
    same pattern as extend_hankel_with_epsilon:

        H_ext_sigma[i+1][j+1] = H_mid_sigma[i][j]
        H_ext_sigma[0][j+1]   = sum_i H_mid_sigma[i][j]
        H_ext_sigma[i+1][0]   = sum_j H_mid_sigma[i][j]
        H_ext_sigma[0][0]     = sum_{i,j} H_mid_sigma[i][j]
                               (probability that the mid symbol = σ).

    This mirrors the construction for H_ext and ensures H_ext_sigma uses
    the same index set (ε + mid-cut prefixes/suffixes).
    """
    if not H_blocks_mid:
        return []
    d = len(H_blocks_mid)
    m = len(H_blocks_mid[0])
    n = len(H_blocks_mid[0][0])
    H_blocks_ext: List[List[List[float]]] = []

    for sigma in range(d):
        H_mid = H_blocks_mid[sigma]
        H_ext = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]

        # interior
        for i in range(m):
            for j in range(n):
                H_ext[i + 1][j + 1] = H_mid[i][j]

        # epsilon row: suffix marginals given σ
        for j in range(n):
            col_sum = 0.0
            for i in range(m):
                col_sum += H_mid[i][j]
            H_ext[0][j + 1] = col_sum

        # epsilon column: prefix marginals given σ
        for i in range(m):
            row_sum = 0.0
            for j in range(n):
                row_sum += H_mid[i][j]
            H_ext[i + 1][0] = row_sum

        # epsilon-epsilon: total mass for σ
        total = 0.0
        for i in range(m):
            for j in range(n):
                total += H_mid[i][j]
        H_ext[0][0] = total

        H_blocks_ext.append(H_ext)

    return H_blocks_ext


# ---------------------------------------------------------------------------
# Sampling and empirical Hankel (mid-cut)
# ---------------------------------------------------------------------------


def sample_sequences(prob_map: Dict[Tuple[int, ...], float], n: int) -> List[Tuple[int, ...]]:
    """
    Draw n i.i.d. samples from a discrete distribution given as prob_map.
    """
    keys = list(prob_map.keys())
    weights = [prob_map[k] for k in keys]
    cumulative: List[float] = []
    total = 0.0
    for w in weights:
        total += w
        cumulative.append(total)
    if total <= 0.0:
        return [tuple()] * n
    samples: List[Tuple[int, ...]] = []
    for _ in range(n):
        r = random.random() * total
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] >= r:
                hi = mid
            else:
                lo = mid + 1
        samples.append(keys[lo])
    return samples


def empirical_midcut_hankel(
    prob_map: Dict[Tuple[int, ...], float],
    prefixes: List[List[int]],
    suffixes: List[List[int]],
    n: int,
) -> List[List[float]]:
    """
    Estimate mid-cut H_mid(u,v) = p(uv) by drawing N samples of length L
    and cutting at t_*.
    """
    m = len(prefixes)
    s = len(suffixes)
    if n <= 0 or m == 0 or s == 0:
        return [[0.0 for _ in range(s)] for _ in range(m)]

    counts = [[0.0 for _ in range(s)] for _ in range(m)]
    samples = sample_sequences(prob_map, n)
    t_star = len(prefixes[0]) if prefixes else 0
    prefix_index = {tuple(p): idx for idx, p in enumerate(prefixes)}
    suffix_index = {tuple(suf): idx for idx, suf in enumerate(suffixes)}

    for seq in samples:
        u = tuple(seq[:t_star])
        v = tuple(seq[t_star:])
        i = prefix_index[u]
        j = suffix_index[v]
        counts[i][j] += 1.0

    inv_n = 1.0 / float(n)
    for i in range(m):
        for j in range(s):
            counts[i][j] *= inv_n
    return counts


def empirical_midcut_hankel_blocks(
    prob_map_Lplus: Dict[Tuple[int, ...], float],
    prefixes: List[List[int]],
    suffixes: List[List[int]],
    n: int,
    d: int,
) -> List[List[List[float]]]:
    """
    Estimate mid-cut blocks H_mid_sigma(u,v) = p(u σ v) from samples of
    length L+1 (same mid-cut).
    """
    m = len(prefixes)
    s = len(suffixes)
    if n <= 0 or m == 0 or s == 0:
        return [[[0.0 for _ in range(s)] for _ in range(m)] for _ in range(d)]

    counts = [[[0.0 for _ in range(s)] for _ in range(m)] for _ in range(d)]
    samples = sample_sequences(prob_map_Lplus, n)
    t_star = len(prefixes[0]) if prefixes else 0
    prefix_index = {tuple(p): idx for idx, p in enumerate(prefixes)}
    suffix_index = {tuple(suf): idx for idx, suf in enumerate(suffixes)}

    for seq in samples:
        # length L+1 = t_* + 1 + (L - t_*)
        u = tuple(seq[:t_star])
        sigma = seq[t_star]
        v = tuple(seq[t_star + 1 :])
        i = prefix_index[u]
        j = suffix_index[v]
        counts[sigma][i][j] += 1.0

    inv_n = 1.0 / float(n)
    for sigma in range(d):
        for i in range(m):
            for j in range(s):
                counts[sigma][i][j] *= inv_n
    return counts


# ---------------------------------------------------------------------------
# Spectral learning (WFA reconstruction from epsilon-extended Hankel)
# ---------------------------------------------------------------------------


def spectral_wfa_from_hankel(
    H_ext: List[List[float]],
    H_blocks_ext: List[List[List[float]]],
    rank: int,
    prefix_anchor: int,
    suffix_anchor: int,
) -> Tuple[List[List[List[float]]], List[float], List[float], float]:
    """
    Standard WFA spectral reconstruction from epsilon-extended H and {H_sigma}:

        H_ext ≈ P S          with  H_ext = U Σ V^T,
        P  = U Σ^{1/2},      S = Σ^{1/2} V^T,
        P^+ = Σ^{-1/2} U^T,  S^+ = V Σ^{-1/2},
        B_σ = P^+ H_ext_σ S^+,
        α^T = h_{u0,*} S^+   (anchor prefix index u0, typically ε-row = 0),
        β   = P^+ h_{*,v0}   (anchor suffix index v0, typically ε-col = 0).

    Returns:
        B_list        : [B_0, B_1, ...] (each r x r)
        alpha, beta   : initial/final vectors (length r)
        sigma_min     : smallest non-zero singular value σ_r(H_ext)
    """
    if not H_ext or not H_ext[0]:
        B_zero = [[[0.0 for _ in range(rank)] for _ in range(rank)] for _ in H_blocks_ext]
        return B_zero, [0.0] * rank, [0.0] * rank, 0.0

    m = len(H_ext)
    n = len(H_ext[0])
    U, S_vals, V = truncated_svd(H_ext, rank)
    r = len(S_vals)
    if r == 0:
        B_zero = [[[0.0 for _ in range(rank)] for _ in range(rank)] for _ in H_blocks_ext]
        return B_zero, [0.0] * rank, [0.0] * rank, 0.0

    sqrt_s = [math.sqrt(max(s, 0.0)) for s in S_vals]
    inv_sqrt_s = [1.0 / (s + 1e-18) if s > 0.0 else 0.0 for s in sqrt_s]

    # P = U Σ^{1/2}  (m x r)
    P: List[List[float]] = [[0.0 for _ in range(r)] for _ in range(m)]
    for i in range(m):
        for k in range(r):
            P[i][k] = U[i][k] * sqrt_s[k]

    # S^T = V Σ^{1/2}   -> S_mat: (r x n)
    S_mat: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(r)]
    for k in range(r):
        sk = sqrt_s[k]
        for j in range(n):
            S_mat[k][j] = sk * V[j][k]

    # Left pseudoinverse P_plus = Σ^{-1/2} U^T   (r x m)
    P_plus: List[List[float]] = [[0.0 for _ in range(m)] for _ in range(r)]
    for k in range(r):
        scale = inv_sqrt_s[k]
        for i in range(m):
            P_plus[k][i] = U[i][k] * scale

    # Right pseudoinverse S_plus = V Σ^{-1/2}   (n x r)
    S_plus: List[List[float]] = [[0.0 for _ in range(r)] for _ in range(n)]
    for j in range(n):
        for k in range(r):
            S_plus[j][k] = V[j][k] * inv_sqrt_s[k]

    # Build B_sigma = P_plus H_sigma S_plus
    B_list: List[List[List[float]]] = []
    for H_sigma in H_blocks_ext:
        # M1 = H_sigma S_plus   (m x r)
        M1 = matmul(H_sigma, S_plus)
        # B = P_plus M1        (r x r)
        B = matmul(P_plus, M1)
        B_list.append(B)

    # Anchor-based α, β
    # α^T = h_{u0,*} S_plus   -> α = S_plus^T h_{u0,*}
    row_anchor = H_ext[prefix_anchor]  # length n
    alpha = matvec(transpose(S_plus), row_anchor)  # (r,)

    # β = P_plus h_{*,v0}
    col_anchor = [H_ext[i][suffix_anchor] for i in range(m)]
    beta = matvec(P_plus, col_anchor)  # (r,)

    sigma_min = S_vals[r - 1]
    return B_list, alpha, beta, sigma_min


def predict_sequence_prob(
    word: Sequence[int],
    B_list: List[List[List[float]]],
    alpha: Sequence[float],
    beta: Sequence[float],
) -> float:
    """
    Evaluate p_hat(x) = α^T B_{x1} ... B_{xL} β
    using right-to-left propagation:

        state_0 = β
        state_{t+1} = B_{x_{L-t}} state_t
        p_hat(x) = α^T state_L
    """
    state = list(beta)
    for sym in reversed(word):
        state = matvec(B_list[sym], state)
    return dot(alpha, state)


def evaluate_model_on_length(
    B_list: List[List[List[float]]],
    alpha: Sequence[float],
    beta: Sequence[float],
    words: List[List[int]],
    prob_true: Dict[Tuple[int, ...], float],
) -> Tuple[float, float]:
    """
    Compute TV and max pointwise error on a given set of words, after
    clamping negatives and re-normalising to get a proper distribution.
    """
    p_hat_raw: Dict[Tuple[int, ...], float] = {}
    total_hat = 0.0

    for word in words:
        key = tuple(word)
        p = predict_sequence_prob(word, B_list, alpha, beta)
        if p < 0.0:
            p = 0.0
        p_hat_raw[key] = p
        total_hat += p

    if total_hat <= 0.0:
        # degenerate case: fallback to uniform
        total_hat = float(len(words))
        for key in p_hat_raw:
            p_hat_raw[key] = 1.0

    # normalise
    inv_total = 1.0 / total_hat
    for key in p_hat_raw:
        p_hat_raw[key] *= inv_total

    # TV and max error
    tv = 0.0
    max_err = 0.0
    for word in words:
        key = tuple(word)
        p_true = prob_true.get(key, 0.0)
        p_hat = p_hat_raw.get(key, 0.0)
        err = abs(p_true - p_hat)
        tv += err
        if err > max_err:
            max_err = err
    tv *= 0.5
    return tv, max_err


# ---------------------------------------------------------------------------
# Utility: coherence (row/column 1-norms) of Hankel
# ---------------------------------------------------------------------------


def hankel_coherence(H: List[List[float]]) -> Tuple[float, float, float]:
    """
    Compute simple coherence-like quantities:

        mu_row = max_u sum_v |H(u,v)|
        mu_col = max_v sum_u |H(u,v)|
        mu     = max(mu_row, mu_col)
    """
    if not H or not H[0]:
        return 0.0, 0.0, 0.0
    m = len(H)
    n = len(H[0])

    mu_row = 0.0
    for i in range(m):
        s = sum(abs(x) for x in H[i])
        if s > mu_row:
            mu_row = s

    mu_col = 0.0
    for j in range(n):
        s = sum(abs(H[i][j]) for i in range(m))
        if s > mu_col:
            mu_col = s

    mu = max(mu_row, mu_col)
    return mu_row, mu_col, mu


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment_for_model_and_length(
    model: str,
    length: int,
    eta: float,
    c_param: float,
    sample_sizes: Sequence[int],
    trials: int,
) -> List[Dict[str, float]]:
    """
    Run Experiment 4 for a given model ('rank1' or 'rank3') and length L.

    Steps (per model & L):
      * build MPS cores up to length L+1;
      * compute true distributions p^{(L)} and p^{(L+1)} via enumeration;
      * construct mid-cut H_mid and H_mid_sigma;
      * Part A: compute spectral quantities (top-3 singular values, log_sigma3,
                coherence, Frobenius/spectral norms);
      * build epsilon-extended H_ext and H_ext_sigma from H_mid, H_mid_sigma;
      * baseline: spectral learning on true H_ext/H_ext_sigma (sample_size = 0);
      * for each sample size N and each trial:
          - estimate empirical H_mid_hat, H_mid_hat_sigma from samples;
          - extend with ε to get H_ext_hat, H_ext_hat_sigma;
          - compute Hankel errors (Frobenius / spectral);
          - run spectral learning and evaluate TV / max errors.
    """
    d = 2
    max_length = length + 1

    # Build MPS
    if model == "rank1":
        cores, alpha, beta = build_rank1_mps(max_length, eta)
        rank = 1
    else:
        cores, alpha, beta = build_rank3_mps(max_length, eta, c_param)
        rank = 3

    # True distributions p^{(L)} and p^{(L+1)}
    prob_L = probability_map_for_length(cores, alpha, beta, length, d)
    prob_Lplus = probability_map_for_length(cores, alpha, beta, length + 1, d)

    # Mid-cut Hankel and blocks (Part A core)
    H_mid_true, prefixes, suffixes = midcut_hankel_from_prob_map(prob_L, length, d)
    H_blocks_mid_true = hankel_blocks_mid_from_prob_map(prob_Lplus, prefixes, suffixes, d)

    # Part A: top-3 singular values, condition number, coherence on mid-cut Hankel
    sigma1, sigma2, sigma3 = top_k_singular_values(H_mid_true, 3)
    log_sigma3 = math.log(max(sigma3, 1e-16))
    mu_row, mu_col, mu_max = hankel_coherence(H_mid_true)
    H_mid_frob = frobenius_norm(H_mid_true)
    H_mid_spec = spectral_norm(H_mid_true)

    # Epsilon-extended Hankel for spectral learning (Part B)
    H_true_ext = extend_hankel_with_epsilon(H_mid_true)
    H_blocks_true_ext = extend_hankel_blocks_with_epsilon(H_blocks_mid_true)

    # Dimensions
    m_ext = len(H_true_ext)
    n_ext = len(H_true_ext[0])

    # Epsilon anchors (row/col 0)
    prefix_anchor_idx = 0
    suffix_anchor_idx = 0

    # All length-L words (for evaluation)
    words_L = all_words(length, d)

    rows: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Baseline: spectral learning on true epsilon-extended Hankel
    # ------------------------------------------------------------------
    B_true, alpha_true, beta_true, sigma_min_true = spectral_wfa_from_hankel(
        H_true_ext, H_blocks_true_ext, rank, prefix_anchor_idx, suffix_anchor_idx
    )
    tv_true, max_true = evaluate_model_on_length(B_true, alpha_true, beta_true, words_L, prob_L)

    # (Optional) sanity warning to stdout if baseline is not tiny
    if tv_true > 1e-6:
        print(
            f"[WARNING] Baseline TV error for model={model}, L={length} is {tv_true:.3e} "
            f"(should be << 1e-6 if the Hankel pipeline is perfectly calibrated)."
        )

    baseline_row: Dict[str, float] = {
        "model": model,
        "length": float(length),
        "rank": float(rank),
        "sample_size": 0.0,
        "trial": 0.0,
        # Part A (mid-cut) spectral structure
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "log_sigma3": log_sigma3,
        "hankel_frob_mid": H_mid_frob,
        "hankel_spec_mid": H_mid_spec,
        "mu_row": mu_row,
        "mu_col": mu_col,
        "mu_max": mu_max,
        # Part B: baseline errors (true epsilon-extended Hankel)
        "frob_error": 0.0,
        "spectral_error": 0.0,
        "tv_error": tv_true,
        "max_error": max_true,
        "sigma_min_emp": sigma_min_true,
        "delta_over_sigma_min": 0.0,
        "prefixes": float(m_ext),
        "suffixes": float(n_ext),
    }
    rows.append(baseline_row)

    # ------------------------------------------------------------------
    # Sample-based experiments (finite N)
    # ------------------------------------------------------------------
    for n in sample_sizes:
        for t in range(trials):
            # Empirical mid-cut Hankels for L and L+1
            H_mid_emp = empirical_midcut_hankel(prob_L, prefixes, suffixes, n)
            H_blocks_mid_emp = empirical_midcut_hankel_blocks(prob_Lplus, prefixes, suffixes, n, d)

            # Epsilon-extended empirical Hankels
            H_emp_ext = extend_hankel_with_epsilon(H_mid_emp)
            H_blocks_emp_ext = extend_hankel_blocks_with_epsilon(H_blocks_mid_emp)

            # Hankel errors (epsilon-extended)
            diff_ext = [
                [H_true_ext[i][j] - H_emp_ext[i][j] for j in range(n_ext)]
                for i in range(m_ext)
            ]
            frob = frobenius_norm(diff_ext)
            spec = spectral_norm(diff_ext)

            # Spectral learning from empirical epsilon-extended Hankels
            B_emp, alpha_emp, beta_emp, sigma_min_emp = spectral_wfa_from_hankel(
                H_emp_ext, H_blocks_emp_ext, rank, prefix_anchor_idx, suffix_anchor_idx
            )

            tv_err, max_err = evaluate_model_on_length(B_emp, alpha_emp, beta_emp, words_L, prob_L)
            delta_over_gamma = spec / (sigma_min_emp + 1e-18)

            row: Dict[str, float] = {
                "model": model,
                "length": float(length),
                "rank": float(rank),
                "sample_size": float(n),
                "trial": float(t + 1),
                # Part A: true mid-cut spectral structure (same as baseline)
                "sigma1": sigma1,
                "sigma2": sigma2,
                "sigma3": sigma3,
                "log_sigma3": log_sigma3,
                "hankel_frob_mid": H_mid_frob,
                "hankel_spec_mid": H_mid_spec,
                "mu_row": mu_row,
                "mu_col": mu_col,
                "mu_max": mu_max,
                # Part B: empirical epsilon-extended Hankel & WFA errors
                "frob_error": frob,
                "spectral_error": spec,
                "tv_error": tv_err,
                "max_error": max_err,
                "sigma_min_emp": sigma_min_emp,
                "delta_over_sigma_min": delta_over_gamma,
                "prefixes": float(m_ext),
                "suffixes": float(n_ext),
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CLI parsing and entry point
# ---------------------------------------------------------------------------


def parse_comma_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4 (epsilon-extended): condition numbers and hard examples (rank-deficient MPS)"
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="6,8,10,12,14,16",
        help="Comma-separated lengths L to evaluate (e.g. '6,10,14').",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.6,
        help="η parameter for diagonal constructions (0 < eta < 1).",
    )
    parser.add_argument(
        "--c-param",
        type=float,
        default=0.5,
        dest="c_param",
        help="c parameter for rank-3 boundaries (0 < c < 1).",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="200,1000,5000",
        help="Comma-separated sample sizes N for empirical Hankel estimates.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Trials per (model, length, sample_size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp4_results.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    lengths = parse_comma_ints(args.lengths)
    sample_sizes = parse_comma_ints(args.sample_sizes)

    rows: List[Dict[str, float]] = []
    for length in lengths:
        for model in ("rank1", "rank3"):
            rows.extend(
                run_experiment_for_model_and_length(
                    model=model,
                    length=length,
                    eta=args.eta,
                    c_param=args.c_param,
                    sample_sizes=sample_sizes,
                    trials=args.trials,
                )
            )

    fieldnames = [
        "model",
        "length",
        "rank",
        "sample_size",
        "trial",
        "sigma1",
        "sigma2",
        "sigma3",
        "log_sigma3",
        "hankel_frob_mid",
        "hankel_spec_mid",
        "mu_row",
        "mu_col",
        "mu_max",
        "frob_error",
        "spectral_error",
        "tv_error",
        "max_error",
        "sigma_min_emp",
        "delta_over_sigma_min",
        "prefixes",
        "suffixes",
    ]

    # Ensure output directory exists (if any)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()



"""
python experiments/exp4_condition_numbers.py --lengths 6,10,14 --eta 0.6 --c-param 0.5 --sample-sizes 200,1000,5000,20000,50000 --trials 20 --seed 0 --output experiments/exp4_results_epsilon.csv
"""