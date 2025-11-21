"""
Experiment 6: Multi-view Hankel stacking and stability (rewritten version).

This script implements the strengthened multi-view experiment (§8.3):

- sample a single left-canonical MPS per trial at fixed length L;
- define K "views" by different cut positions t_* on the same model;
- for each view, build the true Hankel H^{(k)} and its rank-r SVD factorisation
  H^{(k)} = O^{(k)} C^{(k)};
- stack all O^{(k)}, C^{(k)} to form a joint Hankel H_joint = O_joint C_joint
  and compute its smallest non-zero singular value sigma_r(H_joint);
- under finite samples, build empirical Hankels per view, learn a single-view
  spectral model, and also build a joint empirical Hankel via stacking the
  empirical O/C factors, then perform joint whitening and multi-view prediction;
- record coherence, Hankel deviations, and joint/single-view sigma_r and
  downstream errors to validate the benefits predicted by Theorem 8.7 and the
  1/gamma scaling in the end-to-end error bounds.

No visualisation is performed here; all metrics are written to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Sequence, Tuple, Any


# ---------------------------------------------------------------------------
# Basic linear algebra utilities
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
            bk = b[k]
            for j in range(cols):
                out[i][j] += aik * bk[j]
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
    """Power-iteration spectral norm for (possibly complex) matrices."""
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
    """Approximate A^{-1/2} for a positive definite matrix A via Newton–Schulz."""
    n = len(mat)
    trace = sum(mat[i][i].real for i in range(n))
    scale = trace / n if trace != 0 else 1.0
    A = mat_scale(mat, 1.0 / scale)
    Y = [row[:] for row in A]
    Z = identity(n)
    for _ in range(iters):
        ZY = matmul(Z, Y)
        mid = mat_add(identity(n), identity(n))
        mid = mat_add(mid, identity(n))
        mid = mat_add(mid, mat_scale(ZY, -1.0))
        mid = mat_scale(mid, 0.5)
        Y = matmul(Y, mid)
        Z = matmul(mid, Z)
    return mat_scale(Z, 1.0 / math.sqrt(scale))


# ---------------------------------------------------------------------------
# MPS helpers and probability utilities
# ---------------------------------------------------------------------------


def random_left_canonical_mps(
    length: int, bond_dim: int, d: int
) -> Tuple[List[List[List[complex]]], List[complex], List[complex]]:
    """Sample a random left-canonical MPS: sites cores[t][sigma] ∈ C^{D×D}."""
    cores: List[List[List[complex]]] = []
    for _ in range(length):
        raw: List[List[List[complex]]] = []
        for _ in range(d):
            mat = [
                [random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)]
                for _ in range(bond_dim)
            ]
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


def amplitude_for_sequence(
    seq: Sequence[int],
    cores: Sequence[Sequence[Sequence[complex]]],
    alpha: Sequence[complex],
    beta: Sequence[complex],
) -> complex:
    vec = list(alpha)
    for idx, sym in enumerate(seq):
        vec = matvec(cores[idx][sym], vec)
    return dot(vec, beta)


def all_words(length: int, d: int) -> List[List[int]]:
    total = d ** length
    words: List[List[int]] = []
    for val in range(total):
        x = val
        word = [0 for _ in range(length)]
        for i in range(length - 1, -1, -1):
            word[i] = x % d
            x //= d
        words.append(word)
    return words


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
        val = abs(amp) ** 2
        probs[tuple(word)] = val
        total += val
    if total > 0:
        inv_total = 1.0 / total
        for k in list(probs.keys()):
            probs[k] *= inv_total
    return probs


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


# ---------------------------------------------------------------------------
# Hankel utilities
# ---------------------------------------------------------------------------


def enumerate_prefix_suffix(
    length: int,
    d: int,
    max_prefixes: int,
    max_suffixes: int,
    cut: int | None = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Enumerate (or subsample) prefix/suffix sets for a given cut position."""
    cut = cut if cut is not None else length // 2
    all_prefix = all_words(cut, d)
    all_suffix = all_words(length - cut, d)
    rng = random.Random()
    if len(all_prefix) > max_prefixes:
        all_prefix = rng.sample(all_prefix, max_prefixes)
    if len(all_suffix) > max_suffixes:
        all_suffix = rng.sample(all_suffix, max_suffixes)
    return all_prefix, all_suffix


def hankel_from_prob_map(
    prob: Dict[Tuple[int, ...], float],
    prefixes: Sequence[Sequence[int]],
    suffixes: Sequence[Sequence[int]],
) -> List[List[float]]:
    hankel: List[List[float]] = []
    suffix_index = {tuple(s): idx for idx, s in enumerate(suffixes)}
    for u in prefixes:
        row: List[float] = [0.0 for _ in suffixes]
        for v in suffixes:
            x = tuple(u + list(v))
            j = suffix_index[tuple(v)]
            row[j] = prob.get(x, 0.0)
        hankel.append(row)
    return hankel


def truncated_svd_rank(
    hankel: Sequence[Sequence[complex]],
    rank: int,
) -> Tuple[List[List[complex]], List[float], List[List[complex]]]:
    """
    Low-rank SVD via eigen-decomposition of H H^T (power iteration).
    Input hankel is real or complex; we only use the real part in Gram.
    """
    m = len(hankel)
    n = len(hankel[0]) if hankel else 0
    rank = max(1, min(rank, m, n))
    # Gram matrix (real-valued)
    gram: List[List[float]] = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            s = 0.0
            row_i = hankel[i]
            row_j = hankel[j]
            for k in range(n):
                # assume hankel entries are real or nearly real
                s += float(row_i[k].real) * float(row_j[k].real)
            gram[i][j] = s
    U: List[List[complex]] = [[0.0j for _ in range(rank)] for _ in range(m)]
    S: List[float] = [0.0 for _ in range(rank)]
    # power iteration to extract top-`rank` eigenvectors
    for r in range(rank):
        v = normalize([random.random() for _ in range(m)])
        for _ in range(40):
            w = [sum(gram[i][j] * v[j] for j in range(m)) for i in range(m)]
            v = normalize(w)
        sigma_sq = sum(
            v[i] * sum(gram[i][j] * v[j] for j in range(m))
            for i in range(m)
        )
        sigma_sq = float(sigma_sq)
        sigma_sq = max(sigma_sq, 0.0)
        sigma = math.sqrt(sigma_sq)
        S[r] = sigma
        for i in range(m):
            U[i][r] = v[i]
        # deflate
        for i in range(m):
            for j in range(m):
                gram[i][j] -= sigma_sq * v[i] * v[j]
    # V from H^T U / S
    V: List[List[complex]] = [[0.0j for _ in range(rank)] for _ in range(n)]
    for k in range(n):
        for r in range(rank):
            if S[r] > 0:
                acc = 0.0
                for i in range(m):
                    acc += float(hankel[i][k].real) * U[i][r]
                V[k][r] = acc / S[r]
    return U, S, V


def embed_from_svd(
    prefixes: Sequence[Sequence[int]],
    suffixes: Sequence[Sequence[int]],
    U: Sequence[Sequence[complex]],
    S: Sequence[float],
    V: Sequence[Sequence[complex]],
    rank: int,
) -> Tuple[Dict[Tuple[int, ...], List[complex]], Dict[Tuple[int, ...], List[complex]]]:
    """Build prefix/suffix embeddings φ,ψ from SVD factors."""
    rank = min(rank, len(S))
    sqrt_S = [math.sqrt(s) for s in S[:rank]]
    phi: Dict[Tuple[int, ...], List[complex]] = {}
    psi: Dict[Tuple[int, ...], List[complex]] = {}
    for idx, u in enumerate(prefixes):
        phi[tuple(u)] = [U[idx][r] * sqrt_S[r] for r in range(rank)]
    for jdx, v in enumerate(suffixes):
        psi[tuple(v)] = [V[jdx][r] * sqrt_S[r] for r in range(rank)]
    return phi, psi


def compute_coherence(hankel: Sequence[Sequence[float]]) -> float:
    """Coherence μ = max(row-sum, col-sum) of the Hankel entries."""
    if not hankel:
        return 0.0
    row_sums = [sum(row) for row in hankel]
    col_sums = [sum(hankel[i][j] for i in range(len(hankel))) for j in range(len(hankel[0]))]
    return max(max(row_sums), max(col_sums))


def sigma_r_from_hankel(
    hankel: Sequence[Sequence[complex]],
    rank: int,
) -> Tuple[float, List[List[complex]], List[float], List[List[complex]]]:
    if not hankel or not hankel[0]:
        return 0.0, [], [], []
    U, S, V = truncated_svd_rank(hankel, rank)
    sigma_r = S[min(rank, len(S)) - 1] if S else 0.0
    return sigma_r, U, S, V


def hankel_difference(a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]]) -> float:
    """Spectral norm of the difference between two Hankel matrices."""
    rows = min(len(a), len(b))
    cols = min(len(a[0]), len(b[0])) if rows > 0 else 0
    diff: List[List[complex]] = []
    for i in range(rows):
        diff.append([complex(a[i][j]) - complex(b[i][j]) for j in range(cols)])
    return spectral_norm(diff)


def renormalize_distribution(prob: Dict[Tuple[int, ...], float]) -> Dict[Tuple[int, ...], float]:
    cleaned = {k: max(0.0, v) for k, v in prob.items()}
    total = sum(cleaned.values())
    if total > 0:
        inv_total = 1.0 / total
        for k in list(cleaned.keys()):
            cleaned[k] *= inv_total
    return cleaned


def predict_from_embeddings(
    phi: Dict[Tuple[int, ...], Sequence[complex]],
    psi: Dict[Tuple[int, ...], Sequence[complex]],
    cut: int,
    eval_support: Sequence[Sequence[int]],
) -> Dict[Tuple[int, ...], float]:
    preds: Dict[Tuple[int, ...], float] = {}
    for word in eval_support:
        u = tuple(word[:cut])
        v = tuple(word[cut:])
        if u not in phi or v not in psi:
            preds[tuple(word)] = 0.0
            continue
        val = sum(a * b for a, b in zip(phi[u], psi[v])).real
        preds[tuple(word)] = val
    return renormalize_distribution(preds)


def pointwise_errors(
    true_prob: Dict[Tuple[int, ...], float],
    pred_prob: Dict[Tuple[int, ...], float],
) -> Tuple[float, float]:
    keys = set(true_prob.keys()) | set(pred_prob.keys())
    tv = 0.0
    max_err = 0.0
    for k in keys:
        err = abs(pred_prob.get(k, 0.0) - true_prob.get(k, 0.0))
        tv += err
        max_err = max(max_err, err)
    return max_err, 0.5 * tv


def build_joint_hankel_and_labels(
    O_list: Sequence[Sequence[Sequence[complex]]],
    C_list: Sequence[Sequence[Sequence[complex]]],
    prefixes_list: Sequence[Sequence[Sequence[int]]],
    suffixes_list: Sequence[Sequence[Sequence[int]]],
) -> Tuple[
    List[List[complex]],
    List[Tuple[int, Tuple[int, ...]]],
    List[Tuple[int, Tuple[int, ...]]],
]:
    """
    Build joint Hankel H_joint = O_joint C_joint and record row/col labels.

    - O_list[k]: shape (m_k, r)
    - C_list[k]: shape (r, n_k)
    - prefixes_list[k]: list of prefixes for view k (length m_k)
    - suffixes_list[k]: list of suffixes for view k (length n_k)

    Returns:
      H_joint: (sum_k m_k) x (sum_k n_k)
      row_labels: list of (view_index, prefix_tuple)
      col_labels: list of (view_index, suffix_tuple)
    """
    if not O_list or not C_list:
        return [], [], []
    r = len(O_list[0][0]) if O_list[0] else 0
    if r == 0:
        return [], [], []
    # Build O_joint and row_labels
    row_labels: List[Tuple[int, Tuple[int, ...]]] = []
    total_rows = sum(len(P) for P in prefixes_list)
    O_joint: List[List[complex]] = [[0.0j for _ in range(r)] for _ in range(total_rows)]
    row_offset = 0
    for vidx, (O, P) in enumerate(zip(O_list, prefixes_list)):
        for i, u in enumerate(P):
            row_labels.append((vidx, tuple(u)))
            O_joint[row_offset + i] = list(O[i][:r])
        row_offset += len(P)
    # Build C_joint and col_labels
    col_labels: List[Tuple[int, Tuple[int, ...]]] = []
    total_cols = sum(len(S) for S in suffixes_list)
    C_joint: List[List[complex]] = [[0.0j for _ in range(total_cols)] for _ in range(r)]
    col_offset = 0
    for vidx, (C, S) in enumerate(zip(C_list, suffixes_list)):
        for j, v in enumerate(S):
            col_labels.append((vidx, tuple(v)))
        for rr in range(r):
            for j in range(len(S)):
                C_joint[rr][col_offset + j] = C[rr][j]
        col_offset += len(S)
    H_joint = matmul(O_joint, C_joint)
    return H_joint, row_labels, col_labels


# ---------------------------------------------------------------------------
# Experiment logic
# ---------------------------------------------------------------------------


def run_trial(
    trial: int,
    lengths: Sequence[int],
    cuts: Sequence[int | None],
    sample_sizes: Sequence[int],
    bond_dim: int,
    d: int,
    max_prefixes: int,
    max_suffixes: int,
    rank_cap: int,
) -> List[Dict[str, float]]:
    rows_out: List[Dict[str, float]] = []

    # All views must share the same L for a fair multi-view comparison.
    if len(set(lengths)) != 1:
        raise ValueError(
            "All lengths must match for multi-view comparison; "
            "pass repeated L with different cuts."
        )
    base_len = lengths[0]

    # Sample a single ground-truth MPS for this trial.
    cores, alpha, beta = random_left_canonical_mps(base_len, bond_dim, d)
    eval_support = all_words(base_len, d)
    prob_true_global = probability_map(cores, alpha, beta, base_len, d, eval_support)

    # ------------------------------------------------------------------
    # 6A: true per-view Hankels and joint Hankel
    # ------------------------------------------------------------------
    view_data: List[Dict[str, Any]] = []

    # First enumerate prefix/suffix sets for all views to determine global rank.
    prefixes_list: List[List[List[int]]] = []
    suffixes_list: List[List[List[int]]] = []
    cuts_effective: List[int] = []

    for idx, L in enumerate(lengths):
        cut = cuts[idx] if idx < len(cuts) and cuts[idx] is not None else L // 2
        P, S = enumerate_prefix_suffix(L, d, max_prefixes, max_suffixes, cut=cut)
        prefixes_list.append(P)
        suffixes_list.append(S)
        cuts_effective.append(cut)

    # global rank r: same across views, truncated by rank_cap and by smallest view dimension
    global_rank = rank_cap
    for P, S in zip(prefixes_list, suffixes_list):
        global_rank = min(global_rank, len(P), len(S))
    global_rank = max(1, global_rank)

    # build true Hankel per view and its SVD factorisation
    O_true_list: List[List[List[complex]]] = []
    C_true_list: List[List[List[complex]]] = []

    for vidx, (P, S, cut) in enumerate(zip(prefixes_list, suffixes_list, cuts_effective)):
        hankel_true = hankel_from_prob_map(prob_true_global, P, S)
        sigma_true, U, Svals, V = sigma_r_from_hankel(hankel_true, global_rank)
        mu = compute_coherence(hankel_true)
        # build O, C from SVD: H = U Σ V^T = (U Σ^{1/2})(Σ^{1/2} V^T)
        sqrt_S = [math.sqrt(s) for s in Svals[:global_rank]]
        O_true = [
            [U[i][r] * sqrt_S[r] for r in range(global_rank)]
            for i in range(len(P))
        ]
        C_true = [
            [V[j][r] * sqrt_S[r] for j in range(len(S))]
            for r in range(global_rank)
        ]
        O_true_list.append(O_true)
        C_true_list.append(C_true)

        view_data.append(
            {
                "view": vidx,
                "length": base_len,
                "cut": cut,
                "prefixes": P,
                "suffixes": S,
                "hankel_true": hankel_true,
                "sigma_true": sigma_true,
                "mu": mu,
            }
        )

        # log per-view true metrics
        rows_out.append(
            {
                "trial": float(trial),
                "mode": "single_true",
                "view": float(vidx),
                "length": float(base_len),
                "cut": float(cut),
                "sample_size": 0.0,
                "sigma_r_true": sigma_true,
                "sigma_r_emp": sigma_true,
                "delta_hankel": 0.0,
                "mu": mu,
                "rank": float(global_rank),
                "max_error": 0.0,
                "tv_error": 0.0,
            }
        )

    # Build true joint Hankel from true O/C and record its sigma_r
    joint_true, joint_row_labels, joint_col_labels = build_joint_hankel_and_labels(
        O_true_list, C_true_list, prefixes_list, suffixes_list
    )
    joint_rank = global_rank
    sigma_joint_true, _, _, _ = sigma_r_from_hankel(joint_true, joint_rank)

    rows_out.append(
        {
            "trial": float(trial),
            "mode": "joint_true",
            "view": -1.0,
            "length": float(base_len),
            "cut": -1.0,
            "sample_size": 0.0,
            "sigma_r_true": sigma_joint_true,
            "sigma_r_emp": sigma_joint_true,
            "delta_hankel": 0.0,
            "mu": 0.0,
            "rank": float(joint_rank),
            "max_error": 0.0,
            "tv_error": 0.0,
        }
    )

    # ------------------------------------------------------------------
    # 6B: finite-sample single-view vs joint multi-view spectral learning
    # ------------------------------------------------------------------
    for sample_size in sample_sizes:
        # draw samples from the true distribution
        samples = sample_sequences(prob_true_global, sample_size)

        emp_O_list: List[List[List[complex]]] = []
        emp_C_list: List[List[List[complex]]] = []

        # per-view single-view spectral learning
        for vinfo in view_data:
            vidx = vinfo["view"]
            cut = vinfo["cut"]
            P = vinfo["prefixes"]
            S = vinfo["suffixes"]
            hankel_true = vinfo["hankel_true"]
            mu = vinfo["mu"]

            # empirical Hankel
            hankel_emp = [[0.0 for _ in range(len(S))] for _ in range(len(P))]
            pre_index = {tuple(p): i for i, p in enumerate(P)}
            suf_index = {tuple(s): j for j, s in enumerate(S)}
            for seq in samples:
                u = tuple(seq[:cut])
                v = tuple(seq[cut:])
                i = pre_index.get(u)
                j = suf_index.get(v)
                if i is None or j is None:
                    continue
                hankel_emp[i][j] += 1.0
            if sample_size > 0:
                inv_n = 1.0 / float(sample_size)
                for i in range(len(hankel_emp)):
                    row = hankel_emp[i]
                    for j in range(len(row)):
                        row[j] *= inv_n

            sigma_emp, Ue, Se, Ve = sigma_r_from_hankel(hankel_emp, global_rank)
            sqrt_Se = [math.sqrt(s) for s in Se[:global_rank]]
            # empirical O/C for this view
            Oe = [
                [Ue[i][r] * sqrt_Se[r] for r in range(global_rank)]
                for i in range(len(P))
            ]
            Ce = [
                [Ve[j][r] * sqrt_Se[r] for j in range(len(S))]
                for r in range(global_rank)
            ]
            emp_O_list.append(Oe)
            emp_C_list.append(Ce)

            # Hankel deviation and single-view prediction
            delta = hankel_difference(hankel_true, hankel_emp)
            phi, psi = embed_from_svd(P, S, Ue, Se, Ve, global_rank)
            preds_single = predict_from_embeddings(phi, psi, cut, eval_support)
            max_err, tv_err = pointwise_errors(prob_true_global, preds_single)

            rows_out.append(
                {
                    "trial": float(trial),
                    "mode": "single_emp",
                    "view": float(vidx),
                    "length": float(base_len),
                    "cut": float(cut),
                    "sample_size": float(sample_size),
                    "sigma_r_true": vinfo["sigma_true"],
                    "sigma_r_emp": sigma_emp,
                    "delta_hankel": delta,
                    "mu": mu,
                    "rank": float(global_rank),
                    "max_error": max_err,
                    "tv_error": tv_err,
                }
            )

        # joint empirical Hankel from stacked empirical O/C
        joint_emp, row_labels_emp, col_labels_emp = build_joint_hankel_and_labels(
            emp_O_list, emp_C_list, prefixes_list, suffixes_list
        )
        if joint_emp:
            sigma_joint_emp, Uj, Sj, Vj = sigma_r_from_hankel(joint_emp, joint_rank)
            delta_joint = hankel_difference(joint_true, joint_emp)
            # build joint embeddings keyed by (view_index, prefix/suffix)
            sqrt_Sj = [math.sqrt(s) for s in Sj[:joint_rank]]
            phi_joint: Dict[Tuple[int, Tuple[int, ...]], List[complex]] = {}
            psi_joint: Dict[Tuple[int, Tuple[int, ...]], List[complex]] = {}

            for i, label in enumerate(row_labels_emp):
                phi_joint[label] = [Uj[i][r] * sqrt_Sj[r] for r in range(joint_rank)]
            for j, label in enumerate(col_labels_emp):
                psi_joint[label] = [Vj[j][r] * sqrt_Sj[r] for r in range(joint_rank)]

            # multi-view prediction: average over all views where embeddings exist
            preds_joint: Dict[Tuple[int, ...], float] = {}
            for word in eval_support:
                acc_val = 0.0
                used = 0
                for vinfo in view_data:
                    vidx = vinfo["view"]
                    cut = vinfo["cut"]
                    u = tuple(word[:cut])
                    v = tuple(word[cut:])
                    key_row = (vidx, u)
                    key_col = (vidx, v)
                    if key_row in phi_joint and key_col in psi_joint:
                        val = sum(
                            a * b
                            for a, b in zip(phi_joint[key_row], psi_joint[key_col])
                        ).real
                        acc_val += val
                        used += 1
                preds_joint[tuple(word)] = acc_val / used if used > 0 else 0.0
            preds_joint = renormalize_distribution(preds_joint)
            joint_max_err, joint_tv_err = pointwise_errors(prob_true_global, preds_joint)
        else:
            sigma_joint_emp = 0.0
            delta_joint = 0.0
            joint_max_err = 0.0
            joint_tv_err = 0.0

        rows_out.append(
            {
                "trial": float(trial),
                "mode": "joint_emp",
                "view": -1.0,
                "length": float(base_len),
                "cut": -1.0,
                "sample_size": float(sample_size),
                "sigma_r_true": sigma_joint_true,
                "sigma_r_emp": sigma_joint_emp,
                "delta_hankel": delta_joint,
                "mu": 0.0,
                "rank": float(joint_rank),
                "max_error": joint_max_err,
                "tv_error": joint_tv_err,
            }
        )

    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 6: multi-view Hankel stacking (rewritten)")
    parser.add_argument(
        "--lengths",
        type=str,
        default="6,6,6",
        help="Comma-separated view lengths (must be identical for all views)",
    )
    parser.add_argument(
        "--cuts",
        type=str,
        default="2,3,4",
        help="Comma-separated cut points per view (matched to lengths)",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="500,2000",
        help="Comma-separated sample sizes",
    )
    parser.add_argument("--bond-dim", type=int, default=3, help="Bond dimension for the ground-truth MPS")
    parser.add_argument("--alphabet-size", type=int, default=2, help="Alphabet size |Sigma|")
    parser.add_argument("--max-prefixes", type=int, default=128, help="Cap on prefix set size per view")
    parser.add_argument("--max-suffixes", type=int, default=128, help="Cap on suffix set size per view")
    parser.add_argument("--rank", type=int, default=4, help="Rank cap for SVD truncation")
    parser.add_argument("--trials", type=int, default=3, help="Number of independent trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp6_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    cuts: List[int | None] = [int(x) if x else None for x in args.cuts.split(",") if x]
    sample_sizes = [int(x) for x in args.sample_sizes.split(",") if x]

    all_rows: List[Dict[str, float]] = []
    for trial in range(args.trials):
        all_rows.extend(
            run_trial(
                trial=trial,
                lengths=lengths,
                cuts=cuts,
                sample_sizes=sample_sizes,
                bond_dim=args.bond_dim,
                d=args.alphabet_size,
                max_prefixes=args.max_prefixes,
                max_suffixes=args.max_suffixes,
                rank_cap=args.rank,
            )
        )

    fieldnames = [
        "trial",
        "mode",
        "view",
        "length",
        "cut",
        "sample_size",
        "sigma_r_true",
        "sigma_r_emp",
        "delta_hankel",
        "mu",
        "rank",
        "max_error",
        "tv_error",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()


"""
python experiments/exp6_multi_view.py --lengths 6,6,6 --cuts 2,3,4 --sample-sizes 500,2000 --bond-dim 3 --max-prefixes 128 --max-suffixes 128 --rank 4 --trials 3 --seed 0 --output experiments/exp6_results.csv
"""