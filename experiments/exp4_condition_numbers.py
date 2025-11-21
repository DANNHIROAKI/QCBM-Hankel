"""Experiment 4: condition numbers and hard examples (rank-deficient MPS).

This expanded version follows the detailed plan in the user prompt:

* Part A: instantiate the rank-1 (7.2A) and rank-3 (7.2B) diagonal MPS, build
  full Hankel matrices at several lengths, and record the top-3 singular
  values (especially the exponentially decaying ``sigma3`` for the rank-3
  construction).
* Part B: run a simple Hankel spectral-learning pipeline (rank-truncated SVD +
  whitening + WFA-style transitions built from column masks) on i.i.d. samples
  drawn from the target distribution, and measure pointwise/TV errors as a
  function of sample size. The contrast between rank-1 and rank-3 illustrates
  the ``N \gtrsim 1/\sigma_min^2`` dependence.

The implementation is dependency-free (pure Python) and
enumerates all length-L binary strings for accuracy. Hankel column masks are
built from the leading symbol of the suffix to define simple transition
matrices for the WFA-style reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Linear algebra helpers (pure Python, float-only)
# ---------------------------------------------------------------------------


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def norm(u: Sequence[float]) -> float:
    return math.sqrt(dot(u, u))


def normalize(u: Sequence[float]) -> List[float]:
    n = norm(u)
    if n == 0:
        return list(u)
    return [x / n for x in u]


def matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def transpose(a: Sequence[Sequence[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(a[0]) if a else 0
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(b[0]) if b else 0
    mid = len(b)
    out: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(mid):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def outer(u: Sequence[float], v: Sequence[float]) -> List[List[float]]:
    return [[a * b for b in v] for a in u]


def frobenius_norm(a: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(x * x for row in a for x in row))


def spectral_norm(a: Sequence[Sequence[float]], iters: int = 60) -> float:
    if not a or not a[0]:
        return 0.0
    v = normalize([random.random() for _ in range(len(a[0]))])
    for _ in range(iters):
        v = normalize(matvec(transpose(a), matvec(a, v)))
    Av = matvec(a, v)
    return norm(Av)


def power_eigenpair(mat: Sequence[Sequence[float]], iters: int = 80) -> Tuple[float, List[float]]:
    if not mat:
        return 0.0, []
    v = normalize([random.random() for _ in range(len(mat))])
    for _ in range(iters):
        v = normalize(matvec(mat, v))
    lam = dot(v, matvec(mat, v))
    return lam, v


def truncated_svd(mat: Sequence[Sequence[float]], rank: int) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """Return U (m x r), singular values, V (n x r) via Gram deflation."""
    if not mat:
        return [], [], []
    m = len(mat)
    n = len(mat[0]) if m > 0 else 0
    gram = matmul(mat, transpose(mat))  # m x m
    gram_work = [row[:] for row in gram]
    U_cols: List[List[float]] = []
    sigmas: List[float] = []
    for _ in range(rank):
        lam, vec = power_eigenpair(gram_work)
        sigma = math.sqrt(max(lam, 0.0))
        sigmas.append(sigma)
        U_cols.append(vec)
        # Deflate
        outer_part = outer(vec, vec)
        for i in range(m):
            for j in range(m):
                gram_work[i][j] -= lam * outer_part[i][j]
    # Build V using V = (1/sigma) H^T U
    V_cols: List[List[float]] = []
    Ht = transpose(mat)
    for idx, sigma in enumerate(sigmas):
        if sigma == 0.0:
            V_cols.append([0.0 for _ in range(n)])
            continue
        Hv = matvec(Ht, U_cols[idx])
        V_cols.append([x / sigma for x in Hv])
    # Columns -> row-major lists
    U = [[col[i] for col in U_cols] for i in range(m)]
    V = [[col[i] for col in V_cols] for i in range(n)]
    return U, sigmas, V


def top_k_singular_values(mat: Sequence[Sequence[float]], k: int) -> List[float]:
    _, sigmas, _ = truncated_svd(mat, k)
    sigmas += [0.0 for _ in range(max(0, k - len(sigmas)))]
    return sigmas[:k]


# ---------------------------------------------------------------------------
# Combinatorics helpers
# ---------------------------------------------------------------------------


def int_to_word(val: int, length: int, d: int) -> List[int]:
    word = [0 for _ in range(length)]
    for i in range(length - 1, -1, -1):
        word[i] = val % d
        val //= d
    return word


def all_words(length: int, d: int) -> List[List[int]]:
    return [int_to_word(val, length, d) for val in range(d**length)]


# ---------------------------------------------------------------------------
# Rank-1 and rank-3 diagonal MPS instances
# ---------------------------------------------------------------------------


def build_rank1_mps(length: int, eta: float) -> Tuple[List[List[List[float]]], List[float], List[float]]:
    """Rank-1 construction (§7.2A) with A(0)=diag(1,eta), A(1)=diag(eta,1)."""
    A0 = [[1.0, 0.0], [0.0, eta]]
    A1 = [[eta, 0.0], [0.0, 1.0]]
    cores = [[A0, A1] for _ in range(length)]
    alpha = [1.0, 0.0]
    beta = [1.0, 0.0]
    return cores, alpha, beta


def build_rank3_mps(length: int, eta: float, c_param: float) -> Tuple[List[List[List[float]]], List[float], List[float]]:
    """Rank-3 construction (§7.2B) with boundary [1,c]."""
    A0 = [[1.0, 0.0], [0.0, eta]]
    A1 = [[eta, 0.0], [0.0, 1.0]]
    cores = [[A0, A1] for _ in range(length)]
    alpha = [1.0, c_param]
    beta = [1.0, c_param]
    return cores, alpha, beta


def amplitude_for_sequence(
    seq: Sequence[int], cores: Sequence[Sequence[Sequence[float]]], alpha: Sequence[float], beta: Sequence[float]
) -> float:
    vec = list(alpha)
    for site, sym in enumerate(seq):
        A = cores[site][sym]
        vec = matvec(A, vec)
    return dot(vec, beta)


def probability_map(
    cores: Sequence[Sequence[Sequence[float]]], alpha: Sequence[float], beta: Sequence[float], length: int, d: int
) -> Dict[Tuple[int, ...], float]:
    probs: Dict[Tuple[int, ...], float] = {}
    total = 0.0
    for word in all_words(length, d):
        amp = amplitude_for_sequence(word, cores, alpha, beta)
        p = amp * amp
        probs[tuple(word)] = p
        total += p
    if total > 0:
        for key in list(probs.keys()):
            probs[key] /= total
    return probs


# ---------------------------------------------------------------------------
# Hankel construction and sampling
# ---------------------------------------------------------------------------


def hankel_from_prob_map(
    prob_map: Dict[Tuple[int, ...], float], length: int, d: int
) -> Tuple[List[List[float]], List[List[int]], List[List[int]]]:
    t_star = length // 2
    prefixes = all_words(t_star, d)
    suffixes = all_words(length - t_star, d)
    hankel: List[List[float]] = [[0.0 for _ in range(len(suffixes))] for _ in range(len(prefixes))]
    for i, u in enumerate(prefixes):
        for j, v in enumerate(suffixes):
            seq = tuple(u + v)
            hankel[i][j] = prob_map.get(seq, 0.0)
    return hankel, prefixes, suffixes


def sample_sequences(prob_map: Dict[Tuple[int, ...], float], n: int) -> List[Tuple[int, ...]]:
    keys = list(prob_map.keys())
    weights = [prob_map[k] for k in keys]
    cumulative = []
    total = 0.0
    for w in weights:
        total += w
        cumulative.append(total)
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


def empirical_hankel(
    prob_map: Dict[Tuple[int, ...], float], prefixes: List[List[int]], suffixes: List[List[int]], n: int
) -> List[List[float]]:
    counts = [[0.0 for _ in range(len(suffixes))] for _ in range(len(prefixes))]
    samples = sample_sequences(prob_map, n)
    prefix_index = {tuple(p): idx for idx, p in enumerate(prefixes)}
    suffix_index = {tuple(s): idx for idx, s in enumerate(suffixes)}
    for seq in samples:
        t_star = len(seq) // 2
        u = tuple(seq[:t_star])
        v = tuple(seq[t_star:])
        i = prefix_index[u]
        j = suffix_index[v]
        counts[i][j] += 1.0
    for i in range(len(prefixes)):
        for j in range(len(suffixes)):
            counts[i][j] /= float(n)
    return counts


# ---------------------------------------------------------------------------
# Spectral-learning helpers (Part B)
# ---------------------------------------------------------------------------


def hankel_column_masks(suffixes: List[List[int]], d: int) -> List[List[List[float]]]:
    """Build diagonal masks that select columns whose leading symbol is sigma."""
    masks: List[List[List[float]]] = []
    if not suffixes:
        return [[[0.0] for _ in range(1)] for _ in range(d)]
    for sigma in range(d):
        diag = [1.0 if len(suf) > 0 and suf[0] == sigma else 0.0 for suf in suffixes]
        size = len(diag)
        mat = [[0.0 for _ in range(size)] for _ in range(size)]
        for i, val in enumerate(diag):
            mat[i][i] = val
        masks.append(mat)
    return masks


def spectral_wfa_from_hankel(
    H: List[List[float]], masks: List[List[List[float]]], rank: int, prefix_zero: int, suffix_zero: int
) -> Tuple[List[List[List[float]]], List[float], List[float], float]:
    """Return transition operators B_sigma plus initial/final vectors and sigma_min."""
    if not H:
        return [[ [0.0 for _ in range(rank)] for _ in range(rank)] for _ in masks], [0.0]*rank, [0.0]*rank, 0.0

    U, S, V = truncated_svd(H, rank)
    r = len(S)
    inv_sqrt = [1.0 / math.sqrt(s + 1e-18) if s > 0 else 0.0 for s in S]
    # W_L = diag(inv_sqrt) @ U^T
    W_L = [[inv_sqrt[i] * U[row][i] for row in range(len(U))] for i in range(r)]
    # W_R = V @ diag(inv_sqrt)
    W_R = [[V[row][i] * inv_sqrt[i] for i in range(r)] for row in range(len(V))]

    B_list: List[List[List[float]]] = []
    for mask in masks:
        H_mask = matmul(H, mask)
        tmp = matmul(W_L, H_mask)
        B_list.append(matmul(tmp, W_R))

    m = len(H)
    n = len(H[0]) if m else 0
    e_pref = [0.0 for _ in range(m)]
    e_suf = [0.0 for _ in range(n)]
    e_pref[prefix_zero] = 1.0
    e_suf[suffix_zero] = 1.0
    init = matvec(W_L, e_pref)
    final = matvec(transpose(W_R), e_suf)
    sigma_min = S[-1] if S else 0.0
    return B_list, init, final, sigma_min


def predict_sequence_prob(word: Sequence[int], B_list: List[List[List[float]]], init: List[float], final: List[float]) -> float:
    state = init
    for sym in word:
        state = matvec(B_list[sym], state)
    return dot(state, final)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_trial(
    model: str,
    length: int,
    eta: float,
    c_param: float,
    sample_sizes: Sequence[int],
    trials: int,
) -> List[Dict[str, float]]:
    d = 2
    if model == "rank1":
        cores, alpha, beta = build_rank1_mps(length, eta)
        rank = 1
    else:
        cores, alpha, beta = build_rank3_mps(length, eta, c_param)
        rank = 3

    prob = probability_map(cores, alpha, beta, length, d)
    hankel, prefixes, suffixes = hankel_from_prob_map(prob, length, d)
    masks = hankel_column_masks(suffixes, d)
    sigma1, sigma2, sigma3 = top_k_singular_values(hankel, 3)

    # identify all-zero prefix/suffix indices (fallback to 0)
    zero_prefix = tuple([0 for _ in range(length // 2)])
    zero_suffix = tuple([0 for _ in range(length - length // 2)])
    try:
        prefix_zero_idx = prefixes.index(list(zero_prefix))
    except ValueError:
        prefix_zero_idx = 0
    try:
        suffix_zero_idx = suffixes.index(list(zero_suffix))
    except ValueError:
        suffix_zero_idx = 0

    rows: List[Dict[str, float]] = []
    baseline = {
        "model": model,
        "length": length,
        "sample_size": 0,
        "trial": 0,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "log_sigma3": math.log(max(sigma3, 1e-16)),
        "frob_error": 0.0,
        "spectral_error": 0.0,
        "tv_error": 0.0,
        "max_error": 0.0,
        "sigma_min_emp": sigma3,
        "prefixes": len(prefixes),
        "suffixes": len(suffixes),
    }
    rows.append(baseline)

    for n in sample_sizes:
        for t in range(trials):
            emp = empirical_hankel(prob, prefixes, suffixes, n)
            diff = [[hankel[i][j] - emp[i][j] for j in range(len(suffixes))] for i in range(len(prefixes))]
            frob = frobenius_norm(diff)
            spec = spectral_norm(diff)

            B_list, init_vec, final_vec, sigma_min_emp = spectral_wfa_from_hankel(
                emp, masks, rank, prefix_zero_idx, suffix_zero_idx
            )

            max_err = 0.0
            tv_err = 0.0
            for word in all_words(length, d):
                p_true = prob[tuple(word)]
                p_hat = predict_sequence_prob(word, B_list, init_vec, final_vec)
                p_hat = max(p_hat, 0.0)
                max_err = max(max_err, abs(p_true - p_hat))
                tv_err += abs(p_true - p_hat)
            tv_err *= 0.5

            rows.append(
                {
                    "model": model,
                    "length": length,
                    "sample_size": n,
                    "trial": t + 1,
                    "sigma1": sigma1,
                    "sigma2": sigma2,
                    "sigma3": sigma3,
                    "log_sigma3": math.log(max(sigma3, 1e-16)),
                    "frob_error": frob,
                    "spectral_error": spec,
                    "tv_error": tv_err,
                    "max_error": max_err,
                    "sigma_min_emp": sigma_min_emp,
                    "prefixes": len(prefixes),
                    "suffixes": len(suffixes),
                }
            )
    return rows


def parse_comma_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4: condition numbers and hard examples")
    parser.add_argument("--lengths", type=str, default="6,10,14", help="Comma-separated lengths to evaluate")
    parser.add_argument("--eta", type=float, default=0.6, help="Eta parameter for diagonal constructions")
    parser.add_argument("--c-param", type=float, default=0.5, dest="c_param", help="c parameter for rank-3 construction")
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="200,1000,5000",
        help="Comma-separated sample sizes for empirical Hankel estimates",
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials per sample size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="experiments/exp4_results.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    lengths = parse_comma_ints(args.lengths)
    sample_sizes = parse_comma_ints(args.sample_sizes)

    rows: List[Dict[str, float]] = []
    for length in lengths:
        for model in ("rank1", "rank3"):
            rows.extend(run_trial(model, length, args.eta, args.c_param, sample_sizes, args.trials))

    fieldnames = [
        "model",
        "length",
        "sample_size",
        "trial",
        "sigma1",
        "sigma2",
        "sigma3",
        "log_sigma3",
        "frob_error",
        "spectral_error",
        "tv_error",
        "max_error",
        "sigma_min_emp",
        "prefixes",
        "suffixes",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
