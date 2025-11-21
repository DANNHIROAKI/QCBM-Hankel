"""Experiment 6: Multi-view Hankel stacking and stability.

This script implements the expanded multi-view experiment (§8.3):

* sample a medium-conditioned MPS once per trial;
* build multiple Hankel "views" (different lengths / cut points) on the same
  ground-truth model;
* compare single-view vs. jointly stacked Hankels in terms of smallest
  non-zero singular value and empirical stability under finite sampling;
* record coherence, Hankel deviations, and joint/single-view sigma_r values to
  validate the benefit predicted by Theorem 8.7.

The implementation is dependency-free and reuses the simple linear-algebra
helpers from earlier experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple


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
        mid = mat_add(mid, identity(n))
        mid = mat_add(mid, mat_scale(ZY, -1.0))
        mid = mat_scale(mid, 0.5)
        Y = matmul(Y, mid)
        Z = matmul(mid, Z)
    return mat_scale(Z, 1.0 / math.sqrt(scale))


# ---------------------------------------------------------------------------
# MPS helpers and probability utilities
# ---------------------------------------------------------------------------


def random_left_canonical_mps(length: int, bond_dim: int, d: int) -> Tuple[List[List[List[complex]]], List[complex], List[complex]]:
    cores: List[List[List[complex]]] = []
    for _ in range(length):
        raw: List[List[List[complex]]] = []
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
        for k in list(probs.keys()):
            probs[k] /= total
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


def enumerate_prefix_suffix(length: int, d: int, max_prefixes: int, max_suffixes: int, cut: int | None = None) -> Tuple[List[List[int]], List[List[int]]]:
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
    suffix_set = {tuple(s): idx for idx, s in enumerate(suffixes)}
    for u in prefixes:
        row: List[float] = [0.0 for _ in suffixes]
        for v in suffixes:
            x = tuple(u + list(v))
            row[suffix_set[tuple(v)]] = prob.get(x, 0.0)
        hankel.append(row)
    return hankel


def truncated_svd_rank(hankel: Sequence[Sequence[float]], rank: int) -> Tuple[List[List[complex]], List[float], List[List[complex]]]:
    m = len(hankel)
    n = len(hankel[0]) if hankel else 0
    rank = max(1, min(rank, m, n))
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
        sigma_sq = sum(v[i] * sum(gram[i][j] * v[j] for j in range(m)) for i in range(m)).real
        sigma = math.sqrt(max(sigma_sq, 0.0))
        S[r] = sigma
        for i in range(m):
            U[i][r] = v[i]
        for i in range(m):
            for j in range(m):
                gram[i][j] -= sigma_sq * (v[i] * v[j])
    V: List[List[complex]] = [[0.0j for _ in range(rank)] for _ in range(n)]
    for k in range(n):
        for r in range(rank):
            if S[r] > 0:
                V[k][r] = sum(hankel[i][k] * U[i][r] for i in range(m)) / S[r]
    return U, S, V


def embed_from_svd(
    prefixes: Sequence[Sequence[int]],
    suffixes: Sequence[Sequence[int]],
    U: Sequence[Sequence[complex]],
    S: Sequence[float],
    V: Sequence[Sequence[complex]],
    rank: int,
) -> Tuple[Dict[Tuple[int, ...], List[complex]], Dict[Tuple[int, ...], List[complex]]]:
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
    if not hankel:
        return 0.0
    row_sums = [sum(row) for row in hankel]
    col_sums = [sum(hankel[i][j] for i in range(len(hankel))) for j in range(len(hankel[0]))]
    return max(max(row_sums), max(col_sums))


def build_joint_matrix(
    O_list: Sequence[Sequence[Sequence[complex]]],
    C_list: Sequence[Sequence[Sequence[complex]]],
) -> List[List[complex]]:
    if not O_list or not C_list:
        return []
    r = min(
        [len(O[0]) for O in O_list if O] + [len(C) for C in C_list if C]
    )
    if r == 0:
        return []
    rows = sum(len(O) for O in O_list)
    cols = sum(len(C[0]) for C in C_list)
    O_joint: List[List[complex]] = [[0.0j for _ in range(r)] for _ in range(rows)]
    row_offset = 0
    for O in O_list:
        for i in range(len(O)):
            O_joint[row_offset + i] = list(O[i][:r])
        row_offset += len(O)
    C_joint: List[List[complex]] = [[0.0j for _ in range(cols)] for _ in range(r)]
    col_offset = 0
    for C in C_list:
        for i in range(r):
            for j in range(len(C[0])):
                C_joint[i][col_offset + j] = C[i][j]
        col_offset += len(C[0])
    return matmul(O_joint, C_joint)


def sigma_r_from_hankel(hankel: Sequence[Sequence[float]], rank: int) -> Tuple[float, List[List[complex]], List[float], List[List[complex]]]:
    if not hankel or not hankel[0]:
        return 0.0, [], [], []
    U, S, V = truncated_svd_rank(hankel, rank)
    sigma_r = S[min(rank, len(S)) - 1] if S else 0.0
    return sigma_r, U, S, V


def hankel_difference(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    rows = min(len(a), len(b))
    cols = min(len(a[0]), len(b[0])) if rows > 0 else 0
    diff: List[List[complex]] = []
    for i in range(rows):
        diff.append([a[i][j] - b[i][j] for j in range(cols)])
    return spectral_norm(diff)


def renormalize_distribution(prob: Dict[Tuple[int, ...], float]) -> Dict[Tuple[int, ...], float]:
    cleaned = {k: max(0.0, v) for k, v in prob.items()}
    total = sum(cleaned.values())
    if total > 0:
        for k in list(cleaned.keys()):
            cleaned[k] /= total
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
        preds[tuple(word)] = sum(a * b for a, b in zip(phi[u], psi[v])).real
    return renormalize_distribution(preds)


def pointwise_errors(true_prob: Dict[Tuple[int, ...], float], pred_prob: Dict[Tuple[int, ...], float]) -> Tuple[float, float]:
    keys = set(true_prob.keys()) | set(pred_prob.keys())
    tv = 0.0
    max_err = 0.0
    for k in keys:
        err = abs(pred_prob.get(k, 0.0) - true_prob.get(k, 0.0))
        tv += err
        max_err = max(max_err, err)
    return max_err, 0.5 * tv


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
    rows: List[Dict[str, float]] = []
    if len(set(lengths)) != 1:
        raise ValueError("All lengths must match for multi-view comparison; pass repeated L with different cuts.")
    base_len = lengths[0]
    cores, alpha, beta = random_left_canonical_mps(base_len, bond_dim, d)
    eval_support = all_words(base_len, d)
    prob_true_global = probability_map(cores, alpha, beta, base_len, d, eval_support)
    view_data = []
    joint_true: List[List[complex]] = []
    for idx, L in enumerate(lengths):
        cut = cuts[idx] if idx < len(cuts) else None
        prefixes, suffixes = enumerate_prefix_suffix(L, d, max_prefixes, max_suffixes, cut=cut)
        prob_true = {tuple(x): prob_true_global[tuple(x)] for x in eval_support}
        hankel_true = hankel_from_prob_map(prob_true, prefixes, suffixes)
        rank_use = max(1, min(rank_cap, len(prefixes), len(suffixes)))
        sigma_true, U, S, V = sigma_r_from_hankel(hankel_true, rank_use)
        mu = compute_coherence(hankel_true)
        sqrt_S = [math.sqrt(s) for s in S[:rank_use]]
        O = [[U[i][r] * sqrt_S[r] for r in range(rank_use)] for i in range(len(prefixes))]
        C = [[V[j][r] * sqrt_S[r] for j in range(len(suffixes))] for r in range(rank_use)]
        view_data.append(
            {
                "length": L,
                "cut": cut if cut is not None else L // 2,
                "prefixes": prefixes,
                "suffixes": suffixes,
                "prob_true": prob_true,
                "hankel_true": hankel_true,
                "rank": rank_use,
                "sigma_true": sigma_true,
                "mu": mu,
                "O": O,
                "C": C,
            }
        )
        rows.append(
            {
                "trial": trial,
                "mode": "single_true",
                "view": idx,
                "length": L,
                "cut": cut if cut is not None else L // 2,
                "sample_size": 0,
                "sigma_r_true": sigma_true,
                "sigma_r_emp": sigma_true,
                "delta_hankel": 0.0,
                "mu": mu,
                "rank": rank_use,
                "max_error": 0.0,
                "tv_error": 0.0,
            }
        )

        joint_true = build_joint_matrix([v["O"] for v in view_data], [v["C"] for v in view_data])
    joint_rank = min(rank_cap, min(len(v["O"][0]) for v in view_data)) if view_data else 1
    sigma_joint_true, _, _, _ = sigma_r_from_hankel(joint_true, joint_rank)
    rows.append(
        {
            "trial": trial,
            "mode": "joint_true",
            "view": -1,
            "length": -1,
            "cut": -1,
            "sample_size": 0,
            "sigma_r_true": sigma_joint_true,
            "sigma_r_emp": sigma_joint_true,
            "delta_hankel": 0.0,
            "mu": 0.0,
            "rank": joint_rank,
            "max_error": 0.0,
            "tv_error": 0.0,
        }
    )

    for sample_size in sample_sizes:
        samples = sample_sequences(prob_true_global, sample_size)
        emp_O_list: List[List[List[complex]]] = []
        emp_C_list: List[List[List[complex]]] = []
        per_view_cuts: List[int] = []
        for idx, data in enumerate(view_data):
            pre_list = data["prefixes"]
            suf_list = data["suffixes"]
            prob_true = data["prob_true"]
            hankel_emp = [[0.0 for _ in range(len(suf_list))] for _ in range(len(pre_list))]
            pre_index = {tuple(p): i for i, p in enumerate(pre_list)}
            suf_index = {tuple(s): j for j, s in enumerate(suf_list)}
            for seq in samples:
                u = seq[: data["cut"]]
                v = seq[data["cut"] :]
                i = pre_index.get(tuple(u))
                j = suf_index.get(tuple(v))
                if i is None or j is None:
                    continue
                hankel_emp[i][j] += 1.0
            if samples:
                for i in range(len(hankel_emp)):
                    for j in range(len(hankel_emp[0])):
                        hankel_emp[i][j] /= float(len(samples))
            sigma_emp, Ue, Se, Ve = sigma_r_from_hankel(hankel_emp, data["rank"])
            sqrt_Se = [math.sqrt(s) for s in Se[: data["rank"]]]
            Oe = [[Ue[i][r] * sqrt_Se[r] for r in range(data["rank"])] for i in range(len(hankel_emp))]
            Ce = [[Ve[j][r] * sqrt_Se[r] for j in range(len(hankel_emp[0]))] for r in range(data["rank"])]
            emp_O_list.append(Oe)
            emp_C_list.append(Ce)
            delta = hankel_difference(data["hankel_true"], hankel_emp)
            phi, psi = embed_from_svd(pre_list, suf_list, Ue, Se, Ve, data["rank"])
            preds = predict_from_embeddings(phi, psi, data["cut"], eval_support)
            max_err, tv_err = pointwise_errors(prob_true_global, preds)
            per_view_cuts.append(data["cut"])
            rows.append(
                {
                    "trial": trial,
                    "mode": "single_emp",
                    "view": idx,
                    "length": data["length"],
                    "cut": data["cut"],
                    "sample_size": sample_size,
                    "sigma_r_true": data["sigma_true"],
                    "sigma_r_emp": sigma_emp,
                    "delta_hankel": delta,
                    "mu": data["mu"],
                    "rank": data["rank"],
                    "max_error": max_err,
                    "tv_error": tv_err,
                }
            )

        joint_emp = build_joint_matrix(emp_O_list, emp_C_list)
        sigma_joint_emp, _, _, _ = sigma_r_from_hankel(joint_emp, joint_rank)
        delta_joint = hankel_difference(joint_true, joint_emp) if joint_true and joint_emp else 0.0
        # Multi-view prediction: average per-view scores built from the joint embeddings
        joint_preds: Dict[Tuple[int, ...], float] = {}
        if joint_emp:
            # Build embeddings for the union of prefixes/suffixes
            joint_prefixes = list({tuple(p) for v in view_data for p in v["prefixes"]})
            joint_suffixes = list({tuple(s) for v in view_data for s in v["suffixes"]})
            # Ensure deterministic ordering
            joint_prefixes.sort()
            joint_suffixes.sort()
            Uj, Sj, Vj = truncated_svd_rank(joint_emp, joint_rank)
            phi_joint, psi_joint = embed_from_svd(joint_prefixes, joint_suffixes, Uj, Sj, Vj, joint_rank)
            counts: Dict[Tuple[int, ...], float] = {}
            for word in eval_support:
                val = 0.0
                used = 0
                for cut in per_view_cuts:
                    u = tuple(word[:cut])
                    v = tuple(word[cut:])
                    if u in phi_joint and v in psi_joint:
                        val += sum(a * b for a, b in zip(phi_joint[u], psi_joint[v])).real
                        used += 1
                joint_preds[tuple(word)] = val / used if used > 0 else 0.0
            joint_preds = renormalize_distribution(joint_preds)
        joint_max_err, joint_tv_err = pointwise_errors(prob_true_global, joint_preds)
        rows.append(
            {
                "trial": trial,
                "mode": "joint_emp",
                "view": -1,
                "length": -1,
                "cut": -1,
                "sample_size": sample_size,
                "sigma_r_true": sigma_joint_true,
                "sigma_r_emp": sigma_joint_emp,
                "delta_hankel": delta_joint,
                "mu": 0.0,
                "rank": joint_rank,
                "max_error": joint_max_err,
                "tv_error": joint_tv_err,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 6: multi-view Hankel stacking")
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
        help="Optional comma-separated cut points per view (matched to lengths)",
    )
    parser.add_argument("--sample-sizes", type=str, default="500,2000", help="Comma-separated sample sizes per view")
    parser.add_argument("--bond-dim", type=int, default=3, help="Bond dimension for the ground-truth MPS")
    parser.add_argument("--alphabet-size", type=int, default=2, help="Alphabet size |Sigma|")
    parser.add_argument("--max-prefixes", type=int, default=128, help="Cap on prefix set size per view")
    parser.add_argument("--max-suffixes", type=int, default=128, help="Cap on suffix set size per view")
    parser.add_argument("--rank", type=int, default=4, help="Rank cap for SVD truncation")
    parser.add_argument("--trials", type=int, default=3, help="Number of independent trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--output", type=str, default="experiments/exp6_results.csv", help="Output CSV path")
    args = parser.parse_args()

    random.seed(args.seed)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    cuts = [int(x) if x else None for x in args.cuts.split(",") if x]
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
