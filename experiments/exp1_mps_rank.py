"""Experiment 1: MPS bond dimension vs Hankel rank (expanded protocol).

This script follows the extended experiment plan:
* Build random, rank-1 (§7.2A), and rank-3 (§7.2B) MPS instances.
* Enforce per-site left normalisation for random MPS to improve stability.
* Only evaluate configurations where prefix/suffix counts exceed D^2 so the
  Hankel ceiling is not grid-limited.
* Estimate amplitude/probability Hankel ranks via prefix/suffix embeddings with
  a relative Gram–Schmidt tolerance, and record normalised ratios/frequencies.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

SequenceType = Tuple[int, ...]


def generate_sequences(length: int, alphabet: Sequence[int]) -> List[SequenceType]:
    return list(itertools.product(alphabet, repeat=length))


def random_complex() -> complex:
    return random.gauss(0, 1) + 1j * random.gauss(0, 1)


def random_matrix(rows: int, cols: int) -> List[List[complex]]:
    return [[random_complex() / math.sqrt(2 * rows) for _ in range(cols)] for _ in range(rows)]


def normalize(vec: List[complex]) -> List[complex]:
    norm = math.sqrt(sum(abs(x) ** 2 for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def matmul(a: List[List[complex]], b: List[List[complex]]) -> List[List[complex]]:
    rows, cols, mid = len(a), len(b[0]), len(b)
    out = [[0.0j for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(mid):
            aik = a[i][k]
            if abs(aik) < 1e-15:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def eig_hermitian(mat: List[List[complex]]):
    """Tiny Hermitian eigensolver for D<=4 (sufficient for this experiment)."""
    import cmath

    D = len(mat)
    if D == 1:
        return [mat[0][0].real], [[1.0]]
    if D == 2:
        a, b = mat[0][0], mat[0][1]
        c, d = mat[1][0], mat[1][1]
        tr = (a + d).real
        det = (a * d - b * c).real
        disc = cmath.sqrt(tr * tr - 4 * det)
        evals = [(tr + disc).real / 2, (tr - disc).real / 2]
        evecs = []
        for lam in evals:
            if abs(b) > 1e-12 or abs(a - lam) > 1e-12:
                vec = [b, lam - a]
            else:
                vec = [lam - d, c]
            vec = normalize(vec)
            evecs.append(vec)
        return evals, [[evecs[r][c] for r in range(D)] for c in range(D)]

    # Power iteration + Gram-Schmidt deflation for D>2 (up to 4).
    vectors = [[1.0 if i == j else 0.0 for i in range(D)] for j in range(D)]
    evals: List[float] = []
    evecs: List[List[complex]] = []
    for _ in range(D):
        v = vectors.pop()
        for _ in range(30):
            mv = [sum(mat[i][k] * v[k] for k in range(D)) for i in range(D)]
            for ev in evecs:
                dot = sum(x * y.conjugate() for x, y in zip(mv, ev))
                mv = [x - dot * y for x, y in zip(mv, ev)]
            norm = math.sqrt(sum(abs(x) ** 2 for x in mv))
            if norm < 1e-12:
                break
            v = [x / norm for x in mv]
        mv = [sum(mat[i][k] * v[k] for k in range(D)) for i in range(D)]
        lam = sum(v[i].conjugate() * mv[i] for i in range(D)).real
        evals.append(lam)
        evecs.append(v)
    return evals, [[evecs[r][c] for r in range(D)] for c in range(D)]


def build_random_mps(length: int, d: int, D: int):
    tensors: List[List[List[List[complex]]]] = []
    for _ in range(length):
        letter_tensors = [random_matrix(D, D) for _ in range(d)]

        # Left normalisation: enforce sum_sigma A A^dagger ≈ I for stability.
        gram = [[0.0 for _ in range(D)] for _ in range(D)]
        for mat in letter_tensors:
            for i in range(D):
                for j in range(D):
                    gram[i][j] += sum(mat[i][k] * mat[j][k].conjugate() for k in range(D))
        evals, evecs = eig_hermitian(gram)
        inv_sqrt = [[0.0j for _ in range(D)] for _ in range(D)]
        for idx, val in enumerate(evals):
            inv = 1.0 / math.sqrt(max(val, 1e-12))
            for i in range(D):
                for j in range(D):
                    inv_sqrt[i][j] += inv * evecs[i][idx] * evecs[j][idx].conjugate()
        for m_idx, mat in enumerate(letter_tensors):
            letter_tensors[m_idx] = matmul(inv_sqrt, mat)
        tensors.append(letter_tensors)
    alpha = normalize([random_complex() for _ in range(D)])
    beta = normalize([random_complex() for _ in range(D)])
    return tensors, alpha, beta


def build_structured_mps_rank1(length: int, d: int, D: int, eta: float = 0.2):
    """Rank-1 construction from §7.2A (Hankel rank=1)."""
    tensors: List[List[List[List[complex]]]] = []
    base_diag = [1.0 for _ in range(D)]
    if D >= 2:
        base_diag[1] = eta
    for _ in range(length):
        letter_tensors: List[List[List[complex]]] = []
        for idx in range(d):
            diag = base_diag.copy()
            diag[0] = 1.0 if idx == 0 else eta
            mat = [[0.0j for _ in range(D)] for _ in range(D)]
            for i in range(D):
                mat[i][i] = diag[i]
            letter_tensors.append(mat)
        tensors.append(letter_tensors)
    alpha = [1.0] + [0.0 for _ in range(D - 1)]
    beta = [1.0] + [0.0 for _ in range(D - 1)]
    return tensors, alpha, beta


def build_structured_mps_rank3(length: int, d: int, D: int, eta: float = 0.2, c: float = 0.6):
    """Rank-3 style construction from §7.2B (rank ≤ 3, near-singular third mode)."""
    tensors: List[List[List[List[complex]]]] = []
    if D < 2:
        raise ValueError("Rank-3 construction requires D>=2")
    for _ in range(length):
        letter_tensors: List[List[List[complex]]] = []
        mats = [
            [[1.0, 0.0j], [0.0j, eta]],
            [[eta, 0.0j], [0.0j, 1.0]],
        ]
        if D > 2:
            padded = []
            for base in mats:
                mat = [[0.0j for _ in range(D)] for _ in range(D)]
                for i in range(2):
                    mat[i][i] = base[i][i]
                for i in range(2, D):
                    mat[i][i] = 1.0
                padded.append(mat)
            mats = padded
        for idx in range(d):
            if idx < len(mats):
                letter_tensors.append(mats[idx])
            else:
                mat = [[0.0j for _ in range(D)] for _ in range(D)]
                for i in range(D):
                    mat[i][i] = 1.0
                letter_tensors.append(mat)
        tensors.append(letter_tensors)
    alpha = normalize([1.0, c] + [0.0 for _ in range(D - 2)])
    beta = normalize([1.0, c] + [0.0 for _ in range(D - 2)])
    return tensors, alpha, beta


def left_multiply(vec: List[complex], mat: List[List[complex]]) -> List[complex]:
    result = []
    for j in range(len(mat[0])):
        total = 0.0j
        for k, val in enumerate(vec):
            total += val * mat[k][j]
        result.append(total)
    return result


def right_multiply(mat: List[List[complex]], vec: List[complex]) -> List[complex]:
    result = []
    for i in range(len(mat)):
        total = 0.0j
        for k, val in enumerate(vec):
            total += mat[i][k] * val
        result.append(total)
    return result


def kron(vec: List[complex]) -> List[complex]:
    result = []
    for a in vec:
        for b in vec:
            result.append(a * b.conjugate())
    return result


def gram_schmidt_basis(vectors: Iterable[List[complex]], tol: float) -> List[List[complex]]:
    basis: List[List[complex]] = []
    max_norm = 0.0
    for v in vectors:
        v_proj = v.copy()
        for b in basis:
            dot = sum(x * y.conjugate() for x, y in zip(v_proj, b))
            v_proj = [x - dot * y for x, y in zip(v_proj, b)]
        norm = math.sqrt(sum(abs(x) ** 2 for x in v_proj))
        max_norm = max(max_norm, norm)
        if norm > tol * max(1.0, max_norm):
            basis.append([x / norm for x in v_proj])
    return basis


def compute_prefix_embeddings(
    tensors: List[List[List[List[complex]]]],
    alpha: List[complex],
    alphabet: Sequence[int],
    t_star: int,
    probability: bool,
) -> List[List[complex]]:
    embeddings: Dict[SequenceType, List[complex]] = {(): alpha}
    for pos in range(t_star):
        new_embeddings: Dict[SequenceType, List[complex]] = {}
        for seq, state in embeddings.items():
            for symbol in alphabet:
                new_state = left_multiply(state, tensors[pos][symbol])
                new_embeddings[seq + (symbol,)] = new_state
        embeddings = new_embeddings
    vectors = []
    for prefix in generate_sequences(t_star, alphabet):
        state = embeddings[prefix]
        vectors.append(kron(state) if probability else state)
    return vectors


def compute_suffix_embeddings(
    tensors: List[List[List[List[complex]]]],
    beta: List[complex],
    alphabet: Sequence[int],
    t_star: int,
    probability: bool,
) -> List[List[complex]]:
    embeddings: Dict[SequenceType, List[complex]] = {(): beta}
    length = len(tensors)
    for step in range(length - 1, t_star - 1, -1):
        new_embeddings: Dict[SequenceType, List[complex]] = {}
        for seq, state in embeddings.items():
            for symbol in alphabet:
                new_state = right_multiply(tensors[step][symbol], state)
                new_embeddings[(symbol,) + seq] = new_state
        embeddings = new_embeddings
    vectors = []
    for suffix in generate_sequences(length - t_star, alphabet):
        state = embeddings[suffix]
        vectors.append(kron(state) if probability else state)
    return vectors


def matrix_rank_from_factor(L: List[List[complex]], R: List[List[complex]], tol: float) -> int:
    if not L or not R:
        return 0
    cols_R = len(R[0])
    vectors_R = [[R[i][j] for i in range(len(R))] for j in range(cols_R)]
    basis_R = gram_schmidt_basis(vectors_R, tol)
    if not basis_R:
        return 0
    LB_columns: List[List[complex]] = []
    for b in basis_R:
        column = []
        for row in L:
            column.append(sum(x * y for x, y in zip(row, b)))
        LB_columns.append(column)
    return len(gram_schmidt_basis(LB_columns, tol))


def run_experiment(
    alphabet_sizes: Sequence[int],
    lengths: Sequence[int],
    bond_dims: Sequence[int],
    random_samples: int,
    structured_samples: int,
    eta: float,
    c_param: float,
    tol: float,
    seed: int,
    output_path: str,
) -> None:
    random.seed(seed)
    records: List[dict] = []

    for d in alphabet_sizes:
        alphabet = list(range(d))
        for L in lengths:
            t_star = L // 2
            prefix_count = d ** t_star
            suffix_count = d ** (L - t_star)
            for D in bond_dims:
                max_rank_amp = min(D, prefix_count, suffix_count)
                max_rank_prob = min(D ** 2, prefix_count, suffix_count)
                if prefix_count < D ** 2 or suffix_count < D ** 2:
                    print(
                        f"Skipping d={d}, L={L}, D={D} (prefix/suffix cap {prefix_count}/{suffix_count} < D^2)"
                    )
                    continue
                for idx in range(random_samples):
                    tensors, alpha, beta = build_random_mps(L, d, D)
                    L_amp = compute_prefix_embeddings(tensors, alpha, alphabet, t_star, probability=False)
                    R_amp = compute_suffix_embeddings(tensors, beta, alphabet, t_star, probability=False)
                    L_prob = compute_prefix_embeddings(tensors, alpha, alphabet, t_star, probability=True)
                    R_prob = compute_suffix_embeddings(tensors, beta, alphabet, t_star, probability=True)
                    rank_a = matrix_rank_from_factor(L_amp, R_amp, tol)
                    rank_p = matrix_rank_from_factor(L_prob, R_prob, tol)
                    records.append(
                        {
                            "type": "random",
                            "alphabet": d,
                            "length": L,
                            "bond_dim": D,
                            "sample_id": idx,
                            "rank_Ha": rank_a,
                            "rank_Hp": rank_p,
                            "cap_Ha": max_rank_amp,
                            "cap_Hp": max_rank_prob,
                            "ratio_Ha": rank_a / float(max_rank_amp),
                            "ratio_Hp": rank_p / float(max_rank_prob),
                        }
                    )
                for idx in range(structured_samples):
                    tensors, alpha, beta = build_structured_mps_rank1(L, d, D, eta=eta)
                    L_amp = compute_prefix_embeddings(tensors, alpha, alphabet, t_star, probability=False)
                    R_amp = compute_suffix_embeddings(tensors, beta, alphabet, t_star, probability=False)
                    L_prob = compute_prefix_embeddings(tensors, alpha, alphabet, t_star, probability=True)
                    R_prob = compute_suffix_embeddings(tensors, beta, alphabet, t_star, probability=True)
                    rank_a = matrix_rank_from_factor(L_amp, R_amp, tol)
                    rank_p = matrix_rank_from_factor(L_prob, R_prob, tol)
                    records.append(
                        {
                            "type": "structured_rank1",
                            "alphabet": d,
                            "length": L,
                            "bond_dim": D,
                            "sample_id": idx,
                            "rank_Ha": rank_a,
                            "rank_Hp": rank_p,
                            "cap_Ha": max_rank_amp,
                            "cap_Hp": max_rank_prob,
                            "ratio_Ha": rank_a / float(max_rank_amp),
                            "ratio_Hp": rank_p / float(max_rank_prob),
                        }
                    )
                for idx in range(structured_samples):
                    tensors, alpha, beta = build_structured_mps_rank3(L, d, D, eta=eta, c=c_param)
                    L_amp = compute_prefix_embeddings(tensors, alpha, alphabet, t_star, probability=False)
                    R_amp = compute_suffix_embeddings(tensors, beta, alphabet, t_star, probability=False)
                    L_prob = compute_prefix_embeddings(tensors, alpha, alphabet, t_star, probability=True)
                    R_prob = compute_suffix_embeddings(tensors, beta, alphabet, t_star, probability=True)
                    rank_a = matrix_rank_from_factor(L_amp, R_amp, tol)
                    rank_p = matrix_rank_from_factor(L_prob, R_prob, tol)
                    records.append(
                        {
                            "type": "structured_rank3",
                            "alphabet": d,
                            "length": L,
                            "bond_dim": D,
                            "sample_id": idx,
                            "rank_Ha": rank_a,
                            "rank_Hp": rank_p,
                            "cap_Ha": max_rank_amp,
                            "cap_Hp": max_rank_prob,
                            "ratio_Ha": rank_a / float(max_rank_amp),
                            "ratio_Hp": rank_p / float(max_rank_prob),
                        }
                    )
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
                "alphabet",
                "length",
                "bond_dim",
                "sample_id",
                "rank_Ha",
                "rank_Hp",
                "cap_Ha",
                "cap_Hp",
                "ratio_Ha",
                "ratio_Hp",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records to {output_path}")

    for D in bond_dims:
        for d in alphabet_sizes:
            for L in lengths:
                ratios = [
                    rec["ratio_Hp"]
                    for rec in records
                    if rec["bond_dim"] == D and rec["type"] == "random" and rec["alphabet"] == d and rec["length"] == L
                ]
                if not ratios:
                    continue
                mean_ratio = sum(ratios) / len(ratios)
                full_rank = sum(1 for r in ratios if abs(r - 1.0) < 1e-6)
                print(
                    f"Random MPS d={d}, L={L}, D={D}: mean rank ratio={mean_ratio:.3f}, "
                    f"full-rank freq {full_rank}/{len(ratios)}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 on MPS Hankel ranks")
    parser.add_argument("--random-samples", type=int, default=10, help="Number of random MPS per configuration")
    parser.add_argument("--structured-samples", type=int, default=2, help="Number of structured MPS per configuration")
    parser.add_argument("--eta", type=float, default=0.2, help="Eta parameter for structured examples")
    parser.add_argument("--c-param", type=float, default=0.6, help="c parameter for rank-3 structured example")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--tol", type=float, default=1e-6, help="Relative tolerance for Gram-Schmidt rank")
    parser.add_argument("--output", type=str, default="experiments/exp1_results.csv", help="CSV output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        alphabet_sizes=[2, 4],
        lengths=[8, 10, 12],
        bond_dims=[2, 3, 4],
        random_samples=args.random_samples,
        structured_samples=args.structured_samples,
        eta=args.eta,
        c_param=args.c_param,
        tol=args.tol,
        seed=args.seed,
        output_path=args.output,
    )
