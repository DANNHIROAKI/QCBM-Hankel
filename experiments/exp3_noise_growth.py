"""Experiment 3: gate noise and length-dependent Hankel amplification.

This driver follows the expanded protocol: start from left-canonical MPS,
inject spectrally bounded gate noise, measure Hankel differences across
lengths, and compare raw QCBM dynamics against a row-substochastic projection
that mimics the contractive setting of Theorems 5.5/8.1.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Basic linear algebra utilities (complex lists, small dimensions)
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


def mat_add(a: Sequence[Sequence[complex]], b: Sequence[Sequence[complex]]) -> List[List[complex]]:
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def mat_scale(a: Sequence[Sequence[complex]], s: complex) -> List[List[complex]]:
    return [[s * x for x in row] for row in a]


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


def identity(n: int) -> List[List[complex]]:
    return [[1.0 + 0.0j if i == j else 0.0j for j in range(n)] for i in range(n)]


def frobenius_norm(mat: Sequence[Sequence[complex]]) -> float:
    return math.sqrt(sum(abs(x) ** 2 for row in mat for x in row))


def conj_transpose(mat: Sequence[Sequence[complex]]) -> List[List[complex]]:
    rows = len(mat)
    cols = len(mat[0]) if mat else 0
    return [[mat[i][j].conjugate() for i in range(rows)] for j in range(cols)]


def matrix_power_inverse_sqrt(mat: Sequence[Sequence[complex]], iters: int = 8) -> List[List[complex]]:
    """Approximate mat^{-1/2} using Newton–Schulz; mat must be PD."""

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


def spectral_norm_complex(mat: Sequence[Sequence[complex]], iters: int = 30) -> float:
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


def spectral_norm_float(mat: Sequence[Sequence[float]], iters: int = 25) -> float:
    if not mat or not mat[0]:
        return 0.0
    n = len(mat[0])
    v = normalize([random.random() for _ in range(n)])
    for _ in range(iters):
        Av = matvec(mat, v)
        v = normalize(Av)
    Av = matvec(mat, v)
    return norm(Av)


# ---------------------------------------------------------------------------
# MPS construction and amplitudes
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
        site = []
        for mat in raw:
            site.append(matmul(inv_sqrt, mat))
        cores.append(site)
    alpha = normalize([random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)])
    beta = normalize([random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(bond_dim)])
    return cores, alpha, beta


def amplitude_for_sequence(seq: Sequence[int], cores: Sequence[Sequence[Sequence[complex]]], alpha: Sequence[complex], beta: Sequence[complex]) -> complex:
    vec = list(alpha)
    for site, sym in enumerate(seq):
        A = cores[site][sym]
        vec = matvec(A, vec)
    return dot(vec, beta)


def prob_map_for_targets_mps(
    cores: Sequence[Sequence[Sequence[complex]]],
    alpha: Sequence[complex],
    beta: Sequence[complex],
    length: int,
    d: int,
    targets: Sequence[Sequence[int]],
    total_mass: float,
) -> Dict[Tuple[int, ...], float]:
    prob_map: Dict[Tuple[int, ...], float] = {}
    for seq in targets:
        amp = amplitude_for_sequence(seq, cores, alpha, beta)
        prob_map[tuple(seq)] = abs(amp) ** 2
    if total_mass > 0:
        for key in list(prob_map.keys()):
            prob_map[key] /= total_mass
    return prob_map


# ---------------------------------------------------------------------------
# Hankel helpers
# ---------------------------------------------------------------------------


def int_to_word(val: int, length: int, d: int) -> List[int]:
    word = [0 for _ in range(length)]
    for i in range(length - 1, -1, -1):
        word[i] = val % d
        val //= d
    return word


def sample_words(length: int, d: int, max_words: int) -> List[List[int]]:
    total = d ** length
    if total <= max_words:
        return [int_to_word(val, length, d) for val in range(total)]
    chosen = set()
    while len(chosen) < max_words:
        chosen.add(random.randrange(total))
    return [int_to_word(val, length, d) for val in chosen]


def build_hankel_from_map(seq_prob: Dict[Tuple[int, ...], float], prefixes: Sequence[Sequence[int]], suffixes: Sequence[Sequence[int]]) -> List[List[float]]:
    rows = len(prefixes)
    cols = len(suffixes)
    mat: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for p_idx, prefix in enumerate(prefixes):
        for s_idx, suffix in enumerate(suffixes):
            seq = tuple(prefix + list(suffix))
            mat[p_idx][s_idx] = seq_prob.get(seq, 0.0)
    return mat


def mat_diff(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


# ---------------------------------------------------------------------------
# WFA projection helpers (row-substochastic cone)
# ---------------------------------------------------------------------------


def build_Bs(cores: Sequence[Sequence[Sequence[complex]]]) -> List[Dict[int, List[List[complex]]]]:
    Bs: List[Dict[int, List[List[complex]]]] = []
    for core in cores:
        d = len(core)
        D_left = len(core[0])
        D_right = len(core[0][0])
        site: Dict[int, List[List[complex]]] = {}
        for sym in range(d):
            A = core[sym]
            B: List[List[complex]] = [[0.0j for _ in range(D_right * D_right)] for _ in range(D_left * D_left)]
            for i in range(D_left):
                for k in range(D_left):
                    row = i * D_left + k
                    for j in range(D_right):
                        for l in range(D_right):
                            col = j * D_right + l
                            B[row][col] = A[i][j] * A[k][l].conjugate()
            site[sym] = B
        Bs.append(site)
    return Bs


def project_row_substochastic_site(site: Dict[int, List[List[complex]]]) -> Dict[int, List[List[float]]]:
    symbols = list(site.keys())
    rows = len(next(iter(site.values()))) if symbols else 0
    cols = len(next(iter(site.values()))[0]) if symbols else 0
    row_sums = [0.0 for _ in range(rows)]
    abs_mats: Dict[int, List[List[float]]] = {}
    for sym in symbols:
        mat = site[sym]
        real_mat: List[List[float]] = []
        for i in range(rows):
            real_row: List[float] = []
            for j in range(cols):
                val = abs(mat[i][j])
                real_row.append(val)
                row_sums[i] += val
            real_mat.append(real_row)
        abs_mats[sym] = real_mat
    scale = [1.0 if s <= 1.0 else 1.0 / s for s in row_sums]
    proj: Dict[int, List[List[float]]] = {}
    for sym, mat in abs_mats.items():
        proj_mat: List[List[float]] = []
        for i, row in enumerate(mat):
            proj_mat.append([scale[i] * x for x in row])
        proj[sym] = proj_mat
    return proj


def project_Bs_row_substochastic(Bs: Sequence[Dict[int, List[List[complex]]]]) -> List[Dict[int, List[List[float]]]]:
    return [project_row_substochastic_site(site) for site in Bs]


def total_mass_from_Bs(
    Bs: Sequence[Dict[int, List[List[complex]]]],
    alpha: Sequence[complex],
    beta: Sequence[complex],
) -> float:
    dim = len(next(iter(Bs[0].values()))) if Bs else 0
    i_vec = []
    f_vec = []
    for i in range(len(alpha)):
        for k in range(len(alpha)):
            i_vec.append(alpha[i] * alpha[k].conjugate())
    for j in range(len(beta)):
        for l in range(len(beta)):
            f_vec.append(beta[j].conjugate() * beta[l])
    vec = list(i_vec)
    for site in Bs:
        T = [[0.0j for _ in range(dim)] for _ in range(dim)]
        for mat in site.values():
            for r in range(dim):
                for c in range(dim):
                    T[r][c] += mat[r][c]
        vec = matvec(T, vec)
    return dot(vec, f_vec).real if f_vec else 0.0


def total_mass_from_proj_Bs(Bs: Sequence[Dict[int, List[List[float]]]]) -> float:
    dim = len(next(iter(Bs[0].values()))) if Bs else 0
    i_vec = [1.0] + [0.0 for _ in range(dim - 1)]
    f_vec = [1.0 / dim for _ in range(dim)] if dim > 0 else []
    vec = list(i_vec)
    for site in Bs:
        T = [[0.0 for _ in range(dim)] for _ in range(dim)]
        for mat in site.values():
            for r in range(dim):
                for c in range(dim):
                    T[r][c] += mat[r][c]
        vec = matvec(T, vec)
    return dot(vec, f_vec).real if f_vec else 0.0


def prob_map_for_targets_wfa(
    Bs: Sequence[Dict[int, List[List[float]]]],
    length: int,
    d: int,
    targets: Sequence[Sequence[int]],
    total_mass: float,
) -> Dict[Tuple[int, ...], float]:
    dim = len(next(iter(Bs[0].values()))) if Bs else 0
    i_vec = [1.0] + [0.0 for _ in range(dim - 1)]
    f_vec = [1.0 / dim for _ in range(dim)] if dim > 0 else []
    prob_map: Dict[Tuple[int, ...], float] = {}
    for seq in targets:
        vec = list(i_vec)
        for site, sym in enumerate(seq):
            vec = matvec(Bs[site][sym], vec)
        prob = dot(vec, f_vec).real if f_vec else 0.0
        prob_map[tuple(seq)] = prob
    if total_mass > 0:
        for key in list(prob_map.keys()):
            prob_map[key] /= total_mass
    return prob_map


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------


def add_spectral_noise(core: Sequence[Sequence[complex]], eps: float) -> List[List[complex]]:
    D_left = len(core)
    D_right = len(core[0]) if core else 0
    noise = [[random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(D_right)] for _ in range(D_left)]
    spec = spectral_norm_complex(noise)
    scale = eps / spec if spec > 1e-12 else 0.0
    return [[core[i][j] + scale * noise[i][j] for j in range(D_right)] for i in range(D_left)]


def add_noise_to_mps(cores: Sequence[Sequence[Sequence[complex]]], eps: float) -> List[List[List[complex]]]:
    noisy: List[List[List[complex]]] = []
    for site in cores:
        site_noisy = []
        for mat in site:
            site_noisy.append(add_spectral_noise(mat, eps))
        noisy.append(site_noisy)
    return noisy


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_experiment(
    lengths: Sequence[int],
    bond_dim: int,
    num_bases: int,
    epsilons: Sequence[float],
    seed: int,
    max_prefixes: int,
    output: str,
) -> None:
    random.seed(seed)
    d = 2
    records: List[Dict[str, float]] = []
    for L in lengths:
        t_star = L // 2
        prefixes = sample_words(t_star, d, max_prefixes)
        suffixes = sample_words(L - t_star, d, max_prefixes)
        seqs = [list(p) + list(s) for p in prefixes for s in suffixes]
        for base_idx in range(num_bases):
            cores, alpha, beta = random_left_canonical_mps(L, bond_dim, d)
            base_Bs = build_Bs(cores)
            base_total_mass = total_mass_from_Bs(base_Bs, alpha, beta)
            base_prob_map = prob_map_for_targets_mps(cores, alpha, beta, L, d, seqs, base_total_mass)
            base_hankel = build_hankel_from_map(base_prob_map, prefixes, suffixes)
            base_proj_Bs = project_Bs_row_substochastic(base_Bs)
            base_proj_total_mass = total_mass_from_proj_Bs(base_proj_Bs)
            base_proj_map = prob_map_for_targets_wfa(base_proj_Bs, L, d, seqs, base_proj_total_mass)
            base_proj_hankel = build_hankel_from_map(base_proj_map, prefixes, suffixes)

            for eps in epsilons:
                noisy_cores = add_noise_to_mps(cores, eps)
                noisy_Bs = build_Bs(noisy_cores)
                noisy_total_mass = total_mass_from_Bs(noisy_Bs, alpha, beta)
                noisy_map = prob_map_for_targets_mps(noisy_cores, alpha, beta, L, d, seqs, noisy_total_mass)
                noisy_hankel = build_hankel_from_map(noisy_map, prefixes, suffixes)
                diff = mat_diff(base_hankel, noisy_hankel)
                diff_spec = spectral_norm_float(diff)
                diff_fro = frobenius_norm(diff)

                noisy_proj_Bs = project_Bs_row_substochastic(noisy_Bs)
                noisy_proj_total_mass = total_mass_from_proj_Bs(noisy_proj_Bs)
                noisy_proj_map = prob_map_for_targets_wfa(noisy_proj_Bs, L, d, seqs, noisy_proj_total_mass)
                noisy_proj_hankel = build_hankel_from_map(noisy_proj_map, prefixes, suffixes)
                diff_proj = mat_diff(base_proj_hankel, noisy_proj_hankel)
                diff_proj_spec = spectral_norm_float(diff_proj)
                diff_proj_fro = frobenius_norm(diff_proj)

                records.append(
                    {
                        "length": L,
                        "t_star": t_star,
                        "bond_dim": bond_dim,
                        "base_index": base_idx,
                        "epsilon": eps,
                        "num_prefixes": len(prefixes),
                        "num_suffixes": len(suffixes),
                        "hankel_diff_spec": diff_spec,
                        "hankel_diff_fro": diff_fro,
                        "hankel_diff_spec_proj": diff_proj_spec,
                        "hankel_diff_fro_proj": diff_proj_fro,
                    }
                )

    fieldnames = [
        "length",
        "t_star",
        "bond_dim",
        "base_index",
        "epsilon",
        "num_prefixes",
        "num_suffixes",
        "hankel_diff_spec",
        "hankel_diff_fro",
        "hankel_diff_spec_proj",
        "hankel_diff_fro_proj",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(f"Wrote {len(records)} rows to {output}")


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: noise amplification vs. length")
    parser.add_argument("--lengths", type=parse_int_list, default="5,10,15,20")
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--bases", type=int, default=5)
    parser.add_argument("--epsilons", type=parse_float_list, default="0.001,0.003,0.01")
    parser.add_argument("--max-prefixes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/exp3_results.csv")
    args = parser.parse_args()

    lengths: List[int] = args.lengths if isinstance(args.lengths, list) else parse_int_list(str(args.lengths))
    epsilons: List[float] = args.epsilons if isinstance(args.epsilons, list) else parse_float_list(str(args.epsilons))

    run_experiment(
        lengths=lengths,
        bond_dim=args.bond_dim,
        num_bases=args.bases,
        epsilons=epsilons,
        seed=args.seed,
        max_prefixes=args.max_prefixes,
        output=args.output,
    )


if __name__ == "__main__":
    main()
