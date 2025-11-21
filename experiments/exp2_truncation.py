"""Expanded Experiment 2: truncation error vs. Hankel effective rank.

Pure-Python implementation (no external dependencies) that follows the
extended specification: generate base states from random brickwork two-qubit
circuits, perform TT-SVD with controllable bond caps, build fixed-cut Hankel
matrices, and measure tail energies, state/Hankel errors, and effective ranks
at theory-driven tolerances.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import List, Sequence

SequenceType = List[int]


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x]


# ---------------------------------------------------------------------------
# Basic linear algebra helpers (complex lists)
# ---------------------------------------------------------------------------


def vec_add(u: List[complex], v: List[complex]) -> List[complex]:
    return [a + b for a, b in zip(u, v)]


def vec_sub(u: List[complex], v: List[complex]) -> List[complex]:
    return [a - b for a, b in zip(u, v)]


def vec_scale(u: List[complex], s: complex) -> List[complex]:
    return [s * x for x in u]


def dot(u: List[complex], v: List[complex]) -> complex:
    return sum(a.conjugate() * b for a, b in zip(u, v))


def norm(u: List[complex]) -> float:
    return math.sqrt(sum(abs(x) ** 2 for x in u))


def normalize(u: List[complex]) -> List[complex]:
    n = norm(u)
    if n == 0:
        return u[:]
    return [x / n for x in u]


def matvec(mat: List[List[complex]], vec: List[complex]) -> List[complex]:
    out: List[complex] = []
    for row in mat:
        out.append(sum(a * b for a, b in zip(row, vec)))
    return out


def conj_transpose(mat: List[List[complex]]) -> List[List[complex]]:
    rows, cols = len(mat), len(mat[0])
    return [[mat[i][j].conjugate() for i in range(rows)] for j in range(cols)]


def outer(u: List[complex], v: List[complex]) -> List[List[complex]]:
    return [[a * b.conjugate() for b in v] for a in u]


def random_unitary_4() -> List[List[complex]]:
    cols = []
    for _ in range(4):
        col = [random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(4)]
        for prev in cols:
            proj = dot(prev, col)
            col = vec_sub(col, vec_scale(prev, proj))
        col = normalize(col)
        cols.append(col)
    return [[cols[j][i] for j in range(4)] for i in range(4)]


def power_iteration_spec_norm(mat: List[List[float]], iters: int = 25) -> float:
    if not mat or not mat[0]:
        return 0.0
    n = len(mat[0])
    v = [random.random() for _ in range(n)]
    v = normalize(v)
    for _ in range(iters):
        Av = matvec(mat, v)
        v = normalize(Av)
    Av = matvec(mat, v)
    return norm(Av)


# ---------------------------------------------------------------------------
# TT-SVD and tensor utilities (pure Python, truncated power-iteration SVD)
# ---------------------------------------------------------------------------


def reshape_flat_to_matrix(flat: List[complex], rows: int) -> List[List[complex]]:
    cols = len(flat) // rows
    return [flat[i * cols : (i + 1) * cols] for i in range(rows)]


def flatten_matrix(mat: List[List[complex]]) -> List[complex]:
    return [elem for row in mat for elem in row]


def truncated_svd(matrix: List[List[complex]], max_rank: int):
    m = len(matrix)
    n = len(matrix[0]) if matrix else 0
    working = [row[:] for row in matrix]
    frob = sum(abs(x) ** 2 for row in working for x in row)
    singulars: List[float] = []
    left_vecs: List[List[complex]] = []
    right_vecs: List[List[complex]] = []
    for _ in range(min(max_rank, m, n)):
        v = normalize([random.gauss(0, 1) + 1j * random.gauss(0, 1) for _ in range(n)])
        if norm(v) < 1e-14:
            break
        for _ in range(40):
            Av = matvec(working, v)
            normAv = norm(Av)
            if normAv < 1e-12:
                break
            u = [x / normAv for x in Av]
            Atu = matvec(conj_transpose(working), u)
            normAtu = norm(Atu)
            if normAtu < 1e-12:
                break
            v = [x / normAtu for x in Atu]
        Av = matvec(working, v)
        sigma = norm(Av)
        if sigma < 1e-12:
            break
        u = [x / sigma for x in Av]
        singulars.append(sigma)
        left_vecs.append(u)
        right_vecs.append(v)
        uv = outer(u, v)
        for i in range(m):
            for j in range(n):
                working[i][j] -= sigma * uv[i][j]
    if not singulars:
        singulars.append(0.0)
        base_u = [1.0] + [0.0 for _ in range(max(0, m - 1))]
        base_v = [1.0] + [0.0 for _ in range(max(0, n - 1))]
        left_vecs.append(base_u[:m])
        right_vecs.append(base_v[:n])
    kept_energy = sum(s * s for s in singulars)
    tail_energy = max(frob - kept_energy, 0.0)

    U_matrix = [[0.0j for _ in range(len(singulars))] for _ in range(m)]
    for col, u in enumerate(left_vecs):
        for i, val in enumerate(u):
            U_matrix[i][col] = val
    Vh = [[val.conjugate() for val in v] for v in right_vecs]
    SVh = []
    for s, row in zip(singulars, Vh):
        SVh.append([s * x for x in row])
    return singulars, U_matrix, SVh, tail_energy


@dataclass
class TTSVDResult:
    cores: List[List[List[List[complex]]]]
    tail_energies: List[float]
    singulars: List[List[float]]


def tt_svd(state_flat: List[complex], d: int, length: int, max_rank: int) -> TTSVDResult:
    cores: List[List[List[List[complex]]]] = []
    tail_energies: List[float] = []
    singulars: List[List[float]] = []
    remaining = state_flat[:]
    left_rank = 1
    for _ in range(length - 1):
        rows = left_rank * d
        matrix = reshape_flat_to_matrix(remaining, rows)
        s, U, SVh, tail = truncated_svd(matrix, max_rank)
        keep = len(s)
        tail_energies.append(tail)
        singulars.append(s)
        core = [[[0.0j for _ in range(keep)] for _ in range(d)] for _ in range(left_rank)]
        for i in range(left_rank):
            for a in range(d):
                for k in range(keep):
                    core[i][a][k] = U[i * d + a][k]
        cores.append(core)
        remaining = flatten_matrix(SVh)
        left_rank = keep
    final_core: List[List[List[complex]]] = [[[0.0j] for _ in range(d)] for _ in range(left_rank or 1)]
    if left_rank == 0:
        left_rank = 1
    for i in range(left_rank):
        for a in range(d):
            idx = i * d + a
            if idx < len(remaining):
                final_core[i][a][0] = remaining[idx]
    cores.append(final_core)
    tail_energies.append(0.0)
    singulars.append([])
    return TTSVDResult(cores, tail_energies, singulars)


def generate_sequences(length: int, d: int) -> List[List[int]]:
    if length == 0:
        return [[]]
    sequences = [[0] * length for _ in range(d**length)]
    for idx in range(d**length):
        val = idx
        for pos in range(length - 1, -1, -1):
            sequences[idx][pos] = val % d
            val //= d
    return sequences


def amplitude_from_cores(seq: Sequence[int], cores: Sequence[Sequence[Sequence[Sequence[complex]]]]) -> complex:
    vec: List[complex] = [1.0 + 0.0j]
    for site, sym in enumerate(seq):
        core = cores[site]
        next_vec: List[complex] = []
        for j in range(len(core[0][sym])):
            total = 0.0j
            for i, val in enumerate(vec):
                total += val * core[i][sym][j]
            next_vec.append(total)
        vec = next_vec
    return vec[0] if vec else 0.0j


def amplitudes_from_cores(cores: Sequence[Sequence[Sequence[Sequence[complex]]]], length: int, d: int) -> List[complex]:
    seqs = generate_sequences(length, d)
    return [amplitude_from_cores(seq, cores) for seq in seqs]


# ---------------------------------------------------------------------------
# Quantum circuit state preparation
# ---------------------------------------------------------------------------


def apply_two_qubit_gate(state: List[complex], gate: List[List[complex]], i: int, L: int) -> None:
    step = 1 << i
    block = step << 2
    size = 1 << L
    for base in range(0, size, block):
        for offset in range(step):
            idx0 = base + offset
            idx1 = idx0 + step
            idx2 = idx0 + 2 * step
            idx3 = idx0 + 3 * step
            vec = [state[idx0], state[idx1], state[idx2], state[idx3]]
            new = [0.0j, 0.0j, 0.0j, 0.0j]
            for r in range(4):
                new[r] = sum(gate[r][c] * vec[c] for c in range(4))
            state[idx0], state[idx1], state[idx2], state[idx3] = new


def random_brickwork_state(length: int, depth: int) -> List[complex]:
    state = [0.0j for _ in range(1 << length)]
    state[0] = 1.0 + 0.0j
    for layer in range(depth):
        offset = layer % 2
        for i in range(offset, length - 1, 2):
            gate = random_unitary_4()
            apply_two_qubit_gate(state, gate, i, length)
    state = normalize(state)
    return state


# ---------------------------------------------------------------------------
# Hankel construction and ranks
# ---------------------------------------------------------------------------


def hankel_from_probs(probs: List[float], length: int, t_star: int) -> List[List[float]]:
    rows = 1 << t_star
    cols = 1 << (length - t_star)
    mat: List[List[float]] = []
    for r in range(rows):
        row: List[float] = []
        base = r << (length - t_star)
        for c in range(cols):
            row.append(probs[base + c])
        mat.append(row)
    return mat


def frob_norm(mat: List[List[float]]) -> float:
    return math.sqrt(sum(val * val for row in mat for val in row))


def mat_sub(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def effective_rank(matrix: List[List[float]], eps: float) -> int:
    if not matrix:
        return 0
    basis: List[List[float]] = []
    for row in matrix:
        proj = row[:]
        for b in basis:
            dot_rb = sum(p * bb for p, bb in zip(proj, b))
            proj = [p - dot_rb * bb for p, bb in zip(proj, b)]
        n = math.sqrt(sum(p * p for p in proj))
        if n >= eps:
            basis.append([p / n for p in proj])
    return len(basis)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_experiment(
    lengths: Sequence[int],
    bond_max: int,
    num_bases: int,
    epsilons: Sequence[float],
    seed: int,
    depth_scale: float,
    output: str,
) -> None:
    random.seed(seed)
    records = []
    d = 2
    for L in lengths:
        depth = max(1, int(depth_scale * L))
        for base_idx in range(num_bases):
            base_state = random_brickwork_state(L, depth)
            base_state = normalize(base_state)
            base_tt = tt_svd(base_state, d, L, bond_max)
            base_probs = [abs(x) ** 2 for x in base_state]
            t_star = L // 2
            base_hankel = hankel_from_probs(base_probs, L, t_star)
            base_rank_full = effective_rank(base_hankel, 1e-12)

            for D_eff in [bond_max, max(1, bond_max // 2), max(1, bond_max // 4)]:
                trunc_tt = tt_svd(base_state, d, L, D_eff)
                tail_sum = sum(trunc_tt.tail_energies)
                tail_norm = math.sqrt(tail_sum)
                delta_th = 2 * tail_norm
                trunc_state = amplitudes_from_cores(trunc_tt.cores, L, d)
                trunc_state = normalize(trunc_state)
                delta_psi = norm(vec_sub(base_state, trunc_state))
                trunc_probs = [abs(x) ** 2 for x in trunc_state]
                trunc_hankel = hankel_from_probs(trunc_probs, L, t_star)
                diff_hankel = mat_sub(base_hankel, trunc_hankel)
                diff_fro = frob_norm(diff_hankel)
                diff_spec = power_iteration_spec_norm(diff_hankel)
                eff_rank_delta_th = effective_rank(base_hankel, delta_th)
                eff_rank_delta_spec = effective_rank(base_hankel, diff_spec)

                row = {
                    "length": L,
                    "base_index": base_idx,
                    "bond_max": bond_max,
                    "bond_eff": D_eff,
                    "depth": depth,
                    "t_star": t_star,
                    "tail_energy_sum": tail_sum,
                    "tail_energy_norm": tail_norm,
                    "delta_th": delta_th,
                    "delta_psi": delta_psi,
                    "hankel_diff_fro": diff_fro,
                    "hankel_diff_spec": diff_spec,
                    "base_rank": base_rank_full,
                    "rank_delta_th": eff_rank_delta_th,
                    "rank_delta_spec": eff_rank_delta_spec,
                    "rank_success_delta_th": int(eff_rank_delta_th <= D_eff**2),
                    "rank_success_delta_spec": int(eff_rank_delta_spec <= D_eff**2),
                }

                for eps in epsilons:
                    row[f"base_rank_eps_{eps:g}"] = effective_rank(base_hankel, eps)
                    row[f"trunc_rank_eps_{eps:g}"] = effective_rank(trunc_hankel, eps)

                records.append(row)
                print(
                    f"L={L}, base={base_idx}, D_eff={D_eff}: tail_norm={tail_norm:.3e}, "
                    f"Delta_th={delta_th:.3e}, ||H||_2 diff≈{diff_spec:.3e}, rank<=D^2@th={row['rank_success_delta_th']}"
                )

    fieldnames = [
        "length",
        "base_index",
        "bond_max",
        "bond_eff",
        "depth",
        "t_star",
        "tail_energy_sum",
        "tail_energy_norm",
        "delta_th",
        "delta_psi",
        "hankel_diff_fro",
        "hankel_diff_spec",
        "base_rank",
        "rank_delta_th",
        "rank_delta_spec",
        "rank_success_delta_th",
        "rank_success_delta_spec",
    ]
    for eps in epsilons:
        fieldnames.append(f"base_rank_eps_{eps:g}")
        fieldnames.append(f"trunc_rank_eps_{eps:g}")

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded Experiment 2: truncation vs Hankel effective rank")
    parser.add_argument("--lengths", type=parse_int_list, default="8,10,12", help="Comma-separated sequence lengths")
    parser.add_argument("--bond-max", type=int, default=8, help="Maximum bond dimension for base TT-SVD")
    parser.add_argument("--bases", type=int, default=10, help="Number of base random circuits per length")
    parser.add_argument(
        "--epsilons",
        type=parse_float_list,
        default="1e-12,1e-10,1e-8,1e-6",
        help="Tolerances for effective-rank reporting",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Brickwork depth multiplier (depth = depth_scale * L)",
    )
    parser.add_argument("--output", type=str, default="experiments/exp2_results.csv", help="CSV output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        lengths=args.lengths,
        bond_max=args.bond_max,
        num_bases=args.bases,
        epsilons=args.epsilons,
        seed=args.seed,
        depth_scale=args.depth_scale,
        output=args.output,
    )
