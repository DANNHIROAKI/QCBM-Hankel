from __future__ import annotations

"""
Experiment 2 (strengthened): truncation error vs. Hankel effective rank.

- Base states: random brickwork Haar two-qubit circuits on L qubits.
- Compression: TT-SVD with exact dense SVD (via NumPy) and bond cap D_eff.
- Hankel: fixed mid-cut (t_star = floor(L/2)) so each length-L string
  appears exactly once in H_p.
- Metrics per (length, base_index, D_eff):
    * Tail energy: per-bond Frobenius tail eps_t^2 from SVD;
      E_tail = sqrt(sum_t eps_t^2).
    * State error: delta_psi = ||psi - psi_tilde||_2.
    * Probability error: ||p - p_tilde||_2.
    * Hankel Frobenius/spec differences: ||H - H_tilde||_F and ||H - H_tilde||_2.
    * Effective ranks at theory-driven tolerances:
          eps = Delta_th = 2 * E_tail
          eps = ||H - H_tilde||_2
      using an SVD-based definition: rank_eps(H) = # {sigma_i >= eps}.
"""

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def normalize_state(state: np.ndarray) -> np.ndarray:
    """L2-normalise a complex state vector."""
    nrm = np.linalg.norm(state.ravel())
    if nrm == 0.0:
        return state
    return state / nrm


# ---------------------------------------------------------------------------
# Random 2-qubit gates and brickwork circuits
# ---------------------------------------------------------------------------


def random_unitary_4(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random 4x4 unitary from the Haar measure via QR decomposition.
    """
    # Complex Ginibre matrix
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(z)
    # Fix phases on the diagonal of R
    diag = np.diag(r)
    phases = diag / np.abs(diag)
    u = q * phases
    return u.astype(np.complex128)


def apply_two_qubit_gate(
    state: np.ndarray, gate: np.ndarray, i: int, L: int
) -> None:
    """
    Apply a 4x4 two-qubit gate on qubits (i, i+1) to the full state vector.

    Qubits are numbered [0, ..., L-1] from least significant upward.
    The state is a length-2^L vector in the computational basis.
    """
    dim = 1 << L
    step = 1 << i
    block = step << 2  # 4 * step

    for base in range(0, dim, block):
        for offset in range(step):
            idx0 = base + offset
            idx1 = idx0 + step
            idx2 = idx0 + 2 * step
            idx3 = idx0 + 3 * step

            vec = np.array(
                [state[idx0], state[idx1], state[idx2], state[idx3]],
                dtype=np.complex128,
            )
            new_vec = gate @ vec
            state[idx0], state[idx1], state[idx2], state[idx3] = new_vec


def random_brickwork_state(
    length: int, depth: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Prepare a random brickwork Haar-2qubit circuit state on L qubits.

    Initial state: |0...0>.
    Depth: number of brickwork layers (alternating even/odd pairs).
    """
    dim = 1 << length
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0.0j

    for layer in range(depth):
        offset = layer % 2
        for i in range(offset, length - 1, 2):
            gate = random_unitary_4(rng)
            apply_two_qubit_gate(state, gate, i, length)

    return normalize_state(state)


# ---------------------------------------------------------------------------
# TT-SVD (MPS) with exact dense SVD
# ---------------------------------------------------------------------------


@dataclass
class TTSVDResult:
    cores: List[np.ndarray]          # list of cores, each shape (r_left, d, r_right)
    tail_energies: List[float]       # eps_t^2 at each bond
    singulars: List[np.ndarray]      # kept singular values at each bond


def tt_svd(
    state_flat: np.ndarray, d: int, length: int, bond_max: int
) -> TTSVDResult:
    """
    Tensor-train (MPS) factorisation of a length-L, d^L-dimensional state.

    Standard TT-SVD sweep:
        - reshape into (r_left * d) x (rest) at each step,
        - exact SVD with NumPy,
        - truncate to rank ≤ bond_max,
        - reshape left singulars into a TT core and propagate S * Vh.

    At step t, tail_energies[t] stores the Frobenius tail eps_t^2
    (sum of squared discarded singular values). Then

        E_tail = sqrt(sum_t eps_t^2)

    is the canonical TT-SVD bound on ||psi - psi_trunc||_2.
    """
    state = state_flat.reshape(-1).astype(np.complex128)
    cores: List[np.ndarray] = []
    tail_energies: List[float] = []
    singulars: List[np.ndarray] = []

    remaining = state
    r_left = 1

    for _ in range(length - 1):
        rows = r_left * d
        matrix = remaining.reshape(rows, -1)

        # Full SVD (thin)
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        # Truncate rank
        keep = min(bond_max, S.size)
        U_k = U[:, :keep]
        S_k = S[:keep]
        Vh_k = Vh[:keep, :]

        # Tail energy eps_t^2 = sum_{i>keep} S_i^2
        tail = float(np.sum(S[keep:] ** 2))
        tail_energies.append(tail)
        singulars.append(S_k.copy())

        # Core: reshape U_k into (r_left, d, keep)
        core = U_k.reshape(r_left, d, keep)
        cores.append(core)

        # Propagate S_k Vh_k
        remaining = (S_k[:, None] * Vh_k).reshape(-1)
        r_left = keep

    # Final core absorbs the last physical site
    final_core = remaining.reshape(r_left, d, 1)
    cores.append(final_core)
    tail_energies.append(0.0)
    singulars.append(np.array([], dtype=float))

    return TTSVDResult(cores=cores, tail_energies=tail_energies, singulars=singulars)


# ---------------------------------------------------------------------------
# Contract TT cores to recover amplitudes
# ---------------------------------------------------------------------------


def generate_sequences(length: int, d: int) -> List[List[int]]:
    """
    Enumerate all length-L strings over alphabet {0, ..., d-1}.

    Strings are ordered lexicographically in base-d.
    """
    if length == 0:
        return [[]]
    out = [[0] * length for _ in range(d**length)]
    for idx in range(d**length):
        val = idx
        for pos in range(length - 1, -1, -1):
            out[idx][pos] = val % d
            val //= d
    return out


def amplitude_from_cores(seq: Sequence[int], cores: Sequence[np.ndarray]) -> complex:
    """
    Contract TT cores along a single sequence to get psi(seq).
    cores[s] has shape (r_left, d, r_right).
    """
    vec = np.array([1.0 + 0.0j], dtype=np.complex128)  # shape (1,)
    for site, sym in enumerate(seq):
        core = cores[site]  # (r_left, d, r_right)
        # slice for the chosen physical symbol: (r_left, r_right)
        slice_sym = core[:, sym, :]
        vec = vec @ slice_sym  # (1, r_left) @ (r_left, r_right) -> (r_right,)
    return complex(vec[0]) if vec.size > 0 else 0.0 + 0.0j


def amplitudes_from_cores(
    cores: Sequence[np.ndarray], length: int, d: int
) -> np.ndarray:
    """
    Compute psi(x) for all x in Σ^L from TT cores.

    Returns a complex vector of length d^L, matching the sequence ordering
    in generate_sequences.
    """
    seqs = generate_sequences(length, d)
    amps = np.empty(len(seqs), dtype=np.complex128)
    for idx, seq in enumerate(seqs):
        amps[idx] = amplitude_from_cores(seq, cores)
    return amps


# ---------------------------------------------------------------------------
# Hankel construction and SVD-based effective rank
# ---------------------------------------------------------------------------


def hankel_from_probs(
    probs: np.ndarray, length: int, t_star: int
) -> np.ndarray:
    """
    Build mid-cut Hankel H(u,v)=p(uv) with P=Σ^{t_star}, S=Σ^{L-t_star}.

    We index strings by their integer code in binary (qubit 0 is least
    significant bit), and split each integer x into a prefix u and suffix v:
        x = u * 2^{L-t_star} + v.

    This matches the indexing of random_brickwork_state. Each length-L string
    appears exactly once in H.
    """
    rows = 1 << t_star
    cols = 1 << (length - t_star)
    H = np.empty((rows, cols), dtype=float)
    for r in range(rows):
        base = r << (length - t_star)
        H[r, :] = probs[base : base + cols]
    return H


def frob_norm(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat, ord="fro"))


def spectral_norm(mat: np.ndarray) -> float:
    """
    Spectral norm ||mat||_2 = largest singular value.
    """
    if mat.size == 0:
        return 0.0
    svals = np.linalg.svd(mat, compute_uv=False)
    return float(svals[0]) if svals.size > 0 else 0.0


def singular_values_real(mat: np.ndarray) -> np.ndarray:
    """
    Singular values of a real matrix, sorted descending.
    """
    if mat.size == 0:
        return np.array([], dtype=float)
    svals = np.linalg.svd(mat, compute_uv=False)
    return svals.astype(float)


def effective_rank_from_singulars(svals: np.ndarray, eps: float) -> int:
    """
    Effective rank at tolerance eps: number of singular values ≥ eps.
    """
    if eps <= 0.0:
        return int(svals.size)
    return int(np.sum(svals >= eps))


# ---------------------------------------------------------------------------
# Main experiment driver
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
    # Seed both Python's RNG and NumPy's RNG for reproducibility
    random.seed(seed)
    rng = np.random.default_rng(seed)

    d = 2
    records: List[Dict[str, object]] = []

    for L in lengths:
        depth = max(1, int(depth_scale * L))
        t_star = L // 2
        print(f"[length={L}] brickwork depth = {depth}, t_star = {t_star}")

        for base_idx in range(num_bases):
            # 1) Base state and probabilities
            base_state = random_brickwork_state(L, depth, rng)
            base_probs = np.abs(base_state) ** 2

            # 2) Base Hankel and its singular values
            base_hankel = hankel_from_probs(base_probs, length=L, t_star=t_star)
            base_svals = singular_values_real(base_hankel)
            base_rank_full = effective_rank_from_singulars(base_svals, eps=1e-12)

            # 3) Sweep effective bond dimensions
            D_eff_values = sorted(
                {bond_max, max(1, bond_max // 2), max(1, bond_max // 4)}
            )

            for D_eff in D_eff_values:
                # TT-SVD compression from full base state
                tt_res = tt_svd(base_state, d=d, length=L, bond_max=D_eff)

                # Tail energies and theoretical Hankel bound
                tail_energy_sum = float(np.sum(tt_res.tail_energies))
                tail_energy_norm = math.sqrt(tail_energy_sum)
                delta_th = 2.0 * tail_energy_norm

                # Reconstruct truncated state and renormalise
                trunc_state = amplitudes_from_cores(tt_res.cores, length=L, d=d)
                trunc_state = normalize_state(trunc_state)
                delta_psi = float(np.linalg.norm(base_state - trunc_state))

                # Probability vector and L2 error
                trunc_probs = np.abs(trunc_state) ** 2
                prob_diff_l2 = float(np.linalg.norm(base_probs - trunc_probs))

                # Hankel matrices and differences
                trunc_hankel = hankel_from_probs(trunc_probs, length=L, t_star=t_star)
                diff_hankel = base_hankel - trunc_hankel
                diff_fro = frob_norm(diff_hankel)
                diff_spec = spectral_norm(diff_hankel)

                # Effective ranks (base Hankel) at theory-driven epsilons
                rank_delta_th = effective_rank_from_singulars(
                    base_svals, delta_th
                )
                rank_delta_spec = effective_rank_from_singulars(
                    base_svals, diff_spec
                )
                success_delta_th = int(rank_delta_th <= D_eff ** 2)
                success_delta_spec = int(rank_delta_spec <= D_eff ** 2)

                # (Optional sanity: numeric rank of truncated Hankel at a tiny eps)
                trunc_svals = singular_values_real(trunc_hankel)
                trunc_rank_num = effective_rank_from_singulars(
                    trunc_svals, eps=1e-10
                )

                row: Dict[str, object] = {
                    "length": L,
                    "base_index": base_idx,
                    "bond_max": bond_max,
                    "bond_eff": D_eff,
                    "depth": depth,
                    "t_star": t_star,
                    # truncation / state-level metrics
                    "tail_energy_sum": tail_energy_sum,
                    "tail_energy_norm": tail_energy_norm,
                    "delta_th": delta_th,
                    "delta_psi": delta_psi,
                    "delta_psi_over_tail": (
                        delta_psi / tail_energy_norm if tail_energy_norm > 0 else 0.0
                    ),
                    # probability / Hankel metrics
                    "prob_diff_l2": prob_diff_l2,
                    "hankel_diff_fro": diff_fro,
                    "hankel_diff_spec": diff_spec,
                    # base Hankel ranks
                    "base_rank_full": base_rank_full,
                    "rank_delta_th": rank_delta_th,
                    "rank_delta_spec": rank_delta_spec,
                    "rank_success_delta_th": success_delta_th,
                    "rank_success_delta_spec": success_delta_spec,
                    # truncated Hankel numeric rank at a tiny tolerance
                    "trunc_rank_num_1e-10": trunc_rank_num,
                }

                # Ranks at user-specified epsilons (for both base and truncated Hankel)
                for eps in epsilons:
                    key_base = f"base_rank_eps_{eps:g}"
                    key_trunc = f"trunc_rank_eps_{eps:g}"
                    row[key_base] = effective_rank_from_singulars(base_svals, eps)
                    row[key_trunc] = effective_rank_from_singulars(trunc_svals, eps)

                records.append(row)

                print(
                    f"  base={base_idx:02d}, D_eff={D_eff:2d}: "
                    f"E_tail={tail_energy_norm:.3e}, "
                    f"Delta_th={delta_th:.3e}, "
                    f"||H-H_t||_2={diff_spec:.3e}, "
                    f"rank<=D^2@th={success_delta_th}"
                )

    # Write CSV output
    if not records:
        print("No records generated; nothing to save.")
        return

    fieldnames = list(records[0].keys())
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(f"Saved {len(records)} rows to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2 (strengthened): truncation error vs. Hankel effective rank "
            "(brickwork states + TT-SVD + mid-cut Hankel + SVD-based ranks)."
        )
    )
    parser.add_argument(
        "--lengths",
        type=parse_int_list,
        default="8,10,12",
        help="Comma-separated sequence lengths, e.g. '8,10,12'.",
    )
    parser.add_argument(
        "--bond-max",
        type=int,
        default=8,
        help="Maximum bond dimension (D_max) for TT-SVD compression.",
    )
    parser.add_argument(
        "--bases",
        type=int,
        default=10,
        help="Number of random brickwork base states per length.",
    )
    parser.add_argument(
        "--epsilons",
        type=parse_float_list,
        default="1e-12,1e-10,1e-8,1e-6",
        help="Comma-separated tolerances for effective-rank reporting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Brickwork depth multiplier (depth = depth_scale * L).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp2_results.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
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


if __name__ == "__main__":
    main()




"""
python experiments/exp2_truncation.py --lengths 8,10,12 --bond-max 8 --bases 5 --epsilons 1e-12,1e-10,1e-8,1e-6 --depth-scale 1.0 --seed 0 --output experiments/exp2_results.csv
"""