"""heat3d_data - writes HEAT03 (heat3D, 20 x 20 x 20) to examples/benchmarks/data from the model

MATLAB CORA loads the matrix A of heat3D from models/Cora/heat3D/heat<samples>.mat, a Git LFS file
that is not in the repository here, and MATLAB is not available. The matrices are the finite
differences of the heat equation of the ARCH-COMP benchmark (Bak, Tran, Johnson, "Numerical
Verification of Affine Systems with up to a Billion Dimensions", HSCC 2019): on the grid of
step d = 1 / (samples + 1), diffusivity k = 0.01, a = k / d^2 to each neighbor, -6 a plus a per
missing (insulated) neighbor on the diagonal, and a / (1 + h d) (h = 0.5) for the missing
neighbor at x = 1, where heat leaves, added last. `--check` shows that this reproduces the A of
HEAT01 (heat5) and HEAT02 (heat10), which MATLAB exported, bit for bit. HEAT03 is the script
benchmark_linear_reach_ARCH23_heat3D_HEAT03.m: R0 from its aux_getInit, the output C at indMid.

Syntax:   python examples/python/benchmarks/heat3d_data.py [--check] [folder]
See also: examples/matlab/benchmarks/export_benchmark_data.m, benchmark_linear_reach_arch23_heat3d.cpp

Authors:       Claude Opus 5.5
Written:       09-November-2026
Last update:   ---
Reviewed:      ---
"""
import json
import math
import os
import struct
import sys


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "data")


def heat_matrix(samples):
    """The entries {(row, col): value} of A for samples^3 states, x fastest, then y, then z."""
    d = 1.0 / (samples + 1)
    a = 0.01 / d**2
    leaving = a / (1 + 0.5 * d)
    entries = {}
    for z in range(samples):
        for y in range(samples):
            for x in range(samples):
                i = x + y * samples + z * samples * samples
                diag, leaves = -6 * a, False
                # a neighbor on each side, else an insulated side (or at x = 1, the outflow)
                for q, stride in ((x, 1), (y, samples), (z, samples * samples)):
                    if q > 0:
                        entries[(i, i - stride)] = a
                    else:
                        diag += a
                    if q < samples - 1:
                        entries[(i, i + stride)] = a
                    elif stride == 1:
                        leaves = True
                    else:
                        diag += a
                if leaves:
                    diag += leaving
                entries[(i, i)] = diag
    return entries


def initial_state(samples):
    """The script's aux_getInit(A, samples, 1): 1 on the box [0, 0.4] x [0, 0.2] x [0, max_z]."""
    x0 = [0.0] * samples**3
    max_z = 0.1 / (1 / samples) if samples >= 10 else 0.2 / (1 / samples)
    # MATLAB's 0:v runs to floor(v) up to a few ulps of v
    last = lambda v: int(math.floor(v + 1e-9))  # noqa: E731
    for z in range(last(max_z) + 1):
        for y in range(last(0.2 / (1 / samples)) + 1):
            for x in range(last(0.4 / (1 / samples)) + 1):
                x0[x + y * samples + z * samples * samples] = 1.0
    return x0


def coo(shape, entries):
    """A "matrices" entry and its bytes in MATLAB's order of find (column by column)."""
    keys = sorted(entries, key=lambda rc: (rc[1], rc[0]))
    blob = struct.pack(f"<{len(keys)}i", *[r for r, _ in keys])
    blob += struct.pack(f"<{len(keys)}i", *[c for _, c in keys])
    blob += struct.pack(f"<{len(keys)}d", *[entries[k] for k in keys])
    return {"shape": list(shape), "storage": "coo", "nnz": len(keys)}, blob


def read_coo(folder, instance, name):
    """The entries of a coordinate matrix of an exported instance."""
    with open(os.path.join(folder, instance + ".json")) as f:
        entry = json.load(f)["matrices"][name]
    with open(os.path.join(folder, instance + ".bin"), "rb") as f:
        blob = f.read()
    nnz, at = entry["nnz"], entry["offset"]
    rows = struct.unpack_from(f"<{nnz}i", blob, at)
    cols = struct.unpack_from(f"<{nnz}i", blob, at + 4 * nnz)
    vals = struct.unpack_from(f"<{nnz}d", blob, at + 8 * nnz)
    return {(r, c): v for r, c, v in zip(rows, cols, vals)}


def write_heat03(folder):
    """HEAT03.json and HEAT03.bin: A (8000, 8000), B = 0, C at indMid, R0, the point U = 0."""
    samples = 20
    n = samples**3
    x0 = initial_state(samples)
    ones = [i for i in range(n) if x0[i] > 0]
    matrices = {
        "A": heat_matrix(samples),
        "B": {},
        "C": {(0, 4211 - 1): 1.0},  # the script's indMid = 4211, 1-based
        "R0c": {(i, 0): x0[i] for i in ones},
        "R0G": {(i, j): 0.1 * x0[i] for j, i in enumerate(ones)},  # diag(0.1 x0), x0 > 0
        "Uc": {},
    }
    shapes = {"A": (n, n), "B": (n, 1), "C": (1, n), "R0c": (n, 1), "R0G": (n, len(ones)),
              "Uc": (1, 1)}
    entries, blob = {}, b""
    for name, M in matrices.items():
        entry, data = coo(shapes[name], M)
        entries[name] = dict(entry, offset=len(blob))
        blob += data
    entries["UG"] = {"shape": [1, 0], "storage": "dense", "offset": len(blob), "nnz": 0}
    meta = {"format": "cora-benchmark-1", "benchmark": "Heat3D", "instance": "HEAT03",
            "label": "HEAT03", "linAlg": "krylov", "tFinal": 40, "data": "HEAT03.bin",
            "matrices": entries, "specs": [], "expected": {"verified": 1}}
    with open(os.path.join(folder, "HEAT03.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(folder, "HEAT03.json"), "w") as f:
        f.write(json.dumps(meta, indent=2) + "\n")


# ============================================  MAIN  ============================================ #

def main(argv):
    folder = next((a for a in argv if not a.startswith("--")), DATA)
    if "--check" in argv:
        # the model against the matrices MATLAB exported
        for instance, samples in (("HEAT01", 5), ("HEAT02", 10)):
            same = heat_matrix(samples) == read_coo(folder, instance, "A")
            print(f"{instance}: A {'bit-identical' if same else 'DIFFERS'}")
            if not same:
                return 1
        return 0
    write_heat03(folder)
    print(f"wrote HEAT03 to {os.path.abspath(folder)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# ----------------------------------------  END OF CODE  ----------------------------------------- #
