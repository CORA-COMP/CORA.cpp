"""rnd_data - writes the random benchmarks RND01-06 of ARCH-COMP26 AFF to examples/benchmarks/data

The report's instances are "created on the server"; the ARCH-COMP repository (gitlab.com/goranf/
ARCH-COMP) holds the five files that CORA's repeatability package of 2026 loads, in
2026/AFF/cora/code/cora/rand01.json .. rand05.json (also in 2026/AFF/cora/secret-data/): the
systems x' = A x + B u, y = C x of 10, 50, 100, 1000 and 2500 states, intervals X0 and U, one unsafe
interval, the horizon tend. Their file names follow the report's RND01-05, the report gives 500
states for RND04 and these 1000. RND06 (5000 states) is not in the repository: this module generates
a system like rand05 (sparse, diagonally dominant stable A) and places its unsafe halfspace at the
support function maximum of the exact solution plus a tenth of its spread.

The conversion is that of CORA's json2cora_linearSys followed by zonotope(interval): center
(lo + hi) / 2 and generators diag((hi - lo) / 2); an unsafe interval is the polytope [I; -I] y <=
[hi; -lo] without the infinite bounds, as polytope(interval).

Syntax:   python3 examples/python/benchmarks/rnd_data.py [--folder dir] convert <rand0N.json> RND0N
          python3 examples/python/benchmarks/rnd_data.py [--folder dir] generate RND06
See also: examples/benchmarks/README.md, benchmark_linear_verify_arch26_rnd.cpp

Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
import json
import math
import os
import sys

import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "data")
# the algorithm of the script of each instance; RND02 and RND04-06 differ from CORA's (README.md)
VERIFY_ALG = {"RND01": "reachavoid:zonotope", "RND02": "reachavoid:krylovSuppFunc",
              "RND03": "reachavoid:zonotope", "RND04": "reachavoid:krylovSuppFunc",
              "RND05": "reachavoid:krylovSuppFunc", "RND06": "reachavoid:krylovSuppFunc"}


def bounds(value, n):
    """A bound of the json (a number, a list or null) as a vector of n floats; null is infinity."""
    if value is None:
        return np.full(n, math.inf)
    return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()


def interval_zonotope(spec, n):
    """Center (n, 1) and generators diag(radius) (n, n) of the interval {"lowerbound", "upperbound"}."""
    lo, hi = bounds(spec["lowerbound"], n), bounds(spec["upperbound"], n)
    return ((lo + hi) / 2).reshape(n, 1), np.diag((hi - lo) / 2)


def unsafe_polytope(spec, r):
    """The rows and offsets of polytope(interval): [I; -I] y <= [hi; -lo], infinite bounds dropped."""
    lo, hi = bounds(spec["lowerbound"], r), bounds(spec["upperbound"], r)
    rows, offsets = [], []
    for i in range(r):
        if math.isfinite(hi[i]):
            rows.append([1.0 if j == i else 0.0 for j in range(r)])
            offsets.append(hi[i])
    for i in range(r):
        if math.isfinite(lo[i]):
            rows.append([-1.0 if j == i else -0.0 for j in range(r)])
            offsets.append(-lo[i])
    return rows, offsets


def matrix_entry(M, offset):
    """The "matrices" entry of M and its bytes: coordinates (column by column, as MATLAB's find) if
    sparse enough, else float64 values row by row."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    rows, cols = M.shape
    j, i = np.nonzero(M.T)
    if len(i) * 16 < rows * cols * 8:
        blob = (i.astype("<i4").tobytes() + j.astype("<i4").tobytes()
                + M[i, j].astype("<f8").tobytes())
        return {"shape": [rows, cols], "storage": "coo", "offset": offset, "nnz": len(i)}, blob
    return ({"shape": [rows, cols], "storage": "dense", "offset": offset, "nnz": rows * cols},
            M.astype("<f8").tobytes())


def write_instance(folder, name, system, source, note):
    """<name>.json and <name>.bin from a system dict (A, B, C, R0c, R0G, Uc, UG, specs, t_final)."""
    entries, blob = {}, b""
    for key in ("A", "B", "C", "R0c", "R0G", "Uc", "UG"):
        entries[key], data = matrix_entry(system[key], len(blob))
        blob += data
    meta = {"format": "cora-benchmark-1", "benchmark": "Random", "instance": name, "label": name,
            "verifyAlg": VERIFY_ALG[name], "tFinal": system["t_final"], "data": name + ".bin",
            "source": source, "note": note, "matrices": entries,
            "specs": [{"type": "unsafeSet", "A": system["spec_A"], "b": system["spec_b"]}],
            "expected": {"verified": 1}}
    with open(os.path.join(folder, name + ".bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(folder, name + ".json"), "w", newline="\n") as f:
        f.write(json.dumps(meta, indent=2) + "\n")


# Conversion --------------------------------------------------------------------------------------

def convert(path, name, folder):
    """RND0N from a rand0N.json of the ARCH-COMP repository."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    A = np.array(raw["A"], dtype=float)
    n = A.shape[0]
    B = np.array(raw["B"], dtype=float)
    C = np.atleast_2d(np.array(raw["C"], dtype=float))  # a flat list is one output row
    R0c, R0G = interval_zonotope(raw["X0"], n)
    Uc, UG = interval_zonotope(raw["U"], B.shape[1])
    spec_A, spec_b = unsafe_polytope(raw["unsafeSet"], C.shape[0])
    system = {"A": A, "B": B, "C": C, "R0c": R0c, "R0G": R0G, "Uc": Uc, "UG": UG,
              "spec_A": spec_A, "spec_b": spec_b, "t_final": float(raw["tend"])}
    source = "gitlab.com/goranf/ARCH-COMP 2026/AFF/cora/code/cora/" + os.path.basename(path)
    write_instance(folder, name, system, source,
                   f"{n} states; the file of CORA's ARCH26 package, not confirmed to be the "
                   "server's instance")


# Generation --------------------------------------------------------------------------------------

def generate(name, folder, n=5000, seed=26112026):
    """RND06: a sparse stable system like rand05, X0 and U intervals, one unsafe halfspace."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply

    rng = np.random.default_rng(seed)
    # about 3.4 off-diagonal entries per row, heavy rows; the diagonal dominates (stable)
    nnz = int(3.4 * n)
    rows, cols = rng.integers(0, n, nnz), rng.integers(0, n, nnz)
    keep = rows != cols
    N = sp.coo_matrix((rng.normal(0, 2.5, keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    N = sp.csr_matrix(N)
    dominance = np.asarray(abs(N).sum(axis=1)).ravel() + rng.uniform(0.5, 1.5, n)
    A = (N - sp.diags(dominance)).tocsr()
    B = rng.normal(0, 1, (n, 5))
    C = rng.normal(0, 1, (1, n))
    center = rng.normal(0, 15, n)
    radius = rng.uniform(0.05, 0.2, n)
    u_lo = rng.uniform(-2, 1, 5)
    u_hi = u_lo + rng.uniform(0.5, 2.5, 5)
    t_final = 5.0

    # the support function of C x(t) along +1 on a fine grid (the exact solution, not an enclosure)
    steps = 2000
    times = np.linspace(0, t_final, steps + 1)
    Z = expm_multiply(A.T.tocsc(), C.T, start=0, stop=t_final, num=steps + 1, endpoint=True)
    Z = np.asarray(Z).reshape(steps + 1, n)
    hom = Z @ center + np.abs(Z) @ radius
    Zb = Z @ B
    u_center, u_rad = (u_lo + u_hi) / 2, (u_hi - u_lo) / 2
    integrand = Zb @ u_center + np.abs(Zb) @ u_rad
    integral = np.concatenate([[0], np.cumsum((integrand[1:] + integrand[:-1]) / 2 * np.diff(times))])
    h = hom + integral
    limit = float(h.max() + 0.1 * (h.max() - h[0]))
    system = {"A": A.toarray(), "B": B, "C": C, "R0c": center.reshape(n, 1), "R0G": np.diag(radius),
              "Uc": u_center.reshape(5, 1), "UG": np.diag(u_rad), "spec_A": [[-1.0]],
              "spec_b": [-limit], "t_final": t_final}
    write_instance(folder, name, system, "generated by examples/python/benchmarks/rnd_data.py",
                   f"{n} states; own instance like rand05 (sparse, diagonally dominant), seed "
                   f"{seed}; unsafe: y >= {limit:.6g}, the maximum of the support function "
                   f"{h.max():.6g} (on a 2000-point grid) plus a tenth of its rise")


# ============================================  MAIN  ============================================ #

def main(argv):
    folder = DATA
    if argv[:1] == ["--folder"]:
        folder, argv = argv[1], argv[2:]
    if argv[:1] == ["convert"] and len(argv) == 3:
        convert(argv[1], argv[2], folder)
    elif argv[:1] == ["generate"] and len(argv) == 2 and argv[1] == "RND06":
        generate(argv[1], folder)
    else:
        print(__doc__.split("Syntax:")[1].split("See also")[0], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# ----------------------------------------  END OF CODE  ----------------------------------------- #
