"""benchmark_data - loads and runs an ARCH-COMP AFF instance from examples/benchmarks/data

The instance files (<instance>.json and <instance>.bin) are written by
benchmarks/matlab/exportBenchmarkData.m from the original CORA scripts; their format is described
in benchmarks/README.md. load_instance needs only numpy; run_instance verifies one instance with
cora and prints the line 'benchmark,instance,result,time' of the original scripts.

Syntax:   from benchmark_data import run_instance; run_instance("ISSF01_ISS01")
See also: benchmarks/cpp/benchmark_data.h
"""
import json
import os
import sys
import time

import numpy as np

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

DATA = os.environ.get("CORACPP_BENCHMARK_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "data")


# Loading -----------------------------------------------------------------------------------

def read_matrix(entry, binary):
    """The matrix of a "matrices" entry as a dense float64 array (rows, cols)."""
    rows, cols = entry["shape"]
    nnz, offset = entry["nnz"], entry["offset"]
    if entry["storage"] == "dense":
        values = np.frombuffer(binary, dtype="<f8", count=nnz, offset=offset)
        return values.reshape(rows, cols).copy()
    if entry["storage"] != "coo":
        raise ValueError(f"benchmark data: unknown storage {entry['storage']}")
    # coordinates: int32 rows, int32 columns, float64 values (0-based)
    i = np.frombuffer(binary, dtype="<i4", count=nnz, offset=offset)
    j = np.frombuffer(binary, dtype="<i4", count=nnz, offset=offset + 4 * nnz)
    v = np.frombuffer(binary, dtype="<f8", count=nnz, offset=offset + 8 * nnz)
    dense = np.zeros((rows, cols))
    dense[i, j] = v
    return dense


def load_instance(name, folder=DATA):
    """The instance as a dict: the json fields plus "matrices" (name -> array); specs stay lists."""
    with open(os.path.join(folder, name + ".json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta["format"] != "cora-benchmark-1":
        raise ValueError(f"benchmark data: unknown format {meta['format']}")
    with open(os.path.join(folder, meta["data"]), "rb") as f:
        binary = f.read()
    meta["matrices"] = {key: read_matrix(entry, binary) for key, entry in meta["matrices"].items()}
    return meta


# Running -----------------------------------------------------------------------------------

def zonotope(cora, c, G):
    """cora.Zonotope from a center (n, 1) and generators (n, m); none become one zero generator."""
    if G.shape[1] == 0:
        G = np.zeros((c.shape[0], 1))
    return cora.Zonotope(c[:, 0], G)


def specifications(cora, inst):
    """The specifications as cora.Specification, one per halfspace where that is equivalent."""
    specs = []
    for spec in inst["specs"]:
        A, b = np.array(spec["A"], dtype=float), spec["b"]
        if spec["type"] == "safeSet":
            specs += [cora.Specification.safe_set(a, bk) for a, bk in zip(A, b)]
        elif spec["type"] == "unsafeSet" and len(b) == 1:
            specs.append(cora.Specification.unsafe_set(A[0], b[0]))
        elif spec["type"] == "unsafeSet":
            specs.append(cora.Specification.unsafe_set(A, np.array(b)))  # the polytope {A y <= b}
        else:
            raise ValueError(f"benchmark data: unknown specification type {spec['type']}")
    return specs


def run_instance(name):
    """Verifies the instance and prints 'benchmark,instance,result,time'; details go to stderr."""
    import cora

    inst = load_instance(name)
    mat = inst["matrices"]
    algs = {"reachavoid:supportFunc": cora.VerifyAlg.SupportFunc,
            "reachavoid:zonotope": cora.VerifyAlg.Zonotope}
    if inst["verifyAlg"] not in algs:
        raise ValueError(f"benchmark data: unknown verifyAlg {inst['verifyAlg']}")
    system = cora.LinearSys(mat["A"], mat["B"], mat["C"])
    params = cora.VerifyParams(zonotope(cora, mat["R0c"], mat["R0G"]),
                               zonotope(cora, mat["Uc"], mat["UG"]), inst["tFinal"])
    specs = specifications(cora, inst)
    # what will run, before it does
    print(f"{name}: {mat['A'].shape[0]} states, {len(specs)} specifications, "
          f"{inst['verifyAlg']}, t_final {inst['t_final']}", file=sys.stderr)

    start = time.perf_counter()
    res = system.verify(params, algs[inst["verifyAlg"]], specs)
    wall = time.perf_counter() - start

    print(f"{inst['benchmark']},{inst['label']},{int(res.verified)},"
          f"{res.computation_time if res.computation_time > 0 else wall:.6g}", flush=True)
    # the iterations, steps and step size next to those of MATLAB
    detail = f"  iterations {res.iterations}, num_steps {res.num_steps}, time_step {res.time_step:.6g}"
    expected = inst["expected"]
    if expected and "nrSteps" in expected:
        detail += (f"  (MATLAB: iterations {expected['iterations']}, "
                   f"num_steps {expected['num_steps']}, time_step {expected['time_step']:.6g})")
    print(detail, file=sys.stderr)


def main(family):
    """The instances named on the command line, else all of the family."""
    for name in sys.argv[1:] or family:
        run_instance(name)


# ----------------------------------------  END OF CODE  ----------------------------------------- #
