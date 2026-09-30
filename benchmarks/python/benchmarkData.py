"""benchmarkData - loads and runs an ARCH-COMP AFF instance from benchmarks/data

The instance files (<instance>.json and <instance>.bin) are written by
benchmarks/matlab/exportBenchmarkData.m from the original CORA scripts; their format is described
in benchmarks/README.md. loadInstance needs only numpy; runInstance verifies one instance with
cora and prints the line 'benchmark,instance,result,time' of the original scripts.

Syntax:   from benchmarkData import runInstance; runInstance("ISSF01_ISS01")
See also: benchmarks/cpp/benchmarkData.h
"""
import json
import os
import sys
import time

import numpy as np

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

DATA = os.environ.get("CORACPP_BENCHMARK_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data")


# Loading -----------------------------------------------------------------------------------

def readMatrix(entry, binary):
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


def loadInstance(name, folder=DATA):
    """The instance as a dict: the json fields plus "matrices" (name -> array); specs stay lists."""
    with open(os.path.join(folder, name + ".json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta["format"] != "cora-benchmark-1":
        raise ValueError(f"benchmark data: unknown format {meta['format']}")
    with open(os.path.join(folder, meta["data"]), "rb") as f:
        binary = f.read()
    meta["matrices"] = {key: readMatrix(entry, binary) for key, entry in meta["matrices"].items()}
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
            specs += [cora.Specification.safeSet(a, bk) for a, bk in zip(A, b)]
        elif spec["type"] == "unsafeSet" and len(b) == 1:
            specs.append(cora.Specification.unsafeSet(A[0], b[0]))
        else:
            raise ValueError("benchmark data: an unsafe set that is an intersection of halfspaces "
                             "is not supported by Specification")
    return specs


def runInstance(name):
    """Verifies the instance and prints 'benchmark,instance,result,time'; details go to stderr."""
    import cora

    inst = loadInstance(name)
    mat = inst["matrices"]
    algs = {"reachavoid:supportFunc": cora.VerifyAlg.SupportFunc,
            "reachavoid:zonotope": cora.VerifyAlg.Zonotope}
    if inst["verifyAlg"] not in algs:
        raise ValueError(f"benchmark data: unknown verifyAlg {inst['verifyAlg']}")
    system = cora.LinearSys(mat["A"], mat["B"], mat["C"])
    params = cora.VerifyParams(zonotope(cora, mat["R0c"], mat["R0G"]),
                               zonotope(cora, mat["Uc"], mat["UG"]), inst["tFinal"])
    specs = specifications(cora, inst)

    start = time.perf_counter()
    res = system.verify(params, algs[inst["verifyAlg"]], specs)
    wall = time.perf_counter() - start

    print(f"{inst['benchmark']},{inst['label']},{int(res.verified)},"
          f"{res.tComp if res.tComp > 0 else wall:.6g}", flush=True)
    # the iterations, steps and step size next to those of MATLAB
    detail = f"  iterations {res.iterations}, nrSteps {res.nrSteps}, timeStep {res.timeStep:.6g}"
    expected = inst["expected"]
    if expected and "nrSteps" in expected:
        detail += (f"  (MATLAB: iterations {expected['iterations']}, "
                   f"nrSteps {expected['nrSteps']}, timeStep {expected['timeStep']:.6g})")
    print(detail, file=sys.stderr)


def main(family):
    """The instances named on the command line, else all of the family."""
    for name in sys.argv[1:] or family:
        runInstance(name)


# ----------------------------------------  END OF CODE  ----------------------------------------- #
