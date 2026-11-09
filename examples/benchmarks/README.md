# Benchmarks

ARCH-COMP AFF instances of MATLAB CORA (`examples/ARCHcompetition/linear/benchmark_linear_verify*_ARCH23_*.m`)
in C++ and Python, on the same numbers. Each program prints the line of the MATLAB script,
`benchmark,instance,result,time` (e.g. `Spacestation,ISSF01-ISS01,1,0.42`); iterations, nrSteps and
timeStep go to stderr, next to those of MATLAB. `time` is `tComp` of `verify` (wall clock if it is 0),
as `savedata.tComp` in MATLAB. The MATLAB side is the original scripts, run from the CORA repository.

| family | instances | algorithm |
| --- | --- | --- |
| `iss` | ISSC01_ISS02, ISSC01_ISU02, ISSF01_ISS01, ISSF01_ISU01 | reachavoid:supportFunc |
| `beam` | CBC01-03, CBF01-03 | reachavoid:supportFunc |
| `heat3D` | HEAT01, HEAT02 (HEAT03 needs Krylov) | reachavoid:supportFunc |
| `rand` | RAND01, RAND02 (`rand0?_test.json` models) | reachavoid:zonotope |
| `rendezvous` | SRA01-08, SRNA01, SRU01, SRU02 | hybrid reach: zonoGirard or nondetGuard, enclose box / pca |
| `gearbox` | GRBX01, GRBX02 | hybrid reach: zonoGirard, enclose box + pca |
| `powertrain` | DTN01-06 | hybrid reach: zonoGirard, enclose pca |
| `brake` | BRKDC01 (linear reach and resets), BRKNC01 (BRKNP01 needs LinParamSys) | hybrid reach: conZonotope, enclose box + pca |

The hybrid families have no data files: their models are small and are C++ builder functions in
`examples/cpp/benchmarks/benchmark_hybrid_reach_arch23_*.cpp`, the numbers copied from the scripts
and the model files of `models/Cora/` (the powertrain's Jacobians written out from its equations).

## Results

The verdict is CORA's `res` (1 verified, 0 not); every one equals MATLAB CORA's, the expected
verdict of the script (the ARCH-COMP 2023 AFF report: the ISU instances, RAND02, SRU01 and
SRU02 are not verified). CORA.cpp times are wall clock on this machine (16 shared cores, core
preset, 09-November-2026). The ARCH26 columns are the times of CORA (MATLAB) and JuliaReach
(continuous time) in the ARCH-COMP26 AFF report (Althoff, Carraro, Forets, Perschl, Schilling,
"ARCH-COMP26 Category Report: Continuous and Hybrid Systems with Linear Continuous Dynamics", EPiC
Series in Computing 110, pp. 28-51), measured on other machines: compare orders of magnitude only.
ARCH26 has no ISS, uses the new RND01-06 instead of RAND01/02 and adds HEAT04 (Krylov); the
instances here stay those of the ARCH23 scripts. `-`: not in the report, or not ported yet.

| instance | expected | CORA.cpp | CORA.cpp [s] | CORA ARCH26 [s] | JuliaReach ARCH26 [s] |
| --- | --- | --- | --- | --- | --- |
| HEAT01 | 1 | 1 | 0.06 | 0.13 | 0.04 |
| HEAT02 | 1 | 1 | 3.86 | 3.04 | 2.60 |
| HEAT03 | 1 | - (Krylov) | - | 21.8 | - |
| CBC01 | 1 | 1 | 0.21 | 0.28 | 0.05 |
| CBC02 | 1 | 1 | 7.73 | 3.68 | 0.80 |
| CBC03 | 1 | 1 | 67.9 | 37.27 | 3.74 |
| CBF01 | 1 | 1 | 0.15 | 0.40 | 1.99 |
| CBF02 | 1 | 1 | 7.42 | 5.38 | 70.0 |
| CBF03 | 1 | 1 | 89.2 | 67.79 | 869 |
| ISSC01_ISS02 | 1 | 1 | 0.15 | - | - |
| ISSC01_ISU02 | 0 | 0 | 0.52 | - | - |
| ISSF01_ISS01 | 1 | 1 | 0.58 | - | - |
| ISSF01_ISU01 | 0 | 0 | 0.17 | - | - |
| RAND01 | 1 | 1 | 0.002 | - | - |
| RAND02 | 0 | 0 | 0.015 | - | - |
| SRNA01 | 1 | 1 | 0.18 | 51.87 | 1.09 |
| SRA01 | 1 | 1 | 0.07 | 10.94 | 1.17 |
| SRA02 | 1 | 1 | 0.08 | 17.49 | 1.24 |
| SRA03 | 1 | 1 | 0.15 | 27.40 | 1.79 |
| SRA04 | 1 | 1 | 0.30 | 39.05 | 108 |
| SRA05 | 1 | 1 | 0.37 | 67.33 | - |
| SRA06 | 1 | 1 | 0.37 | 71.46 | - |
| SRA07 | 1 | 1 | 5.32 | 120 | - |
| SRA08 | 1 | 1 | 14.0 | 239 | - |
| SRU01 | 0 | 0 | 0.15 | 25.86 | 11.79 |
| SRU02 | 0 | 0 | 15.7 | 244 | 20.00 |
| DTN01 | 1 | 1 | 0.29 | 9.69 | 0.005 |
| DTN02 | 1 | 1 | 0.22 | 10.43 | 0.005 |
| DTN03 | 1 | 1 | 0.23 | 10.27 | 0.006 |
| DTN04 | 1 | 1 | 10.3 | 48.29 | 0.127 |
| DTN05 | 1 | 1 | 18.3 | 90.76 | 0.007 |
| DTN06 | 1 | 1 | 46.5 | 227 | 0.009 |
| PLAA01_BND42 | 1 | - (LinParamSys) | - | 25.02 | 180 |
| PLAA01_BND50 | 1 | - (LinParamSys) | - | 7.77 | 22.01 |
| PLAD01_BND30 | 1 | - (LinParamSys) | - | 4.03 | 62.82 |
| PLAD01_BND42 | 1 | - (LinParamSys) | - | 3.07 | 0.05 |
| PLAN01 | 1 | - (LinParamSys) | - | 162 | - |
| GRBX01 | 1 | 1 | 0.28 | 2.28 | 2.89 |
| GRBX02 | 1 | 1 | 0.29 | 2.44 | 3.15 |
| BRKDC01 | 1 | 1 | 0.04 | 22.66 | 0.10 |
| BRKNC01 | 1 | 1 | 39.6 | 32.07 | 0.17 |
| BRKNP01 | 1 | - (LinParamSys) | - | 52.20 | 16.99 |

In the test suite every ported instance is a test of the label `benchmark` (nightly), and the first
of each family also an `example` test (CONTRIBUTING.md, "Test suites").

## Run

```bash
cmake --build --preset core                                   # build/core/benchmarks/benchmark_linear_*
build/core/benchmarks/benchmark_linear_verify_fast_arch23_iss [ISSF01_ISS01 ...]
build/core/benchmarks/benchmark_hybrid_reach_arch23_rendezvous [SRA01 ...]   # CORACPP_BENCHMARK_VERBOSE=1: the branches
PYTHONPATH=build/python python examples/python/benchmarks/benchmark_linear_verifyFast_ARCH23_iss.py [instance ...]
```

The programs need `LinearSys::verify` (CMake builds them once `linearSys.h` declares it). Without
instance arguments a program runs its whole family. Compare with MATLAB by running the original
script in CORA (`benchmark_linear_verifyFast_ARCH23_iss_ISSF01_ISS01`) and diffing the lines.
`CORACPP_BENCHMARK_DATA` names another data folder.

## Data

`examples/benchmarks/data/` holds the instances, about 0.7 MB in all, written by MATLAB:

```matlab
addpath('examples/matlab/benchmarks'); exportBenchmarkData({}, 'coraRoot', '/path/to/cora')   % all; 'expected', false skips verify
```

`exportBenchmarkData.m` runs a copy of each original script up to (and, for `expected`, through)
its `verify` call and stores what it passed: nothing is retyped, so the numbers are CORA's.
Per instance:

- `<instance>.json`: `format` (`cora-benchmark-1`), `benchmark`, `instance`, `label` (as printed),
  `verifyAlg`, `tFinal`, `data` (the binary file), `matrices`, `specs`, and `expected`
  (`verified`, and `iterations`, `nrSteps`, `timeStep` where `verify` returns them).
- `<instance>.bin`: the matrices `A` (n, n), `B` (n, m), `C` (p, n), `R0c`, `R0G` (initial
  zonotope: center (n, 1), generators), `Uc`, `UG` (input zonotope, in the input space of `B`).
  `matrices` gives shape, `offset` (bytes), `nnz` and `storage` of each: `dense` is float64 row by
  row, `coo` is `nnz` int32 rows, `nnz` int32 columns, `nnz` float64 values (0-based), little endian.
- `specs`: `{"type": "safeSet"|"unsafeSet", "A": rows, "b": values}` over the output `y = C x`;
  a safe set must stay inside `A y <= b`, an unsafe set must avoid its intersection.

A scalar `B` of the beam scripts is stored as `B*I`; an absent `params.U` (ISSC01_ISU02) as `{0}`.
Unsafe sets with several halfspaces (RAND) need a `Specification` that takes them; the programs
stop with an error until then. Test: `python -m unittest tests/python/test_benchmarkData.py`.
