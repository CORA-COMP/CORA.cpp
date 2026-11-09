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
| `heat3D` | HEAT01, HEAT02 | reachavoid:supportFunc |
| `heat3D` (reach) | HEAT03 (heat20, 8000 states, sparse `A`) | linearSys reach, linAlg krylov |
| `rand` | RAND01, RAND02 (`rand0?_test.json` models) | reachavoid:zonotope |
| `rendezvous` | SRA01-08, SRNA01, SRU01, SRU02 | hybrid reach: zonoGirard or nondetGuard, enclose box / pca |
| `gearbox` | GRBX01, GRBX02 | hybrid reach: zonoGirard, enclose box + pca |
| `powertrain` | DTN01-06 | hybrid reach: zonoGirard, enclose pca |
| `brake` | BRKDC01 (linear reach and resets), BRKNC01, BRKNP01 (a LinParamSys flow; by name only, see below) | hybrid reach: conZonotope, enclose box + pca |
| `platoon` | PLAA01_BND42, PLAA01_BND50 (LinParamSys, constParam), PLAD01_BND30, PLAD01_BND42 (LinearSys switching every 5 s), PLAN01 (varParam, then an invariant) | linParamSys reach / linearSys reach |

The hybrid and platoon families have no data files: their models are small and are C++ builder
functions in `examples/cpp/benchmarks/benchmark_hybrid_reach_arch23_*.cpp` and
`benchmark_linear_param_reach_arch23_platoon.cpp`, the numbers copied from the scripts
and the model files of `models/Cora/` (the powertrain's Jacobians written out from its equations).

## Results

The verdict is CORA's `res` (1 verified, 0 not); every one equals MATLAB CORA's, the expected
verdict of the script (the ARCH-COMP 2023 AFF report: the ISU instances, RAND02, SRU01 and
SRU02, PLAN01 and BRKNP01 are not verified; PLAN01 and BRKNP01: below). CORA.cpp times are wall clock (core preset,
09-November-2026, one run each, the programs run one at a time). The ARCH26 columns are the times of CORA (MATLAB) and JuliaReach
(continuous time) in the ARCH-COMP26 AFF report (Althoff, Carraro, Forets, Perschl, Schilling,
"ARCH-COMP26 Category Report: Continuous and Hybrid Systems with Linear Continuous Dynamics", EPiC
Series in Computing 110, pp. 28-51), run on the same machine as CORA.cpp here, so the times compare
directly; other jobs share its 16 CPUs, which can slow a single run.
ARCH26 has no ISS, uses the new RND01-06 instead of RAND01/02 and adds HEAT04 (heat50, 125 000
states, Krylov), a target for later: not run here (its data would be about 14 MB). The instances
here stay those of the ARCH23 scripts. `-`: not in the report, or not ported yet.

| instance | expected | CORA.cpp | CORA.cpp [s] | CORA ARCH26 [s] | JuliaReach ARCH26 [s] |
| --- | --- | --- | --- | --- | --- |
| HEAT01 | 1 | 1 | 0.064 | 0.13 | 0.04 |
| HEAT02 | 1 | 1 | 3.39 | 3.04 | 2.60 |
| HEAT03 | 1 | 1 | 14.6 | 21.8 | - |
| CBC01 | 1 | 1 | 0.162 | 0.28 | 0.05 |
| CBC02 | 1 | 1 | 3.88 | 3.68 | 0.80 |
| CBC03 | 1 | 1 | 44.5 | 37.27 | 3.74 |
| CBF01 | 1 | 1 | 0.255 | 0.40 | 1.99 |
| CBF02 | 1 | 1 | 4.51 | 5.38 | 70.0 |
| CBF03 | 1 | 1 | 66.5 | 67.79 | 869 |
| ISSC01_ISS02 | 1 | 1 | 0.153 | - | - |
| ISSC01_ISU02 | 0 | 0 | 0.522 | - | - |
| ISSF01_ISS01 | 1 | 1 | 0.628 | - | - |
| ISSF01_ISU01 | 0 | 0 | 0.194 | - | - |
| RAND01 | 1 | 1 | 0.0023 | - | - |
| RAND02 | 0 | 0 | 0.017 | - | - |
| SRNA01 | 1 | 1 | 0.473 | 51.87 | 1.09 |
| SRA01 | 1 | 1 | 0.204 | 10.94 | 1.17 |
| SRA02 | 1 | 1 | 0.102 | 17.49 | 1.24 |
| SRA03 | 1 | 1 | 0.178 | 27.40 | 1.79 |
| SRA04 | 1 | 1 | 0.484 | 39.05 | 108 |
| SRA05 | 1 | 1 | 0.464 | 67.33 | - |
| SRA06 | 1 | 1 | 0.472 | 71.46 | - |
| SRA07 | 1 | 1 | 2.49 | 120 | - |
| SRA08 | 1 | 1 | 5.27 | 239 | - |
| SRU01 | 0 | 0 | 0.413 | 25.86 | 11.79 |
| SRU02 | 0 | 0 | 5.93 | 244 | 20.00 |
| DTN01\* | 1 | 1 | 0.508 | 9.69 | 0.005 |
| DTN02\* | 1 | 1 | 0.618 | 10.43 | 0.005 |
| DTN03\* | 1 | 1 | 0.914 | 10.27 | 0.006 |
| DTN04\* | 1 | 1 | 19.4 | 48.29 | 0.127 |
| DTN05\* | 1 | 1 | 35.5 | 90.76 | 0.007 |
| DTN06\* | 1 | 1 | 110 | 227 | 0.009 |
| PLAA01_BND42 | 1 | 1 | 10.5 | 25.02 | 180 |
| PLAA01_BND50 | 1 | 1 | 4.19 | 7.77 | 22.01 |
| PLAD01_BND30 | 1 | 1 | 0.306 | 4.03 | 62.82 |
| PLAD01_BND42 | 1 | 1 | 0.066 | 3.07 | 0.05 |
| PLAN01 | 0 | 0 | 47.9 | 162 (verified, other setup) | - |
| GRBX01 | 1 | 1 | 0.03 | 2.28 | 2.89 |
| GRBX02 | 1 | 1 | 0.037 | 2.44 | 3.15 |
| BRKDC01 | 1 | 1 | 0.037 | 22.66 | 0.10 |
| BRKNC01 | 1 | 1 | 6.58 | 32.07 | 0.17 |
| BRKNP01 | 0 | - (not finished in 30 min) | - | 52.20 | 16.99 |

**PLAN01** is not verified by CORA's ARCH23 script, MATLAB R2024b, CORA v2026.1.0 on this machine
(verdict 0 in 169 s), and CORA.cpp agrees: its 50 s reach with varying parameters reaches x1, x4,
x7 >= -50.65, past the bound -50. The ARCH26 report lists PLAN01 as verified (162 s) with a different
setup, so its time is not comparable. The algorithm is ported step by step (expmIndMixed, priv_tie,
priv_inputSolution, initReach, post) and the constant-parameter instances PLAA01 agree. The
script's invariant check stops after one step: its `aux_inViaProj(Z1, Z2)` tests that Z1 holds Z2
(`contains(Z1proj, Z2proj)`), the reverse of its comment, and the first set holds the start set
(`Zonotope::contains`, by polymax in each two-dimensional projection, as CORA picks below dimension
4; see docs/cora-differences.md).

**HEAT03** runs the script's dual problem: `x' = A' x` from the point `e_mid` with the outputs
`[c G]' x`, by the Krylov subspace algorithm (`alg::Krylov`, CORA's linAlg `krylov`) on the sparse
`A`; the maximum temperature over the steps is 0.0172513, in the goal 0.01716509 +- 1e-4 (its
upper end is 0.0172651). The subspace has order 181, as in CORA: there the error bound
`tau gamma 40^k / k!` is 0 because `k!` overflows from `k = 171` on, while at order 161 it is
4.9e149, far above `krylovError = eps` (docs/cora-differences.md); with `krylovError <= 2 eps`
no error box is added. CORALean shows that CORA's Krylov reach misses trajectories in general (its
time-interval enclosure and the Jawecki bound outside the nonexpansive case,
`Krylov.reach_krylov_counterexample`, `Krylov.printed_time_interval_counterexample`,
`Krylov.jawecki_bound_cora_counterexample`); CORA.cpp follows
CORA, so the same holds here. HEAT03 takes about 15 s, so it is a `benchmark` test only.

**BRKNP01** is not verified by CORA's ARCH23 script, MATLAB R2024b, CORA v2026.1.0 on this machine
(verdict 0 in 883 s, with the warning 'Taylor order not high enough'). The input `U = zonotope(1)`
becomes the constant input `uTrans = 1` with `U = {0}`, which does not hold the origin, so
`priv_inputTie` adds the curvature of the constant input; its series does not converge
(`norm(A, inf) r / (taylorTerms + 2)` is about 18 for `r = 2e-5`). The driver ports the script
(`benchmark_hybrid_reach_arch23_brake BRKNP01`: the LinParamSys flow, conZonotope guards of order 5,
the script's `tMax` loop), but CORA.cpp did not finish within 30 minutes (core preset, 09-November-2026),
so it is not a ctest test and not in the default run of the family. The ARCH26 report lists it as
verified (52.20 s), with a different setup.

In the test suite every ported instance is a test of the label `benchmark` (nightly), and the first
of each family also an `example` test (CONTRIBUTING.md, "Test suites").

\* The JuliaReach times of DTN01-06 are not comparable. Its public code
(github.com/JuliaReach/ARCH2025_AFF_RE, `models/Powertrain/powertrain_benchmark.jl`, unchanged since
2021) runs `powertrain_homog(θ=k)` for DTN0k with the full initial set, where the report defines
θ = 2/22 and the scales 5 %, 30 % and 100 % of the initial set.

## NLN results

The ARCH-COMP NLN instances of MATLAB CORA (`examples/ARCHcompetition/nonlinear/*.m`), ported as
they are in CORA (ARCH23 scripts): `benchmark_nonlinear_reach_arch23_trafficscenario` (TRAF22) and
`benchmark_hybrid_reach_arch23_spacecraft` (SPRE22), both by `NonlinearSys` reach with alg "lin"
(`alg::Linearization`, tensor order 2). The expected verdict is that of the script and of the
ARCH-COMP26 NLN report (Geretti et al., "ARCH-COMP26 Category Report: Continuous and Hybrid
Systems with Nonlinear Dynamics", EPiC Series in Computing 110, pp. 52-84), whose CORA and
JuliaReach times are from the same machine as CORA.cpp's (core preset, 09-November-2026, one run
each; other jobs share its 16 CPUs). The quality metrics of the report (LALO width of x4, ROBE
width of s, LOVO area) belong to instances that are not ported yet; TRAF22 and SPRE22 have none.

| instance | expected | CORA.cpp | CORA.cpp [s] | CORA ARCH26 [s] | JuliaReach ARCH26 [s] | metric |
| --- | --- | --- | --- | --- | --- | --- |
| TRAF22 | 1 | 1 | 0.919 | 152 | 32 | - |
| SPRE22 | 1 | 1 | 1.04 | 37 | 61 | - |

**TRAF22** reads the controls from `data/BEL_Putte-4_2_T-1_controls.csv` (copied from CORA): 33
segments of 0.1 s, each reached in 20 steps from the end of the one before. CORA's script uses a
`nonlinParamSys` whose parameters (K and u_ref of the segment) are numbers, which CORA linearizes as
a `linearSys`; CORA.cpp builds one `NonlinearSys` per segment with the parameters as constants, the
same system (`intermediateTerms` has no effect there; docs/cora-differences.md). Its time includes
the occupancy sets of the script (the car's body over each set, 660 zonotopes and their vertices;
no file is written), as the script's `tComp` does; the ARCH26 report times the reachable sets only.
MATLAB CORA v2026.1.0 (R2024b, this machine) gives the same verdict in 262 s, and the hull of the
applied inputs, `[-0.4988, 0.5001] x [-8.4882, 0.4826]`, agrees with CORA.cpp's to all printed
digits, as do the boxes of the first and the last set. ARCH26 defines the same scenario
(BEL_Putte-4_2_T-1, the same sets Z, U, W) and adds a collision check of the occupancy sets by the
benchmark's Python script, which neither CORA's script nor CORA.cpp runs.

**SPRE22** is the nonlinear spacecraft rendezvous with the passive mode entered for t in [120, 150]
(`rendezvous_nonlinear_passive_nondet.m`), nondetGuard with PCA, time steps 0.2 / 0.05 / 0.2. CORA
checks the flow across the guard x = -100 by a Taylor model, CORA.cpp by a second-order Taylor form,
which agrees for this guard (its flow x' = vx is linear; docs/cora-differences.md). MATLAB CORA v2026.1.0 (R2024b,
this machine) verifies it in 68.8 s with the same three branches (559, 830 and 415 sets, the same
times), and the box of its last set agrees with CORA.cpp's to all printed digits. ARCH26 states the
same specifications as the script (line of sight 20 degrees, velocity bounds, no hit of the station
[-1, 1]^2 after the abort).

## Run

```bash
cmake --build --preset core                                   # build/core/benchmarks/benchmark_linear_*
build/core/benchmarks/benchmark_linear_verify_fast_arch23_iss [ISSF01_ISS01 ...]
build/core/benchmarks/benchmark_hybrid_reach_arch23_rendezvous [SRA01 ...]   # CORACPP_BENCHMARK_VERBOSE=1: the branches
build/core/benchmarks/benchmark_nonlinear_reach_arch23_trafficscenario         # TRAF22
PYTHONPATH=build/python python examples/python/benchmarks/benchmark_linear_verifyFast_ARCH23_iss.py [instance ...]
```

The programs need `LinearSys::verify` (CMake builds them once `linearSys.h` declares it). Without
instance arguments a program runs its whole family. Compare with MATLAB by running the original
script in CORA (`benchmark_linear_verifyFast_ARCH23_iss_ISSF01_ISS01`) and diffing the lines.
`CORACPP_BENCHMARK_DATA` names another data folder.

## Data

`examples/benchmarks/data/` holds the instances, about 1.6 MB in all, written by MATLAB (HEAT03:
by `examples/python/benchmarks/heat3d_data.py`, below):

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

HEAT03 is not exported by MATLAB: CORA's `models/Cora/heat3D/heat20.mat` is a Git LFS file and
MATLAB is not available here. `heat3d_data.py` builds `A` from the finite differences of the
benchmark (Bak, Tran, Johnson, HSCC 2019; see its header), which reproduce the exported `A` of
HEAT01 (heat5) and HEAT02 (heat10) bit for bit (`--check`), and `R0` and `C` from the script;
`linAlg` replaces `verifyAlg`, and `specs` is empty (the goal of the maximum is in the program).

A scalar `B` of the beam scripts is stored as `B*I`; an absent `params.U` (ISSC01_ISU02) as `{0}`.
Unsafe sets with several halfspaces (RAND) need a `Specification` that takes them; the programs
stop with an error until then. Test: `python -m unittest tests/python/test_benchmarkData.py`.

<!-- BEGIN bit-exact -->
| instance | bit-exact vs CORALean | not bit-exact |
|---|---|---|
| BRKDC01 | 4/25 | 21 |
| BRKNC01 | 30/73 | 43 |
| CBC01 | 1/22 | 21 |
| CBC02 | 1/22 | 21 |
| CBC03 | 1/22 | 21 |
| CBF01 | 1/22 | 21 |
| CBF02 | 1/22 | 21 |
| CBF03 | 1/22 | 21 |
| DTN01 | 21/52 | 31 |
| DTN02 | 21/52 | 31 |
| DTN03 | 21/52 | 31 |
| DTN04 | 21/52 | 31 |
| DTN05 | 21/52 | 31 |
| DTN06 | 21/52 | 31 |
| GRBX01 | 30/75 | 45 |
| GRBX02 | 30/75 | 45 |
| HEAT01 | 1/22 | 21 |
| HEAT02 | 1/22 | 21 |
| HEAT03 | 10/29 | 19 |
| ISSC01_ISS02 | 1/22 | 21 |
| ISSC01_ISU02 | 1/22 | 21 |
| ISSF01_ISS01 | 1/22 | 21 |
| ISSF01_ISU01 | 1/22 | 21 |
| PLAA01_BND42 | 23/52 | 29 |
| PLAA01_BND50 | 23/52 | 29 |
| PLAD01_BND30 | 9/30 | 21 |
| PLAD01_BND42 | 9/30 | 21 |
| PLAN01 | - | not measured |
| RAND01 | 5/27 | 22 |
| RAND02 | 5/27 | 22 |
| SPRE22 | 36/86 | 50 |
| SRA01 | 26/63 | 37 |
| SRA02 | 33/75 | 42 |
| SRA03 | 33/75 | 42 |
| SRA04 | 26/63 | 37 |
| SRA05 | 33/75 | 42 |
| SRA06 | 33/75 | 42 |
| SRA07 | 41/87 | 46 |
| SRA08 | 41/87 | 46 |
| SRNA01 | 23/56 | 33 |
| SRU01 | 27/64 | 37 |
| SRU02 | 41/87 | 46 |
| TRAF22 | 15/47 | 32 |
<!-- END bit-exact -->
