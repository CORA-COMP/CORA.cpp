# Benchmarks

## Summary

| category | instances run | verdict = ARCH26 | CORA.cpp fastest | speedup vs best ARCH26 tool (median / geo. mean) | speedup vs CORA ARCH26 (median / geo. mean) |
| --- | --- | --- | --- | --- | --- |
| AFF | 49 | 41 of 43 in ARCH26 (the 47 run in MATLAB CORA equal its verdict; HEAT04 not run there to the end) | 27 of 35 | 2.6 / 4.1 | 13 / 12 (35 instances) |
| NLN | 11 | 11 of 11 | 8 of 11 | 4.2 / 3.4 | 5.9 / 8.7 (11 instances) |
| AINNCS | 11 | 10 of 11 (all 11 equal CORA's) | 8 of 10 | 3.6 / 5.0 | 16 / 20 (10 instances) |

The speedups are over the instances where CORA.cpp's verdict is the expected one and the ARCH26 report
has a time for it: they leave out PLAN01 and BRKNP01 (ARCH26 verifies them with another setup than
the script's), Double Pendulum less-robust (ARCH26: verified by CROWN-Reach and JuliaReach, UNKNOWN
for CORA and CORA.cpp) and the instances the report does not have (ISS, RAND) and RND01-06 (instances that are probably, not certainly, the report's, RND[c]). The AINNCS rows count the Unicycle twice
(the script and the report's disturbance). Times are single runs on a shared machine.

## How to read the tables

- **expected (ARCH26)**: the verdict of the ARCH-COMP26 report of the category (1 or V verified, 0 or U
  not verified, F falsified); `(ARCH23)` where the report has no such instance, `(script: 0)` where
  the report verifies with another setup than CORA's script.
- **CORA.cpp [s]**: wall clock, core preset, one run each, the programs run one at a time on 16 CPUs
  that other jobs share (AFF and NLN: 09-November-2026, AINNCS and HEAT04: 11-November-2026).
- **oracle check (rows)**: the rows of CORA the instance executes that are bit-exact against
  CORALean's `nearest` oracle, of those that can be (`coverage/benchmarks.md`, table at the end), and
  the rows tested by properties only.
- **CORA [s]**: MATLAB R2024b, CORA v2026.1.0, on this machine, `CHECKS_ENABLED` off (CORA's default
  is on and slower), 4 threads, one run; `-` where not measured.
- **CORA ARCH26 [s]**: CORA's time in the report.
- **best ARCH26 [s] (tool)**: the fastest time in the report of any tool with the expected verdict on
  the same instance, from code that solves the report's problem. JuliaReach's discrete-time rows
  (time points only) and immrax (continuous feedback instead of sample-and-hold) do not count. A faster
  result of code that solves another problem is not counted either; a footnote says what differs.
- **speedup** = best / CORA.cpp: above 1, CORA.cpp is faster.

## Programs

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
| `heat3D` (reach) | HEAT03 (heat20, 8000 states), HEAT04 (heat50, 125 000 states), sparse `A` | linearSys reach, linAlg krylov |
| `rand` | RAND01, RAND02 (`rand0?_test.json` models) | reachavoid:zonotope |
| `rnd` | RND01-06 (ARCH26: 10, 50, 100, 1000, 2500 and 5000 states) | reachavoid:zonotope (RND01, RND03), reachavoid:krylovSuppFunc (RND02, RND04-06) |
| `rendezvous` | SRA01-08, SRNA01, SRU01, SRU02 | hybrid reach: zonoGirard or nondetGuard, enclose box / pca |
| `gearbox` | GRBX01, GRBX02 | hybrid reach: zonoGirard, enclose box + pca |
| `powertrain` | DTN01-06 | hybrid reach: zonoGirard, enclose pca |
| `brake` | BRKDC01 (linear reach and resets), BRKNC01, BRKNP01 (a LinParamSys flow, see below) | hybrid reach: conZonotope, enclose box + pca |
| `platoon` | PLAA01_BND42, PLAA01_BND50 (LinParamSys, constParam), PLAD01_BND30, PLAD01_BND42 (LinearSys switching every 5 s), PLAN01 (varParam, then an invariant) | linParamSys reach / linearSys reach |

The hybrid and platoon families have no data files: their models are small and are C++ builder
functions in `examples/cpp/benchmarks/benchmark_hybrid_reach_arch23_*.cpp` and
`benchmark_linear_param_reach_arch23_platoon.cpp`, the numbers copied from the scripts
and the model files of `models/Cora/` (the powertrain's Jacobians written out from its equations).

## AFF results

The expected verdict is that of the ARCH-COMP26 AFF report (Althoff, Carraro, Forets, Perschl,
Schilling, "ARCH-COMP26 Category Report: Continuous and Hybrid Systems with Linear Continuous Dynamics",
EPiC Series in Computing 110, pp. 28-51), whose CORA and JuliaReach times were run on the same machine as
CORA.cpp's. The verdict of CORA.cpp is CORA's `res`; every one equals MATLAB CORA's on the script
(ARCH23: the ISU instances, RAND02, SRU01 and SRU02, PLAN01 and BRKNP01 are not verified; PLAN01 and BRKNP01: below).
ARCH26 has no ISS, uses the new RND01-06 instead of RAND01/02 (RND[c], run here on the files of CORA's package) and adds HEAT04 (heat50, 125 000
states, Krylov), which runs here (its `A` is built from the mesh, not stored). The other instances
here stay those of the ARCH23 scripts.

| instance | expected (ARCH26) | CORA.cpp | CORA.cpp [s] | oracle check (rows) | CORA [s] | CORA ARCH26 [s] | best ARCH26 [s] (tool) | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HEAT01 | 1 | 1 | 0.064 | 1/8 bit-exact, 2 property | - | 0.13 | 0.04 (JuliaReach) | 0.62 |
| HEAT02 | 1 | 1 | 3.39 | 1/8 bit-exact, 2 property | - | 3.04 | 2.6 (JuliaReach) | 0.77 |
| HEAT03 | 1 | 1 | 14.6 | 10/17 bit-exact, 0 property | - | 21.8 | 21.8 (CORA) | 1.49 |
| HEAT04 | 1 | 1 | 1303 | 10/17 bit-exact, 0 property | - | 1193 | 1193 (CORA) [d] | 0.92 |
| CBC01 | 1 | 1 | 0.162 | 1/8 bit-exact, 2 property | - | 0.28 | 0.05 (JuliaReach) | 0.31 |
| CBC02 | 1 | 1 | 3.88 | 1/8 bit-exact, 2 property | - | 3.68 | 0.8 (JuliaReach) | 0.21 |
| CBC03 | 1 | 1 | 44.5 | 1/8 bit-exact, 2 property | - | 37.27 | 3.74 (JuliaReach) | 0.084 |
| CBF01 | 1 | 1 | 0.255 | 1/8 bit-exact, 2 property | - | 0.4 | 0.4 (CORA) | 1.57 |
| CBF02 | 1 | 1 | 4.51 | 1/8 bit-exact, 2 property | - | 5.38 | 5.38 (CORA) | 1.19 |
| CBF03 | 1 | 1 | 66.5 | 1/8 bit-exact, 2 property | - | 67.79 | 67.79 (CORA) | 1.02 |
| ISSC01_ISS02 | 1 (ARCH23) | 1 | 0.153 | 1/8 bit-exact, 2 property | - | - | - | - |
| ISSC01_ISU02 | 0 (ARCH23) | 0 | 0.522 | 1/8 bit-exact, 2 property | - | - | - | - |
| ISSF01_ISS01 | 1 (ARCH23) | 1 | 0.628 | 1/8 bit-exact, 2 property | - | - | - | - |
| ISSF01_ISU01 | 0 (ARCH23) | 0 | 0.194 | 1/8 bit-exact, 2 property | - | - | - | - |
| RAND01 | 1 (ARCH23) | 1 | 0.0023 | 5/13 bit-exact, 2 property | - | - | - | - |
| RAND02 | 0 (ARCH23) | 0 | 0.015 | 5/13 bit-exact, 2 property | - | - | - | - |
| RND01 | 1 | 1 | 0.022 | 5/13 bit-exact, 2 property | 3.88 | 5.72 | 0.18 (JuliaReach) [c] | 7.7 |
| RND02 | 1 | 1 | 0.907 | 2/11 bit-exact, 2 property | 2882 | 0.42 | 0.42 (CORA) [c] | 0.46 |
| RND03 | 1 | 1 | 29.8 | 5/13 bit-exact, 2 property | 37.6 | 37.34 | 37.34 (CORA) [c] | 1.25 |
| RND04 | 1 | 1 | 0.642 | 2/11 bit-exact, 2 property | 6.13 | 7.77 | 7.77 (CORA) [c] | 12.1 |
| RND05 | 1 | 1 | 1.33 | 2/11 bit-exact, 2 property | 22.99 | 45.86 | 45.86 (CORA) [c] | 34.5 |
| RND06 | 1 | 1 | 5.14 | 2/11 bit-exact, 2 property | - | 26.70 | 26.70 (CORA) [c] | 5.2 |
| SRNA01 | 1 | 1 | 0.473 | 23/40 bit-exact, 6 property | - | 51.87 | 1.09 (JuliaReach) | 2.30 |
| SRA01 | 1 | 1 | 0.204 | 26/46 bit-exact, 6 property | - | 10.94 | 1.17 (JuliaReach) | 5.74 |
| SRA02 | 1 | 1 | 0.102 | 34/56 bit-exact, 7 property | - | 17.49 | 1.24 (JuliaReach) | 12.2 |
| SRA03 | 1 | 1 | 0.178 | 34/56 bit-exact, 7 property | - | 27.4 | 1.79 (JuliaReach) | 10.1 |
| SRA04 | 1 | 1 | 0.484 | 26/46 bit-exact, 6 property | - | 39.05 | 39.05 (CORA) | 80.7 |
| SRA05 | 1 | 1 | 0.464 | 34/56 bit-exact, 7 property | - | 67.33 | 67.33 (CORA) | 145 |
| SRA06 | 1 | 1 | 0.472 | 34/56 bit-exact, 7 property | - | 71.46 | 71.46 (CORA) | 151 |
| SRA07 | 1 | 1 | 2.49 | 42/67 bit-exact, 9 property | - | 120 | 120 (CORA) | 48.2 |
| SRA08 | 1 | 1 | 5.27 | 42/67 bit-exact, 9 property | - | 239 | 239 (CORA) | 45.4 |
| SRU01 | 0 | 0 | 0.413 | 27/47 bit-exact, 6 property | - | 25.86 | 11.79 (JuliaReach) | 28.5 |
| SRU02 | 0 | 0 | 5.93 | 42/67 bit-exact, 9 property | - | 244 | 20 (JuliaReach) | 3.37 |
| DTN01 | 1 | 1 | 0.508 | 21/37 bit-exact, 6 property | - | 9.69 | 9.69 (CORA) [a] | 19.1 |
| DTN02 | 1 | 1 | 0.618 | 21/37 bit-exact, 6 property | - | 10.43 | 10.43 (CORA) [a] | 16.9 |
| DTN03 | 1 | 1 | 0.914 | 21/37 bit-exact, 6 property | - | 10.27 | 10.27 (CORA) [a] | 11.2 |
| DTN04 | 1 | 1 | 19.4 | 21/37 bit-exact, 6 property | - | 48.29 | 48.29 (CORA) [a] | 2.49 |
| DTN05 | 1 | 1 | 35.5 | 21/37 bit-exact, 6 property | - | 90.76 | 90.76 (CORA) [a] | 2.56 |
| DTN06 | 1 | 1 | 110 | 21/37 bit-exact, 6 property | - | 227 | 227 (CORA) [a] | 2.06 |
| PLAA01_BND42 | 1 | 1 | 10.5 | 23/35 bit-exact, 7 property | - | 25.02 | 25.02 (CORA) | 2.38 |
| PLAA01_BND50 | 1 | 1 | 4.19 | 23/35 bit-exact, 7 property | - | 7.77 | 7.77 (CORA) | 1.85 |
| PLAD01_BND30 | 1 | 1 | 0.306 | 9/17 bit-exact, 0 property | - | 4.03 | 4.03 (CORA) | 13.2 |
| PLAD01_BND42 | 1 | 1 | 0.066 | 9/17 bit-exact, 0 property | - | 3.07 | 0.05 (JuliaReach) | 0.76 |
| PLAN01 | 1 (script: 0) | 0 | 47.9 | 31/45 bit-exact, 8 property | 173 | 162 | 162 (CORA) | - |
| GRBX01 | 1 | 1 | 0.03 | 30/56 bit-exact, 12 property | - | 2.28 | 2.28 (CORA) | 76.0 |
| GRBX02 | 1 | 1 | 0.037 | 30/56 bit-exact, 12 property | - | 2.44 | 2.44 (CORA) | 65.9 |
| BRKDC01 | 1 | 1 | 0.037 | 4/12 bit-exact, 0 property | - | 22.66 | 0.1 (JuliaReach) | 2.70 |
| BRKNC01 | 1 | 1 | 4.83 | 31/54 bit-exact, 9 property | - | 32.07 | 0.17 (JuliaReach) | 0.035 |
| BRKNP01 | 1 (script: 0) | 0 (reach_set_explosion) | 0.001 | 24/44 bit-exact, 12 property | 438 | 52.2 | 52.2 (CORA) [b] | - |

[a] DTN01-06: JuliaReach's ARCH-COMP 2025 repeatability code (0.005-0.127 s) runs `powertrain_homog(θ=k)` for DTN0k with the full initial set, where the report has θ = 2 (DTN01-03) or 22 (DTN04-06) and initial sets scaled to 5 %, 30 % and 100 %.

[b] BRKNP01: JuliaReach's ARCH-COMP 2025 repeatability code (16.99 s) does not enforce the property (`validate=false`) and runs to t = 0.1, where the report verifies x < x0 up to t = 0.08.

[c] RND01-06: the report's instances are created on the server, these are the files of CORA's package (RND06: our own), so the ARCH26 columns are indicative and the speedups stay out of the summary.

[d] HEAT04: ARCH26 ran a new support-function Krylov verification (Perschl and Althoff, HSCC 2026, not in CORA v2026.1.0); CORA.cpp runs the ARCH23 script's reach with a step of 0.001, and JuliaReach's 3180 s are its discrete-time mode (continuous did not solve it), so CORA's 1193 s is the best.

**PLAN01** is not verified by CORA's ARCH23 script, MATLAB R2024b, CORA v2026.1.0 on this machine
(verdict 0, the CORA column), and CORA.cpp agrees: its 50 s reach with varying parameters reaches x1, x4,
x7 >= -50.65, past the bound -50. The ARCH26 report lists PLAN01 as verified (162 s) with a different
setup, so the speedup is left out. The algorithm is ported step by step (expmIndMixed, priv_tie,
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

**HEAT04** is HEAT03's script on heat50 (125 000 states, `A` of 860 000 entries built from the mesh
in C++, bit-identical to `models/Cora/heat3D/heat50.mat` in its action on a test vector and in
nnz; `R0` has 1386 hot cells, `e_mid` is cell (25, 25, 25)). The maximum temperature over
the steps must lie in 0.01161 +- 1e-4 (the report's Tmax; the true value, from the ODE solution,
is 0.0116118 at t = 18.87). The script's step of 0.005 gives 0.0158, 36 % too high, because the
enlargement of a time interval grows with the 1387 outputs; the step 0.001 gives 0.0116902 and
the verdict 1, in 1303 s on 16 shared CPUs and 4.0 GB peak memory (the dual system keeps `A'`,
the 1387 x 125 000 output matrix, 1.4 GB, and a subspace of order 181 per vector). Sets of 1387
outputs have 2774 generators, so `reach` takes a callback (`LinearSys::reach(problem, alg,
on_time_interval)`) that sees each step's set and stores none: all 40 000 sets would need 2.5 TB.
MATLAB CORA cannot run it on this machine (64 GB limit): the same script to t = 0.2 (200 steps)
takes 32.4 s there against 7.3 s for CORA.cpp, with the same maximum (8.79442e-15). `Interval::zonotope`
no longer forms the dense n x n diagonal of its radii (125 GB here). The Krylov steps are those of
HEAT03, with the same unsound spots (above). The coverage build (3 times slower, over the 3600 s of a test) measured it with `CORACPP_HEAT04_T_FINAL=1`, which shortens the horizon: the files executed are the same.

**BRKNP01** is not verified by CORA's ARCH23 script, MATLAB R2024b, CORA v2026.1.0 on this machine
(verdict 0, the CORA column, with the warning 'Taylor order not high enough'). The input `U = zonotope(1)`
becomes the constant input `uTrans = 1` with `U = {0}`, which does not hold the origin, so
`priv_inputTie` adds the curvature of the constant input; its series does not converge
(`norm(A, inf) r / (taylorTerms + 2)` is 18.3 for `r = 2e-5`) and its remainder is unbounded.
CORA prints the warning and then drops the remainder (`intervalMatrix + interval` adds nothing),
so its sets are finite but miss trajectories. CORA.cpp, as CORALean, keeps the remainder: the
sets are unbounded, `reach` raises `reach_set_explosion` at once (0.001 s) and the driver reports
verdict 0, CORA's. Before that error CORA.cpp ran for over 30 minutes: Inf times a zero entry of
`uTrans` gave NaN sets, which the hybrid loop jumped from again and again at the same time
(docs/cora-differences.md). The ARCH26 report lists BRKNP01 as verified (52.20 s), with a
different setup, so the speedup is left out.

**Guard boxes.** The guards of `nondetGuard` and `conZonotope` cut each set by the polytope (the
rows stay rows on the factors, `ConZonotope::cut`) and box the cut by the option `box`
(`alg::CutBox`, CORALean's `ConZonotope.BoxMethod`). `fm` uses `fm_passes` Fourier-Motzkin passes
(`ConZonotope::interval_tight`, default 10), CORALean's float definition bit for bit; the boxes are
never tighter than CORA's, and for SRA01-08, SRNA01, SRU01/02 and SPRE22 (`nondetGuard`) the
verdicts and the number of sets do not change (the times are the same within the noise of the
shared machine). `cert` takes the dual solution of CORA's two linear programs per coordinate and
bounds with it in directed order (`ConZonotope::Cut::interval_cert`, CORALean's
`intervalCertWith`): as tight as CORA's ranges, outer whatever the solver answers. BRKNC01
(`conZonotope`, order 5) loses its verdict with the passes: 2.7 s and verdict 0 for any number of
passes from 1 to 1000. The driver of BRKNC01 and BRKNP01 therefore sets `box = cert`.

**RND01-06** are the random benchmarks of the ARCH26 report: x' = A x + B u, y = C x with m = 5 inputs, intervals as
initial and input set, and unsafe sets at the border of the reachable set (the report's property (1), so the
expected verdict is 1; it prints times only). The report's instances "are created on the server" and not
published. RND01-05 are the files `rand01.json` to `rand05.json` of the ARCH-COMP repository
(`gitlab.com/goranf/ARCH-COMP`, `2026/AFF/cora/code/cora/`, the same files in `2026/AFF/cora/secret-data/`) that CORA's
repeatability package loads: 10, 50, 100, 1000 and 2500 states.

**Are they the report's instances?** Probably, but nothing confirms it. MATLAB CORA v2026.1.0 takes 37.6 s on RND03
here against 37.34 s in the report and 3.9 s on RND01 against 5.72 s; the report says 500 states for RND04, the file has
1000. RND06 (5000 states) is not in the repository: `examples/python/benchmarks/rnd_data.py generate RND06` writes
a sparse, diagonally dominant (so stable) system in the style of rand05 with the unsafe halfspace y >= 840.2, a tenth
of the rise above the exact maximum 746.9 of the support function. `rnd_data.py convert` turns a json file of
the repository into `data/RND0?.json` and `.bin` (CORA's `json2cora_linearSys` and `zonotope(interval)`;
RND05 and RND06 are stored sparse, 0.3 MB and 0.7 MB).

**All six are safe.** The exact maximum of the support function along the direction of the unsafe halfspace
(scipy, `expm_multiply` on a 4000-point grid, not a proof) is 136.02 against the bound 141.28 for RND04 (3.9 % below),
593.31 against 598.63 for RND05 (0.9 %), 56.936 against 56.941 for the row y <= lo of RND02 (0.009 %) and 0.039 below
the row of RND03 (0.16 %). RND01 is not separated by one row over the whole horizon, but by a different one on each step,
and the zonotope algorithm, the Krylov algorithm and MATLAB CORA all verify it.

**Algorithms.** CORA's package runs `reachavoid:zonotope` for RND01 and RND03, `reachavoid:supportFunc` for RND02 and
for RND04-06 a loop that halves the step until `reachSuppFuncsKrylov` (the competition's copy of CORA, not v2026.1.0)
bounds the support function below the unsafe halfspace. CORA.cpp runs the zonotope algorithm of CORA on RND01 and RND03.
Its `ReachAvoidSupportFunction` takes unsafe sets of one halfspace and the unsafe set of RND02 is an interval of two,
so RND02 and RND04-06 run `alg::ReachAvoidKrylov`, which is not a port of `reachSuppFuncsKrylov` but follows its scheme on
the Krylov machinery of `reach` (`priv_verify_ra_krylov.cpp`; no CORA file, no CORALean operation, tested by properties).
Per halfspace of the specification it takes the Krylov subspace of `C' l` under `A'` (dense or sparse `A`; `l` the
output direction), bounds the support function of the output set at every step by the hull of the support at the two
time points plus the curvature `F(H) e_1` of the subspace, adds the integral of the input (the step times the largest
integrand) and `krylov_error` times the norms of the vectors, and halves the step from t_final / 1000 until every safe
halfspace and, per unsafe set, one of its halfspaces holds on every step. RND04-06 verify in the first pass.

**Soundness of `reach_avoid_krylov`, step by step.**

- The Krylov subspace is exact for the polynomials of degree below its order, which covers the Taylor part of the curvature
  (degree 20, the order is at least 50), and a subspace of the order n is exact (RND02).
- The order is chosen by Jawecki's bound, as in CORA, and that is **unsound**: Thm. 1 holds for nonexpansive matrices only
  (CORALean `Krylov/Counterexamples/JaweckiBoundCora.lean`), and the A of RND04, RND05 and RND06 are not (largest eigenvalue of
  the symmetric part 123.4, 94.0 and 1.9). It picks the orders 101, 101 and 151 there, whose error against scipy's
  `expm_multiply` is at most 3e-14 at t = 1 and t = 5 (measured): the verdicts hold numerically, the bound is no proof.
- The curvature acts on the state at the start of the step, as in the theory of `reach` (sound; CORA's `priv_reach_krylov`
  uses the state at the end).
- The hull of the two time points plus the curvature encloses the solution in between, and the integral is bounded by the step
  times the supremum of the integrand (sound). One halfspace per step of an unsafe polytope is a sufficient test (sound).
- Floating point is rounded to nearest, not outward, as in every float algorithm of CORA.cpp.

`test_linear_sys_verify_krylov` checks a system with a closed-form support function and a 4-state system with two inputs
against their exact maxima (0.2 % above verifies, 0.2 % below does not).

**Soundness of the zonotope algorithm (RND01, RND03).** CORA's verdict is unsound where the unsafe set only touches the reachable
set (CORALean `Counterexamples/VerifyRAOuterVerdictCora.lean`); the gaps of RND01 and RND03 are positive, and the Krylov
algorithm confirms both verdicts.

**MATLAB times (CORA [s]).** MATLAB R2024b, CORA v2026.1.0 on this machine, `CHECKS_ENABLED` off, 4 threads, one run in a fresh
session at a load of 8 to 12: RND01 3.88 s (0.94 s on a second call in the session), RND03 37.6 s, both with
`reachavoid:zonotope`. For RND02 v2026.1.0's `reachavoid:supportFunc` raises an error on the interval of two halfspaces,
so `reachavoid:zonotope` took 2882 s, verdict 1. v2026.1.0 has no Krylov script: RND04 and RND05 are `reachavoid:supportFunc` on the
sparse system with the unsafe halfspace y >= lb (the json has `"upperbound": null`, which CORA's loader reads as the
point lb; the package's script uses `b(1)`), 6.13 s and 22.99 s, both verdict 1. RND06 was not run in MATLAB.

**RND03** takes CORA.cpp 29.8 s with the zonotope algorithm (CORA 37.6 s; 11 iterations, last step 1.3e-5, verdict 1). It took
1330 s (1464 s on a second run) until the simplex of `linprog` was made sparse: the verification solves 1876 linear programs
of up to 1200 variables with one bound row each, and 95% of the time was in their dense tableau (the results of every program
are bit-identical before and after). The Krylov algorithm
verifies RND03 in 0.40 s and RND01 in 0.04 s, an independent confirmation of the verdicts. RND02 with the zonotope algorithm
of CORA.cpp did not finish in 3 minutes (MATLAB: 48 minutes).

**Memory.** RND06 needs 1.2 GB (dense `A` of 5000 x 5000 and its transposed copy for the products), RND05 0.3 GB.

In the test suite every ported instance is a test of the label `benchmark` (nightly), and the first

## NLN results

The ARCH-COMP NLN instances of MATLAB CORA (`examples/ARCHcompetition/nonlinear/*.m`), ported as
they are in CORA (ARCH23 scripts): `benchmark_nonlinear_reach_arch23_trafficscenario` (TRAF22) and
`benchmark_hybrid_reach_arch23_spacecraft` (SPRE22), both by `NonlinearSys` reach with alg "lin"
(`alg::Linearization`, tensor order 2), and `benchmark_nonlinear_reach_arch23_robertson` (ROBE25,
three cases) with alg "lin-adaptive" (`alg::LinearizationAdaptive`), `benchmark_hybrid_reach_arch23_vanderpol`
(CVDP23) with alg "poly" (`alg::Polynomialization`, polynomial zonotopes) and
`benchmark_nonlinear_reach_arch23_laubloomis` (LALO20, three cases),
`benchmark_nonlinear_das_reach_arch23_powersystem` (TSPS25, `NonlinDASys` reach, `alg::DaeLinearization`) and
`benchmark_hybrid_reach_arch23_lotkavolterra` (LOVO25, level set guards and invariants). The expected verdict is that of the
ARCH-COMP26 NLN report (Geretti et al., "ARCH-COMP26 Category Report: Continuous and Hybrid
Systems with Nonlinear Dynamics", EPiC Series in Computing 110, pp. 52-84), whose CORA and
JuliaReach times are from the same machine as CORA.cpp's. The quality metrics of the report (LALO width of x4, ROBE
width of s, LOVO area) belong to the instances that have one; TRAF22, SPRE22 and TSPS25 have none.

| instance | expected (ARCH26) | CORA.cpp | CORA.cpp [s] | oracle check (rows) | CORA [s] | CORA ARCH26 [s] | best ARCH26 [s] (tool) | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAF22 | 1 | 1 | 0.919 | 15/30 bit-exact, 5 property | 138 | 152 | 152 (CORA) [a] | 165 |
| SPRE22 | 1 | 1 | 1.04 | 37/65 bit-exact, 11 property | 34.3 | 37 | 37 (CORA) | 35.6 |
| ROBE25 (1) | 1 | 1 | 25.6 | 10/24 bit-exact, 6 property | 137.5 | 151 | 25 (JuliaReach) [b] | 0.98 |
| ROBE25 (2) | 1 | 1 | 325 | 10/24 bit-exact, 6 property | 225.7 | 239 | 92 (JuliaReach) [b] | 0.28 |
| ROBE25 (3) | 1 | 1 | 100 | 10/24 bit-exact, 6 property | 338.7 | 310 | 217 (JuliaReach) [b] | 2.17 |
| CVDP23 | 1 | 1 | 58.6 | 30/70 bit-exact, 17 property | 461.7 | 571 | 0.4 (JuliaReach) | 0.007 |
| LALO20 W001 | 1 | 1 | 2.24 | 10/24 bit-exact, 6 property | 12.3 | 9.3 | 9.3 (CORA) [c] | 4.15 |
| LALO20 W005 | 1 | 1 | 15.3 | 10/24 bit-exact, 6 property | 21.9 | 17 | 17 (CORA) [c] | 1.11 |
| LALO20 W01 | 1 | 1 | 224 | 7/35 bit-exact, 6 property | 208 | 248 | 248 (CORA) [c] | 1.09 |
| TSPS25 | 1 | 1 | 65 | 17/32 bit-exact, 6 property | 312 | 299 | 299 (CORA) | 4.60 |
| LOVO25 | 1 | 1 [e] | 3.72 | 24/75 bit-exact, 18 property | 11.4 | 9.6 | 9.6 (CORA) [d] | 2.58 |

[a] TRAF22: JuliaReach's ARCH-COMP 2025 repeatability code (32 s) holds the disturbances w constant (w' = 0), where the report lets them vary arbitrarily over time.

[b] ROBE25: KeYmaera X (0.2 s) proves that x + y + z is preserved, not the width bound.

[c] LALO20: JuliaReach's ARCH-COMP 2025 repeatability code (0.7-1.1 s) has x7' = 1.8 x6 - 1.6 x2 x7, where the report has 1.5 x2 x7, and KeYmaera X (12.8 s) and PRoTECT (4.4 s) prove other properties (t in [0, 0.1] and t in [0, inf)).

[d] LOVO25: JuliaReach's ARCH-COMP 2025 repeatability code (1.6 s) integrates the flow without the hybrid automaton and checks only the area, not the crossing counts, and KeYmaera X (1.7 s) proves invariance only.

[e] LOVO25: a sound verdict, not CORA's script (which writes `res = true`; it runs as `LOVO25-script`, 0.09 s, area 8.92e-3, MATLAB 11.4 s). The automaton (with a jump back out of `inside`) is reached from 8 pieces of the initial box with no restart and no cut; 7 pieces give one final branch with 2 jumps, and the piece with the tangential arc (x1 near 1.288 to 1.291) makes CORA's linearization error explode in the jumps in and out, so the flow alone encloses it. The area of all sets at x3 = 3.64 is below 1e-2 and none is in `inside`; exactly 2 and exactly 4 crossings are shown by simulated trajectories (x1 = 1.2891 and 1.288). The CORA time column is the script's.

| instance | quality metric (CORA.cpp; ARCH26 CORA, JuliaReach) |
| --- | --- |
| ROBE25 (1) | width of s 2.28e-6 (CORA 5.8e-6, JuliaReach 1.5e-8) |
| ROBE25 (2) | width of s 7.17e-6 (CORA 8.8e-6, JuliaReach 1.1e-9) |
| ROBE25 (3) | width of s 7.33e-6 (CORA 8.3e-6, JuliaReach 6.5e-10) |
| LALO20 W001 | width of x4 0.00380 (CORA 0.004, JuliaReach 0.003) |
| LALO20 W005 | width of x4 0.0394 (CORA 0.042, JuliaReach 0.019) |
| LALO20 W01 | width of x4 0.0570 (CORA 0.060, JuliaReach 0.031) |
| LOVO25 | area 2.24e-3 (CORA 8.9e-3, JuliaReach 9e-5; the flow alone from the whole box 3.04e-5) |

**TRAF22** reads the controls from `data/BEL_Putte-4_2_T-1_controls.csv` (copied from CORA): 33
segments of 0.1 s, each reached in 20 steps from the end of the one before. CORA's script uses a
`nonlinParamSys` whose parameters (K and u_ref of the segment) are numbers, which CORA linearizes as
a `linearSys`; CORA.cpp builds one `NonlinearSys` per segment with the parameters as constants, the
same system (`intermediateTerms` has no effect there; docs/cora-differences.md). Its time includes
the occupancy sets of the script (the car's body over each set, 660 zonotopes and their vertices;
no file is written), as the script's `tComp` does; the ARCH26 report times the reachable sets only.
MATLAB CORA v2026.1.0 (R2024b, this machine) gives the same verdict in 138 s, and the hull of the
applied inputs, `[-0.4988, 0.5001] x [-8.4882, 0.4826]`, agrees with CORA.cpp's to all printed
digits, as do the boxes of the first and the last set. ARCH26 defines the same scenario
(BEL_Putte-4_2_T-1, the same sets Z, U, W) and adds a collision check of the occupancy sets by the
benchmark's Python script, which neither CORA's script nor CORA.cpp runs.

**SPRE22** is the nonlinear spacecraft rendezvous with the passive mode entered for t in [120, 150]
(`rendezvous_nonlinear_passive_nondet.m`), nondetGuard with PCA, time steps 0.2 / 0.05 / 0.2. CORA
checks the flow across the guard x = -100 by a Taylor model, CORA.cpp by a second-order Taylor form,
which agrees for this guard (its flow x' = vx is linear; docs/cora-differences.md). MATLAB CORA v2026.1.0 (R2024b,
this machine) verifies it in 34.3 s with the same three branches (559, 830 and 415 sets, the same
times), and the box of its last set agrees with CORA.cpp's to all printed digits. ARCH26 states the
same specifications as the script (line of sight 20 degrees, velocity bounds, no hit of the station
[-1, 1]^2 after the abort).

**MATLAB times.** The CORA column was measured with the machine shared (load
average 10 to 15 when the runs started).

**ROBE25** (`..._Robertson.m`, ARCH26 instances (1), (2), (3) as `ROBE25_1` to `ROBE25_3`) is the
Robertson reaction x' = -0.4 x + b y z, y' = 0.4 x - b y z - c y^2, z' = c y^2 from (1, 0, 0) to
t = 40 with (b, c) = (10^2, 10^3), (10^3, 10^5), (10^3, 10^7). The result is the specification of
the report: the width of s = x + y + z at t = 40 below 1e-5 (the script prints the constant 1 and
does not check it; the program prints 1 only if the width is below, and the width on stderr).
CORA's `lin-adaptive` is not sound as written (its Hessian region and its third-order error:
`LinReachAdaptiveCora1D.lean`, `AbstrErrAdaptiveTensor3Cora2D.lean`), so the program runs CORA's
algorithm as CORALean repairs it (`reachAdaptiveFloat`; `docs/cora-differences.md`): the same
passes, tuning and tensor orders 2 and 3 as CORA's, with the Hessian read over the box of the set
joined with the linearization point. It needs 28963, 16296 and 28093 sets (CORA 16501, 22494 and
25960) and a set costs 0.9 ms, 20 ms and 3.6 ms (CORA's 8 ms, 10 ms and 13
ms). The cost of case (2) is the linear step: its Taylor order is raised until `eta + 2 >= 2 ||A dt||`
(CORA's is fixed, which is not sound for a large `||A dt||`), a step with `||A dt||` in the
hundreds takes 29 ms. MATLAB CORA v2026.1.0 (R2024b, this machine) with `CHECKS_ENABLED` off, as
the report's times probably are, needs 137.5 s, 225.7 s and 338.7 s (411 s, 734 s and 1117 s with
the checks, under a load of about 20) and gives widths 5.7785e-6, 8.7772e-6 and 8.3479e-6, as in
the report. CORA.cpp's times are from the core preset under a load of 7 to 10 on 16 CPUs.

The tuning (`alg::LinearizationAdaptive`) is the defaults of CORA (decr_factor 0.9, zeta_phi
0.85, 0.76, 0.68, zeta_k 0.9, 10 Taylor terms, zonotope order 1000) except for `zeta_z`, CORA's
`redFactor`. The script's values (0.00025, 0.0006, 0.017) verify in MATLAB CORA, but the repaired
algorithm is sound where CORA's is not and needs a smaller one in cases (1) and (3): at the script's
values the width of s is 5.6e-5 (17651 sets) and 1.2e-5 (25850 sets), in case (2) it verifies at
6.7e-6. The width is not monotone in `zeta_z`; the values below were found by a sweep.

| | zeta_z (script) | zeta_z | sets | width of s |
| --- | --- | --- | --- | --- |
| ROBE25_1 | 0.00025 | 1e-4 | 28963 | 2.28e-6 |
| ROBE25_2 | 0.0006 | 1.5e-3 | 16296 | 7.17e-6 |
| ROBE25_3 | 0.017 | 1e-2 | 28093 | 7.33e-6 |

**CVDP23** (`benchmark_nonlinear_reach_ARCH23_vanDerPol.m`) is two coupled van der Pol oscillators with
x5 = b as a constant state, three locations that only split the analysis at the plane c'x = d, and 13
boxes of the initial set, each a polynomial zonotope (`PolyZonotope(Interval)`). The program runs
the options of the script (`timeStep` 0.005, `taylorTerms` 3, `zonotopeOrder` 100, `tensorOrder` 3,
`errorOrder` 10, `intermediateOrder` 50, guards by `zonoGirard` with PCA) with `alg::Polynomialization`
and verifies y1, y2 < 2.75 for all 13: the time is that of the 13 runs together, as the script's. The
sets of a location are polynomial zonotopes; the guards act on their enclosing zonotopes and the
set that leaves is turned back into a polynomial zonotope, as CORA's `guardIntersect` does
(docs/cora-differences.md). MATLAB CORA v2026.1.0 (R2024b, this machine, `CHECKS_ENABLED` off) verifies it in
461.7 s with the same verdict; the report's time (571 s) is CORA's.

**LALO20** (`benchmark_nonlinear_reach_ARCH23_laubLoomis.m`, W001, W005, W01) is the Laub-Loomis model from
x0 = (1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45) with boxes of the half-width 0.01, 0.05 and 0.1 up to t = 20; the
specification of the report is x4 < 4.5 (W001, W005) and x4 < 5 (W01) over the whole horizon. W001 and W005 run
the script's alg "lin-adaptive" (`alg::LinearizationAdaptive`, defaults, CORALean's repair) and W01 its alg
"poly-adaptive" (`alg::PolynomializationAdaptive`, the constants of the script, CORALean's repair of
`reachPolyAdaptiveFloat`, !912): 1124 steps (CORA 1193), the largest x4 is 4.62, and the width of x4 at t = 20
is 0.0570 (CORA 0.0600). The time of 224 s (machine load 1 to 3) is 8 % above MATLAB CORA's
208 s; the profile (gprof, t in [0, 4]) has no single hot spot: the merge of equal exponents in the constructor
of a polynomial zonotope 18 %, `Zonotope::reduce_adaptive` 9 %, the adaptive reduction of polynomial zonotopes 7 %,
the column gathers 5 %, the sorts of the rankings 12 %. The independent part of the sets grows over many steps until the volume test restructures, as in CORA. The step with the stand-in that this row replaced (the fixed-step `alg::Polynomialization`, `timeStep` 0.01) took 10.3 s and
gave the width 0.345. MATLAB CORA v2026.1.0 (R2024b, this machine, 4 threads, `CHECKS_ENABLED` off, one run at a time) needs 12.3 s, 21.9 s and 208 s for W001,
W005 and W01 and gives the widths 0.00402, 0.0425 and 0.0600 (W001 and W005 from an earlier run); the widths of CORA.cpp for W001 and W005 are smaller because the repaired algorithm differs in its error terms
(docs/cora-differences.md).

**TSPS25** (`benchmark_nonlinearDAS_reach_ARCH23_powerSystem.m`) is the IEEE 14 bus power system as a
differential-algebraic system of 15 states, 27 algebraic variables and 19 inputs, analysed in four runs, each from the end
set of the one before: normal operation to 0.1 s (time step 0.005), the fault of bus 1 to 0.13 s (0.001), normal operation to
2 s (0.005) and on to 5 s (0.02), zonotope order 400, Taylor terms 6, `errorOrder` 3, `maxError_x` 0.6 on the five speeds
and 0 elsewhere, `maxError_y` 0.01. The verdict is the script's: the interval hull of the set at 5 s lies in the initial box,
which is the report's goal (R(t) returns to R(0)); the script checks t = 5, the report asks for some t > tc. The report's
x* is the steady state the script computes (`steadyState`). The Lagrange remainder is evaluated on the two subsystems of
[Althoff, TPS 2014] (the buses 1, 2, 6 and 3, 8, with the other buses as inputs, CORA's `indexForSubsystems`, ported in
`benchmark_power_system.h`), the sets are those of the whole system, as the report describes. The equations are those
`powerSystem2cora` writes (MATLAB R2024b with the Symbolic Math Toolbox), kept as text in `data/TSPS25.json` by
`examples/python/benchmarks/power_system_data.py`; the program parses them. The end set of 5 s is, in the (x1, x6) plane,
x1 in [0.32976, 0.33330] and x6 in [376.980, 377.037] (the report's figure shows x1 in [0.15, 0.55], x6 in [374, 380]).
MATLAB CORA v2026.1.0 (R2024b, this machine, `CHECKS_ENABLED` off, a pool of 2 workers, the derivatives already
generated, load average 8 to 14) verifies it in 312 s (21, 18, 180 and 94 s for the four runs); a first run that generates
the models and the derivatives and starts a pool of 64 workers needs 644 s, so the report's 299 s is the warm run.
CORA.cpp runs the four runs in {secs} s on one thread (`NonlinDASys` does not run the subsystems in parallel, as
CORA's `parfor` does). **CORA.cpp is not CORA here, on purpose:** CORA's linearization error is unsound in three ways
(docs/cora-differences.md): it adds the algebraic error with `+ C F^-1` where the linearized equations have `- C F^-1`,
it uses each subsystem's own block `Csub*pinv(Fsub)` where the coupling of the subsystems needs the one of the whole
system, and it reads the Hessian on a box that need not contain the linearization point. CORA.cpp follows CORALean's
corrections of all three, so its end set is wider: the widths of x1 and x6 at 5 s are 0.00353 and 0.0568, MATLAB's 0.00273
and 0.0449 (the widths of all 15 states are 1.1 to 1.3 times CORA's). With CORA's three choices switched on in a local
experiment (not in the tree) CORA.cpp's end set of the first run agrees with MATLAB's to 10 digits in the center
(0.333360069691 against 0.333360069687) and to 6 to 7 digits in the widths (x6: 0.429442676 against 0.429442345), and at
5 s the widths are those of MATLAB (0.002734 and 0.044901 against 0.0027 and 0.0449) with x1 in [0.330160, 0.332894]
and x6 in [376.986408, 377.031309] (MATLAB: [0.3302, 0.3329] and [376.9864, 377.0313]): the rest of the algorithm is
CORA's. The tests of `NonlinDASys` fail with those three switched on (the trajectories of x' = x^2 leave the sets).
A step beyond `maxError_x` or `maxError_y` splits the set in CORA, but its split reads an undefined variable and aborts;
the power system never exceeds them, and CORA.cpp refuses such a step.
**LOVO25** (`benchmark_hybrid_reach_ARCH23_lotkaVolterra.m`) is the predator-prey system with a clock, started in
x1 in [1.288, 1.312], x2 = 1, x3 = 0 up to t = 3.64, and a ring of radius 0.161 around (1, 1): the invariant of
'inside' and the **level set** guard (`LevelSet`, `==`) of the jump from 'outside'. The orbit meets the ring
tangentially in its upper arc and fully in the lower arc. The hybrid analysis (alg "lin", tensor order 3, 165
steps) enclosed each guard intersection by `guardIntersect_levelSet` (the box of the sets contracted to the
ring, then `LevelSet::intersect`, the second-order enclosure of Kochdumper et al., HSCC 2020, as a polynomial
zonotope), and stops where a branch has left the ring; the script follows each branch with the continuous
dynamics (the pieces that meet the ring are split along their longest generator, with the clock scaled by
1/1000) and intersects the final sets with the plane x3 = 3.64. The specification of the report has four parts:
the box x1 x x2 of all final sets has an area below 1e-2 (8.92e-3 here), one final set crossed the ring twice
(the branch of the full crossing: entry and exit), one four times (the tangential crossing, then the re-entry:
two branches of 'inside'), and none an odd number of times. The script is kept as `LOVO25-script` and records 1 without testing
the sets. `LOVO25` instead gives the automaton a jump back out of `inside` and reaches 8 pieces of X0 with no restart and no cut;
the piece with the tangential arc makes the linearization error explode and is enclosed by the flow alone, which is sound
because the resets are the identity and both locations have the same dynamics. The area and the location of the sets at
x3 = 3.64 are checked, and the runs with exactly 2 and exactly 4 crossings are simulated trajectories. CORA.cpp
ports the unsolvable branch of `levelSet/and_.m` only (no symbolic solver); CORA also tries the solvable branch,
which the ring can take where one of its two roots lies outside the box, and keeps the smaller error
(docs/cora-differences.md); the areas agree to 0.1%. MATLAB CORA v2026.1.0 (R2024b, this
machine, `CHECKS_ENABLED` off, load average about 10) needs 11.4 s and gives the area 8.916e-3.

## AINNCS results

The ARCH-COMP AINNCS instances of MATLAB CORA (`examples/contDynamics/neurNetContrSys/example_neuralNet_reach_*.m`,
the ARCH25 settings of `run_arch_ainncs_category.m`), run through CORA's `verify` by
`benchmark_neur_net_contr_reach_arch26_ainncs`: Unicycle, ACC, TORA (remain, reach-tanh, reach-sigmoid),
Single Pendulum, Double Pendulum (less-robust, more-robust), Airplane and Attitude Control. The
plants are `NonlinearSys` written out as the model files write them, the networks are
`data/networks/<name>.nn.json` and `.nn.bin` (the layers, their activation and the doubles, written from
the example scripts by `examples/matlab/benchmarks/export_network_data.m`, so the numbers are CORA's).
The expected verdict is that of the specification of the ARCH-COMP26 AINNCS report (Sasaki et al., "ARCH-COMP26
Category Report: Artificial Intelligence and Neural Network Control Systems (AINNCS) for Continuous and
Hybrid Systems Plants", Tables 1 to 4 for CORA, CROWN-Reach, immrax and JuliaReach; probably the same
kind of machine as CORA.cpp's), and the program counts the
instances that differ from CORA's verdict. The time is `verify`: the random simulations (20 runs
here, 10 in CORA) and the reach, with the derivation of the plant's Jacobians and Hessians. The MATLAB
example scripts ran once after `rng(1)`. A verdict `F` is FALSIFIED by a simulated run that the reach from its start
point does not verify, `U` is UNKNOWN, `V` VERIFIED.

| instance | expected (ARCH26) | CORA.cpp | CORA.cpp [s] | oracle check (rows) | CORA [s] | CORA ARCH26 [s] | best ARCH26 [s] (tool) | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unicycle reach | V | V | 0.41 | 21/54 bit-exact, 8 property | 8.87 | 8.015 | 7.093 (JuliaReach) | 17.3 |
| Unicycle reach, with the report's disturbance w | V | V | 0.4 | 21/54 bit-exact, 8 property | - | 8.015 | 7.093 (JuliaReach) | 17.7 |
| ACC safe-distance | V | V | 0.24 | 21/60 bit-exact, 13 property | 4.2 | 3.176 | 0.492 (JuliaReach) [a] | 2.05 |
| TORA remain | V | V | 1.22 | 21/61 bit-exact, 14 property | 9.03 | 8.998 | 2.852 (CROWN-Reach) | 2.34 |
| TORA reach-tanh | V | V | 0.15 | 21/60 bit-exact, 13 property | 2.28 | 2.356 | 0.115 (JuliaReach) [a] | 0.77 |
| TORA reach-sigmoid | V | V | 0.28 | 21/61 bit-exact, 14 property | 5.09 | 4.73 | 0.146 (JuliaReach) [a] | 0.52 |
| Single Pendulum reach | V | V | 0.041 | 21/60 bit-exact, 13 property | 4.03 | 4.083 | 4.083 (CORA) [b] | 99.6 |
| Double Pendulum less-robust | V (CORA: U) | U | 0.13 | 21/54 bit-exact, 8 property | 5.06 | 5.604 | 67.334 (CROWN-Reach) | - |
| Double Pendulum more-robust | F | F | 0.11 | 21/60 bit-exact, 13 property | 4.76 | 5.589 | 0.394 (JuliaReach) | 3.58 |
| Airplane continuous | F | F | 0.53 | 21/60 bit-exact, 13 property | 6.78 | 6.809 | 1.923 (JuliaReach) | 3.63 |
| Attitude Control avoid | V | V | 0.42 | 21/62 bit-exact, 14 property | 5.71 | 5.226 | 5.226 (CORA) [c] | 12.4 |

[a] ACC, TORA reach-tanh and reach-sigmoid: immrax (0.120, 0.040 and 0.043 s) analyses continuous feedback, as the report states, instead of sample-and-hold control.

[b] Single Pendulum: CROWN-Reach's public code (commit 7b90f30, 0.647 s) checks an empty unsafe set (x1 <= 0 and x1 >= 1 at once) instead of theta outside [0, 1].

[c] Attitude Control: CROWN-Reach's public code (commit 7b90f30, 3.237 s) has an empty unsafe box (x4 >= -0.4 and x4 <= -0.6, where the report has x4 in [-0.7, -0.6]).

Double Pendulum less-robust: CORA.cpp's time is for an UNKNOWN, the best time (CROWN-Reach) for a verification.
TORA remain runs with time_step 0.1 and reach-sigmoid with time_step 0.025 (below).

**The loop.** `NeurNetContrSys` is CORA's: per control period the network is enclosed over the current set
together with its input (`NeuralNetwork::evaluate_with_input`), the plant extended by the held outputs
as states of zero derivative (built from the plant's own expressions) is reached from that set by
`NonlinearSys::init_reach` and `post` (alg "lin"), the step size cut to the period as CORA's `tVec`,
and the set at the end of the period projected to the states starts the next. The network is
enclosed by CORA's `poly_method` "singh" of order 1 (the tightest parallelogram of a ReLU,
[Singh et al., NeurIPS 2018, Thm. 3.1 and 3.2] for sigmoid and tanh); on a polynomial zonotope the
factors and exponents stay. The Unicycle, Double Pendulum less-robust start from zonotopes, the
others from polynomial zonotopes, as the scripts. The sets, the boxes and the sets over the
steps agree with MATLAB CORA's to five digits over 50 control periods (TORA reach-tanh, checked at
t = 0.5, 1, ..., 5).

**Soundness, step by step** (docs/cora-differences.md has the rows).

- *The network enclosure* (`computeApproxPoly` "singh" order 1, `evaluatePolyZonotope`): sound for the
  layers here, in exact arithmetic. The ReLU parallelogram is exact outside the crossing
  (CORALean `reluRelaxation_encloses`); for sigmoid and tanh the derivative rises and then falls, so it
  is at least the smaller value at the ends and the two lines of that slope through the ends
  enclose the function (`Real.deepZSShape_encloses`; CORA's transformer is not sound for a non-S-shaped
  monotone function, which `nnActLayerFromHandle` allows and CORA.cpp does not). The rounding is not
  directed: CORA.cpp's float arithmetic is at nearest, like the oracle's `nearest`, so a verdict is
  not a proof (the radius is not rounded up, and one below 0 by rounding is dropped as CORA's `d > 0`).
- *The generator sharing* (`aux_appendU2X`): CORA's stacking is right only while the first
  generators of the network's output are the images of the input's, which singh without order
  reduction gives and its source notes ('only works if no order reduction is applied within nn');
  CORALean says the same (it needs an image that keeps the generators, more than `enclose`
  promises). CORA.cpp carries the input rows through the layers, so it holds by construction.
- *The plant reach*: `NonlinearSys` alg "lin" on the extended system, with the Hessian region joined
  with the linearization point (the repair of `priv_abstrerr_lin`); the sets stay sound at tensor order 3,
  looser than CORA's on polynomial zonotopes.
- *The specification check*: exact on the zonotope (a polynomial zonotope on its zonotope
  enclosure); a falsification is a simulated run (Runge-Kutta, steps of at most 1e-3) from a point of
  the initial set whose reach from that point does not verify, as CORA's, so it is as sound as
  the integration is accurate.

None of the rows is bit-exact against CORALean: its oracle serves no network or controlled-system op
(docs/oracle-requests.md), and the nonlinear plants have no CORALean reach beyond the argument of
`reachWith`.

**Tuned settings.** Two instances need a smaller step than the script's: with the script's 0.25 (TORA
remain) and 0.1 (TORA reach-sigmoid) the sets end just outside the specification (the lower bound of
x2 at t = 5 is -0.906 for the goal -0.9). CORA's tensor order 3 on a polynomial zonotope keeps the
quadratic error term polynomial (`quadMap` of the polynomial zonotope), CORA.cpp's step encloses
the set of the step by a zonotope first (`priv_lin_reach_poly`), which is looser but sound
(docs/cora-differences.md). The settings that verify are 0.1 and 0.025 (others: 0.05 fails for the
sigmoid, 0.1 and 0.05 verify the remain). The Unicycle row "with the report's disturbance" adds
w in 1e-4 [-1, 1] to the speed, which the report states and the script sets to 0; it verifies too.
The Double Pendulum less-robust reach stops at t = 0.05, where the set leaves the safe box (CORA's
is also UNKNOWN); the falsifications are the simulated runs
`x0 = (1.3, 1, 1, 1)` (more-robust: x3 = -1.510 < -1.5 at t = 0.32) and the Airplane run from
`(0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0)` (y = 1.24 > 1 at t = 0.6).

`CORACPP_BENCHMARK_VERBOSE=1` prints the box of every set of the reach, and
`CORACPP_AINNCS_OPTIONS=time_step,tensor_order,taylor_terms,zonotope_order,error_order,intermediate_order`
replaces the settings of the instance (how the two tuned steps were found).

**Not ported here.** VCAS, QUAD, Docking, NAV and CartPole (piece B): the VCAS instances need CORA's
custom discrete algorithm, and the scripts of QUAD, Docking, NAV and CartPole are not all in the CORA
checkout. The plants of `NeurNetContrSys` are `NonlinearSys` only; `LinearSys`, `LinParamSys`
and the discrete-time systems of CORA's class have no instance here, and CORA.cpp has no
discrete-time systems yet. `verify` does not split the initial set (`splitR0` of CORA).


## Run

```bash
cmake --build --preset core                                   # build/core/benchmarks/benchmark_linear_*
build/core/benchmarks/benchmark_linear_verify_fast_arch23_iss [ISSF01_ISS01 ...]
build/core/benchmarks/benchmark_hybrid_reach_arch23_rendezvous [SRA01 ...]   # CORACPP_BENCHMARK_VERBOSE=1: the branches
build/core/benchmarks/benchmark_nonlinear_reach_arch23_trafficscenario         # TRAF22
build/core/benchmarks/benchmark_nonlinear_reach_arch23_robertson [ROBE25_1 ...] # CORACPP_BENCHMARK_VERBOSE=1: step sizes
build/core/benchmarks/benchmark_hybrid_reach_arch23_vanderpol                   # CVDP23, about a minute
build/core/benchmarks/benchmark_nonlinear_reach_arch23_laubloomis [LALO20_W001 ...]
build/core/benchmarks/benchmark_nonlinear_das_reach_arch23_powersystem         # TSPS25, about a minute
build/core/benchmarks/benchmark_hybrid_reach_arch23_lotkavolterra               # LOVO25, CORACPP_BENCHMARK_VERBOSE=1: the branches
build/core/benchmarks/benchmark_neur_net_contr_reach_arch26_ainncs [TORA-remain ...]   # the AINNCS instances, 3 s in all
build/core/benchmarks/benchmark_linear_verify_arch26_rnd [RND01 ...]       # RND03 takes about 20 minutes (zonotope algorithm)
PYTHONPATH=build/python python examples/python/benchmarks/benchmark_linear_verifyFast_ARCH23_iss.py [instance ...]
```

The programs need `LinearSys::verify` (CMake builds them once `linearSys.h` declares it). Without
instance arguments a program runs its whole family. Compare with MATLAB by running the original
script in CORA (`benchmark_linear_verifyFast_ARCH23_iss_ISSF01_ISS01`) and diffing the lines.
`CORACPP_BENCHMARK_DATA` names another data folder.

## Data

`examples/benchmarks/data/` holds the instances, about 3.5 MB in all, written by MATLAB (HEAT03:
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

HEAT04 has no data files: its `A` (860 000 entries, 14 MB) is built from the mesh by `aux_mesh` in
`benchmark_linear_reach_arch23_heat3d.cpp`, the same construction as `heat3d_data.py`, and equals `heat50.mat`.

HEAT03 is not exported by MATLAB: CORA's `models/Cora/heat3D/heat20.mat` is a Git LFS file and
MATLAB is not available here. `heat3d_data.py` builds `A` from the finite differences of the
benchmark (Bak, Tran, Johnson, HSCC 2019; see its header), which reproduce the exported `A` of
HEAT01 (heat5) and HEAT02 (heat10) bit for bit (`--check`), and `R0` and `C` from the script;
`linAlg` replaces `verifyAlg`, and `specs` is empty (the goal of the maximum is in the program).

RND01-06 are not exported by MATLAB either: `examples/python/benchmarks/rnd_data.py convert rand0N.json RND0N`
reads the files of the ARCH-COMP repository (CORA's `json2cora_linearSys` and `zonotope(interval)`), `generate RND06` writes
the 5000-state instance that the repository lacks; `verifyAlg` is `reachavoid:zonotope` or `reachavoid:krylovSuppFunc`,
`source` and `note` say where the numbers come from, and the unsafe set `y >= lb` is the polytope `-y <= -lb`.

A scalar `B` of the beam scripts is stored as `B*I`; an absent `params.U` (ISSC01_ISU02) as `{0}`.
Unsafe sets with several halfspaces (RAND) need a `Specification` that takes them; the programs
stop with an error until then. Test: `python -m unittest tests/python/test_benchmarkData.py`.

The networks of the AINNCS instances are in `data/networks/`: `<name>.nn.json` lists the layers (`out`, `in`,
`activation` identity, relu, sigmoid or tanh, and the offsets of `W` and `b` in doubles) and
`<name>.nn.bin` the little-endian doubles (`W` row by row, then `b`).
`examples/matlab/benchmarks/export_network_data.m` writes them from the example scripts' networks (an
elementwise affine layer of an ONNX network as the layer with `W = diag(scale)`).

## What CORALean backs

The lean backend is bit-exact against CORALean's `nearest` oracle (deliberately unsound rounding) on
the tests' inputs. That pins the algorithms to CORALean's float definitions; it does not make the
results sound, and a verdict above is not a proof (RAND01's is also proved by the oracle at a sound dtype:
`benchmark_sound_verdict_arch23`; hybrid and nonlinear have no such check yet). Eigen and torch agree with lean within `tol`. The
table counts, per instance, the rows it executes whose test is bit-exact; a row that CORALean marks out
of scope cannot be, and is left out of the denominator (`coverage/benchmarks.md` has the rest).

<!-- BEGIN bit-exact -->
| instance | bit-exact vs CORALean | not bit-exact | out of scope (CORALean or CORA.cpp) |
|---|---|---|---|
| ACC-safe-distance | 21/60 | 39 | 11 |
| Airplane-continuous | 21/60 | 39 | 11 |
| Attitude_Control-avoid | 21/62 | 41 | 15 |
| BRKDC01 | 4/12 | 8 | 5 |
| BRKNC01 | 31/54 | 23 | 10 |
| BRKNP01 | 24/44 | 20 | 10 |
| CBC01 | 1/8 | 7 | 4 |
| CBC02 | 1/8 | 7 | 4 |
| CBC03 | 1/8 | 7 | 4 |
| CBF01 | 1/8 | 7 | 4 |
| CBF02 | 1/8 | 7 | 4 |
| CBF03 | 1/8 | 7 | 4 |
| CVDP23 | 30/70 | 40 | 11 |
| DTN01 | 21/37 | 16 | 6 |
| DTN02 | 21/37 | 16 | 6 |
| DTN03 | 21/37 | 16 | 6 |
| DTN04 | 21/37 | 16 | 6 |
| DTN05 | 21/37 | 16 | 6 |
| DTN06 | 21/37 | 16 | 6 |
| Double_Pendulum-less-robust | 21/54 | 33 | 11 |
| Double_Pendulum-more-robust | 21/60 | 39 | 11 |
| GRBX01 | 30/56 | 26 | 10 |
| GRBX02 | 30/56 | 26 | 10 |
| HEAT01 | 1/8 | 7 | 4 |
| HEAT02 | 1/8 | 7 | 4 |
| HEAT03 | 10/17 | 7 | 4 |
| HEAT04 | 10/17 | 7 | 3 |
| ISSC01_ISS02 | 1/8 | 7 | 4 |
| ISSC01_ISU02 | 1/8 | 7 | 4 |
| ISSF01_ISS01 | 1/8 | 7 | 4 |
| ISSF01_ISU01 | 1/8 | 7 | 4 |
| LALO20_W001 | 10/24 | 14 | 7 |
| LALO20_W005 | 10/24 | 14 | 7 |
| LALO20_W01 | 7/35 | 28 | 6 |
| LOVO25 | 24/75 | 51 | 13 |
| LOVO25-script | 27/82 | 55 | 14 |
| PLAA01_BND42 | 23/35 | 12 | 9 |
| PLAA01_BND50 | 23/35 | 12 | 9 |
| PLAD01_BND30 | 9/17 | 8 | 5 |
| PLAD01_BND42 | 9/17 | 8 | 5 |
| PLAN01 | 31/45 | 14 | 9 |
| RAND01 | 5/13 | 8 | 4 |
| RAND02 | 5/13 | 8 | 4 |
| RND01 | 5/13 | 8 | 4 |
| RND02 | 2/11 | 9 | 4 |
| RND03 | 5/13 | 8 | 4 |
| RND04 | 2/11 | 9 | 4 |
| RND05 | 2/11 | 9 | 4 |
| RND06 | 2/11 | 9 | 4 |
| ROBE25_1 | 10/24 | 14 | 7 |
| ROBE25_2 | 10/24 | 14 | 7 |
| ROBE25_3 | 10/24 | 14 | 7 |
| SPRE22 | 37/65 | 28 | 13 |
| SRA01 | 26/46 | 20 | 8 |
| SRA02 | 34/56 | 22 | 10 |
| SRA03 | 34/56 | 22 | 10 |
| SRA04 | 26/46 | 20 | 8 |
| SRA05 | 34/56 | 22 | 10 |
| SRA06 | 34/56 | 22 | 10 |
| SRA07 | 42/67 | 25 | 11 |
| SRA08 | 42/67 | 25 | 11 |
| SRNA01 | 23/40 | 17 | 7 |
| SRU01 | 27/47 | 20 | 8 |
| SRU02 | 42/67 | 25 | 11 |
| Single_Pendulum-reach | 21/60 | 39 | 11 |
| TORA-reach-sigmoid | 21/61 | 40 | 11 |
| TORA-reach-tanh | 21/60 | 39 | 11 |
| TORA-remain | 21/61 | 40 | 11 |
| TRAF22 | 15/30 | 15 | 9 |
| TSPS25 | 17/32 | 15 | 6 |
| Unicycle-reach | 21/54 | 33 | 11 |
| Unicycle-reach-disturbance | 21/54 | 33 | 11 |
<!-- END bit-exact -->
