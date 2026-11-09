# Oracle requests

Operations that CORA.cpp has ported and CORALean has float definitions of, but that the oracle
(`oracle-unsound`, `ping`) does not serve yet. Each entry names the Lean definition, the request it
would take and the reply it would give, in the oracle's JSON style (experiments/oracle/README.md in
CORALean: matrices `{"rows","cols","data"}`, zonotopes `{"c","G","E"}`, polytopes `{"A","b","Ae","be"}`).
Once an op is served, its CORA.cpp test moves to `tests/global/oracle/` with `// ORACLE:` and the
entry is removed. The coverage rows of these operations are oracle debt until then (`coverage.py`).

CORALean's hybrid definitions are its own sound float algorithms (for example the cut of
`guardIntersectZonoGirardFloat` is a dual bound, not Girard's dichotomic search), so a bit-exact
test needs the op to follow CORA's steps, or a CORA.cpp test that compares by containment.

## Hybrid automata (PR "hybrid automata")

| op (proposed) | CORALean definition (file under `src/CORALean/HybridDynamics/Operations/`) | request fields | reply |
|---|---|---|---|
| `location.potInt` | `Location.potInt` (`LocationPotInt.lean`) | `guards` (polytopes), `targets`, `finalLoc`, `R` (zonotopes) | `hits`: `[[transition, set], ...]` (0-based) |
| `location.calcBasis` | `Location.calcBasisBox` (`CalcBasisBox.lean`; `pca` and `flow` have no float definition yet) | `n`, `enclose` (`"box"`) | `B` (matrix) |
| `location.checkFlow` | `Location.checkFlowFloat` (`CheckFlowFloat.lean`) | the location's `A`, `B`, `c`, `U`, the hyperplane `c`, `d`, `R` (zonotopes) | `keep`: indices of `R` |
| `location.guardIntersect_zonoGirard` | `Location.guardIntersectZonoGirardFloat` (`GuardIntersectZonoGirardFloat.lean`), its bound `Location.zonoGirardUpperFloat` (`ZonoGirardUpperFloat.lean`) | `Bs` (matrices), `R0`, `R` (zonotopes), `n` (normal), `gamma` | `zB` (zonotope bundle) |
| `location.guardIntersect_nondetGuard` | `Location.guardIntersectNondetFloat` (`GuardIntersectNondetFloat.lean`) | `Bs`, `R0`, `R`, `guard` (constrained zonotope) | `zB` |
| `location.guardIntersect_conZonotope` | `Location.guardIntersectPolytopeFloat` (`GuardIntersectPolytopeFloat.lean`, with `ConZonotope.intersection` of `ContSet/ConZonotope/Float/Operations/Intersection.lean`) | `Bs`, `R0`, `R`, `guard` (constrained zonotope), `guardOrder` | `zB` |
| `location.guardIntersect` | `Location.guardIntersectFloat` (`GuardIntersectFloat.lean`, the `box` method) | the location's invariant, `guard` (polytopes), `R` | `I` (box) or `null` |
| `transition.evaluate` | `Transition.jumpFloat` (`JumpFloat.lean`) | `W` (matrix), `w` (vector), `Rg` (zonotope) | `Z`; CORA.cpp tests it now through `zonotope.mtimes` and `zonotope.plus` (`test_lean_linear_reset_evaluate`) |
| `location.reach` | `Location.reachFloat` (`LocationReachFloat.lean`) | the location's `A`, `B`, `c`, `U`, `X0`, `tFinal`, `timeStep`, `taylorTerms`, `zonotopeOrder` | `timeInterval`: array of zonotopes, or `null` |
| `hybridAutomaton.reach` | `HybridAutomaton.reachFloat` (`ReachFloat.lean`) | the locations (as above, with `inv` and `trans`: `guard`, `W`, `w`, `target`), `startLoc`, `X0`, `K` (jumps), `tFinal`, the reach options | `sets`: `[[location, zonotope], ...]` or `null` |

`hybridAutomaton.reach` and `location.reach` of CORALean bound every stay by `tFinal` and the number
of jumps by `K`, where CORA stops a stay when the sets leave the invariant; the bit-exact test would
compare the time-interval sets of one stay.

## Linear parametric systems (PR "platoon")

CORALean's float definitions of `LinParamSys` (`src/CORALean/ContDynamics/LinParamSys/Operations/`
and `Definitions/`) are its own sound algorithms, not CORA's steps: they enclose `e^{[A]t}` over
the interval matrix of the set with a Taylor order searched until it is admissible, by scaling
and squaring, and map window `k` from the first one (`[e^{[A] t_k}] H^R(τ_0)`) where CORA's `post`
maps the previous set by `eZ + eI`; their coefficients `r^i/i!` are bracketed (`taylorCoeffBounds`)
where CORA rounds to nearest. None is bit-exact with CORA once served; the two closest are marked.
A bit-exact test needs ops that follow `initReach`/`post` of CORA (the request in the last row).

| CORALean definition (file) | what it is | follows CORA? |
|---|---|---|
| `LinParamSys.reachFloat` (`ReachFloat.lean`), `reachFloatChecked` (`ReachFloatChecked.lean`) | the homogeneous windows over the horizon, `U = {0}` | own: windows from the first by `reachStepkFloat`, order by `reachOrderFloat` |
| `LinParamSys.reachFloatInput` (`ReachFloatInput.lean`) | the windows with `V = B U ⊕ c`, split at its center | own: `constPropagateFloat` and `particularSolutionFloat` instead of `Rinput` per step |
| `LinParamSys.timeIntervalFloat` (`TimeIntervalFloat.lean`) | `linComb(R, [e^{AΔt}] R) ⊕ [F] R` over the interval matrix | own (CORA: `enclose(R, M1 R + M2 R + Rtrans) + F R`, with a matrix zonotope) |
| `LinParamSys.timeIntervalZonoFloat` (`TimeIntervalZonoFloat.lean`) | the same with the exponential as a matrix zonotope (`expMatZonoEnclosure`, `matZonoCorrectionMatrix`) | closest to `initReach`: same structure, but `linComb` and its own `F` and series split |
| `LinParamSys.inputSolutionZonoFloat` (`InputSolutionZonoFloat.lean`), `inputTermZonoFloat` (`InputTermZonoFloat.lean`) | the input solution: orders 0 and 1 as matrix zonotopes, 2..η and `E r` over the interval matrix, "the four pieces stay four products, as in CORA's `priv_inputSolution`" | closest to `priv_inputSolution`: CORA keeps orders 1..intermediateTerms (all η for one generator) as matrix zonotopes and rounds the coefficients to nearest |
| `LinParamSys.inputSolutionFloat` (`InputSolutionFloat.lean`), `inputTaylorSumFloat`, `inputTaylorSumTailFloat` | the input solution over the interval matrix | own |
| `LinParamSys.constInputSolutionFloat`, `constPropagateFloat`, `particularSolutionFloat` | the constant and time-varying input across windows | own (CORA: `Rtrans`, `inputCorr` of `priv_inputTie`) |
| `LinParamSys.expIntervalFloat`, `expSquaringFloat`, `squaringsFloat`, `reachOrderFloat`, `IsOrderAdmissibleFloat` | `[e^{At}]` over a time range, its squarings and the admissible order | own (CORA: `expmOneParam`, `expmMixed`, `expmIndMixed` at a fixed order) |
| `LinParamSys.reachStepkFloat` (`ReachStepkFloat.lean`) | window `k`: `[e^{[A] t_k}]` times the first window | own (CORA's `post` maps the previous set) |
| `LinParamSys.discretizeEnclosureFloat` (`DiscretizeEnclosureFloat.lean`) | the sampled `LinParamSysDT` | no CORA counterpart in reach |

| op (proposed) | request fields | reply |
|---|---|---|
| `linParamSys.initReach` (CORA's steps, as CORA.cpp's `init_reach`) | `A` (matrix zonotope `{"C","G"}`), `B`, `c`, `type` (`"constParam"`/`"varParam"`), `R0`, `U` (zonotopes), `uTrans`, `timeStep`, `taylorTerms`, `intermediateTerms`, `zonotopeOrder`, `compTimePoint` | `ti`, `tp` (zonotopes) and the mappings `M1` (matrix zonotope), `M2`, `F` (interval matrices), `Rinput`, `Rtrans`, `inputCorr` |
| `linParamSys.post` | the mappings of `initReach`, `ti`, `tp`, `zonotopeOrder` | `ti`, `tp` |

The coverage rows `linParamSys`, `initReach` and `post` are oracle debt until then.

## Krylov subspace reach (PR "HEAT03")

CORALean's `ContDynamics/LinearSys/Krylov/` has no float definitions: its operations are over ℝ,
so the oracle serves none of them (no `krylov` op in `ping`). They follow the paper (Althoff, TAC
2020), not CORA's code, except for the Arnoldi iteration:

| CORALean definition (file under `Krylov/Operations/`) | what it is | follows CORA? |
|---|---|---|
| `Krylov.arnoldi` (`Arnoldi.lean`), `gramSchmidtStep` (`GramSchmidtStep.lean`), `arnoldiBasis`, `arnoldiHessenberg` | Alg. 1: `v₁ = v/‖v‖`, `w = C v_k` swept against the earlier vectors one by one, `h_{k+1,k} = ‖w‖`; the happy breakdown is the caller stopping | yes, step for step as `arnoldi.m` (modified Gram-Schmidt), over ℝ |
| `Krylov.augment` (`Augment.lean`) | the integrator matrix `[A u; 0 0]` | yes (`A_int` of `priv_krylov_R_uTrans`, `priv_inputSolution_Krylov`) |
| `Krylov.alg2` (`Alg2.lean`) | Alg. 2 as printed, its subroutines as parameters | the paper's listing, which CORA's `priv_reach_krylov` implements (with its own handling of the inputs) |
| `Krylov.alg2Corrected` (`Alg2Corrected.lean`) | Alg. 2 with lines 8 and 10 repaired | own (sound); CORA keeps the printed lines |
| `Krylov.correctionSet` (`Definitions/CorrectionSet.lean`) | the set of correction matrices of `H` | a set, not CORA's interval matrix `F` |

A bit-exact test of CORA.cpp's `alg::Krylov` needs float ops that follow CORA's code:

| op (proposed) | request fields | reply |
|---|---|---|
| `krylov.arnoldi` (CORA's `arnoldi.m` in float) | `A` (matrix, or coordinates), `v`, `redDim` | `V`, `H`, `Hlast`, `happyBreakdown` |
| `krylov.subspace` (`priv_subspace_Krylov_individual_Jawecki`) | `A`, `v`, `initKrylovOrder`, `krylovError`, `krylovStep`, `tFinal` | `V`, `H`, `krylovOrder` (with CORA's `t^k/k!` in double) |
| `linearSys.reachKrylov` (`priv_initReach_Krylov`, `priv_reach_krylov`) | `A`, `B`, `C`, `R0`, `U`, `tFinal`, `timeStep`, `taylorTerms`, `zonotopeOrder`, `krylovError`, `krylovStep` | `timeInterval`, `timePoint` (output zonotopes) |

The Krylov path is a private algorithm of `reach`, so no coverage row changes; its tests are unit and
property tests (`test_linear_sys_reach_krylov`) until these ops exist.
