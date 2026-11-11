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
| `location.guardGroups` | `Location.guardGroups` (`GuardGroups.lean`: CORA's `aux_groupSets` and `aux_removeGaps`) | `hits`: `[[transition, set], ...]` | `groups`: per crossing `transition`, `sets`, `minInd`, `maxInd`, in CORA's order |
| `location.calcBasis` | `Location.calcBasisBox` (`CalcBasisBox.lean`; `pca` and `flow` have no float definition yet) | `n`, `enclose` (`"box"`) | `B` (matrix) |
| `location.checkFlow` | `Location.checkFlowFloat` (`CheckFlowFloat.lean`) | the location's `A`, `B`, `c`, `U`, the hyperplane `c`, `d`, `R` (zonotopes) | `keep`: indices of `R` |
| `location.guardIntersect_zonoGirard` | `Location.guardIntersectZonoGirardFloat` (`GuardIntersectZonoGirardFloat.lean`), its bound `Location.zonoGirardUpperFloat` (`ZonoGirardUpperFloat.lean`) | `Bs` (matrices), `R0`, `R` (zonotopes), `n` (normal), `gamma` | `zB` (zonotope bundle) |
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

## Nonlinear systems, alg "lin" (PR "port/nln-lin")

CORALean's float definitions of `NonlinearSys` (`src/CORALean/ContDynamics/NonlinearSys/Operations/`)
take the dynamics as a float model (`FloatModel`, `FloatHessModel`: enclosures of `f`, its Jacobian
and its Hessian over boxes), which a JSON request cannot carry as an expression yet. Of the steps
CORA.cpp ports, only `linPointFloat` follows CORA step for step; the others are CORALean's own sound
variants. A bit-exact test needs a request format for the dynamics (an expression tree, as
CORA.cpp's `Expr`) and ops that follow `linReach`.

| CORALean definition (file) | what it is | follows CORA? |
|---|---|---|
| `NonlinearSys.FloatModel.linPointFloat` (`LinPointFloat.lean`) | the expansion point `c + f(c, u*) dt/2`, `u*` the center of `U` | yes, CORA's `linearize` (the flow at `c` is the midpoint of the model's enclosure there) |
| `NonlinearSys.FloatModel.linearizeFloat` (`LinearizeFloat.lean`) | `A`, `B` at the expansion point: the centers of the model's Jacobian enclosures | own (CORA: the symbolic Jacobian at the point) |
| `NonlinearSys.FloatHessModel.hessRemainder` (`HessRemainder.lean`) | `l_i = 1/2 dz' max\|H_i\| dz`, plus the first-order slack of the rounded `A`, `B` | close to `priv_abstrerr_lin` at tensorOrder 2; the slack term and the outward rounding are its own |
| `lagrangeRemainderFloat` (`LagrangeRemainderFloat.lean`) | `1/6 sum_j g_j (g' E_ij g)` with `g` the box radius, the sixth as 11/64 | own (CORA, tensorOrder 3: the interval product `(dz' T_ij dz) dz_j` with `dz` the box, not its radius, and a quadratic set `1/2 quadMap(Z, H)`) |
| `NonlinearSys.oneStepFloat` (`OneStepFloat.lean`) | one affine step with the input `V = f(z*) + B (U - u*) + L` | own (CORA: `oneStep` of the linear system, then the error solution by `particularSolution_timeVarying`) |
| `NonlinearSys.FloatHessModel.linReachFloat` (`LinReachFloat.lean`) | one step: the region checked first, a single pass | own (CORA: the `while perfIndCurr > 1` loop that enlarges the guessed error) |
| `NonlinearSys.FloatModel.reachFloat` (`ReachFloat.lean`) | the steps over the horizon | own, by the steps above |

| op (proposed) | request fields | reply |
|---|---|---|
| `nonlinearSys.linearize` | `f` (expression trees of `x`, `u`), `R`, `U` (zonotopes), `timeStep` | `x_star`, `u_star`, `f0`, `A`, `U` (zonotope), `u_const` |
| `nonlinearSys.abstrerr_lin` | `f`, `R` (zonotope, relative to `x_star`), `x_star`, `u_star`, `U`, `tensorOrder`, `errorOrder`, `intermediateOrder` | `trueError` (vector), `VerrorDyn` (zonotope) |
| `nonlinearSys.linReach` (CORA's steps, as CORA.cpp's `priv_lin_reach`) | `f`, `R`, `error`, `U`, `timeStep`, `taylorTerms`, the orders | `ti`, `tp` (zonotopes), `error` |
| `nonlinearSys.reach` | `f`, `R0`, `U`, `tFinal`, the options of `alg::Linearization` | `timeInterval`, `timePoint`: arrays of zonotopes |

## Nonlinear systems, alg "lin-adaptive" (PR "port/nln-adaptive")

CORA.cpp's `alg::LinearizationAdaptive` is CORA's lin-adaptive as CORALean repairs it
(`reachAdaptiveFloat`, CORALean !906). The oracle serves its ops from the pinned tag
`oracle-20261111-f9723f6f6` on (`lean::NonlinearSys`, `include/cora/global/oracle/nonlinear_sys.h`);
an older oracle makes the tests skip them.

| op | test | agreement |
|---|---|---|
| `nonlinearSys.varphiTotalAdaptive`, `varphiEstAdaptive`, `optimalDeltatAdaptive` | `test_lean_nonlinear_sys_adaptive_tuning` | bit for bit |
| `zonotope.reduce` with method `adaptive` | `test_lean_nonlinear_sys_adaptive_tuning` | bit for bit |
| `nonlinearSys.abstrErrAdaptive` | `test_lean_nonlinear_sys_adaptive_pass` | bit for bit at tensor order 2; order 3 inside the oracle's box (quadMap, see the differences) |
| `nonlinearSys.ordersAdaptive` | `test_lean_nonlinear_sys_adaptive_pass` | exact |
| `nonlinearSys.linReachAdaptivePass` | `test_lean_nonlinear_sys_adaptive_pass` | `xs`, `A`, the zero generators exact; the sets inside the oracle's, 5e-3 tighter at most |
| `nonlinearSys.reachAdaptive` | `test_lean_nonlinear_sys_reach_adaptive` | the same steps and tensor orders while the order is 2; the sets as in a pass |

Not covered yet: `nonlinearSys.linReachAdaptive` (the state-passing step; `reachAdaptive` runs it), and
the bit-exact linear step (CORALean's `reachFloatInput` is the interval-matrix exponential, CORA.cpp's
step is CORA's `oneStep`).

## Polynomial zonotopes and alg "poly" (PR "nln-poly")

The pinned oracle (`oracle-20261111-f9723f6f6`) serves no polynomial zonotope operation and no
`nonlinearSys.reach` with a polynomial zonotope. CORALean has float definitions of the operations
below (`src/CORALean/ContSet/PolyZonotope/Float/Operations/`). Their request is `{"op": ..., "S"
or "pZ": {"c", "G", "GI", "E", "id"}, ...}` with `E` the p x h exponent matrix and `id` the
identifiers, the reply a polynomial zonotope in the same style; a bit-exact test needs the order
of the generators to be CORA's or CORALean's, which CORA.cpp's constructor does not reproduce
(CORA merges equal exponents in the order of a hash, CORA.cpp in the order of first occurrence).
All rows are oracle debt until served; none is known to follow CORA step for step (read as far as
the headers: `reduce` compacts first, so the order bound counts distinct monomials, CORA's
counts columns; `zonotope` and `interval` of CORALean are not CORA's halving of the even
generators).

| op (proposed) | CORALean definition (file) | request fields | reply |
|---|---|---|---|
| `polyZonotope.plus` | `PolyZonotope.plus` (`Plus.lean`) | `pZ`, `S` (a polynomial zonotope) | `pZ` |
| `polyZonotope.exactPlus` | `PolyZonotope.exactPlus` (`ExactPlus.lean`, same identifiers) | `pZ`, `S` | `pZ` |
| `polyZonotope.mtimes` | `PolyZonotope.mtimes` (`Mtimes.lean`) | `M` (matrix), `pZ` | `pZ` |
| `polyZonotope.quadMap` | `PolyZonotope.quadMap` (`QuadMap.lean`) | `Q` (matrices), `pZ` | `pZ` |
| `polyZonotope.enclose` | `PolyZonotope.convHull` (`ConvHull.lean`) | `pZ`, `S` | `pZ` |
| `polyZonotope.cartProd` | `PolyZonotope.cartProd` (`CartProd.lean`) | `pZ`, `S` | `pZ` |
| `polyZonotope.reduce` | `PolyZonotope.reduce` (`Reduce.lean`) | `pZ`, `order` | `pZ` |
| `polyZonotope.project` | `PolyZonotope.project` (`Project.lean`) | `pZ`, `dims` | `pZ` |
| `polyZonotope.zonotope`, `polyZonotope.interval` | `Conversion/Float/Operations` | `pZ` | zonotope, box |
| `nonlinearSys.reachPoly` | `NonlinearSys.FloatHessModel.linReachFloat` at `poly` (`LinReachFloat.lean`), `abstrErrPolyFloat` | the system (`f`, `n`, `m`), `X0` (a polynomial zonotope), `U`, `tFinal`, `timeStep`, `taylorTerms`, the orders | `timeInterval`, `timePoint` (polynomial zonotopes); CORALean's `poly` is its own variant (second-order Lagrange form), so a bit-exact test would need an op that follows `priv_poly_reach` |

## Level sets and Taylor models (PR "nln-levelset-a")

CORALean has float definitions of the Taylor-model arithmetic and of the level-set queries; the
oracle serves none of them, so the coverage rows of `contSet/levelSet` and `contSet/taylm` are
oracle debt. CORALean's `Taylm.times` truncates at the larger order of its operands and bounds the
dropped and cross terms by interval arithmetic, its own bookkeeping of CORA's product; CORA.cpp
follows CORA's (`max_order` of the left operand), so a bit-exact test needs the op to take `maxOrder`.
CORALean has no float `tightenDomain` (only the `linear` contractor, real layer), and none of its
level-set definitions evaluates an `Expr`: the request carries the constraint as the oracle's
expression tree (`{"op":..,"args":..}`, `{"x":i}`, `{"const":text}`).

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `taylm.fromInterval` | `Interval.taylm` (`Conversion/Float/Operations/Basic.lean`) | `I` (box), `maxOrder` | `models`: per coordinate `{"coef": [[exps, c], ...], "rem": [lo, hi]}` |
| `taylm.plus`, `taylm.minus` | `Taylm.plus`, `Taylm.minus` (`Taylm/Float/Operations/Basic.lean`) | `a`, `b` (models) | the model |
| `taylm.times` | `Taylm.times`, `Taylm.smul` (`Taylm/Float/Operations/Times.lean`, `Smul.lean`) | `a`, `b` (models) or `s` | the model |
| `taylm.pow` | `Taylm.pow` (`Taylm/Float/Operations/Pow.lean`) | `a`, `n` | the model |
| `taylm.interval` | `Taylm.interval` (`Conversion/Float/Operations/Basic.lean`) | `a` (model) | `inf`, `sup` |
| `levelSet.contains` | `LevelSet.containsPoint` (`LevelSet/Float/Operations/ContainsPoint.lean`) | `f` (expression trees), `ops`, `points`, `tol` | `res` |
| `levelSet.isIntersecting` | `LevelSet.isIntersecting_uniform`, `_cell` (`LevelSet/Float/Operations/IsIntersecting*.lean`) | `f`, `ops`, `I` (box), `tol` | `res` |
| `levelSet.complement` | `LevelSet.not` (`LevelSet/Real/Operations/Not.lean`, no float layer) | `f`, `op` | `f` (the negated tree), `op` |
| `levelSet.splitEquality`, `levelSet.lift` | `LevelSet.splitEquality`, `LevelSet.lift_` (real layer, route b) | `f`, `ops`, `N`, `dims` | the two level sets, or the lifted one |

## Differential-algebraic systems (PR "port/nln-dae")

The pinned oracle (`oracle-20261111-f9723f6f6`) serves no `nonlinDASys` op. CORALean's float
definitions (`src/CORALean/ContDynamics/NonlinDASys/Operations/`) take the equations as a
`FloatModel` (interval extensions of `f`, `g` and slope enclosures of the Jacobian blocks), which a
request carries as expression trees of `x`, `y` and `u` (CORA.cpp's `Expr::to_json` with a third
tag `{"y":k}`). Two of them are CORA's loops step for step; `reach` has no float definition that
follows CORA's `initReach`.

| CORALean definition (file) | what it is | follows CORA? |
|---|---|---|
| `NonlinDASys.FloatModel.consistentInitialStateFloat` (`ConsistentInitialStateFloat.lean`), `newtonTestFloat` | Newton's loop on `g` in `y`: `l` and `F` the centers of the model's enclosures, `delta_y` from an approximate inverse, the test `\|delta_y\|_2 < tol`; the end point is certified by Krawczyk's test and `none` otherwise | yes, `consistentInitialState.m`, plus a certificate (CORA returns the point unchecked); `F \ (-l)` is an approximate inverse product |
| `NonlinDASys.FloatModel.steadyStateFloat` (`SteadyStateFloat.lean`) | the same loop on the stacked state `(x, y)` with `M = [A C; D F]` | yes, `steadyState.m`, with the same certificate |
| `NonlinDASys.OutputFloatModel.outputSetLinearFloat` (`OutputSetLinearFloat.lean`) | the output set for a linear output | CORA.cpp ports no `outputSet` (the default output is the state) |
| `NonlinDASys.reachFloat` (`ReachFloat.lean`) | the reduced ordinary system `x' = f(x, h(x, u), u)` run as `NonlinearSys.reachFloat`, `y` eliminated by a supplied `ReducedModel` | own: no `initReach`, no linearization of `g` |
| `NonlinDASys.linReach`, `linError`, `initReach` (real layer, `Operations/`) | CORA's step with the sign and the coupling of the error corrected; the split decision an argument | CORA.cpp's `priv_dae_lin_reach` and `priv_dae_lin_error` follow these; no float layer, so no oracle op |

| op (proposed) | request fields | reply |
|---|---|---|
| `nonlinDASys.consistentInitialState` | `f`, `g` (expression trees of `x`, `y`, `u`), `x0`, `y0`, `u0` (vectors), `tol` | `y` (vector) or `null` |
| `nonlinDASys.steadyState` | `f`, `g`, `x0`, `y0`, `u0`, `tol` | `x`, `y` (vectors) or `null` |

The coverage rows `consistentInitialState` and `steadyState` are oracle debt until then; `initReach`,
`linReach` and `post` have no float layer and stay `property`.
## Level set guards (PR "nln-levelset-c")

CORALean states the unsolvable-case enclosure of a level set and a polynomial zonotope as a real
definition with its subset theorem (`LevelSet/Real/Operations/GuardIntersectUnsolvable.lean`,
`Theorems/GuardIntersectUnsolvableSubset.lean`; the counterexample file shows that the printed sign of
`v(x)` in eq. (11) is wrong and CORA's `-` is right); there is no float layer, no `splitDepFactor` /
`splitLongestGen` for a polynomial zonotope, and no PCA reduction on the float layer
(`ReducePCA*` is a cost model on the real layer). CORA.cpp follows CORA's arithmetic (doubles,
nearest), so these ops are oracle debt; the requests that would close it:

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `levelSet.intersectPolyZonotope` | `LevelSet.guardIntersectUnsolvable` (real) needs a float layer | `f` (expression tree), `pZ` (box as a polynomial zonotope) | `pZ` |
| `polyZonotope.splitDepFactor`, `polyZonotope.splitLongestGen` | none | `pZ`, `id` | the two halves |
| `zonotope.reducePCA` | `Zonotope.pcaBasis` (`PCABasis.lean`, real) | `Z` | `Z` |
