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

## Krylov subspaces (PR "port/krylov-sound")

CORALean's `Krylov` float definitions (MR !914) are served as `krylov.arnoldi`, `krylov.errorBound`,
`krylov.subspace` and `krylov.windows`, and the sound Krylov of CORA.cpp (`priv_krylov_float.cpp`,
`priv_krylov_subspace.cpp`) is tested against them, `test_lean_krylov`, on the pinned oracle
(`oracle-20261112-05b8e9279`, CORALean `main` with !914).

| op | tested | against |
|---|---|---|
| `krylov.arnoldi` | bit for bit | `arnoldiFloat`: `nu`, `V`, `H` (random, non-normal with entries 30 above the diagonal, a breakdown, a chain of 30 states, 180 passes on 6 states) |
| `krylov.errorBound` | bit for bit | `errorBoundFloat`, `orderOkFloat`: the residuals, `muA`, `muT`, `bound`, `ok` (the decay product at order 180, where CORA's `k!` is `Inf`) |
| `krylov.subspace` | bit for bit | `subspaceFloat`: order, `ok`, `bound`, `nu`, `V`, `H` |
| `krylov.windows` | time points to 1e-7, time intervals inside | `windowsFloat` is `reachFloat` of `y' = T y`: the lean backend has no `expm`, and CORALean's curvature bounds the Taylor remainder by a norm, so CORA.cpp's standard reach of `T` is not bit for bit |
| `krylov.lyapunovCertificate` | bit for bit | `lyapunovCertificateFloat` (MR !916): `ok`, `lower`, `upper`, `rate` of the weight `P` that CORA.cpp solves (`priv_lyapunov_solve`, Bartels-Stewart on the real Schur form, Cholesky factor of `P - aI`) on a random, a non-normal and a complex-eigenvalue matrix; also a weight that is not symmetric and one the equation fails for |
| `krylov.restartedBound` | bit for bit | `restartedErrorBoundFloat`, `restartedErrorOkFloat`: the bound of five windows of different lengths and local errors in the weight (`RestartedBound`, `priv_krylov_lyapunov.cpp`) |

The sums run from the last term and no product is fused (`-ffp-contract=off` for these files and
`SparseMatrix::matvec_down`), as `sumUp` and `dotUp` do; `p` and `q` are ignored at `nearest`, where
`sqrtUp` and `expUp` are the machine's. `krylov.errorBound` and `krylov.subspace` reply `"ok"`, the
order test, which is also the protocol's success flag: a reply with `"ok": false` and no `"error"` is
a result, and the client reads it so. A request that renames the field (`accepted`) would be cleaner.

What is not formalized, and so oracle debt (the coverage rows stay below `oracle`):

| op (proposed) | CORA.cpp step | what it needs |
|---|---|---|
| `krylov.restart` | `KrylovWindows` (`priv_krylov_windows.cpp`): the longest window whose bound at its end meets `tol`, the restart from `\|s\| V e^{T t} e_1`, the error carried on by `e^{t mu_A}` (and, if the horizon makes that large, in the Lyapunov weight above; the smaller of the two is reported, and inside a window the weighted bound is `(beta + sqrt(upper) eps) / sqrt(lower)`, the decay of `beta` within it not counted) | `Krylov.restarted_error_le` is for a nonexpansive map; the float window length and the Gershgorin carry for any `A` have no definition, the weighted carry only at the window ends (`krylov.restartedBound`) |
| `krylov.lift` | the windows lifted through `\|s\| C V` and the ball of the bound added in the output rows (`\|C_r\|` times it) | `windowsFloat` stops at the small space |
| `linearSys.reachKrylov` | the assembly of `priv_reach_krylov`: the sum over the vectors of `R0`, the generator curvature scaled by `[-1, 1]` | `Krylov.alg2Corrected` for one vector |
| `krylov.inputSolution`, `krylov.constantInput` | `priv_inputSolution_Krylov`, `priv_krylov_R_uTrans`: the subspace of the integrator system `[A u; 0 0]` (the order search above applies to it), the Taylor terms `V H^i e_1 dt^i / i!`, the box of their remainders, the correction `C V G(H) e_1` | the input solution of the printed corollaries fails (`Krylov.partial_input_solution_counterexample`, `varying_input_reach_counterexample`); its corrected form has no float definition |

## Krylov support function verify (PR "port/arch26-rnd")

`alg::ReachAvoidKrylov` (the verify of the random benchmarks RND02, RND04-06) follows no CORA file of
v2026.1.0: the competition's copy of CORA has `reachSuppFuncsKrylov`, which is not in the release. Its
subspaces are CORALean's (`krylov.subspace`, tested above, in the restart windows above); per step it
bounds the support function by the hull of the time points plus the curvature `F(H) y_k` of the
subspace, the integral of the input by the step times the largest integrand, and the error of the
subspace times the length of each vector of `R0` and `B U`. The loop and these bounds are not in CORALean:

| op (proposed) | request fields | reply |
|---|---|---|
| `linearSys.verifyKrylov` (the loop of `priv_verify_ra_krylov`) | `A`, `B`, `C`, `R0`, `U`, `tFinal`, `timeStep`, `taylorTerms`, `krylovError`, `krylovStep`, specifications as rows | `res`, `iterations`, `timeStep`, `nrSteps` |

The algorithm is tested by properties (`test_linear_sys_verify_krylov`: the verdicts of a system whose
support function is known in closed form, agreement with `alg::ReachAvoidSupportFunction`, sparse
against dense `A`).

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
| `NonlinearSys.oneStepFloat` (`OneStepFloat.lean`) | one affine step with the input `V = f(z*) + B (U - u*) + L`: `LinearSys.reachFloatInput` for one step | own (CORA: `oneStep` of the linear system, then the error solution by `particularSolution_timeVarying`); CORA.cpp follows CORALean bit for bit with `linear_step = LinearStep::coralean` (`priv_float_step.cpp`, `alg::Linearization` and `alg::LinearizationAdaptive`); `linear_step = LinearStep::closed_form` takes CORALean's `oneStepClosedFormFloat` (`priv_closed_form_step.cpp`), bit for bit against `linearSys.oneStepClosedForm` as a linear step. Oracle debt: the nonlinear ops (`nonlinearSys.linReach`, `linReachAdaptivePass`, `reachAdaptive`) still use `reachFloatInput`, so `closed_form` inside a nonlinear step is not checked against them; they need a variant (or a switch) that takes `oneStepClosedFormFloat` as the linear step |
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
`oracle-20261112-05b8e9279` on (`lean::NonlinearSys`, `include/cora/global/oracle/nonlinear_sys.h`);
an older oracle makes the tests skip them.

| op | test | agreement |
|---|---|---|
| `nonlinearSys.varphiTotalAdaptive`, `varphiEstAdaptive`, `optimalDeltatAdaptive` | `test_lean_nonlinear_sys_adaptive_tuning` | bit for bit |
| `zonotope.reduce` with method `adaptive` | `test_lean_nonlinear_sys_adaptive_tuning` | bit for bit |
| `nonlinearSys.abstrErrAdaptive` | `test_lean_nonlinear_sys_adaptive_pass` | bit for bit at tensor order 2; order 3 inside the oracle's box (quadMap, see the differences) |
| `nonlinearSys.ordersAdaptive` | `test_lean_nonlinear_sys_adaptive_pass` | exact |
| `nonlinearSys.linReachAdaptivePass` | `test_lean_nonlinear_sys_adaptive_pass` | with `linear_step = coralean`: `xs`, `A`, the zero generators and, at tensor order 2, the sets (`Rmax`, the window, the end set) bit for bit; at order 3 the sets inside the oracle's, 5e-3 tighter at most (quadMap) |
| `linearSys.reach` with an input set (the linear step of `lin` and `lin-adaptive`) | `test_lean_linear_sys_float_step` | bit for bit: the window and the end set of one step, for random systems, orders 4 and 6, reductions to 1 and 10 generators per dimension, a Robertson-like and a van der Pol-like linearization |
| `zonotope.mtimesInterval` | `test_lean_linear_sys_float_step` | bit for bit |
| `linearSys.oneStepClosedForm` (CORA's `oneStep` in closed form, `LinearSys.oneStepClosedFormFloat`, CORALean !911) | `test_lean_linear_sys_closed_form_step` | bit for bit: the end set and the window of one step, for random systems of 3 states at the orders 2, 4, 10 and of 5 at order 4, the oracle's self-test system with and without input generators, a stiff system whose order is raised, and one for which the oracle answers `null` (no order below `eta + 4096`) |
| `nonlinearSys.reachAdaptive` | `test_lean_nonlinear_sys_reach_adaptive` | the same steps and tensor orders while the order is 2; the sets as in a pass |

Not covered yet: `nonlinearSys.linReachAdaptive` (the state-passing step; `reachAdaptive` runs it).

## Nonlinear systems, alg "poly-adaptive"

CORA.cpp's `alg::PolynomializationAdaptive` is CORA's poly-adaptive as CORALean repairs it
(`reachPolyAdaptiveFloat`, CORALean !912). The oracle serves its ops (`nonlinearSys.reachPolyAdaptive`,
`linReachPolyAdaptive`, `linReachPolyAdaptivePass`) from the pinned tag `oracle-20261112-05b8e9279`
on (`lean::NonlinearSys`, `reach_poly_adaptive`, `lin_reach_poly_adaptive_pass`). The linear step (`polyWindowFloat`, `polyPointFloat`, !915) follows the float definition; the free boxes of the sets
agree with the oracle's to 1e-4 on the van der Pol fixture 3201 but not bit for bit: the abstraction error
(`abstrErrPolyAdaptiveFloat`: its `zonotopeEven` error box, `quadMapMixed`) and the error solution are not yet
CORALean's operations step for step, the true error differs by a factor 2 (open). The operations of the polynomial zonotope float
layer behind them (`reduceAdaptive`, `restructure`, `quadMapMixed`, `mtimesIntervalMat`) stay oracle debt: the
oracle serves none of them on its own. `linReachPolyAdaptive` (one step with its tuning) is not tested; the
tuning formulas are the lin-adaptive ones (`dHused` aside).

## Zonotope quadMap and the facets of vertex polytopes (PR "oracle/group-b-1")

`zonotope.quadMap` is served (`Zonotope.quadMap`), but it is not CORA's map: CORALean applies
`PolyZonotope.quadMap` to the zonotope seen as a polynomial zonotope, which returns `(n + m + 1)^2`
generators and the center as a generator (an enclosure, looser than CORA's `1 + m + m(m + 1)/2`
generators with the center kept). CORALean has CORA's structure only over the reals
(`Zonotope.quadMapTri`, `QuadMapTriOuter`); a float `quadMapTri` with its enclosure theorem is the
open work (more than a day), after which CORA.cpp's `quad_map` is tested bit for bit and the 13
instances that use it (`coverage/benchmarks.md`) move to `oracle`. Until then `contSet/zonotope/quadMap`
stays `property`.

`polytope.constraints` of a vertex polytope (CORALean's `VPolytope.hPolytope`, qhull in CORA) has no
op; `test_lean_polytope_constraints` compares the halfspace form only, which is what the benchmarks
build.

## Polynomial zonotopes and alg "poly" (PR "nln-poly")

The pinned oracle (`oracle-20261112-05b8e9279`) serves no polynomial zonotope operation and no
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

The pinned oracle (`oracle-20261112-05b8e9279`) serves no `nonlinDASys` op. CORALean's float
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

## Neural networks and neural network controlled systems (PR "port/ainncs-a")

The pinned oracle (`oracle-20261112-05b8e9279`) serves no `neuralNetwork` or `neurNetContrSys` op.
CORALean's definitions are generic over the set representation (`NeuralNetwork.enclose`, no scalar
named; the float layers instantiate it in `Examples/RunnableFloat.lean`) or over the plant
(`NeurNetContrSys.reachWith`, the plant's reach an argument); only the linear plant has a float
definition (`augmentFloat`, `reachFloat`). CORA.cpp's network enclosure is CORA's "singh" order 1
(`NeuralNetwork::evaluate`, `evaluate_with_input`), whose per-layer enclosure is the interval
rule of CORALean's float layers; the nonlinear plants of the AINNCS instances have no CORALean
reach beyond the hypothesis of `reachWith`.

| CORALean definition (file) | what it is | follows CORA? |
|---|---|---|
| `NeuralNetwork.enclose` (`NN/Operations/Enclose.lean`), float layers (`Examples/RunnableFloat.lean`) | the image of a set through the layers, each activation by its interval rule | CORA's `evaluate_` with `poly_method` "singh"; the zonotope layer rule is CORA's line and radius, the rounding is directed |
| `NeurNetContrSys.augmentFloat` (`NeurNetContrSys/Operations/AugmentFloat.lean`) | the augmented matrices `[A B_nn; 0 0]`, `[B_ext; 0]`, `[c; 0]` of a linear plant | yes, `aux_augmentLinear` |
| `NeurNetContrSys.reachFloat` (`Operations/ReachFloat.lean`) | CORA's loop at a linear plant: per period the network enclosure, the augmented system reached from `X x U` | CORA's loop; `reachFloat_outer` states the enclosure |

| op (proposed) | request fields | reply |
|---|---|---|
| `neuralNetwork.enclose` | `layers` (`W`, `b`, `activation` per layer), `Z` (a zonotope) | `Z` (the enclosure of the outputs) |
| `neuralNetwork.encloseWithInput` | `layers`, `Z` | `Z` of dimension in + out (the pairs) |
| `neurNetContrSys.augment` | `A`, `B`, `c`, `m` (outputs of the network) | `A_aug`, `B_aug`, `c_aug` |
| `neurNetContrSys.reach` | `A`, `B`, `c`, `layers`, `dt`, `Z0`, `U`, `tFinal`, reach options | `timeInterval`, `timePoint` (zonotopes) |

`NeuralNetwork::evaluate`, `evaluate_with_input`, `NeurNetContrSys::reach` and `simulate` are
oracle debt until then; their tests are `property` (the sets hold every simulated run).

### Piece B (PR "port/ainncs-b")

| CORALean definition (file under `src/CORALean/NN/Layers/`) | what CORA.cpp ports | request (proposed op `neuralNetwork.polyRelaxation`) |
|---|---|---|
| `sigmoidPolyRelaxation`, `tanhPolyRelaxation` (`Operations/SigmoidPolyRelaxation.lean`, `Operations/TanhPolyRelaxation.lean`; `Theorems/*PolyRelaxationEncloses.lean`), `leakyReluRelaxation` (ReLU) | `poly_method` "regression" of order 1 in `NeuralNetwork::evaluate`: the line `m x + t` of the 20-point least squares (CORA's `leastSquarePolyFunc`) and its exact error radius; CORALean takes the fit as data and certifies the error | `activation`, `l`, `u`, `coeffs` (`[m, t]`, the fit as CORA.cpp computes it); reply `error`, a certified bound: the test is containment (CORA.cpp's exact radius at most the oracle's), as the fit is not CORALean's |

`NeuralNetwork::sensitivity` (CORA's `calcSensitivity`; CORALean has `Chain.sensitivity`,
`Chain.sensitivity_eq_fderiv` and the interval `IntervalChain.sensitivity`; proposed op
`neuralNetwork.sensitivity`: `layers`, `x` (a point), reply `J`, the Jacobian), the splitting of `NeurNetContrSys::verify`
(CORA's `splitR0`) and the VCAS loop of the benchmark program (the discrete plant step, the
advisory and the acceleration rule, composed of `NeuralNetwork::evaluate` and `PolyZonotope`
affine maps) have no CORALean counterpart beyond `NeurNetContrSysDT` (whose `reach` is the
sampled loop at a discrete-time plant, with the advisory choice an argument); they are oracle debt
with the network ops of the first table.

## Rows the oracle serves that stay `property` or `untested` (Group A, batch 1)

`zonotope.project`, the generic `interval` and `isIntersecting` entries are now `oracle`
(`test_lean_zonotope_project`, `test_lean_cont_set`). The other float rows the benchmarks run were
read against the pinned oracle's op and kept out of `oracle`; each stays until the named change.

| row | op | why it is not bit-exact | what would close it |
|---|---|---|---|
| `zonotope/polytope` | `zonotope.polytope` (`Zonotope.hPolytopeFacets`) | the same facets, in the reverse order of `nchoosek`; sorted, the rows still differ in the last bits on random zonotopes (tested for 2 to 4 dimensions): CORA.cpp projects each cross product onto the orthogonal complement of its generators (`priv_zonotope_facet_normals`, kept for the gradient), CORALean normalizes the cross product itself | an oracle variant, or CORA.cpp dropping the projection (it costs the torch gradient) |
| `zonotope/vertices_` | `zonotope.vertices` | CORA's `aux_vertices2Dfast` walks `V(:,i+1) = V(:,i) + 2 g_i` and shifts by the box, CORALean's `Zonotope.polygon` evaluates each corner as `c - sum g + 2 sum_{j<k} g`: another evaluation order, and for more than two dimensions `2^h` corners instead of the hull | `Zonotope.polygon` as a cumulative walk |
| `zonotope/isIntersecting_` | `zonotope.isIntersecting` | the answer is certified one way (disjointness), CORA solves a linear program | none: a one-sided test stays `property` |
| `zonotope/quadMap` | `zonotope.quadMap` | CORALean maps through a polynomial zonotope, a wider set than CORA's zonotope map | `Zonotope.quadMap` following `quadMap.m` |
| `conZonotope/supportFunc_`, `zonoBundle/supportFunc_` | `conZonotope.supportFunc`, `zonoBundle.supportFunc` | an upper bound from a checked dual certificate, CORA's value is the optimum of the linear program | none |
| `conZonotope/isFullDim`, `conZonotope/rescale` | `conZonotope.isFullDim` | certified one way; `ConZonotope.rescale` takes the ranges of the factors as arguments, CORA finds them by linear programs or Gauss-Jordan | an op `conZonotope.rescale` with the ranges, to test the scaling alone |
| `polytope/constraints`, `polytope/polytope` | none | `VPolytope.hPolytope` encloses the vertices along given directions `Aq`; CORA computes the facets of the hull | a facet enumeration in CORALean |
| `linearSys/taylorMatrices` | `linearSys.taylorMatrices` | the remainder: CORALean bounds it by a norm (`intervalCorrectionMatrix`) or, in `oneStepClosedForm`, by an enclosure of `e^{\|A\| dt}` (`expRemainderEntrywise`); CORA's `W` is `\|expm(\|A\| dt) - series\|` in floats | covered by the closed-form step (`correction_matrix_state` taking `expRemainderEntrywise`), another branch |
| `linearSys/errorSolution_adaptive`, `linearSys/reach`, `contDynamics/reach`, `location/reach` | `linearSys.errorSolution_adaptive`, `linearSys.reach`, `location.reach` | CORALean's step is the interval-matrix one (`reachFloat`, `errorSolutionAdaptiveFloat`), tested by tolerance in `test_lean_linear_sys_steps_agreement` | the closed-form step |
| `linearSys/verify` | none | `verifyFloat` is Alg. 3 with the reach function, the inner approximation and the distances as arguments; no concrete instance serves as an op | an op that instantiates it with CORA's support-function routines |
| `contSet/interval/polyZonotope`, `contSet/zonotope/polyZonotope`, `polyZonotope/*` | none | the oracle has no polynomial zonotope codec until the poly-adaptive branch merges; then the generator order of the constructor (see above) decides | the ops of the section above |
| `nn/layers/*`, `nn/neuralNetwork/*` | none | CORALean's network enclosure is generic (`Examples/RunnableFloat.lean` instantiates it); there is no concrete zonotope-layer float definition to serve | `neuralNetwork.enclose` of the section above |
