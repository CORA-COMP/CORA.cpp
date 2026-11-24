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
| `location.calcBasis` | `Location.calcBasisBox` (`CalcBasisBox.lean`), `Location.calcBasisPcaFloat` (`CalcBasisPcaFloat.lean`; `flow` has no float definition yet) | `n`, `enclose` (`"box"`, `"pca"`), for `pca` the sets `R` and the hyperplane `normal` of the guard | `B` (matrix per entry) |
| `location.checkFlow` | `Location.checkFlowFloat` (`CheckFlowFloat.lean`) | the location's `A`, `B`, `c`, `U`, the hyperplane `c`, `d`, `R` (zonotopes) | `keep`: indices of `R` |
| `location.guardIntersect_zonoGirard` | `Location.guardIntersectZonoGirardFloat` (`GuardIntersectZonoGirardFloat.lean`), its bound `Location.zonoGirardUpperFloat` (`ZonoGirardUpperFloat.lean`), the member with the defect box of a non-orthogonal basis `Location.zonoGirardMemberFloat` (`ZonoGirardMemberFloat.lean`) | `Bs` (matrices), `R0`, `R` (zonotopes), `n` (normal), `gamma` | `zB` (zonotope bundle) |
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
(`oracle-20261112-e3fba3c5b`, CORALean `main` with !914).

| op | tested | against |
|---|---|---|
| `krylov.arnoldi` | bit for bit | `arnoldiFloat`: `nu`, `V`, `H` (random, non-normal with entries 30 above the diagonal, a breakdown, a chain of 30 states, 180 passes on 6 states) |
| `krylov.errorBound` | bit for bit | `errorBoundFloat`, `orderOkFloat`: the residuals, `muA`, `muT`, `bound`, `ok` (the decay product at order 180, where CORA's `k!` is `Inf`) |
| `krylov.subspace` | bit for bit | `subspaceFloat`: order, `ok`, `bound`, `nu`, `V`, `H` |
| `krylov.windows` | time points to 1e-7, time intervals inside | `windowsFloat` is `reachFloat` of `y' = T y`: the lean backend has no `expm`, and CORALean's curvature bounds the Taylor remainder by a norm, so CORA.cpp's standard reach of `T` is not bit for bit |
| `krylov.lyapunovCertificate` | bit for bit | `lyapunovCertificateFloat` (MR !916): `ok`, `lower`, `upper`, `rate` of the weight `P` that CORA.cpp solves (`priv_lyapunov_solve`, Bartels-Stewart on the real Schur form, Cholesky factor of `P - aI`) on a random, a non-normal and a complex-eigenvalue matrix; also a weight that is not symmetric and one the equation fails for |
| `krylov.restartedBound` | bit for bit | `restartedErrorBoundFloat`, `restartedErrorOkFloat`: the bound of five windows of different lengths and local errors in the weight (`RestartedBound`, `priv_krylov_lyapunov.cpp`) |
| `krylov.windowBound` | bit for bit | `windowErrorBoundFloat`, `windowErrorOkFloat`, `carryMinFloat` (MR !921): `windowsOk`, `bound` and `carry` of the next window after the finished ones, from the stable fixtures (every `N` up to five finished windows; the Gershgorin carry below, between and above the bound, and `0`; a NaN is no lean value, so the one case where `carry_min` differs from CORA.cpp's old choice is untested) and from the windows `KrylovWindows` forms on RND02 (all 85) and RND04 (three of 111, in the weight of a fixture: its n = 1000 takes the oracle minutes per request; an overflown carry is passed as the largest float) (`RestartedBound::window_bound`, `carry_min`, `windows_ok`; the oracle takes the Gershgorin carry as given). Needs an oracle tag that includes !921 |

The sums run from the last term and no product is fused (`-ffp-contract=off` for these files and
`SparseMatrix::matvec_down`), as `sumUp` and `dotUp` do; `p` and `q` are ignored at `nearest`, where
`sqrtUp` and `expUp` are the machine's. `krylov.errorBound` and `krylov.subspace` reply `"ok"`, the
order test, which is also the protocol's success flag: a reply with `"ok": false` and no `"error"` is
a result, and the client reads it so. A request that renames the field (`accepted`) would be cleaner.

What is not formalized, and so oracle debt (the coverage rows stay below `oracle`):

| op (proposed) | CORA.cpp step | what it needs |
|---|---|---|
| `krylov.restart` | the window length of `KrylovWindows` (`priv_krylov_windows.cpp`): the longest window whose bound at its end meets `tol` | tuning, not checked: each window it plans is `step_verifyKrylovWindowFloat` (`linearSys.verifyKrylovPass` below) |
| `krylov.lift` | the windows lifted through `\|s\| C V` and the ball of the bound added in the output rows (`\|C_r\|` times it) | `windowsFloat` stops at the small space |
| `linearSys.reachKrylov` | the assembly of `priv_reach_krylov`: the sum over the vectors of `R0`, the generator curvature scaled by `[-1, 1]` | `Krylov.alg2Corrected` for one vector |
| `krylov.inputSolution`, `krylov.constantInput` | `priv_inputSolution_Krylov`, `priv_krylov_R_uTrans`: the subspace of the integrator system `[A u; 0 0]` (the order search above applies to it), the Taylor terms `V H^i e_1 dt^i / i!`, the box of their remainders, the correction `C V G(H) e_1` | the input solution of the printed corollaries fails (`Krylov.partial_input_solution_counterexample`, `varying_input_reach_counterexample`); its corrected form has no float definition |

## Krylov support function verify (PR "verify/krylov-port", part of #45)

`alg::ReachAvoidKrylov` (the verify of the random benchmarks RND02, RND04-06) follows no CORA file of
v2026.1.0: the competition's copy of CORA has `reachSuppFuncsKrylov`, which is not in the release. Its
decision is CORALean's `LinearSys.verifyKrylovPassFloat`, served as `linearSys.verifyKrylovPass`; the
plan of each direction's windows (order, steps), the halving of the step and the stop when no finer
step can help are CORA.cpp's tuning and are not checked.

| op | test | compared |
|---|---|---|
| `linearSys.verifyKrylovPass` (`verifyKrylovWindowsFloat`, `verifyKrylovBoundsFloat`, `verifyKrylovPassFloat`) | `test_lean_linear_sys_verify_krylov` | bit for bit: the verdict of the last pass, every direction's windows (`order`, `nu`, `err`: the next start from the ball of `e^{H L}` with its error, the Gershgorin carry and the smaller weighted one) and its bounds on every step (the iterates of the ball of `e^{H dt}` with their carried error), on systems of two to twenty states: a safe set that holds and one that does not, an unsafe set, inputs, an output matrix, a non-normal matrix, restart windows, halved steps; the Lyapunov weight of `A'` is the one CORA.cpp tried |

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
| `NonlinearSys.oneStepFloat` (`OneStepFloat.lean`) | one affine step with the input `V = f(z*) + B (U - u*) + L`: `LinearSys.reachFloatInput` for one step | own (CORA: `oneStep` of the linear system, then the error solution by `particularSolution_timeVarying`); CORA.cpp follows CORALean bit for bit with `linear_step = LinearStep::coralean` (`priv_float_step.cpp`, `alg::Linearization` and `alg::LinearizationAdaptive`); `linear_step = LinearStep::closed_form`, the default, takes CORALean's `oneStepClosedFormFloat` (`priv_closed_form_step.cpp`), bit for bit against `linearSys.oneStepClosedForm` as a linear step and, inside the nonlinear ops (`nonlinearSys.linReach`, `linReachAdaptivePass`, `reachAdaptive`) with `linearStep: "closedForm"`, by `test_lean_nonlinear_sys_lin_reach`, `test_lean_nonlinear_sys_adaptive_pass` and `test_lean_nonlinear_sys_reach_adaptive` |
| `NonlinearSys.FloatHessModel.linReachFloat` (`LinReachFloat.lean`) | one step: the region checked first, a single pass | own (CORA's `while perfIndCurr > 1` loop is `linReachLoopFloat`, `LinReachLoopFloat.lean`, `ErrorBound::loop`) |
| `NonlinearSys.FloatModel.reachFloat` (`ReachFloat.lean`) | the steps over the horizon | own, by the steps above |

| op (proposed) | request fields | reply |
|---|---|---|
| `nonlinearSys.linearize` | `f` (expression trees of `x`, `u`), `R`, `U` (zonotopes), `timeStep` | `x_star`, `u_star`, `f0`, `A`, `U` (zonotope), `u_const` |
| `nonlinearSys.abstrerr_lin` | `f`, `R` (zonotope, relative to `x_star`), `x_star`, `u_star`, `U`, `tensorOrder`, `errorOrder`, `intermediateOrder` | `trueError` (vector), `VerrorDyn` (zonotope) |
| `nonlinearSys.linReach` (CORA's steps, as CORA.cpp's `priv_lin_reach`) | `f`, `R`, `error`, `U`, `timeStep`, `taylorTerms`, the orders | `ti`, `tp` (zonotopes), `error` |
| `nonlinearSys.linReach`, `nonlinearSys.reach` as CORALean's `linReachFloat` (the region step, `ErrorBound::region`) and `linReachLoopFloat` (CORA's loop, `errorBound: "loop"`, with `abstrerr`) | `f`, `R`, `U`, `timeStep`, `taylorTerms`, `zonotopeOrder`, `linearStep`, `errorBound`, `abstrerr` | `timeInterval`, `timePoint` (and `abstrerr` of the loop): bit for bit by `test_lean_nonlinear_sys_lin_reach` (van der Pol, Robertson, `x' = x^2` where the region check fails and the loop succeeds, recorded steps of TORA, ACC, the TRAF22 car; the loop's error carried by `post` and by `reach`); the closed loop of TRAF22 at ten states has no test (the oracle's trees take an hour) |
| `nonlinearSys.linReachPoly` (CORALean's `linReachPolyLoopFloat`, `linReachPolyFloat`) | `f`, `R` (a polynomial zonotope with `id`), `U`, `timeStep`, `taylorTerms`, `zonotopeOrder`, `abstrerr`, `reduce`, `columnOrder: "first"` | `timeInterval`, `timePoint` (polynomial zonotopes with `id`), `abstrerr`: the step of a polynomial zonotope at `alg = lin` (the AINNCS plants, `priv_lin_reach_poly`), bit for bit by `test_lean_nonlinear_sys_lin_reach_poly` at the closed-form linear step and tensor order 2; `post` reduces as `reduceCora` (`priv_poly_reduce_cora`); the merged columns in the order of their first occurrences (CORA's hash order is not reproduced) |
| `nonlinearSys.reach` | `f`, `R0`, `U`, `tFinal`, the options of `alg::Linearization` | `timeInterval`, `timePoint`: arrays of zonotopes |

## Nonlinear systems, alg "lin-adaptive" (PR "port/nln-adaptive")

CORA.cpp's `alg::LinearizationAdaptive` is CORA's lin-adaptive as CORALean repairs it
(`reachAdaptiveFloat`, CORALean !906). The oracle serves its ops from the pinned tag
`oracle-20261112-e3fba3c5b` on (`lean::NonlinearSys`, `include/cora/global/oracle/nonlinear_sys.h`);
an older oracle makes the tests skip them.

| op | test | agreement |
|---|---|---|
| `nonlinearSys.varphiTotalAdaptive`, `varphiEstAdaptive`, `optimalDeltatAdaptive` | `test_lean_nonlinear_sys_adaptive_tuning` | bit for bit |
| `zonotope.reduce` with method `adaptive` | `test_lean_nonlinear_sys_adaptive_tuning` | bit for bit |
| `nonlinearSys.abstrErrAdaptive` | `test_lean_nonlinear_sys_adaptive_pass` | bit for bit at the tensor orders 2 and 3 (order 3: CORA's quadMap of a zonotope, CORALean's `quadMapTriFiltered`, MR oracle/bitexact-4) |
| `nonlinearSys.ordersAdaptive` | `test_lean_nonlinear_sys_adaptive_pass` | exact |
| `nonlinearSys.linReachAdaptivePass` | `test_lean_nonlinear_sys_adaptive_pass` | bit for bit with `linear_step = coralean` and `closed_form`, at the tensor orders 2 and 3: `xs`, `A`, the sets (`Rmax`, the window, the end set), the true error, the error radius (`errorSolutionAdaptiveFloat`) and the zero generators; the closed form on van der Pol only (the oracle's takes 25 minutes on Robertson) |
| `linearSys.reach` with an input set (the linear step of `lin` and `lin-adaptive`) | `test_lean_linear_sys_float_step` | bit for bit: the window and the end set of one step, for random systems, orders 4 and 6, reductions to 1 and 10 generators per dimension, a Robertson-like and a van der Pol-like linearization |
| `zonotope.mtimesInterval` | `test_lean_linear_sys_float_step` | bit for bit |
| `linearSys.oneStepClosedForm` (CORA's `oneStep` in closed form, `LinearSys.oneStepClosedFormFloat`, CORALean !911, tight `[e^{A dt}]` !913, entrywise `W` at the order `eta_e + 8` !917) | `test_lean_linear_sys_closed_form_step` | bit for bit: the end set and the window of one step, for random systems of 3 states at the orders 2, 4, 10 and of 5 at order 4, the oracle's self-test system with and without input generators, a stiff system whose order is raised, one for which the oracle answers `null` (no order below `eta + 4096`), Jacobians of the Robertson reaction (stiff, with zeros) at four step sizes, a stiff two-state system that conserves x + y (fixtures 3090 to 3093), sparse generators, and matrices of norms from 0.01 to 40 (the halvings `s` of `[e^{A dt}]`); `halvingsFloat` is `none` only where the order search has already failed, so that case is not separate. Beyond a scale of about 100 the powers of `|A| dt` overflow and the oracle prints a number out of range, which the client refuses (oracle debt: print the infinity) |
| `nonlinearSys.reachAdaptive` | `test_lean_nonlinear_sys_reach_adaptive` | bit for bit with the default `closed_form`: the first two steps of van der Pol (times, tensor orders, the sets over them and at the end; the time exact, `ExactTime`); `linear_step = cora` (eigen) the steps to 1e-6 and the sets inside the oracle's |
| `nonlinearSys.linReachAdaptive` | `test_lean_nonlinear_sys_reach_adaptive` | bit for bit: the last step of the van der Pol run (`ratUp` of the exact time left) from CORA.cpp's state before it, the time sent exactly as `m*2^-1074` |

## Nonlinear systems, alg "poly-adaptive"

CORA.cpp's `alg::PolynomializationAdaptive` is CORA's poly-adaptive as CORALean repairs it
(`reachPolyAdaptiveFloat`, CORALean !912). The oracle serves its ops (`nonlinearSys.reachPolyAdaptive`,
`linReachPolyAdaptive`, `linReachPolyAdaptivePass`) from the pinned tag `oracle-20261112-e3fba3c5b`
on (`lean::NonlinearSys`, `reach_poly_adaptive`, `lin_reach_poly_adaptive_pass`). The linear step (`polyWindowFloat`, `polyPointFloat`, !915) follows the float definition; the free boxes of the sets
agree with the oracle's to 1e-5 on the van der Pol and quadratic fixtures, not bit for bit. The true error and the
radius of the error solutions are bit for bit (`priv_poly_quad.cpp`: CORALean's `quadMap`, `quadMapMixed` and
`zonotopeEven`, pairs (i, j) and (j, i) apart). What stays open: `zonotopeEven` moves the shift of the even and
constant columns into the error box of `VerrorDyn`, where `polyWindowFloat` reads it as part of the varying input;
CORA.cpp's zonotopes have no error box and add it to the center, so the window and the end set differ by up to the
error scale (7e-7 on the quadratic fixture). Needs an error box on `Zonotope` (or a shift carried beside it) first.
The operations of the polynomial zonotope float
layer behind them (`reduceAdaptive`, `restructure`, `quadMapMixed`, `mtimesIntervalMat`) stay oracle debt: the
oracle serves none of them on its own. `linReachPolyAdaptive` (one step with its tuning) is not tested; the
tuning formulas are the lin-adaptive ones (`dHused` aside).

### Done in CORALean !926: no box cap on the varying input

`polyWindowFloat` and `polyPointFloat` run `inputSolutionFloat` on `Ũ` itself, not on `Reduce Ũ 1` (the
cap of !918, which widened LALO20 W01's x4 by 2.5 % at t = 3 against CORA). CORA.cpp follows (`aux_cap`
is gone); the oracle tests of the poly-adaptive window and pass run against the oracle of !926 until the
pin is bumped to a tag that has it. Not taken in !926: the constant-offset split (summed series plus
`[G] ũ` as in `oneStepClosedFormFloat`), which matches CORA's step-1 time step exactly but moves W01's
final width only from 0.0582 to 0.0581. CORA's `Zdelta` of the second run of a step is the one of the first
run (the linearization point of the other step size; conservative in the step observed, not shown
unsound); CORALean recomputes it per pass.

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

The pinned oracle (`oracle-20261112-e3fba3c5b`) serves no polynomial zonotope operation and no
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
| `polyZonotope.quadMap` | served: `PolyZonotope.quadMapCora` (`QuadMapCora.lean`, CORA.cpp's order; `test_lean_poly_zonotope` bit for bit) | `P`, `Q` (matrices) | `P` |
| `polyZonotope.enclose` | `PolyZonotope.convHull` (`ConvHull.lean`) | `pZ`, `S` | `pZ` |
| `polyZonotope.cartProd` | `PolyZonotope.cartProd` (`CartProd.lean`) | `pZ`, `S` | `pZ` |
| `polyZonotope.reduce` | `PolyZonotope.reduce` (`Reduce.lean`) | `pZ`, `order` | `pZ` |
| `polyZonotope.project` | `PolyZonotope.project` (`Project.lean`) | `pZ`, `dims` | `pZ` |
| `polyZonotope.zonotope`, `polyZonotope.interval` | `Conversion/Float/Operations` | `pZ` | zonotope, box |
| `nonlinearSys.reachPoly` | `NonlinearSys.FloatHessModel.linReachFloat` at `poly` (`LinReachFloat.lean`), `abstrErrPolyFloat` | the system (`f`, `n`, `m`), `X0` (a polynomial zonotope), `U`, `tFinal`, `timeStep`, `taylorTerms`, the orders | `timeInterval`, `timePoint` (polynomial zonotopes); CORALean's `poly` is its own variant (second-order Lagrange form), so a bit-exact test would need an op that follows `priv_poly_reach` |

### Served in CORALean !936: the operations of CORA's polyZonotope

The ops `polyZonotope.constructor`, `plus` (`Q`, `Z`, `I` or `v`), `mtimes`, `zonotope`, `interval`,
`reduce` (`girard`), `enclose`, `dim`, and `zonotope.polyZonotope`, `interval.polyZonotope` follow CORA's
algorithm, and so do `center`, `cartProd` (a zonotope `Z`), `exactPlus` (the sorted union of the
identifiers, then the constructor) and `isIntersecting` (a polytope `S` and `tol`, CORA's `'approx'`
box test; batch "polyTaylm"), `splitDepFactor` (`factor`, an identifier) and `splitLongestGen` (CORA.cpp's columns
of `getSubset`, the constant ones into the center, then the constructor; replies `P1`, `P2`).
`quadMap` is not served: CORALean's float `quadMap` (all `(h + 1)^2` products, the independent
part absorbed) is not CORA's steps, so it stays debt. Their request is
`{"c", "G", "GI", "Exp", "E", "id"}` (`Exp` the p x h exponent matrix). They are
tested bit for bit by `test_lean_poly_zonotope` (random sets with repeated and zero columns, and the sets of
the ACC network). Equal exponent columns are ordered as CORA's `removeRedundantExponents` does (the hash of
`RandStream('mt19937ar', 'Seed', 0)`, which `std::mt19937` reproduces), a sum of three or more equal
columns is added from the last column to the first, as CORALean's `compactGens`. Open:

| op | what stays oracle debt |
|---|---|
| `polyZonotope.enclose` | unequal numbers of independent generators (CORA's `Gadd` is not halved, CORALean's layout is `Zonotope.linComb`'s); other identifiers with a constant dependent generator (the oracle refuses) |
| `polyZonotope.zonotope`, `interval`, `reduce` | a constant (all-zero exponent) column: CORA halves it, CORALean and CORA.cpp move it whole into the center |

## Level sets and Taylor models (PR "nln-levelset-a")

**Taylor models, served (bit-exactness batch "polyTaylm"):** the ops `taylm.plus`, `minus` (a model
`Q` or a number `s`), `uminus`, `uplus`, `times` (a model `Q` at the order `maxOrder` of the request,
or a number `s`), `power` (`k`) and `interval` follow CORA.cpp's `Taylm` (`Taylm.plusMerged`,
`timesMerged`, `scale`, `powBinary`, `boundsMono` in `Taylm/Float/Operations/`): the terms sorted by
their exponents with equal monomials merged, as its map keeps them; the cross terms, then the term
pairs; CORA's monomial ranges. A model is `{"vars", "maxOrder", "terms": [{"c", "e"}], "rem"}`, the
reply adds the error box `Ierr` (zero at nearest). `test_lean_taylm` compares every term and the
remainder bit for bit. CORA.cpp truncates a product at the first model's order and CORA at the larger
one; the op takes the order, so it serves both. The wire form has no -0, so the test's cases have none.

CORALean has float definitions of the level-set queries; the oracle serves none of them, so the
coverage rows of `contSet/levelSet` are oracle debt. CORALean has no float `tightenDomain` (only the
`linear` contractor, real layer), and none of its level-set definitions evaluates an `Expr`: the
request carries the constraint as the oracle's expression tree (`{"op":..,"args":..}`, `{"x":i}`,
`{"const":text}`).

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `levelSet.contains` | `LevelSet.containsPoint` (`LevelSet/Float/Operations/ContainsPoint.lean`) | `f` (expression trees), `ops`, `points`, `tol` | `res` |
| `levelSet.isIntersecting` | `LevelSet.isIntersecting_uniform`, `_cell` (`LevelSet/Float/Operations/IsIntersecting*.lean`) | `f`, `ops`, `I` (box), `tol` | `res` |
| `levelSet.complement` | `LevelSet.not` (`LevelSet/Real/Operations/Not.lean`, no float layer) | `f`, `op` | `f` (the negated tree), `op` |
| `levelSet.splitEquality`, `levelSet.lift` | `LevelSet.splitEquality`, `LevelSet.lift_` (real layer, route b) | `f`, `ops`, `N`, `dims` | the two level sets, or the lifted one |

## Differential-algebraic systems (PR "port/nln-dae")

The pinned oracle (`oracle-20261112-e3fba3c5b`) serves no `nonlinDASys` op. CORALean's float
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
`v(x)` in eq. (11) is wrong and CORA's `-` is right); there is no float layer and no PCA reduction on the float layer
(`ReducePCA*` is a cost model on the real layer). CORA.cpp follows CORA's arithmetic (doubles,
nearest), so these ops are oracle debt; the requests that would close it:

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `levelSet.intersectPolyZonotope` | `LevelSet.guardIntersectUnsolvable` (real) needs a float layer | `f` (expression tree), `pZ` (box as a polynomial zonotope) | `pZ` |
| `zonotope.reducePCA` | `Zonotope.pcaBasis` (`PCABasis.lean`, real) | `Z` | `Z` |

## Neural networks and neural network controlled systems (PR "port/ainncs-a")

The pinned oracle (`oracle-20261112-e3fba3c5b`) serves no `neuralNetwork` or `neurNetContrSys` op.
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

### AINNCS hard instances (PR "ainncs/hard")

The splitting rule `VerifySplit::Rule::widest` of `NeurNetContrSys::verify` (cut the dimension of the
largest radius relative to the initial set's; CORA.cpp only, CORA's `splitR0` cuts by radius times
sensitivity) and the Docking constraint from inside by a polytope (`|s| >= (sx + sy) / sqrt 2`, the
disk by an inscribed regular polygon) have no CORALean counterpart: they are oracle debt with the
splitting of `verify` above. CORALean needs `NeurNetContrSys.verify` with a split rule, and an
`IsInside` lemma for the inscribed polygon and the Cauchy-Schwarz bound of `|s|`.

## Rows the oracle serves that stay `property` or `untested` (Group A, batch 1)

`zonotope.project`, the generic `interval` and `isIntersecting` entries are now `oracle`
(`test_lean_zonotope_project`, `test_lean_cont_set`), and `zonotope/quadMap` is `oracle` against
`zonotope.quadMapTri` (MR !921; `test_lean_zonotope_quad_map`: random zonotopes and the recorded calls of
TRAF22 and ROBE25_1; the lean backend's matmul and sums already follow its evaluation order, the
`nonzeroFilter` and the box of a single nonzero output are CORA.cpp's own). It needs an oracle tag that
includes !921. `zonotope/isIntersecting_` is `oracle` in its `'approx'` mode against
`zonotope.isIntersectingApprox` (the boxes, the one of `S` widened by `tol`; `test_lean_zonotope_measures`);
the `'exact'` mode is a linear program and stays untested against the oracle. The other float rows the benchmarks run were
read against the pinned oracle's op and kept out of `oracle`; each stays until the named change.

| row | op | why it is not bit-exact | what would close it |
|---|---|---|---|
| `zonotope/polytope` | `zonotope.polytope` (`Zonotope.hPolytopeFacets`) | the same facets, in the reverse order of `nchoosek`; sorted, the rows still differ in the last bits on random zonotopes (tested for 2 to 4 dimensions): CORA.cpp projects each cross product onto the orthogonal complement of its generators (`priv_zonotope_facet_normals`, kept for the gradient), CORALean normalizes the cross product itself | an oracle variant, or CORA.cpp dropping the projection (it costs the torch gradient) |
| `zonotope/vertices_` | `zonotope.vertices` | CORA's `aux_vertices2Dfast` walks `V(:,i+1) = V(:,i) + 2 g_i` and shifts by the box, CORALean's `Zonotope.polygon` evaluates each corner as `c - sum g + 2 sum_{j<k} g`: another evaluation order, and for more than two dimensions `2^h` corners instead of the hull | `Zonotope.polygon` as a cumulative walk |
| `conZonotope/supportFunc_`, `zonoBundle/supportFunc_` | `conZonotope.supportFunc`, `zonoBundle.supportFunc` | an upper bound from a checked dual certificate, CORA's value is the optimum of the linear program | none |
| `conZonotope/isFullDim`, `conZonotope/rescale` | `conZonotope.isFullDim` | certified one way; `ConZonotope.rescale` takes the ranges of the factors as arguments, CORA finds them by linear programs or Gauss-Jordan | an op `conZonotope.rescale` with the ranges, to test the scaling alone |
| `polytope/constraints`, `polytope/polytope` | none | `VPolytope.hPolytope` encloses the vertices along given directions `Aq`; CORA computes the facets of the hull | a facet enumeration in CORALean |
| `linearSys/reach` | `linearSys.reach` | CORALean's step is the interval-matrix one (`reachFloat`), tested by tolerance in `test_lean_linear_sys_steps_agreement` | the closed-form step |
| `contSet/interval/polyZonotope`, `contSet/zonotope/polyZonotope`, `polyZonotope/*` | none | the oracle has no polynomial zonotope codec until the poly-adaptive branch merges; then the generator order of the constructor (see above) decides | the ops of the section above |

## Support-function verify (PRs "oracle/verify-1" to "oracle/verify-4", CORALean MRs !948 to !950 and "oracle/verify-4")

`linearSys/verify` is `oracle`: every instance of the support-function algorithm (CBC, CBF, HEAT01/02,
ISS) runs CORALean's sound `LinearSys.verifySupportFuncFloat`, bit-exact on the lean backend at
`nearest` (`test_lean_linear_sys_verify_support`). The oracle serves its parts and the whole run:
`linearSys.verifyCorrection` (F and G, `test_lean_linear_sys_verify_correction`), `linearSys.expBall`
(`e^{AΔt}` and its balanced radius), `linearSys.energyGrowth` (the energy weight of a second-order
system and its growth factor) and `linearSys.verifySupportFunc` (the verdict, the passes, the last
step and the falsifying time; with an input `B`, `uc`, `GU`: the constant input's solution, the outer
support of the varying input and its inner support for falsification). Departures from CORA, for
soundness: the step's horizon also shrinks with an input set, and no unverified time is dropped.

## Taylor matrices (PR "oracle/taylor-matrices", CORALean MR !938; PR "oracle/bitexact-3", CORALean MR !944)

`linearSys/taylorMatrices` is `oracle`. The default path, which the benchmarks run, is bit-exact:
`LinearSys::correction_matrix_state`, `correction_matrix_input` and `expm_remainder` are CORALean's
`LinearSys.taylorMatricesFloat` (the oracle's `WCora`, `FCora`, `GCora`). That is CORA's algorithm at
`taylorTerms`, with `e^{|A| dt}` of the remainder enclosed by scaling and squaring where CORA uses
`expm`, and CORA's correction coefficient bounded from below (`corrCoeffLowTight`, CORALean MR for
`oracle/bitexact-3b`), so `F` and `G` agree with MATLAB CORA to 1e-13. Not bit-exact, and CORA's `expm` path instead: a batch, a gradient, no halving count below 128,
and matrices that overflow (each commented in `priv_taylor_corrections`). `CORACPP_TAYLOR=entrywise`
takes the entrywise matrices at the admissible order `eta_e` (`order`, `W`, `FEntrywise`,
`GEntrywise`), also bit-exact, but not the default: TSPS25 aborts with it, and DTN01 to DTN03 and
CVDP23 take 1.3x as long.

## Float rows batch (PR "oracle/float-rows")

Moved to `oracle` against the pinned oracle: `zonotope/zonoBundle`, `interval/zonoBundle`,
`zonotope/matZonotope` (`test_lean_conversions`: the conversion of the oracle from the same operand,
centre, generators and bounds equal bit for bit), `linearSys/outputSet` (`C R`, the oracle's with no
input), and, by the runs that already compare bit for bit, `location/reach`, `contDynamics/reach`
(the windows of `NonlinearSys::reach`) and `nonlinearSys/linearize`.
CORALean has no wire format for a constructor (`experiments/oracle/README.md`, "Float rows not
served"). A constructor row is `oracle` only if a run hands the oracle every field the constructor
stores: `specification` (type and set, `test_lean_specification_check`) and `nonlinearSys` (n, m, f,
`test_lean_nonlinear_sys_lin_reach`). The rows `linearSys` (c, D, k, E, F are never read),
`hybridAutomaton` and `location` (the names are never read), `transition` and `linearReset` (the
input matrix of the reset is never read) stay `ported`.

The oracle serves these, but CORA.cpp computes another set from the same operand, so the rows stay
`ported` or `property`:

| row | op | difference |
|---|---|---|
| `zonotope/and_` | `zonotope.and` | CORA.cpp's `intersect(Z2, "averaging")` (CORA's `priv_andAveraging`) differs from the oracle's on random zonotopes in the centre and the generators; an op that follows `priv_andAveraging` would close it |
| `polytope/conZonotope`, `polytope/zonoBundle` | `polytope.conZonotope`, `polytope.zonoBundle` | the oracle's conversions of a vertex polytope give a bundle of one member and a constrained zonotope with one constraint row (5 vertices: G 2x5, A 1x5); CORA.cpp's give one zonotope per facet row (5 members) and Scott et al.'s construction (G 2x7, A 5x7) |

`emptySet/polytope`, `fullspace/polytope`, `interval/project` and `interval/vertices_` have bit-exact
oracle tests (`test_lean_conversions`, `test_lean_interval_project_vertices`), but are `untested` rows
(no Python test), and the coverage checker raises a row to `oracle` only from `ported`.

## Neural networks, polynomials of order k (PR "nn/order-k", CORA.cpp#112)

The powers of a neuron and the layer of order k are served (`polyZonotope.evalPowers`, the layer
`poly` of `nnLayer.evaluatePolyZonotope`, PR "oracle/order-k") and tested bit for bit by
`test_lean_nn_order_k`, so `evaluatePolyZonotopeNeuron` is `oracle`. Still not formalized, so still
oracle debt: the fits of order k (`regression`, `ridgeregression`, `bernstein`, `taylor`, `bounds`,
`throw-catch`) and their errors (`minMaxDiffOrder`), which the oracle takes as data from CORA.cpp
(row `nnActivationLayer`, `property`); `reducePolyZono` (the reduction at `num_generators`, with the
shift `g / 2` of an all-even generator); CORA's `refine` (CORALean has `Schedule.refineLayer`) and
`getRefinableLayers`.

## Activation layers at order 1 (PR "oracle/bitexact-1")

`nnLeakyReLULayer`, `nnSShapeLayer`, `nnSigmoidLayer`, `nnTanhLayer` and `computeApproxPoly` are
`oracle` by `test_lean_nn_evaluate`, which runs the order-1 lines of `evaluate.cpp`, the path of every
AINNCS instance (`coverage/benchmarks.json`: none executes `priv_activation_poly.cpp`). CORA's ReLU is
`nnLeakyReLULayer` at alpha 0, whose `computeApproxPoly` is the ReLU's line (also checked against the
oracle's `leakyRelu` at alpha 0); sigmoid and tanh are `nnSShapeLayer`'s (`getDerBounds` is the
bracket's derivative bracket, `computeApproxPolyCustom` Singh's line); `computeApproxPoly` at order 1
is the regression line through 20 points. The output rows of `evaluate_with_input` (the closed loop's
call) are checked against the oracle's set too. The order-k path stays as above.

## Matrix sets (PR "oracle/matrixsets", CORALean branch "oracle/matrixsets")

| row | op | status |
|---|---|---|
| `matZonotope/reduce` | `matZonotope.reduce` (`Zonotope.reduceMat`) | `oracle`: CORA's `reduce(matZ,'girard',order)` as it is meant to run (CORA's code crashes, see CORALean's ledger), the stacked set reduced and unstacked |
| `matZonotope/contains` | `matZonotope.contains` (`Zonotope.containsMat`) | `property`: a certificate (`true` a proof, `false` refutes nothing) where CORA.cpp solves a linear program, different algorithms; `test_lean_mat_zonotope_contains_agreement` compares the decisions on members near the center and on matrices outside the box | none: a one-sided certificate stays `property` |
| `intervalMatrix/exponentialRemainder` | `intervalMatrix.exponentialRemainder` (`MatExp.exponentialRemainder`) | `oracle`: CORA.cpp now takes `e^|A|` by CORALean's scaling and squaring instead of `expm` (a batch or a gradient keeps `expm`) |
| `linearSys/errorSolution_adaptive` | `linearSys.errorSolution_adaptive` | `oracle`: the radius lin-adaptive tunes by (`priv_error_radius_float`) is the oracle's bit for bit; the public `LinearSys::error_solution_adaptive` still cuts by the gain with `expm` and agrees by tolerance |
| `matZonotope/expmOneParam` | none | debt: CORA.cpp's `K(r)` uses `pow(r, k) / k!` and point products; a float `MatExp.expmOneParam` whose rounding CORA.cpp follows (coefficient brackets, products into the error box) is the missing piece |

## Latent-space verify (PR "nn/verify-cpp", CORA.cpp#213)

`NeuralNetwork::verify` (CORA's `neuralNetwork/verify`, ICLR 2026) is bit-exact on lean where the
oracle serves the op: the batched zonotope with a slot per error at a point center
(`neuralNetwork.evaluate`, `singh`) and Fourier-Motzkin in CORALean's order
(`conZonotope.fourierMotzkinBounds`). Requested:

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `zonotope.zonotackFactors` | `Zonotope.zonotackFactors` (`ContSet/Zonotope/Real/Operations/ZonotackFactors.lean`; no float definition yet) | `Z`, `a` (row) | `beta` (vector) |
| `neuralNetwork.encloseUnsafeInputs` | `SetBasedNet.encloseUnsafeInputs` (`NN/Operations/EncloseUnsafeInputs.lean`; no float definition yet) | `layers`, `Z` (input), `A`, `b` | `C`, `d` (the rows on the input's factors) |
| `neuralNetwork.evaluateZonotopeBatch` | none: CORA's slots (`num_init_gens`, `num_approx_err`), the interval center and the clamps of split neurons | `layers`, `Z`, `dr`, `slots`, `order`, `splits` | `Z`, `dr` |
| `nnLayer.evaluateZonotope` at `poly_method` `"bounds"` | none: CORA's secant line of sigmoid and tanh with the errors at the extreme points of `f(x) - m x` (CORALean has Singh's and the regression lines) | `layer`, `Z` | `Z` |
| `neuralNetwork.verify` | `SetBasedNet.verify` (`NN/Operations/Verify.lean`) with a logged split sequence | `layers`, `X`, `A`, `b`, `splits` | the verdict and the queued sets |
| `conZonotope.dualBound` | none: weak duality at given multipliers (the study's `dualCertificate`) | `C`, `d`, `G`, `lambda` | `h` |
| `neuralNetwork.fradGradient` | none: the gradient of the output's F-radius in each generator's value through the lines' slopes, offsets and errors (CORA's `backpropZonotopeBatch` with `exact_backprop`); CORA.cpp checks it against the central difference | `layers`, `Z`, `dr`, `slots`, `splits` | the gradients |

Phase 2 (CORA.cpp#227, #228) adds steps no CORALean op covers yet:

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `nnLayer.evaluateZonotope` with ReLU slopes | none: the ReLU line `a x + h / 2 +- h / 2`, `h = max(-a l, (1 - a) u)`, `a` in [0, 1] (`zonotope-alpha`); soundness: `relu(x) - a x` in `[0, h]` on `[l, u]` | `layer`, `Z`, `alpha` | `Z` |
| `neuralNetwork.backwardBound` | none: CROWN's backward linear bound of rows of the output over a box, at given pre-activation bounds (chord above, slope below; the zonotope line's parallels for sigmoid/tanh) | `layers`, `x`, `r`, `l`, `u`, `rows`, `alpha` | the lower bounds |
| `neuralNetwork.reluChordRow` | none: the row `ReLU(x) <= u (x - l) / (u - l)` on the factors of a layer's input and output sets (`relu_constraint_faces` `triangle`) | `Zin`, `Zout`, `l`, `u`, `j` | `a`, `b` |

The dual certificate (`dual_certificate`) is the `conZonotope.dualBound` row above, applied to the
specification rows: its request fields are the rows `C`, `d` of the factors (the splits and the
three ReLU faces), `w = G a` and `lambda`.

The slopes' gradient is checked against the central difference, and the witness test runs all of
Phase 2 (0 wrong of 60). Three theorems would close the soundness rows of docs/nn_verification.md:

- `dual_bound_sound` (weak duality): for `C : Matrix p q`, `d : Vec p`, `w : Vec q`, `lambda >= 0`
  and every `beta` in `[-1, 1]^q` with `C beta <= d`: `w . beta >= -sum_j |w_j + (C^T lambda)_j| -
  lambda . d`; and its float form at `nearest` with the rounding bound of the sum.
- `alpha_line_sound`: for `l < 0 < u`, `a` in `[0, 1]` and `x` in `[l, u]`: `relu x - a x` lies in
  `[0, max (-a l) ((1 - a) u)]`, so `a x + h / 2 +- h / 2` is an `IsRelaxationOf` of the ReLU.
- `backward_bound_sound`: for a network of affine layers and activations with pre-activation
  bounds `[l_k, u_k]` that hold on the box, linear bounds `s_l x + b_l <= f(x) <= s_u x + b_u` on
  each `[l_k, u_k]` and a row `a`: CROWN's backward value (`Lambda` through `W_k^T` and the
  positive and negative parts of the slopes, the constant, then `Lambda x - |Lambda| r`) is at
  most `a . N(x')` for every `x'` in the box `(x, r)`.

Phase 3 (convolutional networks) adds layers no CORALean op serves on a zonotope:

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `nnLayer.evaluateZonotope` of a `MaxPoolLayer` | none on a zonotope (CORALean's `MaxPoolLayer` enclosure is of intervals). CORA.cpp's front end lowers a max pool to `max(a, b) = b + relu(a - b)`, pairwise over the window, round by round: a grouped convolution of the differences, its ReLU (the existing line and error), then a gather of the ReLUs plus the second item of each pair as a residual. Request: the same composition at `nearest`, and `maxPool_encloses` (the composition of exact affine maps and the ReLU enclosure contains every `maxpool(x)`, `x` in `Z`) | `kernel`, `stride`, `dilation`, `Z` | `Z` |
| a MatMul of two values (cGAN's self-attention, vit_2023) | none. CORA.cpp lowers `sum_t a_t b_t` to two gathers into one product layer and a 1 x 1 convolution of ones over its first half. Request: `matmul_encloses` for that composition (as `mul_encloses`) | the two branches' `Z` | `Z` |
| `nnLayer.evaluateZonotope` of a floor layer (no CORA layer; ml4acopf's `linear-residual` networks count the turns of an angle, `x - 2 pi floor(x / (2 pi))`) | none. CORA.cpp's line: slope 0, offset `(floor l + floor u) / 2`, radius `(floor u - floor l) / 2` (exact in floats: integers and their halves); no polynomial (a step at every integer). Request: `floorRelaxation` with `encloses` | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of a cube layer (no CORA layer; nn4sys' pensieve_parallel, `Pow(x, 3)`) | none. CORA.cpp's `priv_cube_line`: slope `m = l^2 + l u + u^2` (the chord's), offset and radius the middle and half of the range of `x^3 - m x` over `{l, u, clamp(+-sqrt(m / 3), l, u)}`, the radius widened by `64 eps max(|l|, |u|)^3`; `max(|l|, |u|)^3` beyond the float range fails as out of range. Request: `cubeRelaxation` with `encloses` (for every `x` in `[l, u]`, `x^3` lies in `m x + offset +- radius`) at every `FloatOps` | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of an inverse layer (`1 / x` for `x > 0`; pensieve's `x^3 / sum x^3` as a product with it) | none. CORA.cpp's `priv_inverse_line`: fails as out of range unless `l > 0`; slope `m = -(1 / l)(1 / u)`, offset and radius the middle and half of `[g(sqrt l sqrt u), max(g(l), g(u))]` with `g(x) = 1 / x - m x` (convex), the radius widened by `32 eps / l`. An inverse layer that reads a cube, exp or ReLU layer alone also bounds its input by the image of that layer's box (`[l^3, u^3]`, `[e^l, e^u]`, `[max(l, 0), max(u, 0)]`; `aux_box_image`). Request: `inverseRelaxation` with `encloses` for `0 < l <= u`, and `quotient_encloses` for the composition `x^3 / sum x^3` (the product line of `[x^3; 1 / s]`) | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of a sin or cos layer (no CORA layer; ml4acopf's power flows `cos(t_i - t_j)`, `sin(t_i - t_j)`) | none. CORA.cpp's `priv_trig_line`: the tangent at the middle `c` (slope `f'(c)`, offset `f(c) - f'(c) c`, radius `d^2 / 2` with `d` the larger half width times `1 + 4 eps`, widened by `16 eps (1 + |m| max(|l|, |u|) + d)`), or slope 0 over `priv_trig_box` where narrower: the ends' values `+-16 eps`, 1 (-1) where a maximum (minimum) `a + 2 pi k` may lie in `[l, u]` (`ceil` and `floor` of the widened quotients `(l - a) / 2 pi`, `(u - a) / 2 pi`), clamped to `[-1, 1]`. Request: `sinRelaxation`, `cosRelaxation` with `encloses` at every `FloatOps` | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of an `AddLayer` (a residual) | none (CORALean's networks are chains). CORA.cpp adds the generator rows of an earlier output to the first rows of the later set (the same factors) and the interval-center radii | the two branches' `Z` with their factor counts | `Z` |

The sign line and the square's chord-tangent band are served since oracle-20261123-9e02265cc
(`nnLayer.relaxation` kinds `sign` and `square`, `signRelaxation`, `squareChordTangentRelaxation`,
and the layer kinds `sign` and `square`): `test_lean_nn_relaxation` and `test_lean_nn_evaluate`
compare them bit for bit. Since oracle-20261123-662d313b3 (CORALean !970) the oracle also serves the exp and
reciprocal lines (`expChordRelaxation`, `reciprocalRelaxation`, the closed-form minimum CORA.cpp
follows), the product line (`nnLayer.productLine`, `productLine_encloses`), the box of a softmax's
sums (`nnLayer.softmaxSumBounds`, `softmaxSumBounds_encloses`), the layer `conv` (ONNX's Conv at
its stencil matrix) and the dual certificate (`conZonotope.dualBound`): `test_lean_nn_relaxation`.
A residual Add is `zonotope.exactPlus` at the earlier branch's generators. `softmax_order` is
`softmax_le_iff`; the MaxPool and MatMul identities are `max_eq_add_relu_sub` and
`dot_eq_sum_sq_sub_sq`.

The convolution is checked against its definition and its matrix (`test_neural_network_conv`); a
gather (a permutation, a pad, a nearest upsample) is exact.

The ReLU rows and the iterated refinement are not formalized. The coverage rows `verify`,
`computePGDAttack` and `computeFGSMAttack` are oracle debt until then.

## NN linear layers (PR "nn/port-rest", batch 1)

The oracle (`oracle-20261123-722cbe1e7`) serves the layer kinds `linear`, `relu`, `leakyRelu`, `sigmoid`,
`tanh` and `poly`, and no other linear layer. CORALean has float definitions of the layers below
(`NN/Layers/Definitions/`: `BatchNormLayer`, `AvgPoolLayer`, `WeightLayerFloat`); CORA.cpp's
`NeuralNetwork::Layer::identity`, `elementwise_affine`, `batch_norm` (inference, folded into one map),
`avg_pool2d` and `lip_constr_linear` are tested by hand-computed cases only, so their rows are oracle debt.

| op (proposed) | request fields | reply |
|---|---|---|
| `nnLayer.evaluateZonotope`, layer kind `elementwiseAffine` | `scale`, `offset`, `Z` | `Z` |
| layer kind `batchNorm` (inference) | `scale`, `offset`, `mean`, `var`, `epsilon`, `shape` (C, H, W), `Z` | `Z`; CORALean's order of the three steps decides whether CORA.cpp's folded map is bit-exact |
| layer kind `avgPool` | `pool`, `stride`, `pads`, `shape`, `Z` | `Z` |
| layer kind `lipLinear` | `W`, `b`, `lambda`, `Z` | `Z` |

### Identifier and structure helpers of the polynomial zonotope (port batch pzrest-1)

CORALean has float definitions of these CORA methods (`src/CORALean/ContSet/PolyZonotope/Float/Operations/`).
**Served** (oracle-debt batch 4, `PolyZonotopeOps2.lean`; bit for bit by `test_lean_poly_zonotope_ids` and
`test_lean_poly_zonotope_maps`): `getSubset` and `resolve` (numbers), `replaceId`, `restoreId`, `rmId`,
`onlyId`, `noIndep`, `lift`, `isBounded`, `isFullDim`, `isequal`, `representsa` (point), `sum`, `restructure`,
`jacobian`, `linComb`, `convHull`, `matMap`, `cubMap`, `partZonotope` and `outerApprox`. The sets are `P`, `Q`
(as `polyZonotope.plus`), the identifier argument is `ids` (a request's `id` is the request id), `jacobian`
takes `factor`, the predicates take no tolerance (`tol` is 0). `isequal` is structural and in order (a `true`
is a proof), CORA.cpp's compares up to order: the test restricts to the sets where the two agree. Still oracle
debt: `getSubset` with a box, `subs`, `stack` (and `polyMap`, which has no float definition). The rows below
keep the original proposals; read `id` as `ids` and `pZ` as `P`.

| op (proposed) | CORALean definition | request fields | reply and order |
|---|---|---|---|
| `polyZonotope.getSubset` | `PolyZonotope.getSubset`, `getSubsetCoef` (`GetSubset.lean`) | `pZ`, `id`, `val` (numbers) or `I` (a box) | the columns as CORA.cpp: the original columns, then the appended terms of each expanded generator; the coefficients of `(c + r a)^e` by CORA's `conv` |
| `polyZonotope.replaceId`, `restoreId`, `rmId`, `noIndep` | `ReplaceId.lean`, `PadBack.lean`, `RmId.lean`, `NoIndep.lean` | `pZ`, `id` / `idOld`, `idNew` | exact: nothing rounds; `rmId` adds the columns of equal exponents in order |
| `polyZonotope.onlyId` | `PolyZonotope.onlyId` | `pZ`, `id` | the two parts (`pZId`, `pZr`) |
| `polyZonotope.isequal`, `isFullDim`, `isBounded`, `representsa` | `Isequal.lean` (structural, tolerance 0), `IsFullDim.lean` (certified true only), `Basic.lean` | `pZ`, `S`, `tol` | a Boolean; CORA.cpp's `isequal` compares up to order after `compact`, CORALean's in order, so the two agree on `true` only |
| `polyZonotope.sum` | `PolyZonotope.sum` (`Float/Operations/Sum.lean`) | `pZ`, `L` (a list of sets over `pZ`'s factors) | the columns of `pZ`, then those of each list entry in order, the center `c + (L0.c + (L1.c + ...))` rounded to nearest; no merging |
| `polyZonotope.resolve` | `PolyZonotope.resolve` (`Float/Operations/Basic.lean`, `getSubset` at `[x, x]`) | `pZ`, `id`, `x` | the factors in ascending identifier order, one `getSubset` each with the generator scaled in turn; CORA.cpp's order after that (equal exponents merged, constants in the center) |
| `polyZonotope.restructure` | `PolyZonotope.restructure` (`Float/Operations/Restructure.lean`, `reduceBy` then `absorbRounding`) | `pZ` | `G` then the diagonal of the row sums of `|GI|` (last to first) then `n` zero columns, each on a fresh factor |
| `polyZonotope.subs` | `PolyZonotope.subs`, `subsTerms`, `subsXprod`, `subsPow`, `subsMul` (`Subs*.lean`) | `pZ`, `pZin`, `id` | the terms of each generator in `subsXprod`'s order, the coefficient times the generator once per term; equal exponents merged afterwards (CORA.cpp's order) |
| `polyZonotope.representsa` (`point`) | `PolyZonotope.representsa_point` | `pZ`, `tol = 0` | a Boolean (sufficient only) |
| `polyZonotope.linComb` | `PolyZonotope.linComb` (`Float/Operations/LinComb.lean`, `LinCombGI.lean`) | `pZ1`, `pZ2` | the dependent generators `[G1/2, G2/2, G1/2, -G2/2, (c1 - c2)/2]` and the independent `pairSum` then `pairDiff` columns, the center `(c1 + c2)/2`, factors side by side and `t` last; nothing merged |
| `polyZonotope.lift` | `PolyZonotope.reindex` | `pZ`, `N`, `dims` | exact: the coordinates permuted |
| `polyZonotope.matMap` | `PolyZonotope.matMap` (`Float/Operations/MatMap.lean`) | `pZ1`, `pZ2`, `n`, `k`, `m` | the `(h1 + 1)(h2 + 1)` pair products `matMulDown` (the dot over `k` nested to the right) in `finProdFinEquiv` order, center 0, the three independent blocks `GI1 ext2`, `GI1 GI2`, `ext1 GI2`; factors over the union of the identifiers |
| `polyZonotope.cubMap` | `PolyZonotope.cubMap`, `cubMapDep`, `cubMapGDown`, `Zonotope.cubContractDown` (`Float/Operations/CubMap*.lean`) | `pZ`, `T` | after `absorbRounding`, the generator of the triple `(a, b, c)` is `sum_k a_k (b' T[i][k] c)` with the sums nested to the right; center 0 |
| `polyZonotope.partZonotope` | `PolyZonotope.partZonotope`, `partPatterns` (`Float/Operations/PartZonotope.lean`) | `pZ`, `id` | the groups in the order of `List.dedup` (the last occurrence of each pattern, not the first as the docstring says); the halved block, then its copy |
| `polyZonotope.polyMap` | `PolyZonotope.evalPoly` models the univariate case only | `pZ`, `coeff`, `E` | the general polynomial map has no float definition; the request is for the multivariate term product with `mul` after `absorbRounding` |
| `polyZonotope.convHull` | `PolyZonotope.convHull` (`Float/Operations/ConvHull.lean`) | `pZ1`, `pZ2` | `linComb(linComb(pZ1', pZ1'), linComb(pZ2', pZ2'))` of the sets without independent generators, the independent block `pairSum` then `pairDiff` then one zero column |
| `polyZonotope.jacobian` | `PolyZonotope.jacobian` (`Float/Operations/Jacobian.lean`) | `pZ`, `k` | every column kept, `E k i` copies added from the right, the exponent of `k` lowered (not below 0), center 0 |
| `polyZonotope.outerApprox` | `PolyZonotope.outerApprox`, `boxOut` (`Float/Operations/OuterApprox.lean`, `BoxOut.lean`) | `pZ`, `tol`, `id` | the dependent generators left, then the independent ones with the diagonal of the box appended; the box radius the removed magnitudes plus the taken ones, each sum added up |



## NN nonlinear layers, gradients and forms (PR "nn/port-rest-2", batch 2)

`NeuralNetwork::exp_layer`, `softmax_layer` (1 / sum_j exp(x_j - x_i), two layers; CORA's exp(x - max)
/ sum is the same function but not the same floating-point steps), `max_pool2d_layer` (b + relu(a - b)
over the window's pairs), `evaluate_zonotope_batch`, `backprop`, `backprop_interval_batch`,
`propagate_bounds`, `input_neuron_order` and `normal_form` are tested by hand-computed cases, samples
and the Jacobian of `sensitivity`. `evaluate_zonotope_batch` of dense, conv, relu, sign and square layers
is bit-identical to the oracle's `neuralNetwork.evaluate` (`test_lean_nn_batch`, pin f5ff0c401); max
pool has no layer kind there. `NeuralNetwork::evaluate` of exp and softmax layers is bit-identical to it
since oracle-20261124-406069a06 (`test_lean_nn_evaluate`). CORALean has float models of the activations and of
`neuralNetwork.backwardBound`, but no op for the gradient passes or the forms, so their rows are
oracle debt.

| op (proposed) | request fields | reply |
|---|---|---|
| `neuralNetwork.backprop` | `layers`, `x`, `gradOut` | the input gradient |
| `neuralNetwork.backpropIntervalBatch` | `layers`, `l`, `u`, `gl`, `gu` | `gl`, `gu` (Gowal et al. 2019) |
| `neuralNetwork.propagateBounds` | `layers`, per-layer `l`, `u`, `i` | the bounds per layer |
| `neuralNetwork.normalForm` | `layers` | the merged layers |

## NN softplus, root, inverse square root, group sort and the reduction (PR "nn/port-rest-3", batch 3)

The lines of `Activation::softplus`, `root` and `inv_sqrt_root` (`activation_lines.cpp`) are served since
oracle-20261124-406069a06 (CORALean !975: `nnLayer.relaxation` kinds `softplus`, `root`, `invSqrt` with
`method`, `softplusLineRelaxation`, `rootLineRelaxation`, `invSqrtRootLineRelaxation`), and so are the layer
kinds `exp`, `reciprocal`, `cube`, `inverse`, `softplus`, `root` and `invSqrt` of `nnLayer.evaluateZonotope`
and `neuralNetwork.evaluate` (a softmax as `evaluate` lowers it: linear, exp, linear, reciprocal):
`test_lean_nn_relaxation` and `test_lean_nn_evaluate` compare them bit for bit. The lines read: Singh's slope
f'(l) (softplus) or f'(u) (root, inv_sqrt_root), or the 20-point regression slope m; then the band of
f(x) - m x over the ends and its closed-form inner extremum at f'(x) = m (softplus: -(m log m) - t log t,
t = 1 - m; root: 1 / (4 m); inv_sqrt_root: (3/2) c, c = (-2 m)^(1/3)), not f at a rounded stationary point.

`group_sort` (points and backprop only, as CORA), `composite_layer`, `reduce_generators`, `merge_buckets`,
`find_region_polys` and `compute_reduced_network` have no op in the oracle, so they are tested by properties:
a sort is a permutation whose gradient matches finite differences; the reduced network's output set contains
the original outputs.

| op (proposed) | request fields | reply |
|---|---|---|
| `nnLayer.evaluate`, kind `groupSort` | `groupSize`, `x` | the sorted points (stable, padded with inf) |
| `nnLayer.evaluateZonotopeBatch`, kind `generatorReduction` | `maxGens`, `c`, `G` | `c`, `G` (CORA keeps the shortest generators) |
| `neuralNetwork.computeReducedNetwork` | `layers`, `Z`, `bucketType`, `bucketTol`, `reductionRate` | reduced layers, error intervals, output set |

## NN on Taylor models and constrained zonotopes (PR "nn/port-rest-4", batch 4)

`NeuralNetwork::evaluate_taylm` is served since oracle-20261124-3013bdbc0 (CORALean !976:
`neuralNetwork.evaluateTaylm`, `taylm.constant`, `taylm.fromRange`; `Layer.affineTaylm`,
`Layer.activationTaylmFit`, `reluSinghQuadRelaxation`): `test_lean_nn_evaluate_taylm` compares ReLU
networks at orders 1 and 2 (singh) and per neuron, and sigmoid and tanh at order 1 (singh, regression)
bit for bit. At order 1 a neuron takes the line of `evaluate` (`priv_line`) as it is; the ReLU's parabola
of order 2 has the error of the closed-form ranges of p and p - x on the two pieces (the vertex value c -
b^2 / (4 a), not p at a rounded root) and the radius `max(hi - mid, mid - lo)`. Not served: sigmoid, tanh
and the others at orders 2 and more (the least-squares fit and the sampled error have no CORALean
definition), ReLU at 3 and more (the extrema of a cubic are not closed form), and the poly methods other
than singh and regression.
`evaluate(ConZonotope)` has no op in the oracle, so it is tested by properties: the sampled outputs lie in
the result, exact cases by hand.

| op (proposed) | request fields | reply |
|---|---|---|
| `neuralNetwork.evaluateTaylm` at order >= 2 of sigmoid, tanh, ..., ReLU at order >= 3, the other poly methods | as served, `order`, `poly_method` | one model per output |
| `neuralNetwork.evaluateConZonotope` | `layers` (identity, ReLU), `c`, `G`, `A`, `b`, `boundApprox` | `c`, `G`, `A`, `b` of the output set |

## Explanation of a classification (PR "nn/port-explain", batch 5)

`NeuralNetwork::explain` and `compute_fgsm_attack` have no op in the oracle (newest tag
oracle-20261123-feef03ae2). `explain` is a loop over `neuralNetwork.evaluate` on zonotopes (served) and
`computeReducedNetwork` (requested above), so it is tested by properties: a hand-computed linear case,
the standard method equal to the abstract one at the rate 1, and sampled soundness of the freed set.

| op (proposed) | request fields | reply |
|---|---|---|
| `neuralNetwork.explain` | `layers`, `x`, `label`, `epsilon`, `method`, `featOrder`, `refineSteps`, `outputThreshold` | the freed features, the order tried |
| `neuralNetwork.fgsmAttack` | `layers`, `x`, `t`, `epsilon`, `mask` | the moved points `x - alpha sign(grad)` |

## Zonotope batch gradients (PR "nn/port-rest-6", batch 6)

`NeuralNetwork::backprop_zonotope_batch` has no op in the oracle (newest tag oracle-20261123-26a33fe5b):
`neuralNetwork.backwardBound` is CROWN's bound, not a gradient of the output set, so it is tested by
properties (the chain rule where all neurons are stable, central differences of the exact gradients, the
held slopes by hand). The ConvTranspose layer is tested against the oracle's `conv` kind at stride 1
(a convolution with the flipped kernel and the pads `k - 1 - p`).

| op (proposed) | request fields | reply |
|---|---|---|
| `neuralNetwork.backpropZonotopeBatch` | `layers`, `c`, `G`, `gc`, `gG`, `exact` | the gradients at the input's `c` and `G` |

### Discrete-time nonlinear systems (port batch nlsdt)

CORALean has `NonlinearSysDT.reach`, `simulate` and `linearize` (generic over the scalar and the set
representation, with a `LinStep` certificate whose constant box encloses `f(x*, u*) - A x*`), but the
pinned oracle serves no `nonlinearSysDT` operation, so CORA.cpp tests `reach`, `post`, `outputSet`
and `simulate` as properties (sampled trajectories lie in the sets) and by hand cases. `f` and the
output equation are symbolic expressions; the oracle needs the same expression language as
`nonlinearSys` (`docs/oracle-requests.md`, nonlinear section). CORA.cpp's recipe for one step of
`nonlinearSysDT.reach`, which a CORALean batch can instantiate its `LinStep` with:

| op (proposed) | request fields | reply and order |
|---|---|---|
| `nonlinearSysDT.simulate` | `f`, `x0`, `u` (a constant or one per step), `steps` | the states `f(x, u)` applied `steps` times, each by nearest evaluation of the expression tree in its written order |
| `nonlinearSysDT.post` | `f`, `dt`, `R` (c, G), `U` (c, G), `zonotopeOrder`, `tensorOrder` (2 or 3), `errorOrder` | the zonotope after one step and the added error, as the recipe below |

Recipe of `post` (CORA's `linReach`, `priv_linearize`, `priv_linError_mixed_noInt`, `priv_linError_thirdOrder`):

1. Linearization point: `p_x` the center of `R`, `p_u` the center of `U` (the input 0 without a set). No half step of the flow
   (unlike `nonlinearSys`): `f0 = f(p_x, p_u)`, `A`, `B` the Jacobians there.
2. Affine part: `Rdelta = R - p_x`; `Udelta = B (U - p_u)`; `Rtp = A Rdelta + (Udelta + f0)`, generators in the order
   `[A G_R, B G_U]`, center `f0`.
3. Error set, from the boxes `IHx = interval(Rdelta)`, `IHu = interval(U - p_u)` and `Z = cartProd(reduce(Rdelta, errorOrder), U - p_u)`:
   - Tensor order 2: the Hessian of each component as an interval matrix over `IHx + p_x` and `IHu + p_u` (a second derivative that
     vanishes identically is the exact 0). `H_mid` the `0.5 (inf + sup)`, `H_rad` the `0.5 (sup - inf)` of each entry;
     `error = 0.5 quadMap(Z, H_mid) + zonotope([-e, e])` with `e_i = 0.5 (dz' H_rad_i) dz`, `dz = max(|inf|, |sup|)` of `[IHx; IHu]`
     (the row `dz' H_rad` first, then times `dz`).
   - Tensor order 3: `error = 0.5 quadMap(Z, H(p_x, p_u))` plus the box `1/6 sum_j dz_j (dz' T_ij dz)` over the `j` with a nonzero
     third derivative `T_ij` evaluated over the boxes (the quadratic form `(dz' T_ij) dz` first, times `dz_j`), `dz = [IHx; IHu]`
     (not absolute values).
   - Both: reduced to `zonotopeOrder` n generators (Girard).
4. Result: `reduce(Rtp + error, zonotopeOrder)`; the first set of `reach` (the initial set) is not reduced.

The output set follows `outputSet.m` the same way: `y0 = g(p_x, p_u)`, `J (R - p_x) + D (U - p_u) + y0`, plus the error of the
tensor order of the output equation (order 2: the box `0.5 dz' max|H_i| dz` over `R` and `U` themselves; order 3: the Hessian at the
point maps `Z - p` and the third derivatives over the boxes give the box).


## Specifications (PR "port/spec")

CORALean has `Specification` and `Specification.check` only (route b, no scalar of its own); the
operations below have no Lean definition yet, so their CORA.cpp tests are property tests. A request
would need the definition first; the oracle then serves them over a polytope `{"A","b"}`.

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `specification.robustness` | none (`Specification.distancePolyPointCora` models `aux_distancePolyPoint` only) | `kind`, `A`, `b`, `Ae`, `be`, `p` | the signed distance; the three branches (one hyperplane, the vertex of `n` rows, the 1-norm program) and the equalities as two rows each |
| `specification.inverse` | none | `kind`, the polytopes | the unsafe polytopes covering the complement of the intersection, in CORA's order (`safeSet2unsafeSet`) |
| `specification.project` | none (`polytope.project` is served) | `kind`, `A`, `b`, `dims` | the projected polytope |

## STL time intervals (PR "port/spec2")

CORALean has float definitions of `STLInterval` (`Specification/STLInterval/Operations/*.lean`, generic over the scalar, rounding outward with `addUp`/`subDown`, which is plain arithmetic at `nearest`), but the oracle serves none of them. CORA.cpp follows those definitions; its tests are hand-computed cases and property tests on a grid of times. An interval travels as `{"lb","ub","lc","rc"}` (`ub` null for `inf`) or `{"empty":true}`; every op returns an interval, a bool or an endpoint `{"value","closed"}`.

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `stlInterval.stlInterval` | `STLInterval.stlIntervalBounds` | `lb`, `ub`, `lc`, `rc` | the normalized interval (infinite end open, empty by order) |
| `stlInterval.and_` | `STLInterval.and_` | `I`, `S` | the intersection |
| `stlInterval.plus` | `STLInterval.plus`, `plusNum`, `plusInterval` | `I`, `S` or number `c` or interval `J` | the sum; lower bound `max(0, lb1 + lb2)` for two intervals |
| `stlInterval.minus` | `STLInterval.minus`, `minusNum`, `minusInterval` | as `plus` | the difference, lower bound clamped to 0 (closed there) |
| `stlInterval.minkDiff` | `STLInterval.minkDiff` | `I`, `sub`, `type` | the Minkowski difference; unbounded minus unbounded is unbounded |
| `stlInterval.leftClosure`, `rightClosure` | `STLInterval.leftClosure`, `rightClosure` | `I` | the interval with that end closed |
| `stlInterval.toLeft`, `toRight` | `STLInterval.toLeft`, `toRight` | `I` | `[0, lb]` / `[ub, inf)` with the end flipped; empty input: error |
| `stlInterval.infimum`, `supremum` | `STLInterval.infimum`, `supremum` | `I` | the bound and whether it is closed |
| `stlInterval.interval` | `STLInterval.interval` | `I` | the closed interval of the bounds |
| `stlInterval.contains_` | `STLInterval.containsFloat` | `I`, `S` (interval), `tol` | the bool |
| `stlInterval.isequal` | `STLInterval.isequalFloat` | `I`, `J`, `tol` | the bool |
| `stlInterval.representsa_` | `STLInterval.representsaFloat` | `I`, `type`, `tol` | the bool |

## STL formulas (PR "port/spec3")

CORALean has float definitions of the syntax and the rewriting of CORA's `stl` class (`STL.Expr`, `STL.Tree`, `Specification/STL/Operations/*.lean`), generic over the scalar and independent of any rounding except `maximumTimeFloat`, `minimumTimeFloat` and `combineNextFloat`, but the oracle serves none of them. CORA.cpp follows those definitions; its tests are hand-computed cases and property tests (the meaning of random non-temporal formulas on a grid of states). A formula travels as a tree `{"op": "and" | "or" | "not" | "until" | "release" | "finally" | "globally" | "next" | "true" | "false" | "cmp", ...}` with the window as `{"lb","ub"}`, a comparison as `{"cmp": "<" | "<=" | ">" | ">=", "lhs": expr, "rhs": expr}` and an expression as `{"var": i}`, `{"num": c}`, `{"op": "+" | "-" | "*" | "^", ...}`; every op returns a tree, a list of trees, a bool or a number.

| op (proposed) | CORALean definition | request fields | reply |
|---|---|---|---|
| `stl.lt`, `le`, `gt`, `ge` | `STL.Tree.compare` (`lt`, `le`, `gt`, `ge`) | `e1`, `e2` | the comparison node, the sides arranged as CORA does |
| `stl.plus`, `minus`, `times`, `power` | `STL.Expr.plus`, `minus`, `times`, `power` | `e1`, `e2` or `e`, `k` | the expression, a numeric operand moved as CORA's operators do |
| `stl.desugar`, `negationNormalForm`, `eliminateTrueFalse` | `STL.Tree.desugar`, `negationNormalForm`, `eliminateTrueFalse` | `phi` | the rewritten formula |
| `stl.conjunctiveNormalForm`, `disjunctiveNormalForm` | `STL.Tree.conjunctiveNormalForm`, `disjunctiveNormalForm` | `phi`, `nnf` | the normal form |
| `stl.combineNext` | `STL.Tree.combineNextFloat` | `phi` | the formula with nested `next` merged where the sum is exact |
| `stl.assignIdentifiers`, `getClauses` | `STL.Tree.assignIdentifiers`, `getClauses` | `phi`, `type` | the numbered parts, the clauses |
| `stl.getVariables`, `isPredicate`, `isLinearPredicate` | `STL.Tree.getVariables`, `isPredicate`, `isLinearPredicate` | `phi` | the variable indices, the bool |
| `stl.maximumTime`, `minimumTime` | `STL.Tree.maximumTimeFloat`, `minimumTimeFloat` | `phi` | the time |
