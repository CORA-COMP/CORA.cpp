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
| `nnLayer.evaluateZonotope` of a `ConvLayer` | `ConvLayer` (Dyadic) on points only: the convolution of the center and of every generator row (the generator axis as a batch), groups, strides, dilations, pads | `kernel`, `input` (C, H, W), `stride`, `dilation`, `pads`, `groups`, `Z` | `Z` |
| `nnLayer.evaluateZonotope` of a `MaxPoolLayer` | none on a zonotope (CORALean's `MaxPoolLayer` enclosure is of intervals). CORA.cpp's front end lowers a max pool to `max(a, b) = b + relu(a - b)`, pairwise over the window, round by round: a grouped convolution of the differences, its ReLU (the existing line and error), then a gather of the ReLUs plus the second item of each pair as a residual. Request: the same composition at `nearest`, and `maxPool_encloses` (the composition of exact affine maps and the ReLU enclosure contains every `maxpool(x)`, `x` in `Z`) | `kernel`, `stride`, `dilation`, `Z` | `Z` |
| `nnLayer.evaluateZonotope` of a `SquareLayer` (`x^2`; CORA's `nnSquareLayer` has no zonotope method) | `squareRelaxation` (`Relaxation.ofDeriv`, the chord slope `l + u` in the bracket `[2l, 2u]`): its error is `(u - l)^2`, eight times the tightest parallelogram's. CORA.cpp's verify takes the parallelogram between the chord and the parallel tangent: slope `l + u`, offset `-(l u + m^2) / 2` with `m = (l + u) / 2`, radius `(u - l)^2 / 8` widened by `8 eps max(l^2, u^2)` (CORA.cpp only). Request: that line as a float definition (`squareChordTangentRelaxation`) with `encloses` at every `FloatOps`, and its `nearest` op; CORA.cpp then follows it bit for bit | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of an exp and a reciprocal layer (no CORA layer; a softmax as `1 / sum_t exp(x_t - x_q)`, CORA's `nnSoftmaxLayer` has no zonotope method) | none. CORA.cpp's lines: exp with the chord slope `m`, offset and radius the middle and half range of `exp(x) - m x` over `{l, u, clamp(log m)}`; `1 / max(x, 1)` with `m = -1 / (max(l, 1) max(u, 1))` over `{l, u, clamp(1), clamp(sqrt(max(l, 1) max(u, 1)))}`; each radius widened by `16 eps` of the largest term (CORA.cpp only). Request: `expRelaxation` and `reciprocalRelaxation` with `encloses`, and `softmax_eq` (the composition equals the softmax, the term `t = q` being 1) | `l`, `u` | `slope`, `offset`, `radius` |
| a MatMul of two values (cGAN's self-attention, vit_2023) | none. CORA.cpp lowers `sum_t a_t b_t` to two gathers into one product layer and a 1 x 1 convolution of ones over its first half. Request: `matmul_encloses` for that composition (as `mul_encloses`) | the two branches' `Z` | `Z` |
| the sums of a softmax (a reciprocal layer that reads an exp layer alone) | none. CORA.cpp bounds the sums by the zonotope's interval intersected with the interval image of the exps' box `[e^l, e^u]` (center's image `+-` `abs(W)` times the radius, widened by `n_in + 4` ulps of the largest term). Request: `softmax_sum_bounds`: for every `x` with `l <= d(x) <= u`, `W exp(d(x)) + b` lies in that image | `W`, `b`, `l`, `u` | `lo`, `hi` |
| `nnLayer.evaluateZonotope` of a sign layer (no CORA layer; traffic_signs' binarized networks) | none. CORA.cpp's line: slope 0, offset `(sign l + sign u) / 2`, radius `(sign u - sign l) / 2` (exact in floats). Request: `signRelaxation` with `encloses`, and the dropped output Softmax (`softmax_order`: `softmax_i >= softmax_j` iff `z_i >= z_j`) | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of a product layer (ONNX `Mul` of two branches, lsnc_relu's Lyapunov terms; `Activation::product`, `[a; b] -> [a o b; 0]`) | none. CORA.cpp's `priv_product_line`: on the zonotope `(c, G, dr)` of `[a; b]`, slopes `[c_b; c_a]`, offset `-c_a c_b + (lo + hi) / 2`, radius `(hi - lo) / 2` widened by `(2 q + 16) eps max(hi, -lo) + 16 eps abs(c_a c_b)`, where `[lo, hi] = [max(-D_1^2, -D_l^2) / 4, min(S_1^2, S_l^2) / 4]`, `S_k, D_k = sum abs(k G_a +- G_b / k) + k dr_a + dr_b / k`, `l = sqrt(B / A)` (`A, B` the L1 radii of the halves, 1 if either is 0). Request: `productRelaxation` with `encloses` (for every `beta` in the box, `(c_a + G_a beta)(c_b + G_b beta)` lies in the line's band) at every `FloatOps`, its `nearest` op, and `mul_encloses` for the composition (exact maps around it) | `c`, `G`, `dr` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of a floor layer (no CORA layer; ml4acopf's `linear-residual` networks count the turns of an angle, `x - 2 pi floor(x / (2 pi))`) | none. CORA.cpp's line: slope 0, offset `(floor l + floor u) / 2`, radius `(floor u - floor l) / 2` (exact in floats: integers and their halves); no polynomial (a step at every integer). Request: `floorRelaxation` with `encloses` | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of a cube layer (no CORA layer; nn4sys' pensieve_parallel, `Pow(x, 3)`) | none. CORA.cpp's `priv_cube_line`: slope `m = l^2 + l u + u^2` (the chord's), offset and radius the middle and half of the range of `x^3 - m x` over `{l, u, clamp(+-sqrt(m / 3), l, u)}`, the radius widened by `64 eps max(|l|, |u|)^3`; `max(|l|, |u|)^3` beyond the float range fails as out of range. Request: `cubeRelaxation` with `encloses` (for every `x` in `[l, u]`, `x^3` lies in `m x + offset +- radius`) at every `FloatOps` | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of an inverse layer (`1 / x` for `x > 0`; pensieve's `x^3 / sum x^3` as a product with it) | none. CORA.cpp's `priv_inverse_line`: fails as out of range unless `l > 0`; slope `m = -(1 / l)(1 / u)`, offset and radius the middle and half of `[g(sqrt l sqrt u), max(g(l), g(u))]` with `g(x) = 1 / x - m x` (convex), the radius widened by `32 eps / l`. An inverse layer that reads a cube, exp or ReLU layer alone also bounds its input by the image of that layer's box (`[l^3, u^3]`, `[e^l, e^u]`, `[max(l, 0), max(u, 0)]`; `aux_box_image`). Request: `inverseRelaxation` with `encloses` for `0 < l <= u`, and `quotient_encloses` for the composition `x^3 / sum x^3` (the product line of `[x^3; 1 / s]`) | `l`, `u` | `slope`, `offset`, `radius` |
| `nnLayer.evaluateZonotope` of an `AddLayer` (a residual) | none (CORALean's networks are chains). CORA.cpp adds the generator rows of an earlier output to the first rows of the later set (the same factors) and the interval-center radii | the two branches' `Z` with their factor counts | `Z` |

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

CORALean has float definitions of these CORA methods (`src/CORALean/ContSet/PolyZonotope/Float/Operations/`),
but the pinned oracle serves none of them, so they are tested as properties (evaluation at sampled factor
values) and the rows are oracle debt. Requests as for the section above (`pZ`, `E` the exponent matrix, `id`
the identifiers), reply a polynomial zonotope unless stated. CORALean's factors are positions, so the
requests carry the identifiers and the oracle maps them.

| op (proposed) | CORALean definition | request fields | reply and order |
|---|---|---|---|
| `polyZonotope.getSubset` | `PolyZonotope.getSubset`, `getSubsetCoef` (`GetSubset.lean`) | `pZ`, `id`, `val` (numbers) or `I` (a box) | the columns as CORA.cpp: the original columns, then the appended terms of each expanded generator; the coefficients of `(c + r a)^e` by CORA's `conv` |
| `polyZonotope.replaceId`, `restoreId`, `rmId`, `noIndep` | `ReplaceId.lean`, `PadBack.lean`, `RmId.lean`, `NoIndep.lean` | `pZ`, `id` / `idOld`, `idNew` | exact: nothing rounds; `rmId` adds the columns of equal exponents in order |
| `polyZonotope.onlyId` | `PolyZonotope.onlyId` | `pZ`, `id` | the two parts (`pZId`, `pZr`) |
| `polyZonotope.isequal`, `isFullDim`, `isBounded`, `representsa` | `Isequal.lean` (structural, tolerance 0), `IsFullDim.lean` (certified true only), `Basic.lean` | `pZ`, `S`, `tol` | a Boolean; CORA.cpp's `isequal` compares up to order after `compact`, CORALean's in order, so the two agree on `true` only |
