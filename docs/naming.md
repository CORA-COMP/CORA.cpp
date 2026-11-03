# Naming

CORA.cpp names things the way C++ and Python name them, and it keeps the meaning of CORA's names.
A reader of CORA finds every function under the snake_case form of its CORA name. A few names are
changed, and the reasons are listed below.

## Rules

`tests/global/test_coding_conventions.py` checks these rules.

- **Classes and other types** are PascalCase: `Zonotope`, `LinearSys`, `ReachSet`, `BackendScope`.
- **Functions, methods, arguments, variables and files** are snake_case:
  - `support_function`, `rand_point`, `time_step`;
  - one operation per file, named after the operation (`src/cont_set/zonotope/rand_point.cpp`);
  - folders are snake_case too (`cont_set/`, `linear_sys/`).
- **Private members** end in an underscore: `c_`, `G_`.
- **Constants** are `kName`: `kInf`, `kClassId`.
- **Enumerators** are snake_case: `Error::dimension_mismatch`, `SpecType::unsafe_set`.
- **Auxiliary functions** of a file are `aux_*`, private functions of a class folder `priv_*`
  (as in CORA's `private/`). An unchecked implementation is `<operation>_impl`. CORA writes it
  with a trailing underscore, as in `supportFunc_`.
- **The CORA tag:** every operation file, and every public header of a set, system, network or
  specification, ends its header with one of two lines:
  - `// CORA: <path>.m`, the CORA file it implements;
  - `// CORA.cpp only: <reason>`, if CORA has no such file.
- **Python** uses the C++ names. Sets have properties (`Z.center`, `Z.generators`, `Z.dim`), and
  options are keyword arguments with the same names as the fields of the C++ structs.
- **Options** have the same names in C++ (struct fields) and in Python (keywords):
  - `time_step`, `t_final`, `taylor_terms`, `zonotope_order`;
  - the problem: `initial_set`, `input_set`, `initial_state`, `input`.
  - The CORALean oracle's JSON protocol keeps CORA's keys (`timeStep`, `tFinal`, ...), and so do
    the benchmark data files.

## CORA, C++, Python

| CORA (MATLAB) | C++ | Python |
|---|---|---|
| `contSet` | `ContSet`, type-erased `Set` | `cora.ContSet` (subclass it) |
| `zonotope(c, G)` | `Zonotope(c, G)` | `cora.Zonotope(c, G)` |
| `interval(inf, sup)` | `Interval(inf, sup)` | `cora.Interval(inf, sup)` |
| `intervalMatrix` | `Interval` of matrix bounds | `cora.Interval.matrix(inf, sup)` |
| `Z.c`, `center(Z)` | `Z.center()` | `Z.center` |
| `Z.G`, `generators(Z)` | `Z.generators()` | `Z.generators` |
| `dim(S)` | `S.dim()` | `S.dim` |
| `I.inf`, `I.sup`, `rad(I)` | `I.inf()`, `I.sup()`, `I.rad()` | `I.inf`, `I.sup`, `I.rad` |
| `supportFunc(S, d)` | `S.support_function(d)` | `S.support_function(d)` |
| `randPoint(S, N, type)` | `S.rand_point(N, rng)`, `Z.rand_point(N, rng, type)` | `S.rand_point(N, rng, type)` |
| `interval(S)` | `S.interval()` | `S.interval()` |
| `project(S, dims)` | `S.project(dims)` | `S.project(dims)` |
| `vertices(S)` | `S.vertices()` | `S.vertices()` |
| `display(S)` | `S.display()`, `std::cout << S` | `repr(S)`, `print(S)` |
| `mtimes(M, Z)`, `M * Z` | `Z.linear_map(M)`, `M * Z` | `Z.linear_map(M)`, `M @ Z`, `M * Z` |
| `plus(S1, S2)`, `S1 + S2` | `plus(S1, S2)`, `S1 + S2` | `cora.plus`, `S1 + S2` |
| `and(S1, S2)`, `S1 & S2` | `intersect(S1, S2)`, `S1 & S2` | `cora.intersect`, `S1 & S2` |
| `or(S1, S2)`, `S1 \| S2` | `unite(S1, S2)`, `S1 \| S2` | `cora.unite`, `S1 \| S2` |
| `convHull(S1, S2)` | `convex_hull(S1, S2)` | `cora.convex_hull` |
| `cartProd(S1, S2)` | `cart_prod(S1, S2)` | `cora.cart_prod` |
| `contains(S1, S2, type, tol)` | `contains(S1, S2, tol)`, `I.contains(p, tol)` | `cora.contains`, `p in I` |
| `isIntersecting(S1, S2, type, tol)` | `is_intersecting(S1, S2, tol)` | `cora.is_intersecting` |
| `reduce(Z, 'girard', order)` | `Z.reduce(order)` | `Z.reduce(order)` |
| `abs(Z)`, `box(Z)` | `Z.abs()`, `Z.box()` | `Z.abs()`, `Z.box()` |
| `compact(Z, method, tol)` | `Z.compact(method, tol)` | `Z.compact(method, tol)` |
| `cartProd(Z, S)` | `Z.cart_prod(S)`, `Zonotope::cart_prod(p, Z)` | `Z.cart_prod(S)` |
| `enclose(Z, Z2)` | `Z.enclose(Z2)` | `Z.enclose(Z2)` |
| `convHull(Z, S)` | `Z.convex_hull(S)` | `Z.convex_hull(S)` |
| `and(Z, S, method)` | `Z.intersect(S, method)` | `Z.intersect(S, method)` |
| `constrSat(Z, C, d)` | `Z.constraints_satisfied(C, d)` | `Z.constraints_satisfied(C, d)` |
| `cubMap(Z, T, ind)`, `cubMap(Z, Z2, Z3, T, ind)` | `Z.cubic_map(T, ind)`, `Z.cubic_map(Z2, Z3, T, ind)` | `Z.cubic_map(...)` |
| `dH2box(Z, method)` | `Z.hausdorff_distance_to_box(method)` | `Z.hausdorff_distance_to_box(method)` |
| `dominantDirections(Z, l1, l2)` | `Z.dominant_directions(l1, l2)` | `Z.dominant_directions(l1, l2)` |
| `contains(Z, S, method, tol)` | `Z.contains(S, method, tol)` | `Z.contains(S, method, tol)`, `p in Z` |
| `boundaryPoint(Z, dir, start)` | `Z.boundary_point(dir, start)` | `Z.boundary_point(dir, start)` |
| `linComb(Z1, Z2)` | `Z.linear_combination(Z2)` | `Z.linear_combination(Z2)` |
| `zonotope.generateRandom` | `Zonotope::generate_random(n, m, rng)` | `cora.Zonotope.generate_random` |
| `interval.generateRandom` | `Interval::generate_random(n, rng)` | `cora.Interval.generate_random` |
| `interval.Inf(n)` | `Interval::unbounded(n)` | `cora.Interval.unbounded(n)` |
| `cat(dim, I1, I2)` | `Interval::cat(dim, {I1, I2})` | `cora.Interval.cat(dim, [I1, I2])` |
| `horzcat(I1, I2)` | `Interval::horzcat({I1, I2})` | `cora.Interval.horzcat([I1, I2])` |
| `interval.empty(n)` | `Interval::empty(n)` | `cora.Interval.empty(n)` |
| `interval.enclosePoints(P)` | `Interval::enclose_points(P)` | `cora.Interval.enclose_points(P)` |
| `isFullDim(I)` | `I.is_full_dim()` | `I.is_full_dim()` |
| `isequal(I, S, tol)` | `I.isequal(S, tol)` | `I.isequal(S, tol)` |
| `le(I1, I2)`, `I1 <= I2` | `I.less_equal(I2)`, `I <= I2` | `I.less_equal(I2)`, `I <= I2` |
| `lt(I1, I2)`, `I1 < I2` | `I.less(I2)`, `I < I2` | `I.less(I2)`, `I < I2` |
| `minus(I, S)`, `I - S` | `I.minus(S)`, `I - S`, `v - I` | `I.minus(S)`, `I - S`, `v - I` |
| `mrdivide(I, d)`, `I / d` | `I.divide(d)`, `I / d` | `I.divide(d)`, `I / d` |
| `kron(I1, I2)` | `I.kron(I2)`, `Interval::kron(A, I)` | `I.kron(I2)`, `I.kron(A, left=True)` |
| `mpower(I, k)` | `I.matrix_power(k)` | `I.matrix_power(k)` |
| `max(I, Y)`, `min(I, Y)` | `I.max(Y)`, `I.min(Y)`, `I.max()` (the bound) | the same |
| `minkDiff(I, S, type)` | `I.minkowski_difference(S, type)` | `I.minkowski_difference(S, type)` |
| `or(I, S)` | `I.unite(S)` | `I.unite(S)` |
| `norm_(I, type)` | `I.norm(type)` | `I.norm(type)` |
| `lift_(I, N, proj)` | `I.lift(N, dims)` | `I.lift(N, dims)` |
| `interval.origin(n)` | `Interval::origin(n)` | `cora.Interval.origin(n)` |
| `log(I)` | `I.log()` | `I.log()` |
| `emptySet(n)`, `emptySet.empty(n)` | `EmptySet(n)`, `EmptySet::empty(n)` | `cora.EmptySet(n)`, `cora.EmptySet.empty(n)` |
| `isBounded(E)`, `isFullDim(E)` | `E.is_bounded()`, `E.is_full_dim()` | `E.is_bounded()`, `E.is_full_dim()` |
| `and(E, S)`, `plus(E, S)`, `convHull(E, S)` | `E.intersect(S)`, `E.plus(S)`, `E.convex_hull(S)` | the same |
| `contains(E, S, type, tol)`, `isIntersecting(E, S, type, tol)`, `isequal(E, S, tol)` | `E.contains(S, tol)`, `E.is_intersecting(S, tol)`, `E.isequal(S, tol)` | the same |
| `mtimes(M, E)`, `lift(E, N, dims)` | `E.linear_map(M)`, `M * E`, `E.lift(N, dims)` | `E.linear_map(M)`, `M @ E`, `E.lift(N, dims)` |
| `intervalMatrix(I)` | `I.interval_matrix()` | `I.interval_matrix()` |
| `boundaryPoint(I, dir, start)` | `I.boundary_point(dir, start)` | `I.boundary_point(dir, start)` |
| `linearSys(A, B, c, C)` | `LinearSys(A, B, C)` | `cora.LinearSys(A, B, C)` |
| `nonlinearSys(f)` | `NonlinearSys(f, n)` | `cora.NonlinearSys(f, n)` |
| `neuralNetwork` | `NeuralNetwork` | `cora.NeuralNetwork` |
| `evaluate(nn, X)` | `nn.evaluate(X)` | `nn.evaluate(X)` |
| `reach(sys, params, options)` | `sys.reach(problem, alg)` | `sys.reach(X0, t_final=..., alg=..., ...)` |
| `simulate(sys, params)` | `sys.simulate(problem, alg)` | `sys.simulate(x0, t_final=..., time_step=...)` |
| `simulateRandom(sys, params, options)` | `sys.simulate_random(problem, alg, points, rng)` | `sys.simulate_random(X0, points=..., rng=..., ...)` |
| `verify(sys, params, options, spec)` | `sys.verify(problem, specs, alg)` | `sys.verify(R0, specs=..., alg=..., ...)` |
| `outputSet(sys, R)` | `sys.output_set(R)` | `sys.output_set(R)` |
| `taylorMatrices` (F) | `sys.correction_matrix_state(dt, eta)` | `sys.correction_matrix_state` |
| `taylorMatrices` (G) | `sys.correction_matrix_input(dt, eta)` | `sys.correction_matrix_input` |
| `particularSolution_*` | `sys.input_solution(U, dt, eta)` | — |
| `priv_expmRemainder` | `sys.expm_remainder(dt, eta)` | `sys.expm_remainder` |
| `params.R0` | `ReachProblem::initial_set` | keyword `initial_set` (or the first argument) |
| `params.U` | `ReachProblem::input_set` | keyword `input_set` |
| `params.x0` | `SimulationProblem::initial_state` | keyword `initial_state` (or the first argument) |
| `params.u` | `SimulationProblem::input` | keyword `input` |
| `params.tFinal` | `t_final` | `t_final` |
| `options.timeStep` | `alg::Standard::time_step`, ... | `time_step` |
| `options.taylorTerms` | `taylor_terms` | `taylor_terms` |
| `options.zonotopeOrder` | `zonotope_order` | `zonotope_order` |
| `options.linAlg = 'standard'` | `alg::Standard{...}` | `alg="standard"` |
| `options.linAlg = 'wrapping-free'` | `alg::WrappingFree{...}` | `alg="wrapping_free"` |
| `options.verifyAlg = 'reachavoid:supportFunc'` | `alg::ReachAvoidSupportFunction{}` | `alg="reach_avoid_support_function"` |
| `options.verifyAlg = 'reachavoid:zonotope'` | `alg::ReachAvoidZonotope{}` | `alg="reach_avoid_zonotope"` |
| `options.alg = 'lin'` (nonlinearSys) | `alg::Linearization{...}` | `alg="linearization"` |
| `reachSet` | `ReachSet` | `cora.ReachSet` |
| `R.timeInterval.set` | `R.time_interval` | `R.time_interval` |
| `R.timePoint.set` | `R.time_point` | `R.time_point` |
| `[res, fals, savedata]` of verify | `VerifyResult{verified, falsification, ...}` | `cora.VerifyResult` |
| `savedata.tComp` | `computation_time` | `computation_time` |
| `nrSteps` | `num_steps` | `num_steps` |
| `specification(halfspace, 'safeSet')` | `Specification::safe_set(a, b)` | `cora.Specification.safe_set(a, b)` |
| `specification(..., 'unsafeSet')` | `Specification::unsafe_set(a, b)` | `cora.Specification.unsafe_set(a, b)` |
| `check(spec, S)` | `spec.check(S)` | `spec.check(S)` |
| `CORAerror('CORA:wrongValue', ...)` | `cora::Error`, id `Error::wrong_value` | `cora.errors.WrongValue` (a `ValueError`) |
| `CORAwarning('CORA:...', ...)` | `cora::warn(identifier, message)` | `cora.CORAWarning` |
| `CHECKS_ENABLED` | `ChecksScope`, `set_checks_enabled`, CMake `CORACPP_CHECKS` | `with cora.checks(False):` |
| `withinTol(a, b, tol)` | `within_tol(a, b, tol)` | — |
| default `tol` of each function (eps, 100*eps, 1e-8, ...) | `ToleranceScope`, `set_default_tolerance`, `default_tol(dtype)` | `with cora.tolerance(1e-10):`, `cora.set_default_tolerance`, `cora.default_tol(dtype)` |
| `CORAcolor(identifier)` | `cora_color(identifier)` | `cora.cora_color` |
| `useCORAcolors(scheme)` | `use_cora_colors(scheme)` | `cora.use_cora_colors` |
| `plot(S, dims, ...)` | `plot(fig, S, dims, options)` | `cora.plot(S, dims, ...)`, `S.plot(dims, ...)` |
| `plotOverTime(R, dim, ...)` | `plot_over_time(fig, R, dim, options)` | `cora.plot_over_time(R, dim, ...)` |
| `plotRandPoint(S, dims, N)` | `plot_rand_point(fig, S, dims, N, rng, options)` | `cora.plot_rand_point(S, dims, N, rng)` |
| `FaceColor`, `EdgeColor`, `FaceAlpha`, `LineWidth`, `Unify`, `Set` | `face_color`, `edge_color`, `face_alpha`, `line_width`, `unify`, `set` | the same keywords |
| `vertices(S)` (3-D) | `S.vertices_3d()` | `S.vertices_3d()` |
| `fig2tikz`, matlab2tikz | `fig.tikz(options)`, `fig.save_tikz(path, options)` | `fig.to_tikz(path, ...)`, `cora.subplots` |

## Names that differ from the rule, and why

- **Changed beyond snake_case:**
  - `and` and `or` are keywords of C++ (alternative tokens), so they are `intersect` and
    `unite`. The operators `&` and `|` stay.
  - `mtimes` is MATLAB's name of an operator, so it is `linear_map`.
  - `interval.Inf` is `unbounded`, because `inf` is the lower bound.
  - `linComb` is `linear_combination` and `convHull` is `convex_hull`: words, not
    abbreviations. So are `constrSat` (`constraints_satisfied`), `cubMap` (`cubic_map`) and `dH2box`
    (`hausdorff_distance_to_box`).
  - `le` and `lt` (operators of MATLAB) are `less_equal` and `less`, as Tensor names them;
    `mrdivide` (the division by a number) is `divide`, `mpower` is `matrix_power`, and `minkDiff`
    is `minkowski_difference`: words, not abbreviations.
  - `timeInt` is `time_interval`, `tComp` is `computation_time`, `nrSteps` is `num_steps` and
    `fals` is `falsification`, for the same reason.
  - `Reach` is `ReachSet`, as CORA's `reachSet`.
  - `R0`/`x0`/`U`/`u` are `initial_set`/`initial_state`/`input_set`/`input` as keywords. They stay
    as local variable names in the code, as the formulas write them.
- **Kept against the rule:**
  - CORA's error identifiers (`"CORA:wrongValue"`) and color identifiers (`"CORA:reachSet"`) are
    strings of CORA's protocol.
  - The oracle's JSON keys (`timeStep`, `tFinal`) and the benchmark data keys are files and
    protocols shared with CORALean and MATLAB.
  - `cora.matlab` keeps CORA's names: `intervalMatrix`, `linearSys`, `reach(X0, timeStep, ...)`,
    `outputSet`, `timeInt`, `timePoint`. MATLAB CORA's `toPy`/`fromPy` call them.
  - `examples/matlab/*.m` is MATLAB code and follows CORA's style.
  - Constants are `kName` (Google style), and the single-letter mathematical names (`A`, `B`, `C`,
    `G`, `M`) stay capital, as in the papers.
  - Names of other libraries (Eigen's `cwiseAbs`, unittest's `setUp`) are theirs.
