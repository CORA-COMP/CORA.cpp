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
- **Overrides:** a method of a `ContSet` subclass accepts every argument that `ContSet`'s method
  accepts (it may add keyword arguments with defaults); `tests/python/test_stubs.py` checks it by
  type-checking the stubs with mypy.
- **Options** have the same names in C++ (struct fields) and in Python (keywords):
  - `time_step`, `t_final`, `taylor_terms`, `zonotope_order`;
  - the problem: `initial_set`, `input_set`, `initial_state`, `input`.
  - `generate_random` of every class takes `rng` first, then CORA's name-value options
    (`'Dimension'` is `dimension`, `'NrGenerators'` is `nr_generators`) as optional arguments with
    CORA's defaults.
  - The CORALean oracle's JSON protocol keeps CORA's keys (`timeStep`, `tFinal`, ...), and so do
    the benchmark data files.

## CORA, C++, Python

| CORA (MATLAB) | C++ | Python |
|---|---|---|
| `contSet` | `ContSet`, type-erased `Set` | `cora.ContSet` (subclass it) |
| `zonotope(c, G)` | `Zonotope(c, G)` | `cora.Zonotope(c, G)` |
| `interval(inf, sup)` | `Interval(inf, sup)` | `cora.Interval(inf, sup)` |
| `fullspace(n)`, `fullspace.Inf(n)` | `Fullspace(n)`, `Fullspace::inf(n)` | `cora.Fullspace(n)`, `cora.Fullspace.inf(n)` |
| `intervalMatrix` | `Interval` of matrix bounds | `cora.Interval.matrix(inf, sup)` |
| `interval(inf, sup)` of matrix bounds | `Interval` of matrix bounds | `cora.Interval.matrix(inf, sup)` |
| `intervalMatrix(C, D)`, `intervalMatrix(C)` | `IntervalMatrix(C, D)`, `IntervalMatrix(C)`, `IntervalMatrix(I)` | `cora.IntervalMatrix(C, D)` |
| `abs(A)`, `center(A)`, `infimum(A)`, `contains(A, M, type, tol)` | `A.abs()`, `A.center()`, `A.infimum()`, `A.contains(M, type, tol)` | `A.abs()`, `A.center`, `A.infimum`, `A.contains(M, type, tol)` |
| `mtimes(A, B)`, `A * B` | `A.matmul(B)`, `A * B`, `M * A`, `A * M`, `s * A`, `A * Z` | `A @ B`, `M @ A`, `A @ M`, `s * A`, `A @ Z` |
| `mpower(A, k)` | `A.matrix_power(k)` | `A.matrix_power(k)`, `A ** k` |
| `plus(A, B)`, `A + B` | `A.plus(B)`, `A + B`, `A + M`, `M + A`, `A + s`, `s + A` | `A + B`, `A + M`, `s + A`, `A.plus(B)` |
| `powers(A, N, k, P)` | `A.powers(N)`, `A.powers(N, k, P)` | `A.powers(N, k, P)` (a list, `[i]` of order `k + i`) |
| `rad(A)`, `supremum(A)` | `A.rad()`, `A.supremum()` | `A.rad`, `A.supremum` |
| `representsa(A, type, tol)`, `volume(A)`, `vertices(A)` | `A.representsa(type, tol)`, `A.volume()`, `A.vertices()` | the same |
| `expm`, `expmInd`, `expmMixed` | `A.expm(...)`, `A.expm_independent(...)`, `A.expm_mixed(r, k, N)` | the same |
| `infNorm(A)`, `norm(A, type)` | `A.infinity_norm()`, `A.norm(type)` | the same |
| `dependentTerms(A, r)`, `exactSquare(A)`, `exponentialRemainder(A, N)` | `A.dependent_terms(r)`, `A.exact_square()`, `A.exponential_remainder(N)` | the same |
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
| `isBounded(S)`, `isFullDim(S, tol)` | `S.is_bounded()`, `S.is_full_dim(tol)` | `S.is_bounded()`, `S.is_full_dim(tol)` |
| `enlarge(S, factor)` | `S.enlarge(factor)` | `S.enlarge(factor)` |
| `minus(S, p)`, `minus(p, S)` | `S.minus(p)`, `ContSet::minus(p, S)` | `S.minus(p)` |
| `minkDiff(S, S2, type)` | `S.minkowski_difference(S2, type)` | `S.minkowski_difference(S2, type)` |
| `mldivide(P, S)` | `P.set_difference(S)` (all pieces) | `P.set_difference(S)` |
| `isequal(S1, S2, tol)`, `eq`, `ne` | `isequal(S1, S2, tol)`, `eq`, `ne`, `S1 == S2` | `cora.isequal`, `cora.eq`, `cora.ne`, `S1 == S2` |
| `decompose(S, blocks)` | `S.decompose(blocks)` (0-based, inclusive) | `S.decompose(blocks)` |
| `projectHighDim(S, N, proj)` | `S.project_high_dim(N, proj)` | `S.project_high_dim(N, proj)` |
| `lift(S, N, proj)` | `S.lift(N, proj)` | `S.lift(N, proj)` |
| `norm(S, type, mode)` | `S.norm(type, mode)` | `S.norm(type, mode)` |
| `enclose(S1, S2)`, `enclose(Z, M, Zplus)` | `S1.enclose(S2)`, `Z.enclose(M, Zplus)` | `S1.enclose(S2)`, `Z.enclose(M, Zplus)` |
| `projectOnHyperplane(S, hyp)` | `S.project_on_hyperplane(c, d)` (the constraint) | `S.project_on_hyperplane(c, d)` |
| `contSet.generateRandom(...)` | `ContSet::generate_random(rng, dimension, admissible)` | `cora.generate_random(rng, dimension, admissible)` |
| `and_`, `contains_`, `isIntersecting_` | `detail::intersect_impl`, `contains_impl`, `is_intersecting_impl` | - |
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
| `conZonotope(c, G, A, b)`, `conZonotope(Z)`, `conZonotope(Z, A, b)` | `ConZonotope(c, G, A, b)`, `ConZonotope(Z)`, `ConZonotope::from_matrix(Z, A, b)` | `cora.ConZonotope(c, G, A, b)`, `cora.ConZonotope.from_matrix(Z, A, b)` |
| `conZonotope.empty(n)` | `ConZonotope::empty(n)` | `cora.ConZonotope.empty(n)` |
| `zonoBundle(list)`, `zB.Z`, `zB.parallelSets` | `ZonoBundle(zonotopes)`, `zB.zonotopes()`, `zB.parallel_sets()` | `cora.ZonoBundle(zonotopes)`, `zB.zonotopes`, `zB.parallel_sets` |
| `zonoBundle.empty(n)`, `conZonotope(zB)` | `ZonoBundle::empty(n)`, `zB.con_zonotope()` | `cora.ZonoBundle.empty(n)`, `zB.con_zonotope()` |
| `conIntersect(cZ, cZ2, M)` | `cz.con_intersect(cz2, M)` | `cz.con_intersect(cz2, M)` |
| `intersectStrip(cZ, C, phi, y, method)` | `cz.intersect_strip(C, phi, y, method)` | `cz.intersect_strip(C, phi, y, method)` |
| `approximateBoundsWithGPU(...)` | `ConZonotope::approximate_bounds(c, G, A, b, dr, num_union_const, options)` | `cora.ConZonotope.approximate_bounds(...)` |
| `polytope(V)`, `polytope(A, b)`, `polytope(A, b, Ae, be)`, `polytope(I)` | `Polytope(V)`, `Polytope(A, b)`, `Polytope(A, b, Ae, be)`, `Polytope(I)` | `cora.Polytope(V)`, `cora.Polytope(A, b, Ae, be)`, `cora.Polytope(I)` |
| `polytope.empty(n)`, `polytope.Inf(n)` | `Polytope::empty(n)`, `Polytope::inf(n)` | `cora.Polytope.empty(n)`, `cora.Polytope.inf(n)` |
| `enclosePoints(P, points)` | `Polytope::enclose_points(points)` | `cora.Polytope.enclose_points(points)` |
| `constraints(P)` | `P.constraints()` | `P.constraints()` |
| `center(P, method)` | `P.center(method)` | `P.center` (Chebyshev), `P.center_by(method)` |
| `compact(P, method, tol)` | `P.compact(method, tol)` | `P.compact(method, tol)` |
| `conZonotope(P, method, box)` | `P.con_zonotope(method, box)` | `P.con_zonotope(method, box)` |
| `and(P, S)` | `P.intersect(S, tol)` | `P.intersect(S, tol)` |
| `cartProd(P, S)` | `P.cart_prod(S)` | `P.cart_prod(S)` |
| `convHull(P, S)`, `enclose(P, P2)` | `P.convex_hull(S)`, `P.enclose(P2)` | `P.convex_hull(S)`, `P.enclose(P2)` |
| `hausdorffDist(P, S)` | `P.hausdorff_distance(S)` | `P.hausdorff_distance(S)` |
| `ellipsoid(Q, q, TOL)`, `ellipsoid.empty(n)` | `Ellipsoid(Q, q, tol)`, `Ellipsoid::empty(n)` | `cora.Ellipsoid(Q, q, tol)`, `cora.Ellipsoid.empty(n)` |
| `E.Q`, `E.q`, `E.TOL`, `generators(E)`, `ellipsoidNorm(E, p)` | `E.Q()`, `E.q()`, `E.tol()`, `E.generators()`, `E.ellipsoid_norm(p)` | `E.Q`, `E.q`, `E.tol`, `E.generators()`, `E.ellipsoid_norm(p)` |
| `isBigger(E1, E2)`, `isBadDir(E1, E2, L)` | `E1.is_bigger(E2)`, `E1.is_bad_dir(E2, L)` | the same |
| `ellipsoid.enclosePoints(P, method)` | `Ellipsoid::enclose_points(P, method)` | `cora.Ellipsoid.enclose_points(P, method)` |
| `intersectStrip(E, C, phi, y, sigma_sq_prev, method)` | `E.intersect_strip(C, phi, y, sigma_sq_prev)`, with a `StripMeasurement` for `phi` (the method of Liu) | `E.intersect_strip(C, phi, y, sigma_sq_prev, method)`, a dict `{V, bar_V}` for `phi` |
| `capsule(c, g, r)`, `capsule(c)` | `Capsule(c, g, r)`, `Capsule(c)` | `cora.Capsule(c, g, r)` |
| `capsule.empty(n)`, `capsule.origin(n)`, `capsule.enclosePoints(P)` | `Capsule::empty(n)`, `Capsule::origin(n)`, `Capsule::enclose_points(P)` | `cora.Capsule.empty(n)`, `.origin(n)`, `.enclose_points(P)` |
| `contains_(C, S)`, `isIntersecting_(C, S)`, `convHull_(C, S)`, `isequal(C, S)` | `C.contains(S, tol)`, `C.is_intersecting(S, tol)`, `C.convex_hull(S)`, `C.isequal(S, tol)` | the same, `p in C`, `C == S` |
| `mtimes(M, C)`, `lift_(C, N, proj)` | `C.linear_map(M)`, `M * C`, `C.lift(N, dims)` | `M @ C`, `C.linear_map(M)`, `C.lift(N, dims)` |
| `boundaryPoint(Z, dir, start)` | `Z.boundary_point(dir, start)` | `Z.boundary_point(dir, start)` |
| `linComb(Z1, Z2)` | `Z.linear_combination(Z2)` | `Z.linear_combination(Z2)` |
| `zonotope.generateRandom` | `Zonotope::generate_random(rng, dimension, center, nr_generators, distribution)` | `cora.Zonotope.generate_random` |
| `interval.generateRandom` | `Interval::generate_random(rng, dimension, center, max_radius)` | `cora.Interval.generate_random` |
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
| `enclosePoints(P)`, `lift_(fs, N, dims)`, `mtimes(M, fs)` | `Fullspace::enclose_points(P)`, `F.lift(N, dims)`, `F.linear_map(M)`, `M * F` | `cora.Fullspace.enclose_points(P)`, `F.lift(N, dims)`, `F.linear_map(M)`, `M @ F` |
| `interval.origin(n)` | `Interval::origin(n)` | `cora.Interval.origin(n)` |
| `log(I)` | `I.log()` | `I.log()` |
| `emptySet(n)`, `emptySet.empty(n)` | `EmptySet(n)`, `EmptySet::empty(n)` | `cora.EmptySet(n)`, `cora.EmptySet.empty(n)` |
| `isBounded(E)`, `isFullDim(E)` | `E.is_bounded()`, `E.is_full_dim()` | `E.is_bounded()`, `E.is_full_dim()` |
| `and(E, S)`, `plus(E, S)`, `convHull(E, S)` | `E.intersect(S)`, `E.plus(S)`, `E.convex_hull(S)` | the same |
| `contains(E, S, type, tol)`, `isIntersecting(E, S, type, tol)`, `isequal(E, S, tol)` | `E.contains(S, tol)`, `E.is_intersecting(S, tol)`, `E.isequal(S, tol)` | the same |
| `mtimes(M, E)`, `lift(E, N, dims)` | `E.linear_map(M)`, `M * E`, `E.lift(N, dims)` | `E.linear_map(M)`, `M @ E`, `E.lift(N, dims)` |
| `zonotope.origin(n)` | `Zonotope::origin(n)` | `cora.Zonotope.origin(n)` |
| `radius(Z)`, `rank(Z, tol)` | `Z.radius()`, `Z.rank(tol)` | `Z.radius()`, `Z.rank(tol)` |
| `representsa(Z, type, tol)` | `Z.representsa(type, tol)` | `Z.representsa(type, tol)` |
| `projectHighDim(Z, N, proj)` | `Z.project_high_dim(N, dims)` | `Z.project_high_dim(N, dims)` |
| `rotate(Z, dims, angle)` | `Z.rotate(dims, angle)` | `Z.rotate(dims, angle)` |
| `quadMap(Z, Q)`, `quadMap(Z, Z2, Q)` | `Z.quad_map(Q)`, `Z.quad_map(Z2, Q)` | `Z.quad_map(Q)`, `Z.quad_map(Z2, Q)` |
| `quadMap(I, Q)` | `I.quad_map(Q)`, `I.quad_map(I2, Q)` | `I.quad_map(Q)`, `I.quad_map(I2, Q)` |
| `split(Z)`, `split(Z, N)`, `split(Z, N, maxOrder)`, `split(Z, dir)`, `split(Z, hs)` | `Z.split()`, `Z.split(N)`, `Z.split(N, max_order)`, `Z.split(dir)`, `Z.split(a, d)` | the same |
| `splitFirstGen(Z)`, `splitLongestGen(Z)` | `Z.split_first_gen()`, `Z.split_longest_gen()` | the same |
| `reduceUnderApprox(Z, method, order)` | `Z.reduce_under_approx(method, order)` | `Z.reduce_under_approx(method, order)` |
| `or(Z, S, alg, order)` | `Z.unite(S, method, order)` | `Z.unite(S, method, order)`, `Z \| S` |
| `not(E)`, `not(fs)` | `E.complement()`, `F.complement()` | `E.complement()`, `F.complement()` |
| `radius(E)`, `volume_(E)`, and the same of `fullspace` | `E.radius()`, `E.volume()` (a double) | `E.radius()`, `E.volume()` |
| `representsa_(E, type, tol)` | `E.representsa(type, tol)` (`Fullspace` the same) | `E.representsa(type, tol)` |
| `projectHighDim_(E, N, proj)`, `generateRandom('Dimension', n)` | `E.project_high_dim(N, proj)`, `EmptySet::generate_random(rng, n)` | `E.project_high_dim(N, proj)`, `cora.EmptySet.generate_random(rng, n=None)` |
| `intervalMatrix(I)` | `I.interval_matrix()` | `I.interval_matrix()` |
| `boundaryPoint(I, dir, start)` | `I.boundary_point(dir, start)` | `I.boundary_point(dir, start)` |
| `linearSys(A, B, c, C)` | `LinearSys(A, B, C)` | `cora.LinearSys(A, B, C)` |
| `nonlinearSys(f)` | `NonlinearSys(f, n)` | `cora.NonlinearSys(f, n)` |
| `linParamSys(A, B, c, 'constParam')`, `'varParam'` | `LinParamSys(A, B, c, ParamType::constant)`, `ParamType::varying` | `cora.LinParamSys(A, B, c, type="constant")`, `"varying"` |
| `initReach(sys, Rinit, params, options)`, `post(sys, R, params, options)` (linParamSys) | `sys.init_reach(R_init, U, u_trans, alg)`, `sys.post(R, alg)` (`LinParamStep`) | — |
| `options.intermediateTerms`, `options.compTimePoint` | `alg::LinParamStandard::intermediate_terms`, `comp_time_point` | `intermediate_terms`, `comp_time_point` |
| `neuralNetwork` | `NeuralNetwork` | `cora.NeuralNetwork` |
| `evaluate(nn, X)` | `nn.evaluate(X)` | `nn.evaluate(X)` |
| `reach(sys, params, options)` | `sys.reach(problem, alg)` | `sys.reach(X0, t_final=..., alg=..., ...)` |
| `simulate(sys, params)` | `sys.simulate(problem, alg)` | `sys.simulate(x0, t_final=..., time_step=...)` |
| `simulateRandom(sys, params, options)` | `sys.simulate_random(problem, alg, points, rng)` | `sys.simulate_random(X0, points=..., rng=..., ...)` |
| `verify(sys, params, options, spec)` | `sys.verify(problem, specs, alg)` | `sys.verify(R0, specs=..., alg=..., ...)` |
| `outputSet(sys, R)` | `sys.output_set(R)` | `sys.output_set(R)` |
| `taylorMatrices` (F) | `sys.correction_matrix_state(dt, eta)` | `sys.correction_matrix_state` |
| `taylorMatrices` (G) | `sys.correction_matrix_input(dt, eta)` | `sys.correction_matrix_input` |
| `particularSolution_*` of reach's input (through `B`) | `sys.input_solution(U, dt, eta)` | — |
| `canonicalForm(sys, U, uVec, W, V, vVec)` | `sys.canonical_form(U, u_vec)` | `sys.canonical_form(U, u_vec=None)` |
| `isCanonicalForm(sys)` | `sys.is_canonical_form()` | `sys.is_canonical_form()` |
| `homogeneousSolution`, `affineSolution` | `sys.homogeneous_solution(X, dt, eta)`, `sys.affine_solution(X, u, dt, eta)` | the same |
| `particularSolution_timeVarying`, `particularSolution_constant` | `sys.particular_solution_time_varying(U, dt, eta)`, `sys.particular_solution_constant(U, dt, eta)` | the same |
| `oneStep`, `errorSolution_adaptive`, `trajectoryError` | `sys.one_step(X, U, u, dt, eta)`, `sys.error_solution_adaptive(...)`, `sys.trajectory_error(...)` | the same |
| `initReach`, `post` (with `options` as state) | `sys.init_reach(R_init, U, u_trans, alg)`, `sys.post(state, alg)` | `sys.init_reach(R_init, U, u_trans, alg=..., ...)`, `sys.post(state, alg=..., ...)` |
| `isequal(sys, sys2, tol)`, `eq`, `ne` | `sys.isequal(sys2, tol)`, `==`, `!=` | `sys.isequal(sys2, tol=None)`, `==`, `!=` |
| `display(sys)` | `sys.display()`, `std::cout << sys` | `repr(sys)` |
| `generateRandom('StateDimension', n, ...)` | `LinearSys::generate_random(rng, n, m, p, ...)` | `cora.LinearSys.generate_random(rng, ...)` |
| `nonlinearSys(linsys)` | `sys.nonlinear_sys()` | `sys.nonlinear_sys()` |
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
| `options.linAlg = 'fromStart'` | `alg::FromStart{...}` | `alg="from_start"` |
| `options.linAlg = 'decomp'`, `options.partition` | `alg::Decomposition{...}`, `partition` (0-based blocks) | `alg="decomposition"`, `partition=[[0, 1], [2, 2]]` |
| `options.verifyAlg = 'reachavoid:supportFunc'` | `alg::ReachAvoidSupportFunction{}` | `alg="reach_avoid_support_function"` |
| `options.verifyAlg = 'reachavoid:zonotope'` | `alg::ReachAvoidZonotope{}` | `alg="reach_avoid_zonotope"` |
| `options.alg = 'lin'` (nonlinearSys) | `alg::Linearization{...}` | `alg="linearization"` |
| `options.tensorOrder`, `options.errorOrder`, `options.intermediateOrder` | `alg::Linearization::tensor_order`, `error_order`, `intermediate_order` (also `alg::HybridStandard`) | `tensor_order`, `error_order`, `intermediate_order` |
| `options.alg = 'lin-adaptive'` (nonlinearSys), `options.redFactor`, `decrFactor`, `zetaK`, `zetaTlin`, `zetaTabs`, `zetaphi` | `alg::LinearizationAdaptive{...}`, `zeta_z`, `decr_factor`, `zeta_k`, `zeta_t_lin`, `zeta_t_abs`, `zeta_phi` | `alg="linearization_adaptive"`, the same names |
| `R.timeInterval.time`, `R.timePoint.time` (adaptive step sizes) | `R.time`: the step times t_0, ..., t_K | `R.time` |
| `nonlinearSys(fun)` with `fun(x, u)`; `[A, B] = jacobian(x, u)` | `NonlinearSys(f, n, m)`, `Jacobians{A, B}` | `NonlinearSys(f, dim, inputs=m)` |
| `reachSet` | `ReachSet` | `cora.ReachSet` |
| `R.timeInterval.set` | `R.time_interval` | `R.time_interval` |
| `R.timePoint.set` | `R.time_point` | `R.time_point` |
| `[res, fals, savedata]` of verify | `VerifyResult{verified, falsification, ...}` | `cora.VerifyResult` |
| `savedata.tComp` | `computation_time` | `computation_time` |
| `nrSteps` | `num_steps` | `num_steps` |
| `specification(S, 'safeSet')` | `Specification::safe_set(S)` | `cora.Specification.safe_set(S)` |
| `specification(S, 'unsafeSet')` | `Specification::unsafe_set(S)` | `cora.Specification.unsafe_set(S)` |
| `check(spec, S)` | `spec.check(S)` | `spec.check(S)` |
| `specification(@(S) ..., 'custom')` | `Specification::custom(predicate)` | `cora.Specification.custom(predicate)` |
| `linearReset(A, B, c)`, `evaluate(reset, X, U)` | `LinearReset(A, B, c)`, `reset.evaluate(X, U)` | `cora.LinearReset(A, B, c)`, `reset.evaluate(X, U)` |
| `transition(guard, reset, target)` | `Transition(guard, reset, target)` (0-based target) | `cora.Transition(guard, reset, target)` |
| `location(name, inv, trans, sys)` (sys with c) | `Location(name, inv, trans, flow, c)` | `cora.Location(name, inv, trans, flow, c)` |
| `hybridAutomaton(name, loc)`, `reach(HA, params, options, spec)` | `HybridAutomaton(name, locs)`, `HA.reach(problem, alg, specs)` | `cora.HybridAutomaton`, `HA.reach(X0, t_final=..., ...)` |
| `params.startLoc`, `params.finalLoc`, `params.tStart` | `HybridReachProblem::start_location`, `final_locations`, `t_start` | the same keywords |
| `options.guardIntersect`, `options.enclose`, `options.guardOrder` | `alg::ZonoGirard`, `alg::NondetGuard`, `alg::ConZonotopeGuard{enclose, guard_order}` | `guard_intersect="zono_girard"`, `enclose=["box"]`, `guard_order` |
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
  - `mtimes` is MATLAB's name of an operator, so it is `linear_map` for a set under a matrix and
    `matmul` for the product of matrices (`IntervalMatrix`); `projectOnHyperplane` takes the
    constraint of the hyperplane (a normal and an offset), as CORA.cpp has no hyperplane class.
  - `not` is a keyword of C++ (an alternative token), so the complement of `emptySet` and
    `fullspace` is `complement`.
  - `interval.Inf` is `unbounded`, because `inf` is the lower bound.
  - `linComb` is `linear_combination` and `convHull` is `convex_hull`: words, not
    abbreviations. So are `constrSat` (`constraints_satisfied`), `cubMap` (`cubic_map`) and `dH2box`
    (`hausdorff_distance_to_box`).
  - `infNorm` is `infinity_norm` and `expmInd` is `expm_independent` (`Ind` is *independent*; `Inf` is
    also the unbounded box), as words, not abbreviations. `expm` is a term of art.
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
