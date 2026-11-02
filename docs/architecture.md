# Architecture

Each paragraph below states one decision, how it is built, and where it lives. CONTRIBUTING.md
gives the recipes that follow from these decisions; naming.md lists the names.

**Names.** Classes are PascalCase, and everything else is snake_case. Every name is the
snake_case form of CORA's name, apart from the few listed in naming.md. A reader of CORA finds
each operation under its CORA name, and C++ and Python readers see their own style. Each
operation file says which CORA file it implements (`// CORA: <path>.m`) or why CORA has none
(`// CORA.cpp only: ...`). `tests/global/test_coding_conventions.py` checks the style and the tag.

**A checked boundary around an unchecked core.** Every public operation is a non-virtual entry
point. It begins with `detail::Entry entry;`, checks its input with `CORA_CHECK`, and then calls
the unchecked implementation. For a set operation that is the protected virtual
`<operation>_impl`, which corresponds to CORA's `supportFunc_` behind `supportFunc`.
- `detail::Entry` counts the public entries active on the thread. Only the outermost one checks,
  so an operation that the library calls from inside another one never validates again (CORA's
  `and`/`and_` split).
- `CORA_CHECK(cond, Error::id, [&] { return message; })` builds its message only on failure.
- `CORA_FAIL(id, message)` raises a failure that is not an input error (an option switch with
  no branch left, an algorithm that cannot go on), whether the checks are on or off.
- The checks have three switches:
  - the CMake option `CORACPP_CHECKS=OFF` compiles them out with `if constexpr`, so the guard is
    then empty;
  - `set_checks_enabled(false)` or `CORACPP_CHECKS=0` in the environment sets the process
    default;
  - `ChecksScope off(false)` (Python: `with cora.checks(False):`) switches them off on one
    thread.
- An option struct (`alg::Standard`) and a problem have `validate()` methods guarded by
  `detail::Validation`. These check whenever the checks are on, because only a checking entry
  calls them.
- The conventions test enforces the pattern:
  - the guard comes first;
  - `CORA_CHECK` appears only in guarded functions, never in an `_impl` or under `private/`;
  - there is no raw `throw` outside `global/error.cpp`;
  - every operation's tests include an invalid input and a call with the checks off.
- Files: `include/cora/global/settings.h`, `include/cora/global/error.h`, and
  `src/cont_set/cont_set/` (the `ContSet` entries).

**Errors with CORA's identifiers.**
- `cora::Error` (a `std::runtime_error`) carries one of CORA's identifiers (`Error::Id`) in
  CORA's order: `wrong_value` is `CORA:wrongValue`, `noops` is `CORA:noops`, and so on.
- Python gets one class per identifier in `cora.errors`. Each derives from `cora.CORAError` and
  from the matching builtin: `DimensionMismatch` is a `ValueError`, `RedundantNameValuePair` a
  `TypeError`. Existing `except ValueError` code keeps working.
- Warnings go through one hook (`set_warning_handler`, by default one line on stderr), which
  Python sets to `warnings.warn` with `cora.CORAWarning`.
- Files: `include/cora/global/error.h`, `src/global/error.cpp`,
  `src/global/python/cora/errors.py`.

**One registry for the binary operations.** plus, intersect (CORA's `and`), unite (`or`),
convex_hull, cart_prod, contains and is_intersecting dispatch on the classes of both operands.
- The built-in pairs sit in a `constexpr` table indexed by `ClassId`. A call is one array lookup,
  with no strings and no `dynamic_cast`.
- CORA's precedence decides which operand implements a pair: the lower one comes first, and a
  `static_assert` keeps the table in that order. A commutative operation finds a pair stored the
  other way round.
- If no pair exists, `detail::dispatch` raises `Error::noops` and names both classes.
  `detail::dispatch` is also the single checked entry of these operations: the dimensions
  (except for cart_prod) and `tol`.
- Pairs registered at run time (from Python: `cora.register_binary("plus", MySet, Zonotope, fn)`)
  live in a map behind a mutex. The map is consulted only after the table misses.
- `binary_pairs()` lists everything, and `cora::Set` is the type-erased value these operations
  return.
- Files: `include/cora/cont_set/registry.h`, `include/cora/cont_set/set.h`,
  `src/cont_set/cont_set/registry.cpp`.

**Scoped, thread-local settings.** The backend, the checks and the CORALean dtype each have a
process default and a per-thread scope:
- `BackendScope` (Python: `with cora.backend("torch"):`) with `set_default_backend` and
  `CORACPP_BACKEND`;
- `ChecksScope` with `set_checks_enabled`;
- `lean::DTypeScope` with `lean::set_default_dtype`.

A scope saves the previous value and restores it, so scopes nest and threads do not see each
other's. Reading a setting is one thread-local load plus, without a scope, an atomic load. No
lock is taken. Files: `include/cora/global/tensor/tensor.h` (backend),
`include/cora/global/settings.h`, `include/cora/global/oracle/oracle.h`.

**The data decides the backend.** The scoped or default backend applies only where a tensor is
made from raw numbers: `Tensor({{1, 2}})`, `Tensor::from_data`, `generate_random`, and the gather
of an oracle result.
- Library code makes every other tensor `like` an input (`Tensor::like`, `zeros_like`,
  `eye_like`), so a computation stays on the backend and device of its data.
- The Tensor operation set and the backends are unchanged.
- A few calls make tensors from host numbers by design: the random sets, `Expr` constants
  without a context, and `lean::Tensor::gather`.

**One code path.** Every algorithm and operation exists once, written against `Tensor`.
- No code outside `src/global/backend/` and `src/global/tensor/` branches on the backend,
  device or dtype. There is no `#ifdef CORACPP_TORCH` in library code. The conventions test
  checks this and lists the exceptions: the bindings, and the umbrella header.
- The Python package and the MATLAB bridge convert arguments and compute nothing.
- A token clone detector in the conventions test fails on a repeated sequence of 100 tokens
  (literals abstracted). That is how the duplicated code was found and merged:
  - the seven binary entries;
  - the interval set difference of the two verify algorithms;
  - the distance blocks of the zonotope verifier;
  - `Tensor::cat_last`/`cat_rows`;
  - `Specification::check`/`holds`.
- The competition entry (`cora::comp`) is kept apart on purpose.

**Algorithms are types.** Each algorithm is a struct of exactly its options, with CORA's
defaults, a `name` and a `validate()`:
- `alg::Standard`, `alg::WrappingFree`, `alg::ReachAvoidSupportFunction`,
  `alg::ReachAvoidZonotope` (linear systems);
- `alg::Exact` (linear simulation);
- `alg::Linearization` and `alg::RungeKutta4` (nonlinear systems).

The operations of a system take a `std::variant` of the structs it supports (`alg::LinearReach`,
...). The problem (`ReachProblem`, `SimulationProblem`, `RandomSimulationProblem`) is a separate
struct, as CORA separates `params` from `options`.
- Each struct declares its fields once (`static constexpr auto fields()`). Python builds the
  problem and the algorithm from keyword arguments through these lists. An unknown keyword is
  refused with the closest name ("did you mean 't_final'?").
- The structs are passed by const reference and validated once per call, by the entry point.
- Files: `include/cora/cont_dynamics/algorithms.h`, `include/cora/cont_dynamics/cont_dynamics.h`,
  `include/cora/global/options.h`, `src/global/python/convert.h`.

**Sets are immutable values.**
- An operation returns a new set, and the members are private behind accessors (`center()`,
  `generators()`), so a set cannot be changed after its checks.
- Python sees properties (`Z.center`, `Z.generators`, `Z.dim`), the operators `+ @ * & | in -`
  and `__repr__`. In Python, `@` is the matrix product and `*` the product with a number or a
  matrix.
- Files: `include/cora/cont_set/`, `src/global/python/bindings.cpp`.

**Public headers and an installable package.**
- The public headers live in `include/cora/`. `cora/cora.h` includes all of them, and
  `cora/config.h` (written by CMake) records `CORACPP_CHECKS`. Private headers stay in `src/`.
- `cmake --install build --prefix P` installs the headers, the static library and
  `CORAConfig.cmake`, so a project writes `find_package(CORA)` and links `CORA::cora`. The
  package carries the dependencies: Eigen (bundled if it was fetched), OpenMP and libtorch.
- The ctest `install/consumer` installs the build and compiles `tests/install/consumer` against
  the install tree.
- Files: `scripts/cmake/install.cmake`, `scripts/cmake/CORAConfig.cmake.in`.

**Tolerances as CORA has them.**
- Where CORA's function takes a `tol`, CORA.cpp takes it too, as an explicit last argument with
  CORA's default: `contains` 100·eps, `is_intersecting` and `within_tol` 1e-8. A docstring says
  in one line what `tol` relaxes.
- Every comparison with a tolerance goes through `cora::within_tol(a, b, tol)`. Like CORA's
  `withinTol`, it accepts an absolute or a relative difference, and equal infinities agree.
- Open point: CORA's `isIntersecting` documents `tol` without saying whether it is absolute or
  relative for every pair. CORA.cpp reads it as absolute until a pair needs otherwise.
- File: `include/cora/global/within_tol.h`.

**Extension from Python.**
- A Python subclass of `cora.ContSet` (through a pybind11 trampoline) implements `_dim`,
  `_center`, `_support_function`, `_interval`, `_rand_point`, `_project`, `_vertices` and
  `_display`. It then passes through the same checked entries, the registry (`ClassId::custom`,
  its pairs registered with `cora.register_binary`), plotting and specifications.
- `@cora.register_algorithm(cora.LinearSys, "my_alg")` adds an algorithm.
  `sys.reach(X0, t_final=..., alg="my_alg", ...)` then validates the problem as for a built-in
  algorithm and calls the function with the other keywords. The result is a `ReachSet`, so
  plotting and specifications work on it.
- The building blocks of the reach algorithms are public and checked entries. Inside an
  algorithm they run unchecked, as nested entries:
  - `correction_matrix_state`;
  - `correction_matrix_input`;
  - `expm_remainder`;
  - `input_solution` (C++).
- Files: `src/global/python/bindings.cpp` (`PyContSet`, `run_with_kwargs`),
  `src/global/python/cora/extend.py`.

**No measurable overhead.**
- The guard is one thread-local increment and decrement, and nothing with `CORACPP_CHECKS=OFF`.
- A setting is a thread-local load.
- The registry is an array lookup.
- The algorithm structs are passed by reference.
- No allocation or virtual call was added inside the step loops.
- The ARCH benchmarks and the 5-dimensional example are timed before and after each framework
  change. The bound is 3%.

**Pull requests.** CI runs on pushes to `main` and on pull requests (`.github/workflows/ci.yml`).
Work happens on a branch, which reaches `main` through a reviewed pull request.
