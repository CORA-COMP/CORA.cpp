# Contributing

## Layout

```
include/cora/              the public headers (cora/cora.h includes them all; cora/config.h is written by CMake)
  cont_set/                cont_set.h (ContSet), set.h (Set), registry.h (the binary operations)
    zonotope/  interval/   the class in its header
  cont_dynamics/           cont_dynamics.h (ReachSet, the problems), algorithms.h (the alg:: structs)
    linear_sys/  nonlinear_sys/
  nn/neural_network/  specification/  global/ (tensor, backends, errors, settings, plot, oracle)
src/                       the implementation, one operation per file, mirroring include/cora/
  cont_set/cont_set/       the checked entries of ContSet and the binary registry
  cont_set/zonotope/ ...   the _impl of each operation; private/ holds priv_* algorithms
  global/python/           bindings.cpp, convert.h and the cora/ package (no math)
tests/  examples/          mirror src/; tests/install/consumer builds against the install tree
scripts/competition/       the CORA-COMP entry (cora::comp, kept apart)
scripts/ci/  benchmarks/micro/   the CI helpers (test selection, instruction counts) and the micro-benchmarks they run
scripts/coverage/          the coverage tables of CORA (inventory, CORALean import, the checker)
coverage/                  the generated tables, the inventory, status.json, out-of-scope.md
docs/                      architecture.md (the decisions), naming.md (CORA -> C++ -> Python)
```

The library is namespace `cora`; the competition entry keeps its own sets in `cora::comp`.

## Conventions

`tests/global/test_coding_conventions.py` checks all of these; it runs first in `ctest`, after the coverage check.

- One operation per file, named after the operation in snake_case (`rand_point.cpp` for CORA's
  `randPoint`), with a header (Syntax, Inputs, Outputs, See also; 1-3 lines per entry, no filler)
  that ends with `// CORA: <path>.m` or `// CORA.cpp only: <reason>`. Auxiliary functions
  (`aux_*`) under an `AUXILIARY` marker, the operation under `MAIN` (two empty lines above it),
  all between `BEGIN CODE` and `END OF CODE`; internals in `private/priv_*`.
- Classes PascalCase; functions, arguments, variables, files snake_case; see docs/naming.md.
- Inline comments are one line and say why; function docs are short and explain parameters.
- Every public operation starts with `detail::Entry entry;`, checks its input with `CORA_CHECK`
  and calls an unchecked implementation (`<operation>_impl`, or private code). Never `throw`:
  `CORA_CHECK(cond, Error::<id>, [&] { return message; })` for a wrong input, `CORA_FAIL(id, message)`
  for a failure. An entry that needs another public operation's result for valid input calls
  its unchecked form or checks the input itself: a nested entry does not check.
- Every option switch handles all options and ends in `CORA_FAIL`; a test covers each option.
- The tests of an operation call it with an invalid input (`test::raises(f, Error::<id>)`) and
  once with the checks off (`ChecksScope off(false);`).
- Where CORA's function has a `tol`, ours has it too, as an explicit last argument with CORA's
  default (contains 100·eps, is_intersecting 1e-8); the docstring says in one line what it
  relaxes, and the comparison goes through `cora::within_tol(a, b, tol)`.
- One code path: library code never asks which backend, device or dtype it runs on (only
  `global/backend/` and `global/tensor/` do), makes new tensors `like` an input, and is not
  duplicated (the conventions test fails on a clone of 100 tokens).

## Coverage

`coverage/` says per CORA method how far CORA.cpp is ([README](coverage/README.md), one table per CORA
class). It is computed from the tags and never edited: `python scripts/coverage/coverage.py` rewrites
it, and `ctest` (`coverage`, the first test) fails when it is stale.

- A source names its CORA file (`// CORA: <path>.m`; `// CORA: <path>.m:method` for a method of a
  single-file class). The tests declare what they test with the same line (`// CORA:` in C++, `# CORA:`
  in Python), after the first comment line.
- Status, low to high: `todo`, `untested` (a source, but not both tests), `ported` (a C++ and a Python
  test), `property` (`// PROPERTY:` a test of a stated property on random inputs with fixed seeds),
  `oracle` (`// ORACLE:` a test against CORALean, in `tests/**/oracle/` or `test_lean*`), `out-of-scope`
  (a row with its reason in `coverage/out-of-scope.md`).
- Coverage only goes up: a row below its status in `coverage/status.json` fails the check
  (`--accept-drop <row>` records a drop). The oracle debt, the rows that are `ported` or `property`
  while CORALean has them (`coverage/coralean-status.json`), may not grow (`--accept-debt`).
- `scripts/coverage/inventory.py` regenerates the rows from a CORA checkout (`~/CORA` or `CORA_DIR`);
  `scripts/coverage/import_coralean.py <CORALean>` refreshes what CORALean has. Both run on demand.
- `tests/global/test_coverage.py` tests the checker.

## Examples

CMake compiles every C++ example with `-include cora/global/banner.h`, which prints the
`CORA START` block before `main` and a `CORA END` block after it; an example does not include or call it.

## Adding things

**An operation** of a set class (say `Zonotope::volume`):

1. Declare the checked entry in the class header. For an operation of every set, declare the
   non-virtual entry in `include/cora/cont_set/cont_set.h` and a protected pure
   `volume_impl` that each class overrides.
2. Write `src/cont_set/zonotope/volume.cpp` (or `src/cont_set/cont_set/volume.cpp` for the
   entry of every set). The header names the CORA file. The function begins with
   `detail::Entry entry;`, then `CORA_CHECK`s, then the computation against `Tensor`. CMake
   finds the file.
3. Bind it in `src/global/python/bindings.cpp` (`.def("volume", &Zonotope::volume)`).
4. Test it in `tests/cont_set/zonotope/test_zonotope_volume.cpp`: the results, an invalid input
   with `test::raises`, and a call under `ChecksScope off(false)`. Add a Python test if the
   binding does more than forward.

**A set class** (say `Ellipsoid`):

1. Derive it from `ContSet` with `ClassId::ellipsoid` (the ids are in CORA's precedence order).
2. Write the `_impl` overrides, one file each, and the constructor with its checks in
   `ellipsoid.cpp`.
3. Add it to `set_of`/`object_of` in `src/global/python/convert.h` and bind it.

**A binary pair** (say `plus(Ellipsoid, Zonotope)`): write the function and add one line to
`aux_build()` in `src/cont_set/cont_set/registry.cpp`:
`aux_add_set<BinaryOp::plus, Ellipsoid, Zonotope>(table, [](const Ellipsoid &E, const Zonotope &Z) { ... })`.
The lower-precedence class comes first, and a `static_assert` says so. The entry checks the
dimensions for every pair. From Python, `cora.register_binary("plus", MySet, cora.Zonotope, fn)`
adds a pair at run time.

**An algorithm** (say a new `reach` of `LinearSys`):

1. Add a struct to `include/cora/cont_dynamics/algorithms.h` with its options and CORA's
   defaults, a `static constexpr const char *name`, `fields()` and `validate()`.
2. Add the struct to the variant (`alg::LinearReach`).
3. Write the algorithm in `src/cont_dynamics/linear_sys/private/` and dispatch to it in `reach.cpp`
   with `std::visit`.
4. Python gets it by name with its fields as keywords, with no binding code.

To prototype in Python first, decorate a function with
`@cora.register_algorithm(cora.LinearSys, "my_alg")`. `sys.reach(X0, t_final=..., alg="my_alg")`
then checks the problem and calls it (see `src/global/python/cora/extend.py`).

**A backend:** implement `Tensor::Impl` and `Tensor::Backend` (`src/global/backend/eigen.cpp` is
the model) and register it in `aux_make_backend` (`src/global/tensor/tensor.cpp`). Only the backend and tensor
folders may know it.

## Tensor

Sets and dynamics are written once against `Tensor`, a runtime type-erased wrapper of Eigen
(double, CPU, unbatched) or libtorch (batched, GPU, autograd).
- Layout is `(..., rows, cols)`; generic code broadcasts the leading dimensions, which is the
  automatic batching.
- The backend of new tensors made from numbers is a per-thread scope: `BackendScope s("torch")`
  in C++, `with cora.backend("torch"):` in Python. Otherwise it is the process default
  (`set_default_backend`, `CORACPP_BACKEND`).
- Everything else stays where its data is, and a tensor can live on its own device
  (`Tensor(data, "gpu")`, `t.to("cpu")`).
- Runtime dispatch costs about 17 ms over 250 steps of the 5-dim example.

## Build and test

```bash
scripts/setup_local.sh                                       # once
scripts/with_env.sh scripts/build.sh                          # everything (or: a target, --debug, --eigen)
scripts/with_env.sh scripts/test.sh -j4                        # conventions, C++ tests, examples, Python
scripts/with_env.sh scripts/build.sh example_zonotope_01      # one target, e.g. a C++ example
scripts/with_env.sh python -m unittest discover -s tests/python
scripts/with_env.sh cmake --install build --prefix /tmp/cora   # the package of find_package(CORA)
```

`with_env.sh` enters the conda environment and puts the built `cora` package on `PYTHONPATH`.
Options (`scripts/build.sh -D...`, see `CMakeLists.txt`): `CORACPP_CHECKS` (OFF compiles the input checks out), `CORACPP_TORCH`, `CORACPP_TORCH_DIR`, `CORACPP_NATIVE`,
`CORACPP_BLAS`; `.coracpp-build` names another build directory. Tests run on every backend and the GPU;
the `_torch` ones need libtorch.

## CI

Three tiers (`.github/workflows/`); the one check to require in the ruleset is `ci-gate`.
- **Every push to a pull request** (`ci.yml`, under 10 minutes): `checks` (format, conventions,
  coverage when `scripts/coverage/coverage.py` exists), then `linux-core` and `linux-python`, which
  build everything and run only the tests the change can affect, and `instructions`, which fails
  when a micro-benchmark of `benchmarks/micro` runs more than 3% more instructions than at the base
  commit. A documentation-only change skips the builds. A new push cancels the older run.
- **Before a merge** (`merge.yml`, merge queue): every test of `core`, `torch`, `python`, `python-torch` on Linux.
- **Every night** (`nightly.yml`): the full matrix of three operating systems, ASan and UBSan, the
  competition binary; skipped when `main` has not moved since the last successful night. A failed
  night opens an issue labelled `nightly`.

Which tests a change affects, from the `#include` graph (`ALL` after a change to the tensor, the
backends, the global headers, the build or CI; `NONE` for documentation):

```bash
python3 scripts/ci/affected_tests.py origin/main HEAD          # a regex for ctest -R
ctest --preset core -R "$(python3 scripts/ci/affected_tests.py origin/main HEAD)"
scripts/ci/instruction_counts.sh origin/main                    # the instruction counts (needs valgrind)
```

A full run by hand: Actions > `merge` > Run workflow (or `gh workflow run merge.yml --ref <branch>`);
`nightly` can be started the same way and then never skips.

## VS Code

`scripts/setup_local.sh` writes `.vscode/settings.json` and `c_cpp_properties.json` (git-ignored:
they hold this machine's paths); `tasks.json` and `launch.json` are shared and read those paths
(`${config:coracpp.*}`). Rerun `scripts/with_env.sh python scripts/setup_vscode.py` after moving
the checkout or recreating the environment. On Windows, `WSL: Reopen Folder in WSL` first.

- `F5` runs the open example, C++ or Python, through `scripts/run_example.py` (the first entry of
  the Run and Debug dropdown): a Python one runs under the Python debugger, so breakpoints in it
  work; a C++ one is built and run. Choose `C++: debug current example` (gdb, `-O0 -g` build) to
  step through C++; it needs the C/C++ extension installed *in WSL* (extensions view >
  `Install in WSL`), else VS Code says "debug type 'cppdbg' not supported".
- `Ctrl+Shift+B`: build and run the open C++ example.
- `Terminal > Run Task`: `test (conventions and C++)`, `test (python)`, `build python package`.
- The Python configurations run the environment's interpreter (`coracpp.envPrefix`), whatever the
  editor has selected; for IntelliSense pick it once with `Python: Select Interpreter` (`coracpp`).
  "The minimum Python version for the debugger is 3.9" means the system Python was used.

## CORALean oracle

`cora::lean` (C++) and `cora.lean` (Python) compute sound floating-point sets with the
[CORALean](https://gitlab.lrz.de/cps/coralean) project: a nominal zonotope plus an error box that
encloses every rounding error, in `binary64`, `ieee:<format>`, `dyadic:<p>` or `fixedpoint:<f>`. The
computation runs in a separate process, the CORALean `oracle`, that CORA.cpp starts and talks to with
one JSON line per request.

Two executables serve it: the `oracle` (sound, every dtype above) and the `oracle-unsound`, which
offers `nearest` (round-to-nearest, no guarantees). `nearest` is what the `lean` Tensor backend
(`BackendScope s("lean")`) must match bit for bit; `tests/global/oracle/test_lean_backend.cpp` checks that. The
dtype `nearest` goes to the unsound oracle automatically.

Use a local build (CORALean needs Lean via elan, leanprover.github.io):

```bash
git clone https://gitlab.lrz.de/cps/coralean.git && cd coralean
lake exe cache get && lake build oracle              # .lake/build/bin/oracle (oracle.exe on Windows)
cmake --preset core -DCORACPP_ORACLE=/path/to/coralean/.lake/build/bin/oracle
cmake --build --preset core && ctest --preset core -R lean
```

`CORACPP_ORACLE` (and `CORACPP_ORACLE_UNSOUND`) is OFF, `DOWNLOAD`, or the executable or a shell
command that starts it (`cd /path/to/coralean && lake exe oracle`); without an oracle the tests and
examples named `*lean*` skip. Outside CMake, set the environment variable of the same name. From WSL,
the Windows `oracle.exe` works over its pipes. `DOWNLOAD` fetches the release pinned in
`scripts/coralean.cmake` (Linux x86_64) and checks its hash. Protocol and operations:
`experiments/oracle/README.md` in CORALean; the C++ side is `src/global/oracle/`. A dtype is set with
`lean::DTypeScope s("ieee:binary32")` (Python: `with cora.lean.dtype("ieee:binary32"):`); values cross into CORA.cpp only when they are exact doubles.

### Still to do for a fresh install

The `oracle` CI job (`.github/workflows/nightly.yml`) is defined and disabled (`if: false`). To enable it:

1. Publish CORALean (public repository or public releases); CORA.cpp may move to GitLab later, only
   the URL in `scripts/coralean.cmake` changes.
2. CORALean CI builds `oracle` and `oracle-unsound` for Linux x86_64 and attaches them to a release
   as `oracle-linux-x86_64` and `oracle-unsound-linux-x86_64`, with their SHA-256.
3. Merge CORALean's `feat/oracle-unsound` (the `oracle-unsound` executable with `nearest`, under
   `Examples/Bench/`, where the false axiom behind it is allowed; the sound `oracle` never reaches it).
4. Fill `CORALEAN_VERSION` and both hashes in `scripts/coralean.cmake`, remove `if: false`.
5. Compare `ping`'s `version` (CORALean's commit, with a `dirty` flag) with the pinned one in CORA.cpp
   (not done yet), so a mismatch is refused.

Later, without Lean and without a binary: CORALean releases the generated C of the oracle with
`lean.h` and the runtime sources, and CORA.cpp compiles it as a library. It needs a JSON codec of the
oracle's own (`Lean.Data.Json` pulls the Lean elaborator in, 107 MB); that reaches about 5 MB.

## Plotting

`global/plot/plot.h` decides everything about a plot: the projection, the vertices, the color
scheme (`use_cora_colors`), the next color of CORA's order, widths and what lies on top (`zorder`).
It puts resolved layers into a `Figure`. `Figure::svg()` writes them for C++; `python/cora/plot.py`
takes them out (`Figure.take_layers()`) and draws them with matplotlib, and nothing else. A new
frontend (or a new plot option) is added to the core, never to a wrapper.
