# Contributing

## Layout

```
src/
  tensor/                  Tensor, the backend wrapper (eigen.cpp, torch.cpp)
  contSet/                 contSet.h, the abstract set
    zonotope/  interval/   the class in its header, one operation per file
  contDynamics/            contDynamics.h, the abstract dynamic system
    linearSys/             linearSys.h, reach, simulate, ...; private/ has the algorithms
    nonlinearSys/          nonlinearSys.h, reach (linearization), simulate; dynamics are Expr
  nn/neuralNetwork/        neuralNetwork.h, evaluate (points, and sets through the layers)
  specification/           specification.h, check.cpp
  global/                  random numbers, threads, symbolic expressions (expr.h),
                           plot/ (colors, figure, plot: the plotting logic for every language)
  python/                  bindings.cpp and the cora/ package (plot.py, tensor.py)
  lean/                    cora::lean: sets and dynamics computed by CORALean (needs CORACPP_ORACLE)
tests/  examples/          mirror src/
competition/               the CORA-COMP entry
```

The library is namespace `cora`; the competition entry keeps its own sets in `cora::comp`.

## Conventions

- One operation per file, named as in CORA, with a header (Syntax, Inputs, Outputs, See also);
  auxiliary functions (`aux_*`) first, under an `AUXILIARY` marker, then the operation under
  a `MAIN` marker (two empty lines above it), all between `BEGIN CODE` and `END OF CODE`; internals in `private/priv_*`.
- camelCase operations, CORA argument names (`timeStep`, `tFinal`, `taylorTerms`, `linAlg`).
- Inline comments are one line and say why; function docs are short and explain parameters.
- Every option switch handles all options explicitly and throws a descriptive error otherwise,
  and a test covers each option and the error.
- `tests/global/test_codingConventions.py` checks naming and layout; it is the first test of `ctest`.

## Examples

CMake compiles every C++ example with `-include global/banner.h`, which prints the
`CORA START` block before `main` and a `CORA END` block after it; an example does not include or call it.

## Adding things

- **Operation:** declare it in the class, add the file to its folder; the CMake file finds sources.
- **Set or dynamics class:** write it against `Tensor` only, then bind it in `src/python/bindings.cpp`.
- **Backend:** implement `Tensor::Impl` and `Tensor::Backend` (`src/tensor/eigen.cpp` is the model)
  and register it in `makeBackend`.

## Tensor

Sets and dynamics are written once against `Tensor`, a runtime type-erased wrapper of Eigen
(double, CPU, unbatched) or libtorch (batched, GPU, autograd). Layout is `(..., rows, cols)`;
generic code broadcasts leading dimensions, which is the automatic batching. The backend is set
with `setBackend` or `CORACPP_BACKEND`; a tensor can also live on its own device
(`Tensor(data, "gpu")`, `t.to("cpu")`). Runtime dispatch costs about 17 ms over 250 steps of the
5-dim example.

## Build and test

```bash
scripts/setup_local.sh                                       # once
scripts/with_env.sh scripts/build.sh                          # everything (or: a target, --debug, --eigen)
scripts/with_env.sh scripts/test.sh -j4                        # conventions, C++ tests, examples, Python
scripts/with_env.sh scripts/build.sh example_zonotope_01      # one target, e.g. a C++ example
scripts/with_env.sh python -m unittest discover -s tests/python
```

`with_env.sh` enters the conda environment and puts the built `cora` package on `PYTHONPATH`.
Options (`scripts/build.sh -D...`, see `CMakeLists.txt`): `CORACPP_TORCH`, `CORACPP_TORCH_DIR`, `CORACPP_NATIVE`,
`CORACPP_BLAS`; `.coracpp-build` names another build directory. Tests run on every backend and the GPU;
the `_torch` ones need libtorch.

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
(`setBackend("lean")`) must match bit for bit; `tests/lean/test_lean_backend.cpp` checks that. The
dtype `nearest` goes to the unsound oracle automatically.

Use a local build (CORALean needs Lean via elan, leanprover.github.io):

```bash
git clone https://gitlab.lrz.de/cps/coralean.git && cd coralean
lake exe cache get && lake build oracle              # .lake/build/bin/oracle (oracle.exe on Windows)
cmake --preset cpp -DCORACPP_ORACLE=/path/to/coralean/.lake/build/bin/oracle
cmake --build --preset cpp && ctest --preset cpp -R lean
```

`CORACPP_ORACLE` (and `CORACPP_ORACLE_UNSOUND`) is OFF, `DOWNLOAD`, or the executable or a shell
command that starts it (`cd /path/to/coralean && lake exe oracle`); without an oracle the tests and
examples named `*lean*` skip. Outside CMake, set the environment variable of the same name. From WSL,
the Windows `oracle.exe` works over its pipes. `DOWNLOAD` fetches the release pinned in
`scripts/coralean.cmake` (Linux x86_64) and checks its hash. Protocol and operations:
`experiments/oracle/README.md` in CORALean; the C++ side is `src/lean/`. A dtype is set with
`lean::setDType("ieee:binary32")`; values cross into CORA.cpp only when they are exact doubles.

### Still to do for a fresh install

The `oracle` CI job (`.github/workflows/ci.yml`) is defined and disabled (`if: false`). To enable it:

1. Publish CORALean (public repository or public releases); CORA.cpp may move to GitLab later, only
   the URL in `scripts/coralean.cmake` changes.
2. CORALean CI builds `oracle` and `oracle-unsound` for Linux x86_64 and attaches them to a release
   as `oracle-linux-x86_64` and `oracle-unsound-linux-x86_64`, with their SHA-256.
3. Put `nearest` into the unsound oracle only: `nearest` rests on a false axiom that
   CORALean's hook allows under `Examples/Bench/` alone, so it needs an executable of its own there.
   The branch `experiment/oracle-native-dtypes` (local, in CORALean) has the dtype code.
4. Fill `CORALEAN_VERSION` and both hashes in `scripts/coralean.cmake`, remove `if: false`.
5. Check the oracle's `ping` version against the pinned one (not done yet), so a mismatch is refused.

Later, without Lean and without a binary: CORALean releases the generated C of the oracle with
`lean.h` and the runtime sources, and CORA.cpp compiles it as a library. It needs a JSON codec of the
oracle's own (`Lean.Data.Json` pulls the Lean elaborator in, 107 MB); that reaches about 5 MB.

## Plotting

`global/plot/plot.h` decides everything about a plot: the projection, the vertices, the color
scheme (`useCORAcolors`), the next color of CORA's order, widths and what lies on top (`zorder`).
It puts resolved layers into a `Figure`. `Figure::svg()` writes them for C++; `python/cora/plot.py`
takes them out (`Figure.takeLayers()`) and draws them with matplotlib, and nothing else. A new
frontend (or a new plot option) is added to the core, never to a wrapper.
