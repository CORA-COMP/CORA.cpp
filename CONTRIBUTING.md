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

`cora::lean` (and `cora.lean` in Python) computes sound float sets with the CORALean project. Build
its `oracle` (`lake build oracle` on the `oracle` branch of CORALean) and point `CORACPP_ORACLE` at
the executable or a shell command that starts it; the tests and examples named `*lean*` use it and
skip without it. From WSL the Windows `oracle.exe` works over its pipes.

## Plotting

`global/plot/plot.h` decides everything about a plot: the projection, the vertices, the color
scheme (`useCORAcolors`), the next color of CORA's order, widths and what lies on top (`zorder`).
It puts resolved layers into a `Figure`. `Figure::svg()` writes them for C++; `python/cora/plot.py`
takes them out (`Figure.takeLayers()`) and draws them with matplotlib, and nothing else. A new
frontend (or a new plot option) is added to the core, never to a wrapper.
