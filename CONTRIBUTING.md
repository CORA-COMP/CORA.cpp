# Contributing

## Layout

```
src/
  tensor/                  Tensor, the backend wrapper (eigen.cpp, torch.cpp)
  contSet/                 contSet.h, the abstract set
    zonotope/  interval/   the class in its header, one operation per file
  contDynamics/            contDynamics.h, the abstract dynamic system
    linearSys/             linearSys.h, reach, simulate, ...; private/ has the algorithms
  specification/           specification.h, check.cpp
  global/                  random numbers, threads
  python/                  bindings.cpp and the cora/ package (plot.py, colors.py, tensor.py)
tests/  examples/          mirror src/
competition/               the CORA-COMP entry
```

## Conventions

- One operation per file, named as in CORA, with a header (Syntax, Inputs, Outputs, See also);
  auxiliary functions (`aux_*`) first, under an `AUXILIARY` marker, then the operation under
  a `MAIN` marker, all between `BEGIN CODE` and `END OF CODE`; internals in `private/priv_*`.
- camelCase operations, CORA argument names (`timeStep`, `tFinal`, `taylorTerms`, `linAlg`).
- Inline comments are one line and say why; function docs are short and explain parameters.
- Every option switch handles all options explicitly and throws a descriptive error otherwise,
  and a test covers each option and the error.
- `tests/global/test_codingConventions.py` checks naming and layout; `make test` runs it first.

## Adding things

- **Operation:** declare it in the class, add the file to its folder; the Makefile finds sources.
- **Set or dynamics class:** write it against `Tensor` only, then bind it in `src/python/bindings.cpp`.
- **Backend:** implement `Tensor::Impl` and `Tensor::Backend` (`src/tensor/eigen.cpp` is the model)
  and register it in `makeBackend`.

## Tensor

Sets and dynamics are written once against `ct::Tensor`, a runtime type-erased wrapper of Eigen
(double, CPU, unbatched) or libtorch (batched, GPU, autograd). Layout is `(..., rows, cols)`;
generic code broadcasts leading dimensions, which is the automatic batching. The backend is set
with `setBackend` or `CORACPP_BACKEND`; a tensor can also live on its own device
(`Tensor(data, "gpu")`, `t.to("cpu")`). Runtime dispatch costs about 17 ms over 250 steps of the
5-dim example.

## Build and test

```bash
scripts/setup_local.sh                          # once
scripts/with_env.sh make test                   # conventions, then all C++ tests
scripts/with_env.sh make example                # C++ examples
scripts/with_env.sh make python
PYTHONPATH=build scripts/with_env.sh python -m unittest discover -s tests/python
make DEBUG=1 ...                                # -O0 -g
```

Variables: `TORCH` (libtorch path, auto-detected from pip torch), `BUILD`, `OPT`, `PYTHON`;
put local overrides in `local.mk`. Tests run on every backend, and on the GPU when present;
the `_torch` ones need libtorch.

## VS Code

Install the recommended extensions, then `F1` > `WSL: Reopen Folder in WSL` (the conda
environment lives in WSL). With an example open:

- `Ctrl+Shift+B` builds and runs a C++ example; `F5` debugs it (gdb, `-O0 -g` build in
  `~/.cache/coracpp/build-debug`) or, for a `.py` file, runs it with the `cora` package built.
- `Terminal > Run Task` has the builds, `test (conventions and C++)`, `test (python)`.
- The interpreter is `~/miniforge3/envs/coracpp/bin/python`; select it once if VS Code asks.
