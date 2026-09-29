# CORA.cpp

A C++ and Python implementation of parts of [CORA](https://cora.in.tum.de), the toolbox for
set-based computing, that runs on the CPU and on GPUs, batches automatically and is
differentiable. Folders, file names, function names and arguments follow MATLAB CORA, so if
you know CORA you know where to look: `contSet/zonotope/mtimes.cpp` is `@zonotope/mtimes.m`.

```cpp
#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"
using namespace cora::ct;

Tensor A({{-0.1, 1}, {-1, -0.1}});                                   // x' = A x
Zonotope X0(Tensor({1, 0}), Tensor({{0.1, 0}, {0, 0.1}}));           // center, generators
Reach R = LinearSys(A).reach(X0, /*timeStep=*/0.1, /*tFinal=*/5.0, /*taylorTerms=*/8);
Specification spec = Specification::unsafeSet(Tensor({1, 0}), -0.75);  // stay out of x1 <= -0.75
bool safe = spec.check(R.timeInt);
```

```python
import coracpp
R = coracpp.reach(A, c, G, timeStep=0.1, tFinal=5.0)     # A, c, G: torch tensors or numpy arrays
coracpp.plot(R)
```

## Build

Needs a C++17 compiler and [Eigen](https://eigen.tuxfamily.org). [libtorch](https://pytorch.org)
is optional and adds the GPU, batching, gradients and the Python package; the libtorch of a
pip-installed `torch` works (`python -c "import torch, os; print(os.path.dirname(torch.__file__))"`).

```bash
make test example                          # Eigen only: tests, then the C++ examples
make TORCH=/path/to/libtorch test example  # with libtorch (TORCH_ABI=0 if it was built with the old ABI)
make python TORCH=/path/to/libtorch        # the Python package, into build/coracpp
PYTHONPATH=build python examples/python/example_linearSys_reach.py
```

## What is implemented

| | CORA | Operations (one file each) |
| --- | --- | --- |
| `Zonotope` | `zonotope` | `mtimes` (matrix or interval matrix), `plus`, `linComb`, `supportFunc`, `interval`, `randPoint`, `generateRandom` |
| `Interval` | `interval` (also as interval matrix) | `mtimes`, `plus`, `supportFunc`, `contains`, `randPoint`, `generateRandom`, `center`, `rad` |
| `ContSet` | `contSet` | the abstract set: `dim`, `center`, `supportFunc`, `interval`, `randPoint` |
| `LinearSys` | `linearSys` | `reach` (`standard`, `wrapping-free`), `simulate`, `simulateRandom`, `correctionMatrixState` |
| `Specification` | `specification` | `safeSet`, `unsafeSet` for halfspaces; `check`, `holds`, `firstViolation` |
| `coracpp.plot` | `plot` | reachable sets, zonotopes, simulations, points, specifications, in CORA's colors |

`reach` and `correctionMatrixState` match MATLAB CORA R2024b to 1e-12 on the systems in
`tests/contDynamics/linearSys/`. Inputs to the system and other set representations are not
implemented yet.

Differences from MATLAB: vectors and points are columns `(n, 1)`; the steps of a result are
`R.timeInt[k]` over `[k*timeStep, (k+1)*timeStep]` and `R.timePoint[k]` at `k*timeStep`
(0-based); sets are classes with methods, not operator overloads (`Z.mtimes(M)`); in Python
the result stacks the steps, `R.timeInt_c` is `(steps, n)` and `R.timeInt_G` `(steps, n, m)`.

## Backends, devices, batching, gradients

Sets and dynamics are written once against `ct::Tensor`, which wraps Eigen or libtorch. The
backend is chosen once, in code or from outside, and every tensor is made on it:

```cpp
setBackend("torch");                   // "eigen", "torch", "torch:cuda"; or CORACPP_BACKEND=eigen
Tensor A({{0, 1}, {-1, 0}});           // on that backend
Tensor B({{0, 1}, {-1, 0}}, "gpu");    // or on a device of its own; t.to("cpu") moves it
```

| | |
| --- | --- |
| Batching | Leading dimensions of `A` `(…, n, n)` and of a set broadcast: a batch of systems, of sets or of both is the same call as one. Eigen holds one set. |
| Gradients | With libtorch, autograd differentiates through everything; `setBackend("torch,customBackward")` uses a hand-written backward pass for the matrix exponential instead. |
| GPU | `"torch:cuda"` as default, or `"gpu"` per tensor. |
| Python | Torch tensors run on libtorch, numpy arrays on Eigen; the call picks by its arguments. |

## Simulation, specifications, plotting

```cpp
Rng rng(0);                                                     // a seed repeats the draw
Tensor start = X0.randPoint(20, rng);                           // or type "extreme": corners
std::vector<Tensor> x = LinearSys(A).simulate(start, 0.05, 5.0);  // x[k]: the points at k*0.05
```

```python
x = coracpp.simulate(A, coracpp.randPoint(c, G, 20, seed=1), timeStep=0.05, tFinal=5.0)
coracpp.plot(R, label="reachable set"); coracpp.plot(x, label="simulations")
coracpp.plot(spec, label="unsafe set")           # draw the sets first: it shades to the limits
```

## Examples and tests

`examples/cpp/` and `examples/python/` have one small file per topic
(`example_linearSys_reach`, `_reach_eigen`, `_reach_gpu`, `_reach_batch`, `_gradient`,
`_simulate`, `_specification`, `_plot`, …). `tests/` mirrors `src/`; each test runs on every
backend and on the GPU when there is one, and the `_torch` ones need libtorch. `make test`
runs the C++ tests; `PYTHONPATH=build python -m unittest discover -s tests/python` the Python ones.

## Layout

```
src/
  tensor/                  Tensor, the backend wrapper (eigen.cpp, torch.cpp)
  contSet/                 contSet.h, the abstract set
    zonotope/  interval/   the class in its header, one operation per file
  contDynamics/linearSys/  linearSys.h, reach, simulate, ...; private/ has the algorithms
  specification/           specification.h, check.cpp
  global/                  random numbers, threads
  python/                  bindings.cpp and the coracpp/ package (plot.py, colors.py)
tests/  examples/          mirror src/
competition/               the CORA-COMP entry, see competition/README.md
```

An operation is one file named as in CORA with a short header (Syntax, Inputs, Outputs, See
also) and its auxiliary functions at the bottom. To add one, add the declaration to the
class and the file to its folder; the Makefile finds sources itself. To add a set or a
dynamics class, write it against `Tensor` only. To add a backend, implement `Tensor::Impl` and
`Tensor::Backend` (`src/tensor/eigen.cpp` is the model) and register it in `makeBackend`.

## CORA-COMP

This repository is also a tool in the CORA competition; how it is run there, the catalog's
own tuned set implementations, and the harness are in [`competition/`](competition/README.md).
