# CORA.cpp

Parts of [CORA](https://cora.in.tum.de), the toolbox for set-based computing, in C++ and Python.
Runs on CPU and GPU, batches automatically, is differentiable. Names follow MATLAB CORA:
`contSet/zonotope/mtimes.cpp` is `@zonotope/mtimes.m`.

C++:

```cpp
#include "contDynamics/linearSys/linearSys.h"
#include "plot/plot.h"
#include "specification/specification.h"
using namespace cora::ct;

Tensor A({{-0.1, 1}, {-1, -0.1}});                                     // x' = A x
Zonotope X0(Tensor({1, 0}), 0.1 * Tensor::eye(2));                     // center, generators
Reach R = LinearSys(A).reach(X0, /*timeStep=*/0.1, /*tFinal=*/5.0);
Specification spec = Specification::unsafeSet(Tensor({1, 0}), -0.75);  // stay out of x1 <= -0.75
bool safe = spec.check(R.timeInt);
plot(R, {0, 1});                                                       // dimensions to show
figure().save("reach.svg");                                            // C++ writes SVG
```

Python:

```python
from cora import Tensor, Zonotope, LinearSys, Specification, eye, plot

A = Tensor([[-0.1, 1], [-1, -0.1]])                                    # x' = A x
X0 = Zonotope(Tensor([1, 0]), 0.1 * eye(2))                            # center, generators
R = LinearSys(A).reach(X0, timeStep=0.1, tFinal=5.0)
spec = Specification.unsafeSet(Tensor([1, 0]), -0.75)                  # stay out of x1 <= -0.75
safe = spec.check(R.timeInt)
plot(R)                                                                # dims default to (0, 1)
```

## Install

Needs a C++20 compiler and Eigen; libtorch (via `pip install torch`) adds GPU, batching,
gradients and the Python package. On Linux and WSL2 one script sets everything up:

```bash
scripts/setup_local.sh                                       # once: conda env, torch, VS Code
scripts/with_env.sh make run-example_linear_reach_01_5dim    # build and run a C++ example
scripts/with_env.sh make python                              # the Python package
scripts/with_env.sh python examples/python/example_linear_reach_01_5dim.py
scripts/with_env.sh make test                                # conventions and all tests
```

In VS Code (on Windows: `WSL: Reopen Folder in WSL` first), open an example and press `F5` to run
or debug it, `Ctrl+Shift+B` for a C++ example without the debugger.

## Usage

- **Sets:** `Zonotope(c, G)`, `Interval(inf, sup)`; any set works wherever a set is expected.
- **Batches:** a leading dimension in `c`, `G` or `A` is a batch; the call is the same as for one set.
- **Backends:** `setBackend("torch" | "torch:cuda" | "eigen")` or `CORACPP_BACKEND`. In Python, torch
  tensors run on libtorch, numpy arrays on Eigen. `device="gpu"` places a tensor on the GPU.
- **Gradients:** autograd works through `reach` and `simulate` on libtorch;
  `setBackend("torch,customBackward")` uses a hand-written backward pass for the matrix exponential.
- **Simulation:** `sys.simulate(x0, timeStep, tFinal)`, `sys.simulateRandom(X0, n, ...)`.
- **Plotting:** `plot(S, dims, options)` projects a set (or a `Reach`, `Specification`, simulation)
  onto two dimensions and draws it; reachable sets are drawn as their union. C++ writes SVG
  (`plot(R, {0, 1}, {.label = "reachable set"}); figure().save("reach.svg");`), Python uses
  matplotlib (`plot(R)`). Colors are CORA's: `{.color = "CORA:red"}`.
- **Options:** `linAlg` is `"standard"` or `"wrapping-free"`; an unknown value throws.

Differences from MATLAB: points are columns `(n, 1)` in C++ (1-D in Python); `R.timeInt[k]`
covers `[k*timeStep, (k+1)*timeStep]` and `R.timePoint[k]` is at `k*timeStep` (0-based);
sets are objects with methods (`Z.mtimes(M)`), not operators.

## Examples and tests

`examples/cpp/` and `examples/python/`: one file per topic, sectioned like CORA's
`example_linear_reach_01_5dim`. `make test` runs the tests, `make example` the C++ examples.

## CORA-COMP

The competition entry is in [`competition/`](competition/README.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) to work on the library.
