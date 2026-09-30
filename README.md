# CORA.cpp

Parts of [CORA](https://cora.in.tum.de), the toolbox for set-based computing, in C++ and Python.
Runs on CPU and GPU, batches automatically, is differentiable. Names follow MATLAB CORA:
`contSet/zonotope/mtimes.cpp` is `@zonotope/mtimes.m`.

C++:

```cpp
#include "contDynamics/linearSys/linearSys.h"
#include "global/plot/plot.h"
#include "specification/specification.h"
using namespace cora;

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

Needs a C++20 compiler and CMake 3.22 or newer, on Linux, macOS or Windows. Pick what to add:

| preset | adds |
| --- | --- |
| `cpp` | nothing: the C++ library on Eigen |
| `torch` | libtorch for GPU, batching and gradients; it is downloaded, no Python needed |
| `python` | the Python package on numpy (creates `.venv`) |
| `python-torch` | the Python package with torch tensors (creates `.venv`, installs torch into it) |

```bash
cmake --preset cpp                    # or: torch, python, python-torch
cmake --build --preset cpp
ctest --preset cpp                    # tests and examples
```

Eigen is fetched if it is not installed. The Python presets need Python 3; they run
`pip install -r requirements-python.txt` (`requirements.txt` for torch) in `.venv` at the project
root, nothing else is touched. Use the package with `PYTHONPATH=build/python` (Windows:
`set PYTHONPATH=build\python`) and `.venv`'s Python. `-DCORACPP_TORCH_VARIANT=cu126` downloads a CUDA
libtorch instead of the CPU one. On Linux and WSL2, `scripts/setup_local.sh` alternatively sets up a
conda environment (`scripts/with_env.sh scripts/build.sh`). In VS Code, open an example and press `F5`.

## Examples

One file per topic, in [`examples/cpp`](examples/cpp) and [`examples/python`](examples/python):
reachability of linear and nonlinear systems, specifications, backends, batching, GPU, gradients,
learning, conformal prediction, neural network verification, and the CORALean oracle.

Differences from MATLAB: points are columns `(n, 1)` in C++ (1-D in Python); `R.timeInt[k]` covers
`[k*timeStep, (k+1)*timeStep]` and `R.timePoint[k]` is at `k*timeStep` (0-based).

[`competition/`](competition/README.md) holds the CORA-COMP entry; see
[CONTRIBUTING.md](CONTRIBUTING.md) to work on the library.
