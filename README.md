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

Needs a C++20 compiler, CMake and Eigen (fetched if missing). The **torch backend** (GPU, batching,
gradients, the Python package) is used whenever the Python that CMake finds has `torch`; without it
the build is Eigen only. Linux, macOS and Windows work; build inside a virtual environment:

```bash
python -m venv .venv                          # or: uv venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.txt               # torch, numpy, matplotlib (uv: uv pip install -r ...)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build -j4                    # tests
PYTHONPATH=build python examples/python/example_linear_reach_01_5dim.py    # Windows: set PYTHONPATH=build
```

Skip the first three lines for an Eigen-only build, or pass `-DCORACPP_TORCH=OFF`.
On Linux and WSL2, `scripts/setup_local.sh` does all of this in a conda environment
(`scripts/with_env.sh scripts/build.sh`, `scripts/with_env.sh scripts/test.sh -j4`). In VS Code, open an
example and press `F5`.

## Examples

One file per topic, in [`examples/cpp`](examples/cpp) and [`examples/python`](examples/python):
reachability of linear and nonlinear systems, specifications, backends, batching, GPU, gradients,
learning, conformal prediction, neural network verification, and the CORALean oracle.

Differences from MATLAB: points are columns `(n, 1)` in C++ (1-D in Python); `R.timeInt[k]` covers
`[k*timeStep, (k+1)*timeStep]` and `R.timePoint[k]` is at `k*timeStep` (0-based).

[`competition/`](competition/README.md) holds the CORA-COMP entry; see
[CONTRIBUTING.md](CONTRIBUTING.md) to work on the library.
