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

Linux or WSL2 (Windows: `wsl --install`, then `WSL: Reopen Folder in WSL` in VS Code; macOS and
native Windows are not supported). One script creates a conda environment with the compiler,
CMake and Eigen, and installs torch (the CUDA build if there is an NVIDIA GPU, `--cpu` forces the
CPU one):

```bash
scripts/setup_local.sh                                   # once
scripts/with_env.sh scripts/build.sh                     # build the library, examples, Python package
scripts/with_env.sh scripts/test.sh -j4                   # run the tests
scripts/with_env.sh python examples/python/example_linear_reach_01_5dim.py
```

Without conda: `cmake -S . -B build && cmake --build build` needs a C++20 compiler, CMake and
Eigen; libtorch (found through `pip install torch`) adds GPU, batching, gradients and the Python
package, and `-DCORACPP_TORCH=OFF` builds Eigen only. In VS Code, open an example and press `F5`.

## Usage

- **Sets:** `Zonotope(c, G)`, `Interval(inf, sup)`; any set works wherever a set is expected.
  Operators as in CORA: `A * Z` (linear map), `2 * Z`, `Z + Z2` (Minkowski sum), `Z + v`, `-Z`;
  `std::cout << Z` / `print(Z)` shows a set (dimension, center, generators).
- **Tensors:** `Tensor::eye(n)`, `Tensor::ones(shape)`, `t.cos()` in C++; `Tensor.eye(n)`,
  `Tensor.ones(2, 3)`, `Tensor.cos(x)` (also `sin`, `tan`, `exp`, `log`, `sqrt`) in Python,
  without `math` or numpy.
- **Batches:** a leading dimension in `c`, `G` or `A` is a batch; the call is the same as for one set.
- **Backends:** `setBackend("torch" | "torch:cuda" | "eigen")` or `CORACPP_BACKEND`. In Python, torch
  tensors run on libtorch, numpy arrays on Eigen. `device="gpu"` places a tensor on the GPU.
- **Learning:** `A` can be a parameter of a torch `nn.Module` (Python): see
  `example_linear_learn_01_dynamics`, which learns a system matrix from measurements.
- **Gradients:** autograd works through `reach` and `simulate` on libtorch;
  `setBackend("torch,customBackward")` uses a hand-written backward pass for the matrix exponential.
- **Nonlinear systems:** write the dynamics on symbolic states, `NonlinearSys(f, n)` derives it,
  and `reach` linearizes step by step (CORA's `lin`); see `example_nonlinear_reach_01_vanDerPol`.
  The remainder is of order 2, so it needs small steps and initial sets (no inputs yet). `reach`
  and `simulate` run on tensors, so on libtorch gradients flow to the initial set.
- **Conformal prediction:** `example_linear_learn_02_conformal` (Python) learns a linear model of a
  nonlinear system; `example_linear_conformal_01_pendulum` (C++) uses its linearization. Both
  enlarge the model's reachable set by a conformal radius, so that new trajectories stay inside
  with the promised probability (on average over calibration sets).
- **Neural networks:** `NeuralNetwork({{W1, b1}, {W2, b2}})` (Python: a list of `(W, b)`) has
  ReLU layers; `nn.evaluate(x)` maps points and `nn.evaluate(X)` a zonotope to a zonotope that
  contains all outputs (affine layers exactly, ReLUs by the tightest one-slope parallelogram). It
  is differentiable and batched. See `example_neuralNetwork_verify_01`, which certifies a classifier,
  and `example_neuralNetwork_train_01_setBased`, which trains with the set-based loss of Koller,
  Ladner and Althoff (loss of the output center plus the size of the output set) through autograd.
- **Simulation:** `sys.simulate(x0, timeStep, tFinal)`, `sys.simulateRandom(X0, n, ...)`.
- **Plotting:** `plot(S, dims, options)` projects a set (or a `Reach`, `Specification`, simulation)
  onto two dimensions and draws it; reachable sets are drawn as their union. C++ writes SVG
  (`plot(R, {0, 1}, {.label = "reachable set"}); figure().save("reach.svg");`), Python uses
  matplotlib (`plot(R)`). Sets are filled, each in the next color of CORA's order (blue, red,
  yellow, ...); `useCORAcolors("CORA:contDynamics")` draws a set as an initial set and a reachable
  set in blue, as CORA does for reachability. Options: `{.color = "CORA:red"}`, `.filled = false`
  (Python `filled=False`), `.timePoints`, `.step`. All of this is decided in C++; Python only draws.
- **Options:** `linAlg` is `"standard"` or `"wrapping-free"`; an unknown value throws.

Differences from MATLAB: points are columns `(n, 1)` in C++ (1-D in Python); `R.timeInt[k]`
covers `[k*timeStep, (k+1)*timeStep]` and `R.timePoint[k]` is at `k*timeStep` (0-based);
sets are objects with methods (`Z.mtimes(M)`), not operators.

## Examples and tests

`examples/cpp/` and `examples/python/`: one file per topic, sectioned like CORA's
`example_linear_reach_01_5dim`. `scripts/test.sh` (ctest) runs the conventions, the tests and the C++ examples (`-L test`, `-L example`).

## CORA-COMP

The competition entry is in [`competition/`](competition/README.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) to work on the library.
