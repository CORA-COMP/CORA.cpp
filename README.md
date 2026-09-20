# CORA.cpp — CORA-COMP submission

A C++ entry for [CORA-COMP](https://github.com/CORA-COMP/cora-eval-platform): the
[benchmark catalog](https://github.com/CORA-COMP/benchmarks)'s set representations as
[Eigen](https://eigen.tuxfamily.org) matrices, written for speed on the CPU. Plain `double`
arithmetic, no rounding control — the counterpart to
[CORALean-Cpp](https://github.com/CORA-COMP/CORALean-Cpp), which computes the same
operations with outward rounding, and to [CORA.py](https://github.com/CORA-COMP/CORA.py)
and [CORA.rust](https://github.com/CORA-COMP/CORA.rust), which reach the GPU through
libtorch.

Submit it like any other tool: this repository and a commit, plus a Debian- or
Ubuntu-based image, e.g. `ubuntu:24.04`. `install_tool.sh` installs `g++`, Eigen and GLPK
if the image lacks them, as root or through the node's passwordless `sudo`.

## What it runs

| | |
| --- | --- |
| Benchmarks | `test`, `interval`, `zonotope`, and both `-batched` twins |
| Operations | `startup`, `generateRandom`, `randPoint`, `supportFunc`, `matMul`, `minkSum`, `contains` |
| Devices | `cpu` |
| Batched | yes, one matrix per batch |

The `gpu` instances report `unsupported`, which is what the catalog asks of a library
with no GPU support — the point of this entry is what a native CPU implementation does.

| Operation | Inputs (generated in the run) | Repeated |
| --- | --- | --- |
| `startup` | — | the unit zonotope `(zeros(n), eye(n))` |
| `generateRandom` | — | a random set |
| `randPoint` | a random set | `points` points, `c + G·β` with `β ~ U[-1, 1]^m` (uniform in the box for an interval) |
| `supportFunc` | a random set, a random unit direction | `cᵀd + ‖Gᵀd‖₁` (`cᵀd + rᵀ\|d\|` for an interval) |
| `matMul` | `M = randn(n)`, a random set | `M·S`; the interval hull `Mc ± \|M\|r` for an interval |
| `minkSum` | two random sets | `S1 + S2` |
| `contains` | a random set, `points` points drawn from it | point containment, all points at once |

Random sets follow CORA's `generateRandom`, as the catalog specifies.

**Layout.** A set holds its whole batch in one matrix: an interval is `lo`/`hi` of size
`n × B`, one set per column, a zonotope a centre `n × B` and the batch's generator blocks
side by side in one `n × (m·B)` matrix. Point clouds are `n × (points·B)`. `matMul` is one
product for the whole batch, since `M` maps every generator alike.

**Where the speed comes from.**

- Eigen's vectorized `double` kernels at `-O3 -march=native`, built on the worker, so the
  products use whatever SIMD that machine has.
- OpenMP across the batch, and inside the elementwise and reduction passes once they are
  large enough to pay for the fan-out. Eigen leaves its own threading off inside a
  parallel region, so the two never nest.
- A xoshiro256++ stream with Box–Muller for the normals, in the one translation unit
  compiled with `-ffast-math` so the logarithm and the sine vectorize through libmvec.
  `generateRandom` is bound by how fast numbers can be produced, not by any product.
- A warm daemon, so no measurement pays for process start.

**Automatic differentiation.** The set representations and their operations are templates
on the scalar type, so the same source instantiates for `double` and for a differentiable
scalar — `Eigen::AutoDiffScalar`, [autodiff](https://autodiff.github.io)'s `dual`, CoDiPack
and the like all work as Eigen scalars. Overload `cora::value_of` for the new scalar and
the non-differentiable parts (the containment LP, the verdicts) keep working too. The
catalog measures `double`, which is the only instantiation that reaches the vectorized
paths.

**Zonotope containment** is an LP in CORA. Here it is exact without one for all but a
handful of points:

- *Few facets* (`C(m, n−1) ≤ 300`, so up to 5d in the catalog): the zonotope is the
  intersection of the slabs `|hᵀ(x − c)| ≤ Σⱼ |hᵀgⱼ|` over the normals `h` of every `n − 1`
  generators.
- *Otherwise*: alternating projections between the affine set `{β | Gβ = p − c}` (via one
  thin QR of `Gᵀ`) and the box `[-0.9, 0.9]^m`. Every answer is certified: a `β` in the
  affine set with `‖β‖∞ ≤ 1` proves containment, a `d` with `dᵀ(p − c) > ‖Gᵀd‖₁` proves the
  opposite. The shrunk box makes interior points converge in about 15 iterations; a point
  neither proves in 60 goes to an exact LP (GLPK).

`make test` checks the operations against their definitions and both containment paths
against the LP, on points placed just inside and just outside the boundary, plus the
moments of the random numbers.

## What is measured

The harness times `run_instance.sh`. A fresh process pays for its thread pool, its first
allocations and its page faults, so `prepare_instance.sh` (untimed) starts a daemon once
([`src/server.cpp`](src/server.cpp)) that runs every operation once. `run_instance.sh`
then only sends `params` to the daemon over a localhost socket (bash's `/dev/tcp`, no
extra process) and waits for the verdict. The daemon runs the whole instance: generate the
inputs, then repeat the operation.

A daemon that does not answer — dead, or still busy with an instance the harness timed
out — is replaced by the next `prepare_instance.sh`. Without a daemon, `run_instance.sh`
runs the instance in a fresh process.

The result file also carries the daemon's own numbers, kept per instance next to the
harness wall-clock:

| Column | |
| --- | --- |
| `time_generate` | generating the inputs |
| `time_operation` | the operation, repeated `repetition` times |
| `dtype` | the element type the numbers were produced in |

## Layout

| File | |
| --- | --- |
| `install_tool.sh` | installs the compiler, Eigen and GLPK, builds, then prints what the worker will run on |
| `prepare_instance.sh` | starts or replaces the daemon |
| `run_instance.sh` | the timed script: sends the instance to the daemon |
| `coracpp_lib.sh` | the shell client of the daemon |
| `src/sets.h` | the library: `Interval`, `Zonotope` and their operations, over any scalar |
| `src/contains.cpp` | exact zonotope containment: facets, projections, LP |
| `src/lp.cpp` | the containment LP, on GLPK |
| `src/rng.cpp` | the random numbers |
| `src/instance.cpp` | one instance: inputs, the repeated operation, the verdict |
| `src/server.cpp` | the daemon |
| `src/main.cpp` | the `serve` / `run` / `env` / `check` commands |

## Configuration

All optional:

| Variable | Default | |
| --- | --- | --- |
| `CORACPP_PORT` | `47916` | the daemon's localhost port |
| `CORACPP_SERVER_DIR` | `~/.coracpp_server` | the daemon's pid file and logs |
| `CORACPP_START_TIMEOUT` | `120` | seconds `prepare_instance.sh` waits for the daemon |
| `CORACPP_BIN` | `build/coracpp` | the built binary |
| `EIGEN` | `/usr/include/eigen3` | where Eigen's headers are |
| `CXXFLAGS` | `-O3 -march=native -std=c++17 -fopenmp …` | the build flags |
| `OMP_NUM_THREADS` | all cores | how many threads the operations use |

## Running one instance locally

With the tool built and this repository as the working directory:

```bash
P='{"set": "zonotope", "operation": "contains", "dim": 100, "generators": 200, "device": "cpu", "repetition": 100, "points": 10}'
./prepare_instance.sh v1 zonotope contains-100d-cpu "$P"
./run_instance.sh     v1 zonotope contains-100d-cpu "$P" /tmp/result.csv
cat /tmp/result.csv
```

`build/coracpp run "$P" /tmp/result.csv` runs it without the daemon, `build/coracpp env`
prints what the worker runs on, and `build/coracpp check` runs every operation once.
