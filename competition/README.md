# CORA.cpp in CORA-COMP

The CORA-COMP entry of CORA.cpp (the library itself is in the [main README](../README.md)): the
[benchmark catalog](https://github.com/CORA-COMP/benchmarks)'s set representations in two
backends — [Eigen](https://eigen.tuxfamily.org) matrices for speed on the CPU, libtorch
tensors for the GPU and for gradients. Plain `double` arithmetic, no rounding control: the
counterpart to [CORALean-Cpp](https://github.com/CORA-COMP/CORALean-Cpp), which computes
the same operations with outward rounding, and a sibling of
[CORA.py](https://github.com/CORA-COMP/CORA.py),
[CORA.rust](https://github.com/CORA-COMP/CORA.rust) and
[CORA.jax](https://github.com/CORA-COMP/CORA.jax).

Submit it like any other tool: this repository and a commit, with the script directory set
to `competition`, plus a Debian- or Ubuntu-based image. `install_tool.sh`
installs `g++`, Eigen and GLPK if the image lacks them, as root or through the node's
passwordless `sudo`, and builds the libtorch backend in whenever it finds libtorch.

**Base image.** `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` gets both backends and
the GPU; the install reuses that image's own libtorch, headers and all, rather than
downloading one. A plain `ubuntu:24.04` installs too — it just builds the Eigen backend
alone, and then `gpu` instances report `unsupported`.

## What it runs

| | |
| --- | --- |
| Benchmarks | `test`, `interval`, `zonotope`, and both `-batched` twins |
| Operations | `startup`, `generateRandom`, `randPoint`, `supportFunc`, `matMul`, `minkSum`, `contains` |
| Devices | `cpu` (Eigen), `gpu` (libtorch) |
| Batched | yes |

Built without libtorch, or run on a worker with no CUDA device, the `gpu` instances report
`unsupported` — what the catalog asks of a library that cannot run them.

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

## Two backends

| | Eigen | libtorch |
| --- | --- | --- |
| Runs | `cpu` instances, when pinned | `cpu` and `gpu` instances |
| Scalar | `double`, or any differentiable Eigen scalar | `double`, `float` on request |
| Batch | one matrix per batch, OpenMP across sets | leading tensor axes, one call per batch |
| Gradients | at compile time, by substituting the scalar | at run time, through autograd |

`CORACPP_BACKEND` pins a run to one of them, so the same commit enters the catalog twice
and the two sets of results sit side by side:

| Tool environment | CPU | GPU |
| --- | --- | --- |
| unset (default) | libtorch (Eigen if built without libtorch) | libtorch |
| `CORACPP_BACKEND=eigen` | Eigen | `unsupported` |
| `CORACPP_BACKEND=torch` | libtorch | libtorch |

The `backend` column of every result says which one actually ran.

**Which is faster on the CPU depends on the instance**, so `CORACPP_BACKEND=eigen` enters
the Eigen backend as a second tool next to the default. libtorch costs a fixed ~0.3–1.5 ms of dispatch per
operation, so Eigen wins everything small — up to 130× on `supportFunc` at 10d. Above
about 100 dimensions libtorch's kernels pull ahead on `matMul` and `minkSum` (1.3–3.5×).
Batched `contains` is the other way round: Eigen gives each set of the batch a thread,
which suits a problem that is per-point branching more than arithmetic, and it stays an
order of magnitude ahead on the batched instances — 0.11 s against 2.4 s at 50d over a
hundred sets — besides finishing two that libtorch cannot finish inside the catalog's
60 s. On the GPU, where Eigen cannot go at all, libtorch finishes every one of them.

**Eigen layout.** A set holds its whole batch in one matrix: an interval is `lo`/`hi` of
size `n × B`, one set per column, a zonotope a centre `n × B` and the batch's generator
blocks side by side in one `n × (m·B)` matrix. Point clouds are `n × (points·B)`. `matMul`
is one product for the whole batch, since `M` maps every generator alike.

**Tensor layout.** Leading batch dimensions that may be empty: `lo`/`hi` and `c` are
`(…, n)`, `G` is `(…, n, m)`, point clouds are `(…, N, n)`. Every operation touches only
the trailing dimensions, so a batch is one call and the GPU gets work worth its latency.

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
- Threads matched to the work rather than to the machine. OpenMP and Eigen both default
  to every hyperthread — 160 on the competition's two-socket worker, against 80 cores —
  and at that width a region over ten sets, or a 40 MFLOP product, spends an order of
  magnitude longer gathering threads than computing. The pool is set to the physical
  cores once; a region either runs serially or gets the whole pool, and only a product,
  which Eigen splits itself, is narrowed further ([`../src/global/threads.h`](../src/global/threads.h)). An
  intermediate team is the worst of both, since OpenMP parks and wakes the difference
  between one region and the next. On the worker that is 33 s against 0.13 s on a
  mid-sized batched `matMul`, and 1.25 s against 0.011 s on a batched `randPoint`.

**Automatic differentiation**, two ways.

- *Through libtorch*, at run time: the operations are ordinary tensor expressions, so
  autograd differentiates them with respect to a set whose `c`, `G`, `lo` or `hi` requires
  a gradient — on either device, with no second implementation. This is the path the tests
  check, against the analytic gradient of the support function. Measurement runs under
  `NoGradGuard`, so the catalog never times a graph being recorded.
- *Through the Eigen templates*, at compile time: the catalog's `Interval`, `Zonotope`, `support_func`,
  `mat_mul`, `mink_sum` and `origin` ([`sets/sets.h`](sets/sets.h)) are templates on the scalar type, so the same source
  instantiates for `Eigen::AutoDiffScalar`, [autodiff](https://autodiff.github.io)'s
  `dual`, CoDiPack and the like; overload `cora::value_of` for the new scalar and the
  non-differentiable parts keep working. `generateRandom`, `randPoint` and `contains` are
  `double` only — drawing numbers and deciding a verdict are not operations a gradient
  passes through.

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

The libtorch backend runs the same two paths on tensors, the whole batch at once, with the
LP on the host for whatever the certificates leave open. Three things about it are chosen
by what the hardware is good at rather than by the mathematics, and each is worth between
three- and tenfold on the instances it touches:

- The points are the columns of one matrix per set. Held the other way up, the set would
  broadcast over the points in every product, and torch materializes that — at 1000d an
  iteration read a 1.6 GB expansion of a 160 MB tensor.
- The projector comes from a QR of `Gᵀ` on the host and from a Cholesky of `G Gᵀ` on the
  device. Both describe the same subspace; cuSOLVER is far from peak on a batched QR that
  large, while MKL is not, so either choice is three- to sixfold wrong on the other
  device.
- Where `B·m²` fits in 128 MB the projector is held as a matrix, which makes an iteration
  one fused product instead of three calls. At small `m` these are tiny tensors over a
  hundred repetitions, where an iteration costs almost only its dispatches.

`make test-competition` (the tests in `tests/competition/`) checks the operations against their definitions and both containment paths
against the LP, on points placed just inside and just outside the boundary, plus the
moments of the random numbers. With libtorch built in it also runs the *same* sets and the
*same* points through both backends and compares the answers, on the CPU and on the GPU:
each backend agreeing with its own LP would not catch the two reading the catalog
differently.

## What is measured

The harness times `run_instance.sh`. A fresh process pays for its thread pool, its first
allocations and its page faults, so `prepare_instance.sh` (untimed) starts a daemon once
([`server.cpp`](server.cpp)) that runs every operation once. `run_instance.sh` then
only sends `params` to the daemon over a localhost socket (bash's `/dev/tcp`, no extra
process) and waits for the verdict. The daemon runs the whole instance: generate the
inputs, then repeat the operation.

A CUDA call only queues work, so the timer stops after a synchronization — what is measured
is work done, not work queued.

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
| `backend` | `eigen` or `torch` |


## Layout

Everything here is what the competition runs. The library it uses is in [`../src`](../src).

| File | |
| --- | --- |
| `install_tool.sh`, `prepare_instance.sh`, `run_instance.sh`, `coracpp_lib.sh` | the scripts the harness calls |
| `main.cpp`, `server.cpp` | the `serve` / `run` / `env` / `check` commands, and the daemon |
| `instance.cpp`, `catalog.cpp`, `json.cpp` | one instance: inputs, the repeated operation, the verdict; the catalog's names and `params` |
| `backend.h`, `torch_backend.cpp`, `torch_none.cpp` | what the two catalog backends have in common; the libtorch runner and its stand-in when libtorch is absent |
| `sets/` | the catalog's Eigen and libtorch sets (tuned for speed on one library each), their exact zonotope containment, and its LP |

## Configuration

All optional:

| Variable | Default | |
| --- | --- | --- |
| `CORACPP_BACKEND` | libtorch on both devices | `eigen` or `torch` pins the whole run to one backend (the library reads it too, see the main README) |
| `CORACPP_DTYPE` | `float64` | `float32` for what a GPU can do instead (libtorch only) |
| `CORACPP_PORT` | `47916` | the daemon's localhost port |
| `CORACPP_SERVER_DIR` | `~/.coracpp_server` | the daemon's pid file and logs |
| `CORACPP_START_TIMEOUT` | `120` | seconds `prepare_instance.sh` waits for the daemon |
| `CORACPP_BIN` | `build/coracpp` | the built binary |
| `EIGEN` | `/usr/include/eigen3` | where Eigen's headers are |
| `LIBTORCH` | the image's Python torch | where libtorch is, when it is not found by itself |
| `CORACPP_PYTHON` | `python3`, then `python` | the interpreter whose torch to build against |
| `OPT` | `-O3 -march=native -std=c++17 -fopenmp …` | the compiler flags of the build (`make OPT=…`) |
| `OMP_NUM_THREADS` | the machine's physical cores | how many threads the operations use; set it and the tool leaves it alone |

## Running one instance locally

With the tool built and this repository as the working directory:

```bash
P='{"set": "zonotope", "operation": "contains", "dim": 100, "generators": 200, "device": "cpu", "repetition": 100, "points": 10}'
competition/prepare_instance.sh v1 zonotope contains-100d-cpu "$P"
competition/run_instance.sh     v1 zonotope contains-100d-cpu "$P" /tmp/result.csv
cat /tmp/result.csv
```

`build/coracpp run "$P" /tmp/result.csv` runs it without the daemon, `build/coracpp env`
prints what the worker runs on, and `build/coracpp check` runs every operation once on
every device it has, gradients included.

