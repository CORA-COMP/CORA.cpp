# Neural-network verification (VNN-COMP)

CORA.cpp verifies VNN-COMP instances: an ONNX network, a VNN-LIB property and a timeout. The answer is
`sat` with a counterexample, `unsat`, or `unknown`. The work is tracked in issue #213, phase by phase.
This page describes the design and what Phase 0 implements.

## Pipeline

```
onnx file ──> cora.onnx_network.OnnxNetwork ──> batched torch run (any device, float32 search)
                         └── dense_layers() ──> cora.NeuralNetwork (float64) ──> zonotope bounds
vnnlib file ─> cora.vnnlib.Property: cases = input box + union of polytopes {y : A y <= b}
                         └── NnProblem (loaded once) ──> verify(problem, timeout)
                                   ├── falsification (cora.nn_attack, by name)
                                   ├── branch and bound: bounding method + split heuristic (by name)
                                   └── every candidate counterexample ──> cora.counterexample check
```

- **Network.** `OnnxNetwork` runs the graph node by node on torch, one sample of the declared shape at
  a time and the rows by `torch.func.vmap`, so ONNX shapes keep their meaning (a `Reshape` to a
  constant `[1, 18]`, `Shape`/`Gather` subgraphs).
  - **Operators:** Add, Sub, Mul, Div, Neg, MatMul, Gemm, Conv (1-3 D, any pads), BatchNormalization,
    Relu, Sigmoid, Tanh, LeakyRelu, Softmax, MaxPool, AveragePool, GlobalAveragePool, Pad, Flatten,
    Reshape, Transpose, Squeeze, Unsqueeze, Concat, Identity, Dropout, Cast, Shape, Gather, Slice and
    Constant. That covers the starting benchmarks.
  - **Lowering:** a graph that is a chain of affine parts and relu/sigmoid/tanh lowers to
    `cora.NeuralNetwork`, one dense layer per affine part. The layer's matrix is the image of the unit
    vectors with the additive constants dropped, so a Gemm keeps its weights bit for bit. Its bias is
    the image of zero.
  - **Not lowered:** a residual connection, a max pool, or a layer beyond `max_elements` (the collins
    window-40 and soundnessbench convolutions) stays torch-only. Such a graph gets falsification only,
    until Phase 3 brings conv layers on the generator axis.
- **Property.** VNN-LIB 1.0 is read with the semantics of the scoring's parser. VNN-LIB 2.0 is read by
  the reference parser (`vnnlib`), every assertion in DNF and the conjunction multiplied out. The
  package's own `transform_to_compat` drops the input box of a conjunction of two disjunctions (ACAS
  Xu property 6), so it is not used. On every official file of the starting benchmarks the 1.0 and 2.0
  versions read to the same arrays, bit for bit.
- **Problem.** `NnProblem.load` reads the network and the property once. They are reused across
  restarts, rounds and branch and bound.
  - **Attack boxes:** the attacks search the float32 box inside each case's box: the smallest float32
    at or above `lo`, the largest at or below `hi`. A point there passes the scoring's input check
    under either dtype rule.
  - **Bounding boxes:** the bounds cover the outer float64 box, which also holds `lo` and `hi`
    rounded to the nearest float32. The scoring compares a float32 input with the bound rounded to
    float32, so a competitor's counterexample may lie there.

## Strategies, chosen by name

The option names follow CORA's `options.nn` (`nn/+nnHelper/validateNNoptions.m`, `nn/@neuralNetwork/verify.m`).
Each strategy is a small interface with a registry, so a new one is a class or a function and a decorator.

| Option (`VerifyOptions`) | Interface | Registry | Phase 0 | Later |
| --- | --- | --- | --- | --- |
| `falsification_method` | `nn_attack.Attack.run(problem, deadline) -> AttackResult` | `@register_attack` | `pgd`, `fgsm`, `center` | `zonotack` (stub, P1), BaB-attack |
| `falsification_order` | `verify` | — | `attack-first`, `bound-first` | interleaved |
| `bounding_method` | `nn_verify.Bounding.prepare(problem)`, `.lower(lo, hi, case)` | `@register_bounding` | `zonotope` (`cora.NeuralNetwork`, DeepZ) | `conzonotope` (stub, P1), backward |
| `input_split_heuristic` | `f(problem, lo, hi, case) -> dimension per box` | `@register_input_split_heuristic` | `most-sensitive-input-radius`, `largest-radius` | enclosure gradient (P1) |
| `neuron_split_heuristic` | — | `NEURON_SPLIT_HEURISTICS` | `none` | P1 |

Further options:
- `num_splits` and `num_dimensions` (the pieces and dimensions of an input split);
- `verify_dequeue_type` (`front`: breadth first; `back`: depth first);
- `max_verif_iter`;
- `attack_time_share` (the timeout fraction PGD gets in attack-first);
- `attack_options` (per method: PGD's `steps`, `restarts`, `step_size`, `decay`, `decay_at`, `seed`);
- `verified_margin` (see below);
- `memory_budget`, `device` and `dtype`.

**PGD** is CORA's `computePGDAttack` (iterated FGSM with step decay), batched:
- every box gets `restarts` samples per round: the center first, the rest uniform;
- the samples take sign steps of `step_size` times the box radius on the property's loss
  `min over polytopes of max_k (A_k y - b_k)`, which is negative inside the unsafe set;
- each step is projected back into the box;
- rounds repeat until the deadline;
- every point with a negative loss is replayed by onnxruntime and the scoring's check, the most
  negative first. Only a `CORRECT` one ends the search.

**Branch and bound** in Phase 0 is CORA's naive refinement:
- the zonotope of each box in a batch gives lower bounds of every row of `A y - b`;
- a polytope is excluded when some row's bound exceeds `verified_margin`, and a box is verified when
  every polytope of its case is excluded;
- the open boxes' centers are tried as counterexamples, and the boxes are split by the heuristic;
- `unsat` when no box is left.

`verified_margin` (1e-4) stands in for the float64 re-check of Phase 2. The enclosure is of the real
network, but the network that onnxruntime runs is float32. On cora_2024 the two differ by up to 4e-4.

## Speed and memory

- **Batching:** batches follow `memory_budget` (bytes, default 8 GiB, sized like a GPU). An attack batch
  holds `memory_budget / (4 · activation_elements · itemsize)` samples. A bounding batch holds
  `memory_budget / box_bytes`, where a box costs about three copies of its generator matrix.
- **Generators:** the zonotope gets generators only for the dimensions that some box of the batch
  perturbs.
- **Gradients:** every evaluation outside a gradient step runs under `torch.no_grad`.
- **Measurement:** the runner records the wall time and the peak RSS of every `run_instance.sh`
  (`wait4`), torch's peak on a GPU, and the seconds of the attack and of the branch and bound. These
  are the baseline for later phases.

## Counterexamples and their check

`cora.counterexample.check_counterexample` replays a counterexample as the VNN-COMP 2026 scoring does
(`vnncomp2026_results/SCORING`, files `counterexamples.py`, `counterexamples_v2.py`, `vnnlib_v1.py`,
`settings.py`). Its rules:

- **Versions:** onnxruntime 1.26.0, onnx 1.21.0 and vnnlib 1.0.2 (`scripts/requirements-vnncomp.txt`),
  on the CPU execution provider. The 1.0 session has 12 intra and 12 inter threads; the 2.0 session has
  the default threads. Neither uses the CPU memory arena.
- **Outputs:** the written outputs are ignored. The input is parsed as float64 and cast to the ONNX input
  type (float32), and onnxruntime computes the output from it.
- **Input check:** `CORRECT` when the input lies in the box with tolerance 0, and `CORRECT_UP_TO_TOLERANCE`
  within `COUNTEREXAMPLE_ATOL = 1e-4`. The scoring credits the latter but does not let it establish `sat`.
- **Output check:** tolerance 0 (`output_tol = 0.0`; `IGNORE_CE_Y`).
- **Dtypes:** NumPy 2 compares a float32 scalar with a Python float in float32. In VNN-LIB 1.0 the input
  bounds are therefore compared in float32 and the rows `A·y` in float64. In 2.0 every comparison is
  in the variables' float32.
- **When a sat counts:** our `sat` is written only for a `CORRECT` check.

`write_counterexample` writes VNN-LIB 1.0 (`((X_i v) … (Y_j v))`) or 2.0 (a `X float32 [1, 5]` header
per variable, one value per line). Every value is the shortest decimal that parses back to the same float32.

## Instances

**Rule:** all development and tuning use our own dev sets. The official 2026 instances and their score
come only at the very end.

- **Generation:** `scripts/vnncomp/make_dev_set.py <benchmark> --seed S --out DIR` runs the benchmark's
  own `generate_properties.py`, from the source commit that the 2026 benchmark repository names, with
  our seed. It writes VNN-COMP's layout in 1.0 and in 2.0 (`to_vnnlib2`), a `manifest.json` (seed,
  source, method, the number of instances identical to official ones) and, where the generator knows
  the verdicts (sat_relu), an `expected.csv`.
- **Where a generator does not help:**
  - acasxu's generator is deterministic, so its instances get random sub-boxes of the official
    properties;
  - dist_shift's generator is not public. Its structure is reconstructed from the official files: an
    MNIST test image, the prior's latent mean ± 0.13 standard deviations, unsafe if the runner-up
    class reaches the top class.
- **Generator environments:** `--python` picks the Python of a generator, and the manifest records it.
  sat_relu needs a torch whose ONNX exporter writes the weights inline (torch 2.4; 2.15 writes external
  data files). safenlp needs NumPy < 1.25 (Python 3.10), because it compares an array with `[]`.
- **Data:** networks and instances stay outside the repository (e.g. `~/vnncomp/dev_sets`).

## Running

- **Harness:** `scripts/vnncomp/prepare_instance.sh` and `run_instance.sh` are VNN-COMP's tool interface.
  `run_instance.py` writes the result file and `RESULT.stats.json`.
- **Runner:** `scripts/vnncomp/run_dev_set.py DIR --out results.csv` runs a dev set as
  `run_single_instance.sh` does. Its CSV has the scoring's six columns, then ours.
- **Scoring:** `scripts/vnncomp/score.py results.csv --expected DIR/expected.csv` scores the CSV with the
  2026 rules for one tool:
  - +10 for a correct `holds` or `violated`;
  - −150 for an incorrect one, or for a missing or failing counterexample;
  - solves over the timeout count as timeouts;
  - no normalization to a best tool.

## The oracle

All set propagation in Phase 0 goes through `cora.NeuralNetwork`:
- its layer types (linear, relu, sigmoid, tanh) are checked bit for bit against CORALean's
  `nnLayer.evaluateZonotope`, through `cora.lean.evaluate_network`;
- the network of every ONNX operator that lowers, and small end-to-end runs (one per verdict), are
  replayed through the oracle (`tests/python/test_lean_nn_verify.py`).

Oracle gaps: CORALean has no op for a convolution on the generator axis, a max pool, a batched layout
or an interval center. Phase 0 avoids them by lowering, and Phase 1 and 3 request them
(`docs/oracle-requests.md`).
