# Neural-network verification (VNN-COMP)

CORA.cpp verifies VNN-COMP instances: an ONNX network, a VNN-LIB property and a timeout. The answer is
`sat` with a counterexample, `unsat`, or `unknown`. The work is tracked in issue #213, phase by phase.
The verifier runs in C++ (`NeuralNetwork::verify`, `include/cora/nn/neural_network/verify.h`); Python
is the front end (ONNX and VNN-LIB into the C++ objects) and the tooling (dev sets, run scripts,
scoring, the scoring-exact counterexample check).

## The verifier

```
NeuralNetwork (dense layers) + NnProperty (boxes, unsafe polytopes A y <= b) + confirm check
        └── NeuralNetwork::verify(property, timeout, NnVerifyOptions, confirm)
                ├── NnProblem: the search net (float32) and the bound net (float64), the boxes
                ├── Attack (by name): pgd, fgsm, center, zonotack
                └── Bounding (by name): zonotope (DeepZ + input splits), conzonotope (CORA's verify)
```

- **Order:** `falsification_order` `attack-first` runs the attack for `attack_time_share` of the
  timeout (at most `attack_first_rounds` rounds), then the branch and bound, then the attack until the
  deadline; `bound-first` skips the first attack.
- **Boxes:** the search runs in `dtype` (float32) inside the float32 box of each case (the smallest
  float32 at or above `lo`, the largest at or below `hi`); the bounds cover the float64 box that also
  holds `lo` and `hi` rounded to the nearest float32, since the scoring compares a float32 input with
  the bound rounded to float32.
- **Counterexamples:** every candidate with a loss at most 0 goes through the `confirm` callback, the
  least loss first, at most `max_candidates` per step. From Python it is the scoring's check
  (onnxruntime, then VNN-COMP's checker of the written counterexample); in C++ without one, the
  float64 output of the network. Only a confirmed point is `sat`.
- **Backends:** everything is written on `Tensor`, so the verifier runs on every backend: lean (the
  oracle's arithmetic, float64 only: the oracle tests), eigen (the CPU) and torch (batched, any
  `device`). The gradients are written out (CORA's backprop), not taken from autograd: library code
  does not ask which backend it runs on, and the same code then runs on lean and eigen too.

## The front end (Python)

```
onnx file ──> cora.onnx_network.OnnxNetwork ──> layers() ──> cora.NeuralNetwork (eigen or torch)
vnnlib file ─> cora.vnnlib.Property: cases = input box + union of polytopes {y : A y <= b}
                    └── cora.nn_verify.load ──> verify(instance, timeout, cora.vnncomp_options(...))
                              └── NeuralNetwork.verify (C++), confirm = the scoring's check
```

- **Network.** `OnnxNetwork` runs the graph node by node on torch (one sample of the declared shape
  at a time, the rows by `torch.func.vmap`), so ONNX shapes keep their meaning. A graph that is a
  chain of affine parts and relu/sigmoid/tanh lowers to dense layers: the layer's matrix is the image
  of the unit vectors with the additive constants dropped (a Gemm keeps its weights bit for bit), its
  bias the image of zero. A graph with a Conv, ConvTranspose or AveragePool lowers layer by layer
  instead, and no convolution becomes a matrix: Conv is a C++ `Layer::conv2d` (groups, strides,
  dilations, asymmetric pads; Conv1D as width 1), ConvTranspose its transpose
  (`Layer::conv_transpose2d`), BatchNormalization and AveragePool depthwise convolutions, the data
  moves (Transpose, Pad, Slice, Gather, Concat with constants, nearest Upsample) a `Layer::gather`
  found by running the node on the indices, Flatten and Reshape nothing, Gemm and MatMul dense
  layers, and an added constant the bias of the layer before (a BatchNormalization after a Conv or a
  Gemm folds into it, one of features elsewhere is a diagonal dense layer). A Slice or Gather of a
  value of the input is such a data move, not shape arithmetic (linearizenn slices its input).
  Two branches joined by an Add (a ResNet, soundnessbench's carriers) become a graph: each value is
  a pending sum of maps of earlier outputs, emitted as layers when an activation or the output
  needs it, the longest term as a chain and the others as residuals of its last layer (`source`,
  `residuals` of a `Layer`). A Concat of branches is their sum, each term padded into its place by
  a gather (linearizenn's controller beside its linear part). A MaxPool without pads lowers to `max(a, b) = b + relu(a - b)` pairwise
  over its window (a grouped convolution of the differences, a ReLU, a gather plus a residual
  convolution of the second items, per round). What does not lower, or what C++ refuses, is
  `unknown` at once with the reason, before the property is read.
- **Products (Phase 4).** A product of two values (lsnc_relu's Lyapunov terms `x P x`) lowers to
  a layer of `Activation::product`: a dense map to `[a; b]`, the product `[a o b; 0]`, a dense map
  that takes the first half (at most 4096 entries). A difference of branches takes `-I`, a
  constant minus a value and a scaling by constants (Mul, Div, Neg, Cast) fold into the map before
  or become a diagonal layer, and ReduceSum is a dense layer of ones. A graph without a
  convolution that is no chain (a product, a join) lowers the same way. verify encloses the
  product on the zonotope itself, not on the bounds (`priv_product_line`): with `a = c_a + x`,
  `b = c_b + y`, `x = G_a beta`, `y = G_b beta` (the interval center's radii as factors of their
  own), `a b = c_a c_b + c_b x + c_a y + x y`, the linear part exact (slopes `[c_b; c_a]`) and the
  rest `x y = ((lambda x + y / lambda)^2 - (lambda x - y / lambda)^2) / 4` in
  `[-D^2 / 4, S^2 / 4]` with `S, D = |lambda G_a +- G_b / lambda|_1`, intersected over `lambda = 1`
  (the squares of `(a +- b) / 2`, exact for `a = b`) and `lambda = sqrt(|G_b|_1 / |G_a|_1)` (at
  most `|G_a|_1 |G_b|_1`, the bilinear bound), widened by the rounding. When one factor varies far
  less than the other (a softmax's weights times the values, `x` times a constant-like `P x`) the
  rest is that much smaller than the squares': on lsnc_relu's first box the products' errors drop
  from 17 and 82 to 6.1 and 5.2. The interval gradient takes `a_m b_m +- (|a_m| r_b + |b_m| r_a +
  r_a r_b)`, the backward bound the line's two parallels through the transposed slopes, the
  gradients `[b; a]`; the set operations outside verify refuse the layer. `Activation::square`
  (slope `l + u`, radius `(u - l)^2 / 8` widened by the rounding) stays for `x^2`. A negative slope
  (x^2 left of 0, the reciprocal) maps the interval center's radius by its absolute value.
- **Sign (Phase 4, traffic_signs).** `Activation::sign` (CORA.cpp only): on `[l, u]` the line of
  slope 0 between `sign(l)` and `sign(u)` (exact: the values are halves), the interval `[sign(l),
  sign(u)]`, the derivative 0 in the bounds' heuristics; the attack's gradient takes the
  straight-through estimator (1 on `[-1, 1]`), so that PGD moves on a binarized network. The
  weights' Sign folds as a constant. A Softmax that makes the output is dropped when the property
  only compares outputs (`y_i - y_j <= 0`: it keeps their order); with any other property the
  Softmax stays, lowered as below. traffic_signs' three networks lower and give onnxruntime's
  probabilities to 7e-8.
- **Softmax and attention (Phase 4, cGAN's transformer).** A Softmax inside the network lowers to
  `1 / sum_t exp(x_t - x_q)`: the differences by two gathers of the same layer (one with the
  factor -1), a layer of `Activation::exp`, their sums by a 1 x 1 convolution, then a layer of
  `Activation::reciprocal`, `1 / max(x, 1)`, which is `1 / x` there because the term `t = q` is
  exactly 1. A MatMul of two values (`theta^T phi`, `g attn^T`) is a sum of products: two gathers
  into one product layer `[a; b]`, then a 1 x 1 convolution of ones over the first half; nothing
  is dense. A reciprocal layer that reads an exp layer alone (the sums of a softmax) also bounds
  its input by the image of the exps' box `[e^l, e^u]` (at least 1, as the term `t = q` is), so the
  reciprocal's chord spans the sums' true range, not the zonotope's. The lines (CORA.cpp only): exp takes the chord's slope `m`, the range of
  `exp(x) - m x` at the ends and at `log m`; the reciprocal the chord's slope on `[max(l, 1),
  max(u, 1)]`, the range of `f(x) - m x` at `l`, `u`, 1 and `sqrt(l u)`; both widened by 16 ulps of
  the largest term. A bound beyond the float range (an exp that overflows) has no line: that batch
  stays open (`out_of_range_sets`), the instance is never unsat for it, and the attack goes on.
  cGAN's `small_transformer` lowers to 90 layers and gives onnxruntime's output to 4e-7.
- **Vision transformers (Phase 4, vit_2023).** The graph builds its class token from the batch
  size (Shape, Gather, Unsqueeze, Concat, ConstantOfShape), which folds as shape arithmetic; the
  token joins the patches as a Concat with a constant. Slice takes its bounds from the attributes
  before opset 10. A BatchNormalization of a token sequence `(C, L)` is a gather with factors, and
  ReduceMean (the mean over the tokens) a dense layer of `1 / count`. Both networks lower (71 and
  100 layers) and give onnxruntime's output to 4e-7. With the 1 GiB batch budget the slot budget
  keeps 768 of the 3072 input generators and boxes the rest, and an exp's bound then overflows; an
  8 GiB budget keeps them all (2.6 GiB peak) and finishes a first iteration, but the margin's
  lower bound stays below -20 where the network's linearization varies by 0.33 over the box: the
  products and the softmax's lines are far too loose to verify yet. Measured on a 40 GB A100:
  the slot estimate is conservative (16 GiB kept 1536 input generators of ibp's 3072 and an
  interval of radius 28 on the output; the zonotope's root margin -5.7, CROWN's -0.18), so the vit
  preset's `memory_budget` is 32 GiB: all generators, 13 slots, ibp's root margins -0.03 to -0.06
  (CROWN) and pgd's some roots verified at once. The ReLU-slope steps (`alpha_iterations`) left the
  root bound unchanged (the softmax and the products are the loose part), so the preset turns them
  off. The preset only changes the budgets, not an enclosure.
- **nn4sys (Phase 4).** A Gather's negative index counts from the end (pensieve's last entry of a
  row, index -1). lindex and the three pensieve_simple networks lower and give onnxruntime's
  output to 6e-7. pensieve_parallel's input `[12, 8]` has no batch axis: only a Reshape to a batch
  of 1 may read it. Its `x^3 / sum x^3` lowers as `Activation::cube` (Pow(x, 3)), the sum to
  `Activation::inverse` (`1 / x`, verify only for `x > 0`: a bound `l <= 0` is out of range, and that
  batch stays open), and a product layer of the cubes and the broadcast inverse. The lines (CORA.cpp
  only): x^3 the chord's slope `l^2 + l u + u^2` with the range of `x^3 - m x` at the ends and at
  `+-sqrt(m / 3)`, `1 / x` the chord's slope `-1 / (l u)` with the convex `1 / x - m x` least at
  `sqrt(l u)`, both widened by the rounding. A cube or an inverse layer that reads a ReLU, cube or
  exp layer alone also bounds its input by the image of that layer's box, so the sums of cubes of
  ReLUs stay at least the cubes of the ReLUs' lower bounds. Both pensieve_parallel networks lower
  and equal the float64 graph to 1e-13 (onnxruntime's float32 to 3e-5).
  lindex's properties have thousands of cases, each a small box with its own two polytopes.
  The attack's loss and gradient take each row's margins over its own case's polytopes only
  (`NnProblem` per case, when the most polytopes of a case times the outputs is below all of
  them), the batch budget counts the rows of the polytopes, and the branch and bound goes chunk
  by chunk of cases of at most 256 rows (`NnProblem::subproblem`; unsat when every chunk is).
  lindex_60000 took 44 GiB and ran past its deadline; now 1.2 GiB within it, and lindex_10000 is
  unsat in 17 s with `conzonotope` bounding.
- **ml4acopf (Phase 4).** Its `linear-nonresidual` networks replace sin, cos and the sigmoid by
  sums of ReLUs over thresholds: `x - t_k` broadcasts a value against a constant to more entries,
  which lowers as a gather that repeats the value first. They also compute `C h^T` with the batch
  axis transposed: a Transpose of a row `(1, n)` makes a column (a flag on the value, the same
  entries), a MatMul of a constant matrix from the left keeps it a column as a dense layer, and a
  Transpose makes it a row again; any other operator on a column refuses. A dense chain whose layer
  exceeds `max_elements` now goes layer by layer. The three networks lower (122 layers) and give
  onnxruntime's output to 2e-6. The residual ones reduce the angles mod 2 pi by Floor. The plain
  ones take `Activation::sin` and `Activation::cos` of angle differences (CORA.cpp only): the
  tangent at the middle with radius `d^2 / 2` (`|f''| <= 1`), or slope 0 over the range's box (the
  ends' values, 1 or -1 where an extremum may lie inside) where that is narrower, both widened by
  the rounding; Where, Equal and Expand fold as constants.
- **Convolutions in C++.** A convolution's rows are the patches of its input times the kernel as a
  matrix (im2col); its transpose multiplies by the kernel's transpose and folds the patches back
  onto the input pixels (col2im), so `W^T y` costs what `W x` does (`conv2d.cpp`). The generator rows of a batch of
  zonotopes `(B, q, n)` are just more rows, so the propagation, the reverse pass, CROWN's bound and
  the attacks run unchanged on them; the set operations other than `verify` take dense layers.
- **Slot budget.** When one set of the batch does not fit `memory_budget` (a wide convolutional
  net with an error slot per neuron), the conzonotope bounding halves the error slots per layer,
  then the input generators, until it fits; the errors without a slot go to the interval center
  (sound, looser) and the backward bound keeps the margin. `stats` reports `slot_budget` and
  `init_gen_budget`. How many sets go into one batch is `batch_memory_budget` (0: `memory_budget`):
  on a GPU, run_instance.py sets it to 0.8 of the free memory, so the batch grows and one set's cuts
  stay those of the 1 GiB default (the larger sets cost more per box than the batch saves).
- **Memory check.** The front end builds a Gemm's matrix after a convolution from chunks of unit
  vectors (malbeware's identity of 61504 entries alone was 29 GiB). `run_dev_set.py
  --one-per-network` runs the first instance of each network of a dev set, and `score.py
  --max-rss-mib M` lists every instance whose peak RSS exceeds M and exits with 1.
- **Property.** VNN-LIB 1.0 is read with the semantics of the scoring's parser, 2.0 by the
  reference parser (`vnnlib`), every assertion in DNF and the conjunction multiplied out; the cases'
  polytopes are stacked and padded with the rows `0 <= +inf`.
- **Check.** C++ calls the Python check back for every candidate: onnxruntime's output of the
  float32 point, then VNN-COMP's checker of the written counterexample (`cora.counterexample`); only
  a `CORRECT` one ends the search.
- **Backend.** `load(..., backend="eigen")` gives the network numpy weights (Eigen), `"torch"` torch
  tensors on `device`; `run_instance.py --backend` (or `CORACPP_VNNCOMP_BACKEND`) picks it, CUDA
  takes torch. The default `"auto"` (`choose_backend`) takes torch for a network with a convolution
  and eigen for a dense one. At `CORACPP_NATIVE=OFF`, one BaB iteration of a cifar100 ResNet takes
  160 s on eigen and 61 s on torch, and of a tinyimagenet one 562 s and 273 s (2 threads).

## Strategies, chosen by name

The option names follow CORA's `options.nn` (`nn/+nnHelper/validateNNoptions.m`,
`nn/@neuralNetwork/verify.m`). Each strategy kind has a registry; a program registers its own under a
name (`register_attack`, `register_bounding`, `register_heuristic`, `register_input_split_heuristic`).

| Option (`NnVerifyOptions`) | Interface | Built-ins |
| --- | --- | --- |
| `falsification_method` | `Attack::run(problem, deadline, rounds)` | `pgd`, `fgsm`, `center`, `zonotack` |
| `bounding_method` | `Bounding::prepare`, `Bounding::branch_and_bound` | `zonotope`, `conzonotope`, `zonotope-alpha` |
| `input_split_heuristic` | zonotope: `(problem, lo, hi, cases) -> dimension`; conzonotope: a `Heuristic` | `largest-radius`, `most-sensitive-input-radius`; CORA's ten |
| `neuron_split_heuristic`, `relu_constraint_heuristic`, `input_generator_heuristic` | `Heuristic`: `score(l, u, er, sens, grad)`, `look_ahead` | CORA's ten (`computeHeuristic`), `fsb` |

CORA's ten heuristics: `input-radius`, `least-unstable`, `least-unstable-gradient`, `most-unstable`,
`gap-sensitivity`, `product-sensitivity`, `centered-sensitivity`, `most-sensitive-approx-error`,
`most-sensitive-input-radius`, `zono-norm-gradient`. `cora_vnncomp_options(category)` is CORA's
`getDefaultVNNCOMPoptions` with the benchmark's own changes. `vnncomp_options(category)` adds
Phase 2's bounds (`zonotope-alpha`, `backward_bound`; `alpha_method` `zonotope` for acasxu, `crown`
else), except for sat_relu: its many small boxes keep CORA's `conzonotope` bounding without CROWN,
which scores 980 on dev seeds 1001 and 1002 against 960 and 940 (the slopes cost more than they
decide). traffic_signs gives the attack nine tenths of the time (`attack_time_share` 0.9): its
binarized networks' bounds rarely decide, and the share scores 330 and 250 on seeds 1001 and 1002
against 300 and 240. metaroom, soundnessbench and tinyimagenet take no alpha-CROWN steps
(`alpha_iterations` 0): on the first 40 instances of seed 1001 they score 380, 60 and 60 against
370, 50 and 50, tinyimagenet in half the time; cifar100 keeps them (70 against 60), and cgan and
collins score the same either way. On the three, 3 steps or `optimize_gap` 0.1 score no more.
nn4sys keeps CORA's `conzonotope` bounding without CROWN and gives the attack a twentieth of the
time (`attack_time_share` 0.05): lindex's thousands of small cases go chunk by chunk, and the
preset scores 450 against 400 on dev seed 1001 (19 lindex instances unsat against 14; pensieve and
mscn the same).

lsnc_relu takes no alpha-CROWN steps either. Its boxes are verified by the refinement (an empty
child), and on instance 0 of seed 1001 the steps took 27 s of 57 s and verified none of 7410 sets.
Without them, the 8 sample rows of each dev seed (25 s) give 2 sat against 1 sat (0 unsat either
way), and instance 0 closes as unsat in 250 s. Its children also leave the refinement once a pass
moves their box by at most 0.03 (`refinement_min_move`), and its networks run on torch (a product:
`choose_backend`). Together, instance 0 closes in 126 s (156 s on torch alone); the sample rows'
verdicts stay the same (2 sat, 14 unknown), since the official timeout is 25 s.

On a CUDA device (A100) the sets of lsnc_relu are tiny (6 inputs), so a batch of 1024 sets is bound
by kernel launches, not arithmetic: instance 0 of seed 1001 closes as unsat in 46 s at 1024 sets
per batch, 18 s at 4096 and 12 s at 32768 (3.4 GiB peak). `run_instance.py` therefore takes
`mini_batch_size` 32768 for lsnc_relu on CUDA (`GPU_MINI_BATCH`). Soundness: the sets of a batch
are propagated and bounded independently (every step is per row), so the batch size changes no
bound and only the order in which the open boxes are taken; the verdict is unsat only when the
queue is empty, whatever the order. The 8 sample rows go from 3 sat and 5 unknown to 3 sat and
5 unsat.

- **PGD** (CORA's `computePGDAttack`): per box `pgd_restarts` samples (as many as the memory budget
  holds), the center first, sign steps of `pgd_step_size` times the box radius on the objective (per
  polytope the violated rows' excess plus a tenth of the worst row, the best polytope of the case),
  decayed by `pgd_decay` at `pgd_decay_at`, projected into the box, in rounds until the deadline.
  When the first step shows that `pgd_min_steps` steps would not fit in the time left (a wide conv
  net), each box keeps fewer restarts (CORA.cpp only; small networks never shrink). **FGSM** is one step of the full radius from the center, **center** the centers.
- **zonotope:** the output zonotope of `NeuralNetwork::evaluate` (DeepZ) per box, the naive branch and
  bound of input splits (`num_splits` pieces), the centers of the open boxes as candidates.

## CORA's verify in the latent space

`bounding_method` `conzonotope` is CORA's `neuralNetwork/verify` with `refinement_method` `zonotack`
(Koller et al., ICLR 2026, Algorithm 1), batched: `private/priv_zono_batch` propagates the zonotopes,
`private/priv_verify_zonotack.cpp` holds the algorithm.

- **Sets:** a batch of zonotopes is the center `c (B, n)`, the generators as rows `G (B, q, n)` and
  the radius `dr (B, n)` of CORA's interval center. The input gets `num_init_gens` generators (chosen
  by `input_generator_heuristic`), each activation layer at most `num_approx_err` error generators
  (chosen by `approx_error_order`); the rest of the errors goes into `dr`. The output's first factors
  are the input's, so constraints on the output are constraints on the input. The sensitivity that
  orders the slots (`sensitivity*length`) is computed only if some layer has slots for some but not
  all of its neurons (tinyimagenet's preset has none, and the 200-output Jacobian costs a pass).
- **Lines:** a ReLU takes CORALean's relaxation, which is CORA's `poly_method` `bounds` in the reals.
  Sigmoid and tanh take CORA's `bounds` (the secant, the errors at the extreme points of
  `f(x) - m x`) or CORALean's `singh`.
- **One iteration:**
  1. the check (ICLR Eq. 8): a box is verified when every polytope of its case is excluded by a row
     whose lower bound exceeds `verified_margin`;
  2. `bab_falsification_method`: `zonotack` tries, per row of every open polytope, the input point of
     the output's vertex least in that row (`beta = -sign(A_i G_y)`); `center` and `fgsm` too;
  3. the split rows on the factors: `num_dimensions` input splits, `num_neuron_splits` neuron splits
     (at zero or the middle) into `num_splits` pieces, the ReLU rows `ReLU(x) >= 0`, `ReLU(x) >= x`
     of `num_relu_constraints` neurons, each chosen by its heuristic;
  4. every combination of pieces is a child; per child and open polytope Prop. 2 turns the unsafe set
     into rows on the factors, Fourier-Motzkin (`polytope_bound_approx_max_iter` passes) bounds the
     factors, the hull over the polytopes rescales the input, which is propagated again,
     `refinement_max_iter - 1` times. An empty child is verified. A child that is empty, or whose
     input box did not move in a pass (the next pass would repeat it), leaves the passes (CORA.cpp
     only: the same children and boxes as CORA's fixed count, with fewer propagations; lsnc_relu
     instance 0 needs half the Fourier-Motzkin and propagation time). `refinement_min_move` > 0
     (CORA.cpp only, default 0) also lets a child leave once its box moves by at most that much in
     the factors' `[-1, 1]` (a looser box, never an unsound one).
- **Queue:** `mini_batch_size` sets per batch (`half-half`: half from the front, half from the back),
  the children in front (`verify_enqueue_type`), at most what `batch_memory_budget` holds; a batch starts
  only if the last one's time fits before the deadline, and the deadline is checked per chunk. Out
  of memory, the ReLU rows and then the batch shrink as in CORA. `stats`: `batch_size` (at the
  start), `queue_peak`, and `verified_sets` as the sum of `check_verified` (boxes the check decides),
  `empty_children` and `point_children` (children of the refinement that are empty or points).
- **Deadline inside the steps:** every pass (the zonotope forward, the sensitivity, the reverse and
  backward passes, the interval gradient, the attack's gradient) checks the deadline once per layer
  (`check_deadline`), and candidate points are evaluated in chunks. A pass that reaches it raises
  `special_error` (`is_deadline` tells it apart): the batch stays open, an attack step finds
  nothing, and verify answers within about one layer's time of its budget (a cifar100 ResNet: 26.3 s for 26 s, was 39.6 s; tinyimagenet:
  30.5 s for 30 s, was 114 s). alpha-CROWN returns its best bound instead, once it has one.
- **Fourier-Motzkin:** `fourier_motzkin_order` `cora` takes a row's sum minus the term (fast);
  `coralean` sums the other terms in CORALean's order (`contractHalfspace`), bit for bit the oracle's
  on lean. `ConZonotope::approximate_bounds` has the same switch (`BoundsOptions::order`).
- **Gradients** (`zono-norm-gradient`, `least-unstable-gradient`): the gradient of the output's
  F-radius with respect to each generator's value. With `exact_backprop` (CORA's default, on) it is a
  hand-written reverse pass through every layer: the slopes, offsets and errors of the lines depend
  on `l` and `u`, and so on `G` and `dr`, through the split clamps too (the Python reference of
  Phase 1 did this by autograd). The pass is written on the Tensor operations, so it runs on every
  backend; libtorch's autograd would tie it to torch, and no code outside the backends branches on
  one (docs/architecture.md). With `exact_backprop` off the slopes are held, which is
  `|G_g|^2 / (v_g F)` for the row `g` of the value `v_g`. The input generators' heuristic
  (`num_init_gens` below the input dimension) takes the gradient through the interval propagation,
  written out.
- **zonotope-alpha:** CORA's verify whose check runs `alpha_iterations` projected Adam steps on the
  ReLU slopes of the open sets (from `u / (u - l)`), ascending the sum over the open polytopes of
  their best row's lower bound. The gradient is a hand-written reverse pass (the one of
  `exact_backprop`, seeded with the margins), exact through the bounds of the later layers; a ctest
  checks it against the central difference. A set leaves the loop once it is verified.
- **alpha_method:** `zonotope` (the default) optimizes the slopes of the zonotope's lines, so every
  later layer's bounds move and a step costs a forward and a reverse pass of the sets; `crown`
  optimizes CROWN's lower ReLU slopes per specification row in the backward pass (alpha-CROWN) over
  the zonotope's bounds, so a step costs one backward pass and its reverse, `(B, P K, n)` per layer.
- **optimize_gap:** the slopes and the dual multipliers are optimized only for open sets whose worst
  open polytope's best row is below 0 by at most this share of the row's width (inf: all).
- **backward_bound:** the check also bounds every row by CROWN's backward pass over the input box and
  takes the better of it and the zonotope's bound. Its cost is `(B, R, n)` per layer, so it stays
  cheap on wide layers. R counts only the rows of the polytopes that the zonotope left open in some
  set of the batch (each row's bound is its own, so the verdicts do not change); alpha-CROWN does
  the same. A robustness property of 199 rows whose zonotope excludes most classes costs a few.
- **dual_certificate:** the check also bounds every row of the open sets by weak duality under the
  rows `C beta <= d` of the factors: each split of the set (`x >= 0` or `x <= 0` of the neuron's
  input) and the three faces of the `dual_relu_rows` crossing ReLUs of the largest error. For any
  `lambda >= 0`, `min a^T y >= a^T c - |a| dr - |G a + C^T lambda|_1 - lambda^T d` (beta-CROWN's
  bound in the latent space); `dual_iterations` projected Adam steps choose `lambda` in the search
  dtype, and the bound of the best `lambda` is evaluated again in float64.
- **Frozen zonotope (the groundwork of a ReLU-split branch and bound):** `priv_split_table` takes one
  propagation's crossing ReLU neurons once (their input generators, centers and error radii). Split
  in a fixed order, the nodes of one depth share these rows `R` and differ in their signs `s`: node
  `n` holds `s_j G_j beta <= -s_j c_j + dr_j` (`s_j = -1`: the input at least 0, `+1`: at most 0,
  as the splits above), so a step's `lambda C` costs `(lambda * s) R` and no step needs a node's
  own matrix (`priv_dual_lower_signs`; a child starts at its parent's multipliers, 0 for its new
  row). The bound is the same weak duality, sound for any `lambda >= 0`, evaluated again in
  float64. The steps work on rows of 1-norm 1; the bound takes `lambda / norm` on each node's
  unscaled rows and offsets, so no rounded scaled row enters it: one certificate,
  `priv_dual_certificate`, for `priv_dual_lower`, `priv_dual_lower_signs` and
  `priv_dual_lower_phases`; the latter two on rows all nodes share, one per coefficient pair of
  `G_in`, `G_out` (`R`, `-R`; the phases' seven: `G_in`, `-G_in`, `G_out`, `-G_out`, `G_out - G_in`,
  `G_in - G_out`, `G_out - sigma G_in`), a node's multipliers and offsets on its own blocks' rows.
- **relu-bab (`refinement_method`):** a ReLU-split branch and bound on the frozen root zonotope.
  - **Root:** one propagation per input box, its slopes optimized (`zonotope-alpha`,
    `relu_bab_alpha_iterations` steps without patience), with a generator for each input dimension
    (at most `relu_bab_init_gens`) and a slot for every crossing neuron's error, as
    `relu_bab_memory_budget` (0: 3/4 of the batch budget; the nodes' chunks take a quarter)
    allows: fewer slots first, then fewer generators. Two passes count the crossing neurons per layer (the second, of the tighter net,
    fewer), so no slot is spent on a stable neuron. `relu_bab_slotless` counts the crossing neurons
    whose error stays in `dr` (slack in each row). The open polytopes go weakest first; one left
    open ends the search (unknown).
  - **Nodes:** the crossing neurons split in the order `relu_bab_order`: `output`, the largest
    `|w_k|` at the neuron's slot factor first (its error's slack in the polytope's row), or `error`,
    the largest line error first. The open nodes are parts of at most 1/16 of
    `relu_bab_frontier_bytes` (0: a quarter of the batch budget), all of them at most that: the
    last part's children first, so breadth first while they fit, then deeper. Each node is a sign
    vector, its bound `priv_dual_lower_phases` (`relu_bab_dual_steps` steps from its parent's
    multipliers, kept in the search dtype; the root `relu_bab_root_steps`), in chunks of a quarter
    of the batch budget. Besides its splits a node holds the triangle faces of the next
    `relu_bab_faces` neurons of the order. A node is verified when some row's float64 bound exceeds
    `verified_margin`; it is dropped as infeasible when Farkas' certificate holds
    (`relu_bab_farkas_steps`, 0: off; on cifar100 it pruned almost no node at the cost of a second
    ascent per node), some `mu >= 0` with `min_beta mu^T (C beta - d) > verified_margin` over the
    box (the same bound with `w = 0`; an infeasible node's own bound grows too). The weakest nodes' Lagrangian minimizers go to the
    confirm check (`relu_bab_witnesses`).
  - **Rows of split j:** `s_j = +1` (inactive, `y = 0`): `G_in beta <= -c_in + dr_in` and
    `|G_out beta + c_out| <= dr_out`; `s_j = -1` (active, `y = z`): `-G_in beta <= c_in + dr_in`
    and `|(G_out - G_in) beta + c_out - c_in| <= dr_in + dr_out`. They share the table's rows, so
    `lambda C` is two products with the shared `G_in`, `G_out`. An unsplit neuron's faces over its
    bounds `[l, u]`: `y >= 0`, `y >= z` and the chord `y <= sigma z + max(-sigma l, u - sigma u)`,
    `sigma = u / (u - l)` rounded up (a larger slope is a true face too; the offset holds at both
    ends, `priv_relu_chord_offset`), each with `delta` at its worst.
  - **Soundness:** for every input of the box there is one `beta` (the input factors and the slots'
    error factors) at which every layer is its affine form up to `delta`, `|delta| <= dr`:
    `z = c_in + G_in beta + delta_in`, `y = c_out + G_out beta + delta_out`. A phase is a true fact
    of the inputs on its side (inactive: `z <= 0`, `y = 0`; active: `z >= 0`, `y = z`), and its rows
    are those facts with each `delta` at its worst, so a split only intersects. This needs every
    error that is not a factor in `dr` (never dropped) and one `beta` for `G_in` and `G_out` (one
    column order, later columns zero, as `priv_split_table` pads them). The frozen lines and clamps
    of later layers are not re-derived. unsat only when every open polytope's frontier is empty.
- **relu_constraint_faces:** `triangle` adds the chord above each chosen crossing ReLU as a third row
  (CORA's `num_relu_constraints` rows are the two lower faces), so the factor system holds the
  triangle relaxation.
- **fsb:** the neuron split heuristic filters `look_ahead_candidates` neurons by zono-norm-gradient,
  then splits the one whose worse child has the best backward bound (FSB's look-ahead).
- **dual-iter:** `conzonotope_bounding_method` `dual-iter` tightens only the box of the inputs, which
  the refinement does not read (it rescales by the factor box), so it never decides; it is accepted
  and does the same as `fourier-motzkin`.

### CORA's verify, step by step

| Step (CORA) | Sound? | Evidence | Here |
| --- | --- | --- | --- |
| zonotope propagation, `evaluateZonotopeBatch` | yes, in the reals | CORALean `SetBasedNet.forward`, `IsRelaxationOf` (errors bound `f(t) - m t` pointwise, which the extreme points give) | float64; the ReLU and Singh lines bit-exact with CORALean |
| interval center, slots | yes, in the reals | interval arithmetic of the errors without a slot | as CORA |
| check, Eq. 8 | yes | `Zonotope.inter_ineqs_eq_empty_of_row` | as CORA, with `verified_margin` |
| zonotack candidates, Eq. 9 | no claim (a candidate is checked) | `Zonotope.zonotackFactors`, CORA's sign | as CORA, checked by the confirm callback |
| unsafe inputs, Prop. 2 | yes, with pointwise errors | `SetBasedNet.encloseUnsafeInputs_outer` | as CORA, the offset plus `verified_margin` |
| Fourier-Motzkin, Prop. 3 and App. D | yes | `ConZonotope.fourierMotzkinBounds_outer`, `fourierMotzkinIter_outer` | CORA's order, or CORALean's (bit-exact) |
| rescaling, `aux_scaleAndOffsetZonotope` | yes | the factors' box holds every feasible factor | as CORA |
| input and neuron splits | yes (the pieces cover the factor box) | `SetBasedNet.verify_sound` for any covering split | as CORA |
| ReLU rows `ReLU(x) >= 0`, `>= x` | yes (the true factors satisfy them) | not formalized | as CORA |
| split and ReLU rows in later refinement iterations | **no** | the rows hold the error factors of the first propagation; the next propagations put new error factors under the same indices | repaired (`project_stale_factors`): after the first iteration the rows keep the input factors, the error factors' share moves into the offsets |
| dummy splits counted as verified (`isDummySplit`) | **no** | a set without an unstable neuron is not verified by having none; the flags are also tiled (`repmat`) over children ordered by `repelem` | repaired (`drop_dummy_duplicates`): the duplicate child is dropped, the other is refined |
| a neuron split's bound under an interval center | **no** | the split hyperplane is at the interval's middle, so `x >= 0` does not follow on the child | repaired (`remember_point_splits_only`): the bound is remembered only at a point center |
| single precision on the GPU | **no** (no rounding control) | CORA's `verify.m` casts to `single` | float64 bounds and `verified_margin`; the float64 re-check of the leaves is Phase 2 |

Beyond CORA (Phase 2), every step sound for any choice it optimizes:

| Step | Sound? | Evidence | Here |
| --- | --- | --- | --- |
| ReLU slope `alpha` in `[0, 1]` (`zonotope-alpha`) | yes, in the reals | `relu(x) - a x` lies in `[0, max(-a l, (1 - a) u)]` on `[l, u]` for `a` in `[0, 1]` (convex, 0 at 0); not formalized | every iteration's bound is a bound; a polytope excluded once stays excluded |
| alpha-CROWN slopes (`alpha_method` `crown`) | yes, in the reals | any lower slope in `[0, 1]` through 0 bounds a crossing ReLU from below; every step's bound is a bound, the best is kept; not formalized | float64, `verified_margin`, the clamped bounds of the zonotope |
| CROWN backward bound (`backward_bound`) | yes, in the reals | the chord above and a slope in `[0, 1]` through 0 below a crossing ReLU; the zonotope line's parallels for sigmoid/tanh; not formalized | float64, `verified_margin`, the clamped bounds of the zonotope |
| chord rows `ReLU(x) <= u (x - l) / (u - l)` (`relu_constraint_faces` `triangle`) | yes (the true factors satisfy them on the clamped `[l, u]`) | the chord bounds a ReLU on `[l, u]` from above; the offsets add `dr_out + s dr_in`; not formalized | with the other ReLU rows, under `project_stale_factors` |
| dual certificate (`dual_certificate`) | yes, for any `lambda >= 0` | weak duality: `w^T beta >= w^T beta + lambda^T (C beta - d)` on the feasible factors, whose least value over the box is `-\|w + C^T lambda\|_1 - lambda^T d`; the rows hold the true factors (splits on the clamped bounds, the ReLU faces with `dr` at its worst); not formalized | the bound of the best `lambda`, evaluated again in float64, and never below the zonotope's (`lambda = 0`) |
| FSB look-ahead (`fsb`) | no claim (it only picks the neuron) | the children are as for any neuron split | the backward bound of both children, the other bounds held |

The repairs are options, on by default. `tests/nn/neural_network/test_neural_network_verify_witness.cpp`
runs 60 random 2-input ReLU networks whose properties have unsafe inputs: with the repairs no verdict
is `unsat`; with `drop_dummy_duplicates` off some are.

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
  cifar100_2024 and tinyimagenet_2024 need torch < 2.6 (e.g. 2.4.1 with NumPy < 2): their datasets
  are pickles holding NumPy scalars, which `torch.load`'s `weights_only` default refuses since 2.6.
  make_dev_set itself runs in a Python with NumPy and the built `cora` package
  (`PYTHONPATH=build/python`), which writes the VNN-LIB 2.0 form; that need not be the `--python`.
  linearizenn needs h5py.
  relusplitter's `generate_properties.py` installs uv by `curl | sh` and a CUDA torch, so
  make_dev_set runs its `scripts/gen_props.py` in our venv with CPU torch instead. It needs
  auto_LiRPA (`git submodule update --init` in the source; install.sh's one-line patch is applied to
  the copy), and beartype from `--libs` (`genlibs/relusplitter_2026`, outside the venv: torch 2.4's
  ONNX exporter type-checks with an installed beartype and fails). Its node names carry random ids,
  so a seed gives the same weights and properties, not the same bytes.
  vit's generator shuffles each model's pool of CIFAR-10 images (242 for ibp_3_3_8, 4847 for
  pgd_2_3_16) and takes 100 of each, so about 40 of a seed's 200 are official ones, left out.
  vggnet16 preprocesses its ImageNet images with mxnet 1.9, which needs NumPy < 1.24
  (`genvenv-vgg`, Python 3.10; its generator pip-installs mxnet into any Python that lacks it). Its
  528 MB VGG16 comes from the ONNX model zoo once, into the cache (`--cache`).
  challenging_certified_training's generator downloads its six networks from a password-protected
  share (the password is in its source). make_dev_set decompresses the three that the 2026 benchmark
  repository ships, identical to the share's; the generator fetches the three wide ones. Only the
  sampled CIFAR-10 and TinyImageNet test images are new.
  tllverifybench's generator draws each network's output threshold from NumPy's global generator,
  which it never seeds; make_dev_set seeds it with the seed (the networks are the official ones).
  nn4sys runs its three generators (video streaming, learned index, cardinality) with the seed.
  ml4acopf's generator perturbs each bus' loads with the seed (69 instances per seed).
- **Data:** networks and instances stay outside the repository (e.g. `~/vnncomp/dev_sets`).
- **Releases:** `scripts/vnncomp/freeze_dev_sets.py DIR --write --release v1` writes the release's
  `MANIFEST.json`: per benchmark and seed the instances, the official ones left out, the source
  commit, the method, the known nondeterminism and the sha256 of every file; `--verify` checks a copy
  against it. Both refuse absolute, dangling or escaping links, work folders and logs. A release is
  read-only; a change becomes the next release in a new directory. The subset safenlp_2024_sub
  (every fourth instance of safenlp) links its networks and properties relatively.

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

Bit for bit on the lean backend against the CORALean oracle at dtype nearest
(`tests/global/oracle/test_lean_nn_verify.cpp`):
- the batched zonotope with an error generator per neuron and a point center, member by member, for
  ReLU, sigmoid and tanh at `poly_method` `singh`, against `neuralNetwork.evaluate`;
- Fourier-Motzkin in CORALean's order against `conZonotope.fourierMotzkinBounds`
  (`lean::fourier_motzkin_bounds`);
- a whole verify of an ACAS Xu-sized network on lean (`unsat` after splits), and the sets of its first
  batch: the output zonotope and the factor system of Prop. 2.

The gaps are in `docs/oracle-requests.md` ("Latent-space verify").
