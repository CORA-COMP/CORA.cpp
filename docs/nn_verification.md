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

## Strategies, chosen by name

The option names follow CORA's `options.nn` (`nn/+nnHelper/validateNNoptions.m`,
`nn/@neuralNetwork/verify.m`). Each strategy kind has a registry; a program registers its own under a
name (`register_attack`, `register_bounding`, `register_heuristic`, `register_input_split_heuristic`).

| Option (`NnVerifyOptions`) | Interface | Built-ins |
| --- | --- | --- |
| `falsification_method` | `Attack::run(problem, deadline, rounds)` | `pgd`, `fgsm`, `center`, `zonotack` |
| `bounding_method` | `Bounding::prepare`, `Bounding::branch_and_bound` | `zonotope`, `conzonotope` |
| `input_split_heuristic` | zonotope: `(problem, lo, hi, cases) -> dimension`; conzonotope: a `Heuristic` | `largest-radius`, `most-sensitive-input-radius`; CORA's ten |
| `neuron_split_heuristic`, `relu_constraint_heuristic`, `input_generator_heuristic` | `Heuristic`: `score(l, u, er, sens, grad)` | CORA's ten (`computeHeuristic`) |

CORA's ten heuristics: `input-radius`, `least-unstable`, `least-unstable-gradient`, `most-unstable`,
`gap-sensitivity`, `product-sensitivity`, `centered-sensitivity`, `most-sensitive-approx-error`,
`most-sensitive-input-radius`, `zono-norm-gradient`. `vnncomp_options(category)` is CORA's
`getDefaultVNNCOMPoptions` with the benchmark's own changes.

- **PGD** (CORA's `computePGDAttack`): per box `pgd_restarts` samples (as many as the memory budget
  holds), the center first, sign steps of `pgd_step_size` times the box radius on the objective (per
  polytope the violated rows' excess plus a tenth of the worst row, the best polytope of the case),
  decayed by `pgd_decay` at `pgd_decay_at`, projected into the box, in rounds until the deadline.
  **FGSM** is one step of the full radius from the center, **center** the centers.
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
  are the input's, so constraints on the output are constraints on the input.
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
     `refinement_max_iter - 1` times. An empty child is verified.
- **Queue:** `mini_batch_size` sets per batch (`half-half`: half from the front, half from the back),
  the children in front (`verify_enqueue_type`), at most what `memory_budget` holds; a batch starts
  only if the last one's time fits before the deadline, and the deadline is checked per chunk. Out
  of memory, the ReLU rows and then the batch shrink as in CORA.
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

The repairs are options, on by default. `tests/nn/neural_network/test_neural_network_verify_witness.cpp`
runs 60 random 2-input ReLU networks whose properties have unsafe inputs: with the repairs no verdict
is `unsat`; with `drop_dummy_duplicates` off some are.

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
