# Tensor

`cora::Tensor` is the one array type that sets, systems and networks are written against. Each
algorithm exists once; the backends (Eigen, libtorch, lean) implement a small core, and the rest is
portable code on top of it. `include/cora/global/tensor/ops.def` is the table of operations; this
page lists it, `tensor_ops()` (C++) and `cora.tensor_ops()` (Python) return it, and the conventions
test keeps all three in step.

## Rules

- **The data decides.** A tensor carries its backend, device and dtype (float64, or float32 on
  torch and Eigen; lean computes in float64 only). An operation runs where its operands live.
  Operands on two backends, devices or dtypes are an error that names both, checks on or off.
  Library code makes new tensors like an operand (`zeros_like`, `eye_like`, `Tensor::like`); the
  current backend (`BackendScope`) applies only to raw-number entry points, functions marked
  `CORA_FROM_NUMBERS` (the conventions test checks this).
- **Layout and batches.** Row-major, the last two axes are the matrix (a vector is a column
  `(n, 1)`), leading axes are batches on every backend and broadcast as in numpy. Eigen and lean
  loop over the matrices of a batch, in threads when the work pays (`cora::threads`).
- **Masks and indices** are tensors of the operand's dtype holding 0/1 or integer values.
- **Copy-on-write.** Storage is shared and immutable to users. An operation on an rvalue tensor
  that alone owns its memory (`std::move(x).add(y)`, `a + b` on a temporary) writes in place.
- **Zero copy with Python.** A C-contiguous float64 (float32) numpy array becomes an Eigen tensor
  over the array's memory; a result goes back as a numpy array over the tensor's memory, read-only
  unless nobody else holds it. A torch tensor stays the same `at::Tensor`, with its autograd.

## Core, derived and fused

A backend implements the core: the 26 pure virtual functions of `Tensor::Impl` (one `binary`,
`scalar` and `unary` cover the elementwise rows). A derived operation has a portable default made
of core operations (`src/global/tensor/derived.cpp`); a fused one is a derived hot-spot primitive
whose default is its specification (`src/global/tensor/fused.cpp`). On lean every default runs as
written, so its rounding and summation order is that of the core operations: sums nest to the right
from zero, as CORALean's `sumUp`. A backend may override a derived operation natively; the override
must agree with the default (`tests/global/tensor/test_tensor_overrides.cpp`):

| backend | native versions of derived and fused operations |
|---|---|
| eigen | `expm` (Eigen's Padé), `zonotope_linear_map`, `zonotope_interval_map`, `zonotope_linear_combination`, `interval_matmul`, `matmul_interval` (products with M⁺ and M⁻), `girard_split`, `abs_sum`, and the sums along the last axis and `matmul` (Eigen's kernels) |
| torch | `expm` (`matrix_exp`, or the hand-written backward with `,custom_backward`), `sort`, `diag`, `matrix_norm`, `vector_norm`, `interval_matmul`, `matmul_interval` |
| lean | none: lean is the specification (`expm` is an error: CORALean's exponential is an interval series) |

## Gradients

Each operation has a tag. **differentiable**: gradients pass. **piecewise**: the gradient of the
branch taken, the choice held fixed, as `torch.sort` (selections are made on detached values).
**not_differentiable**: on torch, given tensors that need gradients, the operation computes on the
detached tensors and warns once per operation (`CORA:notDifferentiable`, a `cora.CORAWarning`);
`GradientsScope` (Python: `with cora.strict_gradients():`) makes that an error. Every operation that
passes gradients has a float64 gradcheck against central differences
(`tests/global/tensor/test_tensor_gradcheck_torch.cpp`, required by the conventions test).

## Agreement with lean

The lean backend rounds as CORALean at dtype `nearest` (`tests/global/oracle/test_lean_tensor_ops.cpp`
compares its primitives with the oracle on point intervals, bit for bit). Eigen and torch on the CPU
agree with lean per operation (`tests/global/tensor/test_tensor_equivalence.cpp`), as the table says:

- **exact**: the same bits;
- **ulp k**: within k units in the last place of lean's result;
- **sum k**: within k·n·u·S, u = 2⁻⁵³, n the number of terms, S the same operation on |operands|;
- **norm k**: within k·n·u·max|lean's result|, n the matrix size, on well-conditioned operands
  (the factorizations have no CORALean counterpart; lean's are Eigen's).

Because an algorithm is the same code on every backend, a bit-exact oracle test of it on lean
carries over to the other backends up to these bounds of its primitives.

## Operations

### Construction

| operation | kind | gradient | agreement | what it computes |
|---|---|---|---|---|
| `like` | core | not_differentiable | exact | host values as a tensor of the backend, device and dtype of another |
| `full` | core | not_differentiable | exact | a tensor of one value, like another |
| `arange` | core | not_differentiable | exact | 0, 1, ..., n-1 as a 1-D tensor, like another |
| `eye` | derived | not_differentiable | exact | the identity (..., n, n), like another |
| `to` | core | differentiable | exact | the values on another device or in another dtype |
| `data` | core | not_differentiable | exact | the values on the host, row-major |

### Elementwise

| operation | kind | gradient | agreement | what it computes |
|---|---|---|---|---|
| `add` | core | differentiable | exact | a + b, broadcast |
| `subtract` | core | differentiable | exact | a - b, broadcast |
| `multiply` | core | differentiable | exact | a * b elementwise, broadcast |
| `divide` | core | differentiable | exact | a / b elementwise, broadcast |
| `pow` | core | differentiable | ulp 2 | a ** b elementwise, broadcast |
| `maximum` | core | piecewise | exact | the larger of a and b, broadcast |
| `minimum` | core | piecewise | exact | the smaller of a and b, broadcast |
| `remainder` | core | piecewise | exact | fmod(a, b), plus b where their signs differ (MATLAB mod) |
| `atan2` | core | differentiable | ulp 2 | the angle of (b, a), broadcast |
| `equal` | core | not_differentiable | exact | 1 where a == b, else 0 |
| `not_equal` | core | not_differentiable | exact | 1 where a != b, else 0 |
| `less` | core | not_differentiable | exact | 1 where a < b, else 0 |
| `less_equal` | core | not_differentiable | exact | 1 where a <= b, else 0 |
| `greater` | core | not_differentiable | exact | 1 where a > b, else 0 |
| `greater_equal` | core | not_differentiable | exact | 1 where a >= b, else 0 |
| `negative` | core | differentiable | exact | -x |
| `abs` | core | piecewise | exact | |x| |
| `sign` | core | piecewise | exact | -1, 0 or 1 |
| `floor` | core | piecewise | exact | the largest integer <= x |
| `ceil` | core | piecewise | exact | the smallest integer >= x |
| `round` | core | piecewise | exact | the nearest integer, halves away from zero (MATLAB round), +0 for a zero |
| `sqrt` | core | differentiable | ulp 1 | the square root (correctly rounded on the host; torch vectorized: 1 ulp) |
| `exp` | core | differentiable | ulp 2 | e^x |
| `log` | core | differentiable | ulp 2 | the natural logarithm |
| `sin` | core | differentiable | ulp 2 | sine |
| `cos` | core | differentiable | ulp 2 | cosine |
| `tan` | core | differentiable | ulp 2 | tangent |
| `asin` | core | differentiable | ulp 2 | arc sine |
| `acos` | core | differentiable | ulp 2 | arc cosine |
| `atan` | core | differentiable | ulp 2 | arc tangent |
| `sinh` | core | differentiable | ulp 2 | hyperbolic sine |
| `cosh` | core | differentiable | ulp 2 | hyperbolic cosine |
| `tanh` | core | differentiable | ulp 2 | hyperbolic tangent |
| `asinh` | core | differentiable | ulp 2 | inverse hyperbolic sine |
| `acosh` | core | differentiable | ulp 2 | inverse hyperbolic cosine |
| `atanh` | core | differentiable | ulp 2 | inverse hyperbolic tangent |
| `isnan` | core | not_differentiable | exact | 1 where x is NaN, else 0 |
| `isinf` | core | not_differentiable | exact | 1 where x is infinite, else 0 |
| `isfinite` | core | not_differentiable | exact | 1 where x is finite, else 0 |
| `where` | core | piecewise | exact | a where the condition is nonzero, else b, broadcast |
| `square` | derived | differentiable | exact | x * x |
| `sigmoid` | derived | differentiable | ulp 4 | 1 / (1 + e^-x) |
| `clip` | derived | piecewise | exact | x limited to [low, high] |

### Reductions

| operation | kind | gradient | agreement | what it computes |
|---|---|---|---|---|
| `sum` | core | differentiable | sum 1 | the sum along an axis (lean: nested to the right from zero) |
| `prod` | core | differentiable | sum 1 | the product along an axis |
| `max` | core | piecewise | exact | the largest entry along an axis |
| `min` | core | piecewise | exact | the smallest entry along an axis |
| `argmax` | core | not_differentiable | exact | the index of the first largest entry along an axis |
| `argmin` | core | not_differentiable | exact | the index of the first smallest entry along an axis |
| `cumsum` | core | differentiable | sum 1 | the running sum along an axis |
| `mean` | derived | differentiable | sum 1 | the mean along an axis |
| `any` | derived | not_differentiable | exact | 1 if some entry along an axis is nonzero |
| `all` | derived | not_differentiable | exact | 1 if every entry along an axis is nonzero |
| `count_nonzero` | derived | not_differentiable | exact | the number of nonzero entries along an axis |
| `softmax` | derived | differentiable | ulp 8 | e^x / sum(e^x) along an axis, shifted by the max |

### Shape and Selection

| operation | kind | gradient | agreement | what it computes |
|---|---|---|---|---|
| `reshape` | core | differentiable | exact | the same values in another shape (row-major) |
| `permute_dims` | core | differentiable | exact | the axes in another order |
| `broadcast_to` | core | differentiable | exact | repeated along axes of size 1 or new leading axes (torch expand) |
| `slice` | core | differentiable | exact | x[..., start:stop:step, ...] along one axis |
| `take` | core | differentiable | exact | the entries at the given indices along an axis |
| `take_along_axis` | core | differentiable | exact | per position, the entry at the index of the same position |
| `concat` | core | differentiable | exact | tensors joined along an axis |
| `matrix_transpose` | derived | differentiable | exact | the last two axes swapped |
| `expand_dims` | derived | differentiable | exact | a new axis of size 1 |
| `squeeze` | derived | differentiable | exact | an axis of size 1 removed |
| `stack` | derived | differentiable | exact | tensors of one shape along a new axis |
| `flip` | derived | differentiable | exact | the order along an axis reversed |
| `tile` | derived | differentiable | exact | copies side by side along each axis (MATLAB repmat) |
| `diag` | derived | differentiable | exact | a column (..., n, 1) as the diagonal matrix (..., n, n) |
| `diagonal` | derived | differentiable | exact | the diagonal of (..., n, m) as a column (..., min(n, m), 1) |
| `tril` | derived | differentiable | exact | the entries on and below a diagonal, the rest zero |
| `triu` | derived | differentiable | exact | the entries on and above a diagonal, the rest zero |
| `block_diag` | derived | differentiable | exact | matrices along the diagonal of one, zeros elsewhere (MATLAB blkdiag) |

### Sorting

| operation | kind | gradient | agreement | what it computes |
|---|---|---|---|---|
| `argsort` | core | not_differentiable | exact | the indices that sort along an axis, stable |
| `sort` | derived | piecewise | exact | the values sorted along an axis |
| `topk` | derived | piecewise | exact | the k largest (or smallest) along an axis and their indices, stable |
| `nonzero` | derived | not_differentiable | exact | the flat indices of the nonzero entries, on the host (MATLAB find) |

### Linear Algebra

| operation | kind | gradient | agreement | what it computes |
|---|---|---|---|---|
| `matmul` | core | differentiable | sum 2 | the matrix product, batched and broadcast (lean: sums nested to the right) |
| `solve` | core | differentiable | norm 64 | x with A x = B, batched |
| `cholesky` | core | differentiable | norm 64 | the lower factor L of A = L L^T |
| `qr` | core | differentiable | norm 64 | the reduced Q R, R with a nonnegative diagonal |
| `svd` | core | differentiable | norm 64 | the reduced U S V^T, S a column, descending, signs fixed |
| `eigh` | core | differentiable | norm 64 | eigenvalues (ascending column) and vectors of a symmetric matrix, signs fixed |
| `inv` | derived | differentiable | norm 64 | the inverse, by solve with the identity |
| `lstsq` | derived | differentiable | norm 64 | the least-squares solution of A x = B (full column rank), by qr |
| `pinv` | derived | differentiable | norm 64 | the pseudo-inverse, by svd |
| `matrix_rank` | derived | not_differentiable | exact | the number of singular values above a tolerance |
| `trace` | derived | differentiable | sum 1 | the sum of the diagonal, (..., 1, 1) |
| `kron` | derived | differentiable | exact | the Kronecker product |
| `expm` | derived | differentiable | norm 256 | the matrix exponential (Pade 13, scaling and squaring) |
| `matrix_norm` | derived | piecewise | sum 2 | the 1, 2, inf or Frobenius norm of (..., n, m), (..., 1, 1) |
| `vector_norm` | derived | piecewise | sum 2 | the 1, 2 or inf norm along an axis |

### Fused Primitives

| operation | kind | gradient | agreement | what it computes |
|---|---|---|---|---|
| `zonotope_linear_map` | fused | differentiable | sum 2 | M c and M G of a zonotope in one call |
| `zonotope_interval_map` | fused | piecewise | sum 2 | T c and [T G, diag(S sum|[c G]|)] for the interval matrix T +- S (CORA's mtimes) |
| `zonotope_linear_combination` | fused | differentiable | exact | (c1 + c2)/2 and [(G1 + G2)/2, (G1 - G2)/2, (c1 - c2)/2] (CORA's linComb) |
| `interval_matmul` | fused | piecewise | sum 2 | the bounds of I M for an interval matrix I (CORA's mtimes) |
| `matmul_interval` | fused | piecewise | sum 2 | the bounds of M I for an interval matrix I (CORA's mtimes) |
| `girard_split` | fused | not_differentiable | exact | the generators by Girard's score |g|_1 - |g|_inf, the k kept and the rest |
| `abs_sum` | fused | piecewise | sum 1 | sum(|x|) along the last axis, kept (the radius of a zonotope) |
