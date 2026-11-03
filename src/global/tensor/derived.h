// derived - the portable defaults of the derived and fused Tensor operations
//
// Also the checks of the front end that they share.
//
// Each default is made of core operations only, so it runs on any backend and, on lean, rounds in
// the order of those operations. A backend that overrides one natively must agree with it (ops.def,
// tests/global/tensor/test_tensor_overrides.cpp calls both).
//
// Syntax:   Tensor E = derived::expm(A);   detail::check_pair(a, b, "add");
// See also: tensor.h, derived.cpp, fused.cpp

#pragma once

#include "cora/global/tensor/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

namespace detail {

/// An error unless a and b share backend, device and dtype; it says which is where.
void check_pair(const Tensor &a, const Tensor &b, const char *op);

/// The axis (negative counts from the end) of a tensor of `rank` axes, checked.
int axis_of(int axis, int64_t rank, const char *op);

/// t as it may enter the not_differentiable `op`: detached, with a warning once per op (an error
/// under GradientsScope), when it needs gradients; else t itself.
Tensor without_gradient(TensorOp op, const Tensor &t);

} // namespace detail

namespace derived {

// Derived operations with a virtual hook (a backend may override them)
Tensor expm(const Tensor &A);
Tensor sort(const Tensor &x, int axis, bool descending);
std::pair<Tensor, Tensor> topk(const Tensor &x, int64_t k, int axis, bool largest);
Tensor kron(const Tensor &A, const Tensor &B);
Tensor matrix_norm(const Tensor &A, Tensor::Norm ord);
Tensor vector_norm(const Tensor &x, Tensor::Norm ord, int axis, bool keepdims);
Tensor diag(const Tensor &v);

// Fused primitives
std::pair<Tensor, Tensor> zonotope_linear_map(const Tensor &M, const Tensor &c, const Tensor &G);
std::pair<Tensor, Tensor> zonotope_interval_map(const Tensor &inf, const Tensor &sup,
                                                const Tensor &c, const Tensor &G);
std::pair<Tensor, Tensor> zonotope_linear_combination(const Tensor &c1, const Tensor &G1,
                                                      const Tensor &c2, const Tensor &G2);
std::pair<Tensor, Tensor> interval_matmul(const Tensor &inf, const Tensor &sup, const Tensor &M);
std::pair<Tensor, Tensor> matmul_interval(const Tensor &M, const Tensor &inf, const Tensor &sup);
std::pair<Tensor, Tensor> girard_split(const Tensor &G, int64_t keep);
Tensor abs_sum(const Tensor &x);

} // namespace derived

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
