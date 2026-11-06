// torch - the libtorch backend of Tensor: batched tensors on the CPU or a CUDA device
//
// Differentiable through autograd. Only built with libtorch.
//
// Syntax:   BackendScope scope("torch");   Tensor t = from_torch(x);
//           torch::Tensor x = to_torch(t);
// See also: tensor.h, eigen.h
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <torch/torch.h>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The libtorch backend on `device`, for `backend(spec)` and `BackendScope`. `custom_backward`
/// differentiates the matrix exponential with a hand-written backward pass, not autograd's own.
std::shared_ptr<const Tensor::Backend> torch_backend(const torch::Device &device,
                                                     bool custom_backward);

/// Sets libtorch's intra-op pool to `n` threads, and its inter-op pool too the first time
/// (libtorch allows that once, before any parallel work).
void torch_set_threads(int n);

/// The device named "cpu", "gpu" (CUDA), "cuda" or "cuda:<index>"; anything else is an error.
torch::Device torch_device(const std::string &name);

/// Wraps an existing tensor; `custom_backward` carries over to every tensor made from it.
Tensor from_torch(const torch::Tensor &t, bool custom_backward = false);

/// The tensor inside one made by `from_torch`, or by an operation on one.
torch::Tensor to_torch(const Tensor &t);

/// Whether the tensor belongs to the libtorch backend.
bool is_torch(const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
