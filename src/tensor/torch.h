// CoraTensor on libtorch: tensors with leading batch dimensions on the CPU or a CUDA
// device, differentiable through autograd. Only built with libtorch.

#pragma once

#include "tensor/tensor.h"

#include <torch/torch.h>

namespace cora::ct {

/// The libtorch backend on `device`, for `set_backend`. `custom_backward` differentiates
/// the matrix exponential with a hand-written backward pass instead of autograd's own.
std::shared_ptr<const Tensor::Backend> torch_backend(const torch::Device &device,
                                                     bool custom_backward);

/// Wraps an existing tensor; `custom_backward` carries over to every tensor made from it.
Tensor from_torch(const torch::Tensor &t, bool custom_backward = false);

/// The tensor inside one made by `from_torch`, or by an operation on one.
torch::Tensor to_torch(const Tensor &t);

} // namespace cora::ct
