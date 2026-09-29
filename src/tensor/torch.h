// CoraTensor on libtorch: tensors with leading batch dimensions on the CPU or a CUDA
// device, differentiable through autograd. Only built with libtorch.

#pragma once

#include "tensor/tensor.h"

#include <torch/torch.h>

namespace cora::ct {

/// The libtorch backend on `device`, for `setBackend`. `customBackward` differentiates
/// the matrix exponential with a hand-written backward pass instead of autograd's own.
std::shared_ptr<const Tensor::Backend> torchBackend(const torch::Device &device,
                                                     bool customBackward);

/// Wraps an existing tensor; `customBackward` carries over to every tensor made from it.
Tensor fromTorch(const torch::Tensor &t, bool customBackward = false);

/// The tensor inside one made by `fromTorch`, or by an operation on one.
torch::Tensor toTorch(const Tensor &t);

} // namespace cora::ct
