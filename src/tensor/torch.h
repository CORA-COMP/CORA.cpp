// torch - the libtorch backend of Tensor: batched tensors on the CPU or a CUDA device
//
// Differentiable through autograd. Only built with libtorch.
//
// Syntax:   setBackend("torch");   Tensor t = fromTorch(x);   torch::Tensor x = toTorch(t);
// See also: tensor.h, eigen.h

#pragma once

#include "tensor/tensor.h"

#include <torch/torch.h>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

/// The libtorch backend on `device`, for `setBackend`. `customBackward` differentiates
/// the matrix exponential with a hand-written backward pass instead of autograd's own.
std::shared_ptr<const Tensor::Backend> torchBackend(const torch::Device &device,
                                                     bool customBackward);

/// The device named "cpu", "gpu" (CUDA), "cuda" or "cuda:<index>"; anything else is an error.
torch::Device torchDevice(const std::string &name);

/// Wraps an existing tensor; `customBackward` carries over to every tensor made from it.
Tensor fromTorch(const torch::Tensor &t, bool customBackward = false);

/// The tensor inside one made by `fromTorch`, or by an operation on one.
torch::Tensor toTorch(const Tensor &t);

/// Whether the tensor belongs to the libtorch backend.
bool isTorch(const Tensor &t);

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
