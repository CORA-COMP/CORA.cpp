// eigen - the Eigen backend of Tensor: double matrices on the CPU, no batch dimensions
//
// Syntax:   BackendScope scope("eigen");   Tensor t = from_eigen(m);
//           Eigen::MatrixXd m = to_eigen(t);
// See also: tensor.h, torch.h

#pragma once

#include "cora/global/tensor/tensor.h"

#include <Eigen/Dense>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The Eigen backend, for `backend(spec)` and `BackendScope`.
std::shared_ptr<const Tensor::Backend> eigen_backend();

/// Wraps an Eigen matrix as a tensor.
Tensor from_eigen(const Eigen::MatrixXd &m);

/// The matrix inside a tensor made by `from_eigen`, or by an operation on one.
Eigen::MatrixXd to_eigen(const Tensor &t);

/// Whether the tensor belongs to the Eigen backend.
bool is_eigen(const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
