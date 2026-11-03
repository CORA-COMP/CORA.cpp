// eigen - the Eigen backend of Tensor: float64 or float32 tensors of any shape on the CPU
//
// Syntax:   BackendScope scope("eigen");   Tensor t = from_eigen(m);
//           Eigen::MatrixXd m = to_eigen(t);
// See also: tensor.h, host.h, torch.h

#pragma once

#include "cora/global/tensor/tensor.h"

#include <Eigen/Dense>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The Eigen backend, for `backend(spec)` and `BackendScope`.
std::shared_ptr<const Tensor::Backend> eigen_backend();

/// A copy of an Eigen matrix as a tensor.
Tensor from_eigen(const Eigen::MatrixXd &m);

/// A copy of a float64 matrix (n, m) of a host backend as an Eigen matrix.
Eigen::MatrixXd to_eigen(const Tensor &t);

/// Whether the tensor belongs to the Eigen backend.
bool is_eigen(const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
