// eigen - the Eigen backend of Tensor: double matrices on the CPU, no batch dimensions
//
// Syntax:   setBackend("eigen");   Tensor t = fromEigen(m);   Eigen::MatrixXd m = toEigen(t);
// See also: tensor.h, torch.h

#pragma once

#include "tensor/tensor.h"

#include <Eigen/Dense>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The Eigen backend, for `setBackend`.
std::shared_ptr<const Tensor::Backend> eigenBackend();

/// Wraps an Eigen matrix as a tensor.
Tensor fromEigen(const Eigen::MatrixXd &m);

/// The matrix inside a tensor made by `fromEigen`, or by an operation on one.
Eigen::MatrixXd toEigen(const Tensor &t);

/// Whether the tensor belongs to the Eigen backend.
bool isEigen(const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
