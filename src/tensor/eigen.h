// CoraTensor on Eigen: `double` matrices on the CPU, no batch dimensions.

#pragma once

#include "tensor/tensor.h"

#include <Eigen/Dense>

namespace cora::ct {

/// The Eigen backend, for `setBackend`.
std::shared_ptr<const Tensor::Backend> eigenBackend();

Tensor fromEigen(const Eigen::MatrixXd &m);

/// The matrix inside a tensor made by `fromEigen`, or by an operation on one.
Eigen::MatrixXd toEigen(const Tensor &t);

} // namespace cora::ct
