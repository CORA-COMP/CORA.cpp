// CoraTensor on Eigen: `double` matrices on the CPU, no batch dimensions.

#pragma once

#include "tensor/tensor.h"

#include <Eigen/Dense>

namespace cora::ct {

/// The Eigen backend, for `set_backend`.
std::shared_ptr<const Tensor::Backend> eigen_backend();

Tensor from_eigen(const Eigen::MatrixXd &m);

/// The matrix inside a tensor made by `from_eigen`, or by an operation on one.
Eigen::MatrixXd to_eigen(const Tensor &t);

} // namespace cora::ct
