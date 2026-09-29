// linearSysTesting - what the linearSys tests share: the reference systems as tensors, and the
// conversions between host (Eigen) matrices and tensors on the current backend.

#pragma once

#include "contDynamics/linearSys/linearSys.h"
#include "contDynamics/linearSys/matlabReference.h"
#include "testing.h"

#include <unsupported/Eigen/MatrixFunctions>

namespace test::lin {

using namespace cora::ct;
using matlab_reference::System;

const Algorithm kAlgorithms[] = {Algorithm::Standard, Algorithm::WrappingFree};

inline std::string name(Algorithm a) {
    return a == Algorithm::Standard ? "standard" : "wrapping-free";
}

inline Zonotope set_of(const System &s) {
    return {Tensor::fromData(s.c, {s.n, 1}), Tensor::fromData(s.G, {s.n, s.m})};
}

inline Tensor system_matrix(const System &s) { return Tensor::fromData(s.A, {s.n, s.n}); }

inline double support(const Zonotope &Z, const Eigen::VectorXd &d) {
    return Z.supportFunc(Tensor::fromData({d.data(), d.data() + d.size()}, {d.size(), 1}))
        .data()[0];
}

/// A host matrix as a tensor on the current backend.
inline Tensor tensor_of(const Eigen::MatrixXd &M) {
    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> rows = M;
    return Tensor::fromData({rows.data(), rows.data() + rows.size()}, {M.rows(), M.cols()});
}

inline Eigen::MatrixXd host_matrix(const std::vector<double> &row_major, int rows, int cols) {
    return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(row_major.data(), rows,
                                                                            cols);
}

inline Eigen::MatrixXd host_of(const Tensor &t) {
    const std::vector<int64_t> s = t.shape();
    const std::vector<double> v = t.data();
    return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(v.data(), s[0], s[1]);
}

inline Eigen::MatrixXd oscillator3() {
    Eigen::MatrixXd A(3, 3);
    A << -0.3, 1.0, 0.0, -1.0, -0.2, 0.4, 0.0, -0.5, -0.1;
    return A;
}

} // namespace test::lin
