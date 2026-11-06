// linear_sys_testing - what the linear_sys tests share: the reference systems as tensors, the
//
// conversions between host (Eigen) matrices and tensors on the current backend, and the reach
// algorithms by name.
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/linear_sys/matlab_reference.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "testing.h"

#include <unsupported/Eigen/MatrixFunctions>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test::lin {

using namespace cora;
using matlab_reference::System;

/// The reach algorithms of a linear system, by their names.
inline const std::vector<std::string> kAlgorithms = {"standard", "wrapping_free"};

inline std::string name(const std::string &a) { return a; }

/// The algorithm `a` ("standard" or "wrapping_free") with these options.
inline alg::LinearReach algorithm(const std::string &a, double time_step, int taylor_terms,
                                  int zonotope_order = 50) {
    if (a == "standard")
        return alg::Standard{
            .time_step = time_step, .taylor_terms = taylor_terms, .zonotope_order = zonotope_order};
    return alg::WrappingFree{.time_step = time_step, .taylor_terms = taylor_terms};
}

inline Zonotope set_of(const System &s) {
    return {Tensor::from_data(s.c, {s.n, 1}), Tensor::from_data(s.G, {s.n, s.m})};
}

inline Tensor system_matrix(const System &s) { return Tensor::from_data(s.A, {s.n, s.n}); }

inline double support(const Zonotope &Z, const Eigen::VectorXd &d) {
    return Z.support_function(Tensor::from_data({d.data(), d.data() + d.size()}, {d.size(), 1}))
        .data()[0];
}

/// A host matrix as a tensor on the current backend.
inline Tensor tensor_of(const Eigen::MatrixXd &M) {
    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> rows = M;
    return Tensor::from_data({rows.data(), rows.data() + rows.size()}, {M.rows(), M.cols()});
}

inline Eigen::MatrixXd host_matrix(const std::vector<double> &row_major, Eigen::Index rows,
                                   Eigen::Index cols) {
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

// ---------------------------------------  END OF CODE  ---------------------------------------- //
