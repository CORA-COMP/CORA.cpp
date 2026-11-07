// linear_sys_steps_testing - what the tests of the step operations of linear_sys share: two
//
// small systems with a known solution and the sets that go with them
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "testing.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test::steps {

using namespace cora;

constexpr double kStep = 0.1;
constexpr int kTerms = 10;

/// A = diag(-1, -2): e^{A t} = diag(e^-t, e^-2t), invertible.
inline Tensor diagonal() { return Tensor({{-1.0, 0.0}, {0.0, -2.0}}); }

/// The double integrator A = [0 1; 0 0]: nilpotent, singular, e^{A t} = I + A t.
inline Tensor integrator() { return Tensor({{0.0, 1.0}, {0.0, 0.0}}); }

/// The initial set of the examples: center (1, 2), generators diag(1, 0.5).
inline Zonotope initial_set() {
    return {test::column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}})};
}

/// The point u as a set.
inline Zonotope point(const std::vector<double> &u) {
    return {test::column(u), Tensor::zeros({int64_t(u.size()), 0})};
}

/// The set {0} of the dimension n.
inline Zonotope origin(int64_t n) { return point(std::vector<double>(n, 0.0)); }

/// The support function of Z along d.
inline double support(const Zonotope &Z, const std::vector<double> &d) {
    return test::support(Z, d);
}

/// The state e^{A t} x0 of x' = A x.
inline Tensor flow(const Tensor &A, double t, const Tensor &x0) {
    return (A * t).expm().matmul(x0);
}

} // namespace test::steps

// ---------------------------------------  END OF CODE  ---------------------------------------- //
