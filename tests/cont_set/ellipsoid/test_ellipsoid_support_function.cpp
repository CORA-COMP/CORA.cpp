// test_ellipsoid_support_function - Ellipsoid support_function: d'q + sqrt(d'Qd)
//
// CORA: contSet/@ellipsoid/supportFunc_.m
// PROPERTY: contSet/@ellipsoid/supportFunc_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "ellipsoid_testing.h"

#include <cmath>
#include <limits>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed along the axes and a diagonal; a point; the empty ellipsoid has -Inf.
void support_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(close(E.support_function(column({1.0, 0.0})), {3.0}, 1e-14), b + ": d = (1, 0)");
    check(close(E.support_function(column({0.0, 1.0})), {3.0}, 1e-14), b + ": d = (0, 1)");
    check(close(E.support_function(column({1.0, 1.0})), {3.0 + std::sqrt(5.0)}, 1e-14),
          b + ": d = (1, 1)");
    check(close(Ellipsoid(Tensor::zeros({2, 2}), column({1.0, 2.0}))
                    .support_function(column({2.0, 1.0})),
                {4.0}, 0),
          b + ": a point");
    check(Ellipsoid::empty(2).support_function(column({1.0, 0.0})).item() ==
              -std::numeric_limits<double>::infinity(),
          b + ": empty");
    const Ellipsoid batch(Tensor::stack({Tensor({{4.0}}), Tensor({{9.0}})}, 0),
                          Tensor::stack({column({0.0}), column({1.0})}, 0));
    check(close(batch.support_function(column({1.0})),
                Tensor::stack({Tensor({{2.0}}), Tensor({{4.0}})}, 0), 0),
          b + ": a batch");
}

/// Stated property: no boundary point goes beyond it, and q + Q d / sqrt(d'Qd) attains it.
void support_property(const std::string &b) {
    Rng rng(89);
    for (int k = 0; k < 6; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k == 5 ? 2 : -1);
        const std::vector<double> d = test::random_direction(rng, 3);
        const double value = test::support(E, d);
        const Tensor P = test::ellipsoid_points(E, rng, 40, true);
        check(column(d).matrix_transpose().matmul(P).max().item() <= value + 1e-9,
              b + ": the points stay below");
        const Tensor Qd = E.Q().matmul(column(d));
        const double spread = std::sqrt(column(d).matrix_transpose().matmul(Qd).item());
        check(close(column(d).matrix_transpose().matmul(E.q() + Qd / spread).item(), value, 1e-9),
              b + ": the tip attains it");
    }
}

/// A direction of the wrong size is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(test::raises([&] { E.support_function(column({1.0})); }, Error::dimension_mismatch),
          "support_function: dimension");
    ChecksScope off(false);
    check(close(E.support_function(column({1.0, 0.0})), {3.0}, 1e-14),
          "support_function: same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        support_known(b);
        support_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
