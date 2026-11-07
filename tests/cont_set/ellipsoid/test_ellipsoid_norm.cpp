// test_ellipsoid_norm - Ellipsoid norm: the root of the largest eigenvalue of Q
//
// CORA: contSet/@ellipsoid/norm_.m
// PROPERTY: contSet/@ellipsoid/norm_.m
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

/// Hand-computed: the largest semi-axis; the empty ellipsoid has -Inf.
void norm_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(close(E.norm(), {2.0}, 1e-14), b + ": 2");
    check(close(Ellipsoid(Tensor({{2.0, 1.0}, {1.0, 2.0}})).norm(), {std::sqrt(3.0)}, 1e-14),
          b + ": a rotated one");
    check(E.norm(Tensor::Norm::two, "exact").item() == E.norm().item(), b + ": modes agree");
    check(Ellipsoid::empty(2).norm().item() == -std::numeric_limits<double>::infinity(),
          b + ": empty");
}

/// Stated property: no point of the ellipsoid is longer, and the longest semi-axis attains it.
void norm_property(const std::string &b) {
    Rng rng(103);
    for (int k = 0; k < 5; ++k) {
        const Ellipsoid E(test::random_ellipsoid(rng, 3).Q());
        const double value = E.norm().item();
        check(test::ellipsoid_points(E, rng, 40, true).square().sum(0).sqrt().max().item() <=
                  value + 1e-9,
              b + ": the points stay below");
        check(close(E.radius().item(), value, 1e-9), b + ": the largest radius");
    }
}

/// Another norm and a nonzero center are refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    check(test::raises([] { Ellipsoid(Tensor::eye(2)).norm(Tensor::Norm::one); },
                       Error::no_exact_alg),
          "norm: type");
    check(test::raises([] { Ellipsoid(Tensor::eye(2), column({1.0, 0.0})).norm(); },
                       Error::not_supported),
          "norm: center");
    check(test::raises([] { Ellipsoid(Tensor::eye(2)).norm(Tensor::Norm::two, "best"); },
                       Error::wrong_value),
          "norm: mode");
    ChecksScope off(false);
    check(close(Ellipsoid(Tensor::eye(2)).norm(), {1.0}, 1e-14), "norm: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        norm_known(b);
        norm_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
