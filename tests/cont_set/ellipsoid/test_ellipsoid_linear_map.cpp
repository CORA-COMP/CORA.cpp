// test_ellipsoid_linear_map - Ellipsoid linear_map: M Q M' and M q
//
// CORA: contSet/@ellipsoid/mtimes.m
// PROPERTY: contSet/@ellipsoid/mtimes.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a row map, a rotation, the empty ellipsoid takes the rows of M.
void linear_map_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Ellipsoid R = E.linear_map(Tensor({{1.0, 2.0}}));
    check(close(R.Q(), {8.0}, 1e-14) && close(R.q(), {5.0}, 1e-14), b + ": M = (1 2)");
    const Ellipsoid S = E.linear_map(Tensor({{0.0, 1.0}, {1.0, 0.0}}));
    check(close(S.Q(), {1, 0, 0, 4}, 1e-14) && close(S.q(), {2, 1}, 0), b + ": a swap");
    check(Ellipsoid::empty(2).linear_map(Tensor({{1.0, 2.0}})).dim() == 1, b + ": empty");
    check(close(E.linear_map(Tensor::eye(2)).Q(), E.Q(), 0), b + ": the identity");
}

/// Stated property: the image of every point of E is a point of the image, whose Q stays symmetric.
void linear_map_property(const std::string &b) {
    Rng rng(97);
    for (int k = 0; k < 5; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k == 4 ? 2 : -1);
        const Tensor M = test::normal_matrix(rng, 2, 3);
        const Ellipsoid R = E.linear_map(M);
        check(test::all_inside(R, M.matmul(test::ellipsoid_points(E, rng, 30)), 1e-6),
              b + ": the images are inside");
        check(close(R.Q(), R.Q().matrix_transpose(), 0), b + ": Q stays symmetric");
    }
}

/// A matrix of the wrong width is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(test::raises([&] { E.linear_map(Tensor({{1.0, 2.0, 3.0}})); }, Error::dimension_mismatch),
          "linear_map: width");
    ChecksScope off(false);
    check(E.linear_map(Tensor::eye(2)).dim() == 2, "linear_map: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        linear_map_known(b);
        linear_map_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid linear_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
