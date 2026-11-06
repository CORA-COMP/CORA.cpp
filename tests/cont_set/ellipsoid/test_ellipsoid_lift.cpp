// test_ellipsoid_lift - Ellipsoid lift: the permutation of the coordinates
//
// CORA: contSet/@ellipsoid/lift_.m
// PROPERTY: contSet/@ellipsoid/lift_.m
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

/// Hand-computed: a permutation by lift, the identity, the empty ellipsoid.
void lift_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 1.0, 0.0}, {1.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}),
                      column({1.0, 2.0, 3.0}));
    const Ellipsoid L = E.lift(3, {1, 2, 0});
    check(close(L.q(), {3, 1, 2}, 0) && close(L.Q(), {3, 0, 0, 0, 4, 1, 0, 1, 2}, 0),
          b + ": lift permutes");
    check(close(E.lift(3, {0, 1, 2}).Q(), E.Q(), 0), b + ": the identity permutation");
    check(Ellipsoid::empty(3).lift(3, {1, 2, 0}).dim() == 3, b + ": empty");
}

/// Stated property: lifted points are the permuted points.
void lift_property(const std::string &b) {
    Rng rng(103);
    const Ellipsoid E = test::random_ellipsoid(rng, 3);
    const Ellipsoid L = E.lift(3, {1, 2, 0});
    check(test::all_inside(L, test::ellipsoid_points(E, rng, 30).take({2, 0, 1}, 0), 1e-6),
          b + ": the permuted points are inside");
}

/// More dimensions and a non-permutation are refused; with the checks off a valid call works.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor::eye(3), column({1.0, 2.0, 3.0}));
    check(test::raises([&] { E.lift(4, {0, 1, 2}); }, Error::not_defined), "lift: more dimensions");
    check(test::raises([&] { E.lift(3, {0, 0, 2}); }, Error::wrong_value), "lift: no permutation");
    ChecksScope off(false);
    check(E.lift(3, {1, 0, 2}).dim() == 3, "lift: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        lift_known(b);
        lift_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
