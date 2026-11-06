// test_ellipsoid_convex_hull - Ellipsoid convex_hull: the hull is unite in mode outer
//
// CORA: contSet/@ellipsoid/convHull_.m
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

/// A point inside and an empty operand give the ellipsoid; the registry finds the pair.
void convex_hull_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Ellipsoid inside(Tensor::zeros({2, 2}), column({1.0, 2.0}));
    check(E.convex_hull(inside).isequal(E) && E.convex_hull(Ellipsoid::empty(2)).isequal(E),
          b + ": E");
    check(convex_hull(E, inside).is<Ellipsoid>(), b + ": registry");
}

/// Two proper ellipsoids need an SDP and a dimension mismatch is refused; checks off works.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(test::raises([&] { E.convex_hull(E); }, Error::not_supported),
          "convex_hull: two proper ones");
    check(
        test::raises([&] { E.convex_hull(Ellipsoid(Tensor::eye(3))); }, Error::dimension_mismatch),
        "convex_hull: dimension");
    ChecksScope off(false);
    check(E.convex_hull(Ellipsoid::empty(2)).isequal(E), "convex_hull: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { convex_hull_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
