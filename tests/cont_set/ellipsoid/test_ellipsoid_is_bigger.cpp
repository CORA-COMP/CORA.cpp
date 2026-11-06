// test_ellipsoid_is_bigger - Ellipsoid::is_bigger: E2 - q2 lies in E - q
//
// CORA: contSet/@ellipsoid/isBigger.m
// PROPERTY: contSet/@ellipsoid/isBigger.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: scaled and rotated shapes, the tolerance, the dimension 1, degenerate sets.
void is_bigger_known(const std::string &b) {
    const Tensor Q = Tensor({{4.0, 0.0}, {0.0, 4.0}});
    const Ellipsoid E(Q), small(Q * 0.25), skewed(Tensor({{5.0, 0.0}, {0.0, 1.0}}));
    check(E.is_bigger(small) && !small.is_bigger(E), b + ": a quarter of the shape is smaller");
    check(E.is_bigger(E), b + ": an ellipsoid is not smaller than itself");
    check(!E.is_bigger(skewed) && !skewed.is_bigger(E), b + ": neither contains the other");
    check(E.is_bigger(Ellipsoid(Q, column({5.0, 5.0}))), b + ": the centers do not matter");
    // the ratio 1 + 5e-7 is within the tolerance 1e-6, but not at the default one
    check(Ellipsoid(Q, 1e-6).is_bigger(Ellipsoid(Q * (1.0 + 5e-7), 1e-6)), b + ": within tol");
    check(!E.is_bigger(Ellipsoid(Q * (1.0 + 5e-7))), b + ": beyond the default tolerance");
    // CORA answers true in dimension 1 without comparing; here the comparison is made
    check(!Ellipsoid(Tensor({{1.0}})).is_bigger(Ellipsoid(Tensor({{4.0}}))),
          b + ": [-1, 1] does not contain [-2, 2]");
    check(Ellipsoid(Tensor({{4.0}})).is_bigger(Ellipsoid(Tensor({{1.0}}))), b + ": dimension 1");
    // degenerate sets: a segment contains a shorter one on its line, and no full-dimensional one
    const Ellipsoid seg(Tensor({{1.0, 0.0}, {0.0, 0.0}})),
        shorter(Tensor({{0.25, 0.0}, {0.0, 0.0}}));
    check(seg.is_bigger(shorter) && !shorter.is_bigger(seg), b + ": segments");
    check(!seg.is_bigger(E) && E.is_bigger(seg), b + ": a segment is smaller than a ball");
    check(Ellipsoid(Tensor::zeros({2, 2})).is_bigger(Ellipsoid(Tensor::zeros({2, 2}))),
          b + ": two points");
}

/// Stated property: E - q contains E2 - q2 exactly when the boundary points of E2 are in E.
void is_bigger_property(const std::string &b) {
    cora::Rng rng(29);
    for (int k = 0; k < 10; ++k) {
        const Ellipsoid E1 = test::random_ellipsoid(rng, 3), E2 = test::random_ellipsoid(rng, 3);
        const Ellipsoid E1c(E1.Q()), E2c(E2.Q());
        // a shape that is certainly inside: a third of E1 shifted to the center of E2
        const Ellipsoid inner(E1.Q() * 0.3, E2.q());
        check(E1.is_bigger(inner) && !inner.is_bigger(E1), b + ": a shrunk copy is inside");
        const bool res = E1.is_bigger(E2);
        const Tensor P = test::ellipsoid_points(E2c, rng, 40, true);
        const bool inside = test::all_inside(E1c, P, 1e-6);
        check(!res || inside, b + ": a true answer holds for the boundary points");
        // a scaled copy that is larger sticks out of E1 at its boundary points
        const Ellipsoid outer(E1.Q() * 1.5);
        check(!E1.is_bigger(outer) &&
                  !test::all_inside(E1c, test::ellipsoid_points(outer, rng, 5, true)),
              b + ": a larger copy is not inside");
    }
}

/// A different dimension, an empty ellipsoid and a batch are refused; with the checks off the
/// answer is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 4.0}})), line(Tensor({{1.0}}));
    check(test::raises([&] { E.is_bigger(line); }, Error::dimension_mismatch),
          "is_bigger: the dimensions agree");
    check(test::raises([&] { E.is_bigger(Ellipsoid::empty(2)); }, Error::empty_set),
          "is_bigger: an empty ellipsoid");
    check(test::raises([&] { Ellipsoid(Tensor::zeros({2, 2, 2})).is_bigger(E); },
                       Error::not_supported),
          "is_bigger: a batch is not supported");
    const bool on = E.is_bigger(E);
    ChecksScope off(false);
    check(E.is_bigger(E) == on, "is_bigger: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        is_bigger_known(b);
        is_bigger_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid is_bigger");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
