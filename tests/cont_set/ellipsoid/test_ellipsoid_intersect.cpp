// test_ellipsoid_intersect - Ellipsoid::intersect: with a point, and the ellipsoids that are points
//
// CORA: contSet/@ellipsoid/and_.m
// PROPERTY: contSet/@ellipsoid/and_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a point in E is the result, a point outside gives the empty ellipsoid; an
/// ellipsoid that is a point, an empty one, and the cases that need a solver.
void intersect_known(const std::string &b) {
    const Ellipsoid E(vec({1.0, 2.0}), Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    const Ellipsoid in = E.intersect(vec({2.0, 2.5}));
    check(close(in.Q(), Tensor::zeros({2, 2}), 0) && close(in.q(), {2, 2.5}, 0),
          b + ": a point in E is the result, as an ellipsoid with Q = 0");
    check(E.intersect(vec({5.0, 2.0})).is_empty(), b + ": a point outside E: empty");
    check(E.intersect(vec({3.0, 2.0})).is_empty() ||
              close(E.intersect(vec({3.0, 2.0})).q(), {3, 2}, 0),
          b + ": a point on the boundary: the point, up to the machine epsilon");
    // an ellipsoid that is a point: itself if the other holds it, else nothing
    const Ellipsoid point(vec({1.0, 2.5}), Tensor::zeros({2, 2}));
    check(close(point.intersect(E).q(), {1, 2.5}, 0), b + ": a point and an ellipsoid holding it");
    check(close(E.intersect(point).q(), {1, 2.5}, 0), b + ": an ellipsoid and a point in it");
    const Ellipsoid far(vec({9.0, 9.0}), Tensor::zeros({2, 2}));
    check(far.intersect(E).is_empty() && E.intersect(far).is_empty(), b + ": a point outside");
    check(E.intersect(Ellipsoid::empty(2)).is_empty() &&
              Ellipsoid::empty(2).intersect(E).is_empty(),
          b + ": with the empty ellipsoid");
    check(E.intersect(point, "inner").q().numel() == 2, b + ": the mode 'inner' for a point");
    // the intersection of two ellipsoids that are no points needs the solvers of CORA
    check(test::raises([&] { E.intersect(Ellipsoid(Tensor({{1.0, 0.0}, {0.0, 1.0}}))); },
                       Error::not_supported),
          b + ": two proper ellipsoids need a solver");
}

/// Stated property: the result is a subset of both operands, and a point of E stays.
void intersect_property(const std::string &b) {
    cora::Rng rng(19);
    for (int k = 0; k < 6; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3);
        // a point at the machine epsilon of the boundary is not reliably in, as in CORA
        const Tensor inside = test::ellipsoid_points(E, rng, 1).squeeze(0);
        const Ellipsoid kept = E.intersect(inside);
        check(!kept.is_empty() && close(kept.q(), inside, 0), b + ": a point of E stays");
        const Tensor boundary = test::ellipsoid_points(E, rng, 1, true).squeeze(0);
        const Tensor outside = E.q() + (boundary - E.q()) * 3.0;
        check(E.intersect(outside).is_empty(),
              b + ": a boundary point seen from three times as far is out");
    }
}

/// A point of the wrong shape, a mode, a dimension and a batch are refused; with the checks off
/// the call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { E.intersect(vec({1.0})); }, Error::dimension_mismatch),
          "intersect: the point has the dimension of E");
    check(test::raises([&] { E.intersect(Ellipsoid(Tensor({{1.0}}))); }, Error::dimension_mismatch),
          "intersect: the ellipsoids agree in dimension");
    check(test::raises([&] { E.intersect(E, "exact"); }, Error::wrong_value),
          "intersect: the mode is outer or inner");
    check(test::raises([&] { Ellipsoid(Tensor::zeros({2, 2, 2})).intersect(vec({0.0, 0.0})); },
                       Error::not_supported),
          "intersect: a batch is not supported");
    const Tensor p = vec({2.0, 2.0});
    const Ellipsoid on = E.intersect(p);
    ChecksScope off(false);
    check(close(E.intersect(p).q(), on.q(), 0), "intersect: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        intersect_known(b);
        intersect_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
