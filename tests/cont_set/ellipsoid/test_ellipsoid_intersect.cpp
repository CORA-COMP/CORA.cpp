// test_ellipsoid_intersect - Ellipsoid::intersect: with a point, and the ellipsoids that are points
//
// CORA: contSet/@ellipsoid/and_.m
// PROPERTY: contSet/@ellipsoid/and_.m

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: a point in E is the result, a point outside gives the empty ellipsoid; an
/// ellipsoid that is a point, an empty one, and the cases that need a solver.
void intersect_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Ellipsoid in = E.intersect(column({2.0, 2.5}));
    check(close(in.Q(), Tensor::zeros({2, 2}), 0) && close(in.q(), {2, 2.5}, 0),
          b + ": a point in E is the result, as an ellipsoid with Q = 0");
    check(E.intersect(column({5.0, 2.0})).Q().numel() == 0, b + ": a point outside E: empty");
    check(E.intersect(column({3.0, 2.0})).Q().numel() == 0 ||
              close(E.intersect(column({3.0, 2.0})).q(), {3, 2}, 0),
          b + ": a point on the boundary: the point, up to the machine epsilon");
    // an ellipsoid that is a point: itself if the other holds it, else nothing
    const Ellipsoid point(Tensor::zeros({2, 2}), column({1.0, 2.5}));
    check(close(point.intersect(E).q(), {1, 2.5}, 0), b + ": a point and an ellipsoid holding it");
    check(close(E.intersect(point).q(), {1, 2.5}, 0), b + ": an ellipsoid and a point in it");
    const Ellipsoid far(Tensor::zeros({2, 2}), column({9.0, 9.0}));
    check(far.intersect(E).Q().numel() == 0 && E.intersect(far).Q().numel() == 0,
          b + ": a point outside");
    check(E.intersect(Ellipsoid::empty(2)).Q().numel() == 0 &&
              Ellipsoid::empty(2).intersect(E).Q().numel() == 0,
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
        const Tensor inside = test::ellipsoid_points(E, rng, 1);
        const Ellipsoid kept = E.intersect(inside);
        check(kept.Q().numel() > 0 && close(kept.q(), inside, 0), b + ": a point of E stays");
        const Tensor boundary = test::ellipsoid_points(E, rng, 1, true);
        const Tensor outside = E.q() + (boundary - E.q()) * 3.0;
        check(E.intersect(outside).Q().numel() == 0,
              b + ": a boundary point seen from three times as far is out");
    }
}

/// A point of the wrong shape, a mode, a dimension and a batch are refused; with the checks off
/// the call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { E.intersect(column({1.0})); }, Error::dimension_mismatch),
          "intersect: the point has the dimension of E");
    check(test::raises([&] { E.intersect(Ellipsoid(Tensor({{1.0}}))); }, Error::dimension_mismatch),
          "intersect: the ellipsoids agree in dimension");
    check(test::raises([&] { E.intersect(E, "exact"); }, Error::wrong_value),
          "intersect: the mode is outer or inner");
    check(test::raises([&] { Ellipsoid(Tensor::zeros({2, 2, 2})).intersect(column({0.0, 0.0})); },
                       Error::not_supported),
          "intersect: a batch is not supported");
    const Tensor p = column({2.0, 2.0});
    const Ellipsoid on = E.intersect(p);
    ChecksScope off(false);
    check(close(E.intersect(p).q(), on.q(), 0), "intersect: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        intersect_known(b);
        intersect_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid intersect");
}
