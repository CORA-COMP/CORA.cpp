// test_ellipsoid_contains - Ellipsoid::contains: points, a zonotope, a box and trivial ellipsoids
//
// CORA: contSet/@ellipsoid/contains_.m
// PROPERTY: contSet/@ellipsoid/contains_.m

#include "ellipsoid_testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

namespace {

/// Hand-computed with E = {(1, 2) + (x, y) : x^2/4 + y^2 <= 1}.
void contains_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(E.contains(column({3.0, 2.0})) && E.contains(column({1.0, 2.0})),
          b + ": the boundary and the center");
    check(!E.contains(column({3.0, 3.0})), b + ": a point outside");
    check(E.contains(Tensor({{1.0, 2.0}, {2.0, 2.0}})) &&
              !E.contains(Tensor({{1.0, 9.0}, {2.0, 2.0}})),
          b + ": all points must be inside");
    check(E.contains(column({3.0000001, 2.0}), "exact", 1e-6), b + ": the tolerance admits a hair");
    check(!E.contains(column({3.0000001, 2.0}), "exact", 0.0), b + ": no tolerance, no hair");
    // zonotopes and boxes
    const Zonotope inside(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}}));
    const Zonotope sticks_out(column({1.0, 2.0}), Tensor({{2.5, 0.0}, {0.0, 0.0}}));
    check(E.contains(inside) && !E.contains(sticks_out), b + ": zonotopes");
    const Interval box(column({0.0, 1.5}), column({2.0, 2.5})),
        wide(column({-4.0, 0.0}), column({6.0, 4.0}));
    check(E.contains(box) && !E.contains(wide),
          b + ": boxes (the corners of [0, 2] x [1.5, 2.5] are in)");
    const double inf = std::numeric_limits<double>::infinity();
    check(!E.contains(Interval(column({0.0, 0.0}), column({inf, 1.0}))), b + ": an unbounded box");
    check(E.contains(Interval::empty(2)), b + ": the empty box");
    check(E.contains(Zonotope(column({1.0, 2.0}), Tensor::zeros({2, 0}))),
          b + ": a zonotope that is a point");
    // ellipsoids that are points, empty, or contained trivially
    const Ellipsoid point(Tensor::zeros({2, 2}), column({1.0, 2.0}));
    check(point.contains(column({1.0, 2.0})) && !point.contains(column({1.0, 2.1})),
          b + ": E a point");
    check(point.contains(Ellipsoid(Tensor::zeros({2, 2}), column({1.0, 2.0}))) &&
              !point.contains(E) && point.contains(Ellipsoid::empty(2)),
          b + ": a point contains the same point and the empty set");
    check(!Ellipsoid::empty(2).contains(column({0.0, 0.0})) &&
              Ellipsoid::empty(2).contains(Ellipsoid::empty(2)),
          b + ": the empty ellipsoid contains nothing but the empty set");
    check(E.contains(point) && E.contains(Ellipsoid::empty(2)) &&
              !E.contains(Ellipsoid(Tensor::zeros({2, 2}), column({9.0, 9.0}))),
          b + ": E contains a point, the empty set, not a far point");
    // a degenerate E contains points of its segment only
    const Ellipsoid segment(Tensor({{1.0, 0.0}, {0.0, 0.0}}));
    check(segment.contains(column({0.5, 0.0})) && !segment.contains(column({0.5, 0.1})),
          b + ": a segment");
}

/// Stated property: the search over the generators agrees with all 2^m vertices, also in 3D where
/// CORA enumerates; the points sampled in the set are inside exactly when its vertices are.
void contains_property(const std::string &b) {
    cora::Rng rng(37);
    for (int k = 0; k < 12; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k % 4 == 3 ? 2 : -1);
        const double scale = 0.15 + 0.1 * k;
        const Zonotope Z(E.q() + test::normal_matrix(rng, 3, 1, 0.1),
                         test::normal_matrix(rng, 3, 4, scale * 0.3));
        bool all = true;
        for (int mask = 0; mask < 16; ++mask) {
            Tensor v = Z.center();
            for (int j = 0; j < 4; ++j)
                v = v + Z.generators().slice(1, j, j + 1) * ((mask >> j) & 1 ? 1.0 : -1.0);
            all = all && E.contains(v);
        }
        check(E.contains(Z) == all, b + ": the search agrees with the vertices");
        const Interval I(Z.center() - Z.generators().abs().sum(1, true) * 0.5,
                         Z.center() + Z.generators().abs().sum(1, true) * 0.5);
        bool corners = true;
        for (int mask = 0; mask < 8; ++mask) {
            std::vector<double> sel;
            for (int j = 0; j < 3; ++j) sel.push_back(((mask >> j) & 1) ? 1.0 : 0.0);
            const Tensor w = Tensor::like(I.inf(), sel, {3, 1});
            corners = corners && E.contains(I.inf() + (I.sup() - I.inf()).multiply(w));
        }
        check(E.contains(I) == corners, b + ": a box agrees with its corners");
    }
}

/// The method, the dimension, a batch and an ellipsoid that needs the SDP are refused; with the
/// checks off the answer is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    const Zonotope Z(column({0.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    check(test::raises([&] { E.contains(column({0.0, 0.0}), "sloppy"); }, Error::wrong_value),
          "contains: an unknown method");
    check(test::raises([&] { E.contains(Z, "approx"); }, Error::not_supported),
          "contains: approx needs an SDP solver");
    check(E.contains(column({0.0, 0.0}), "approx"),
          "contains: points are exact whatever the method");
    check(test::raises([&] { E.contains(column({0.0})); }, Error::dimension_mismatch),
          "contains: the points have the dimension of E");
    check(test::raises([&] { E.contains(Tensor::zeros({2, 2, 2})); }, Error::wrong_value),
          "contains: the points are a matrix");
    check(test::raises([&] { E.contains(Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "contains: the zonotope has the dimension of E");
    check(test::raises([&] { Ellipsoid(Tensor::zeros({2, 2, 2})).contains(column({0.0, 0.0})); },
                       Error::not_supported),
          "contains: a batch is not supported");
    check(test::raises([&] { E.contains(Ellipsoid(Tensor({{0.25, 0.0}, {0.0, 0.25}}))); },
                       Error::not_supported),
          "contains: an ellipsoid in an ellipsoid needs an SDP solver");
    check(test::raises([&] { E.contains(column({0.0, 0.0}), "exact", -1.0); }, Error::wrong_value),
          "contains: the tolerance is nonnegative");
    const bool on = E.contains(Z);
    ChecksScope off(false);
    check(E.contains(Z) == on, "contains: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        contains_known(b);
        contains_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid contains");
}
