// test_level_set_intersect - LevelSet intersect: the level set with a polynomial zonotope
//
// The second-order expansion of the equation solved for the state of the largest derivative
// (Kochdumper et al., HSCC 2020). The parabola x2 + x1^2 = 0 on [-1, 1] x [-2, 2] by hand (the
// sign of the quadratic term), the stated property (the solutions in the box stay) on random
// circles, a point, an empty contraction, a flat equation, and the arguments.
//
// CORA: contSet/@levelSet/and_.m
// PROPERTY: contSet/@levelSet/and_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "level_set_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: every point of a random circle in the box of the set is in the result.
void intersect_property(const std::string &b) {
    Rng rng(17);
    for (int k = 0; k < 10; ++k) {
        std::vector<double> p(4);
        rng.uniform(p.data(), 4, 0.5, 1.5);
        const double r = 0.3 * p[2];
        const Interval I = test::box({p[0] - 0.5, p[1] - 0.5}, {p[0] + 0.5, p[1] + 0.5});
        const LevelSet ls(test::circle(1.0, 1.0, r), 2, CompOp::eq);
        const std::optional<PolyZonotope> R = ls.intersect(PolyZonotope(I));
        const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
        for (int s = 0; s < 200; ++s) {
            const double x = 1.0 + r * std::cos(0.0314 * s), y = 1.0 + r * std::sin(0.0314 * s);
            if (x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1]) continue;
            check(R.has_value() && R->interval().contains(column({x, y}), 1e-9),
                  b + ": a solution stays in the result");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // x2 + x1^2 = 0 on [-1, 1] x [-2, 2]: x2 = -x1^2, in [-1, 0]
        const Expr x1 = Expr::var(0), x2 = Expr::var(1);
        const LevelSet parabola(x2 + x1 * x1, 2, CompOp::eq);
        const auto R = parabola.intersect(PolyZonotope(test::box({-1, -2}, {1, 2})));
        const Interval J = R->interval();
        check(R && close(J.inf(), column({-1.0, -1.0}), 1e-12) &&
                  close(J.sup(), column({1.0, 0.0}), 1e-12),
              b + ": the parabola x2 = -x1^2");
        // the same set with the other sign of the constraint
        const LevelSet flipped(-(x2 + x1 * x1), 2, CompOp::eq);
        const Interval K = flipped.intersect(PolyZonotope(test::box({-1, -2}, {1, 2})))->interval();
        check(close(K.inf(), J.inf(), 1e-12) && close(K.sup(), J.sup(), 1e-12),
              b + ": the sign of f does not matter");
        // a point on the level set stays, one off it is empty
        const LevelSet line(x1 - 1.0, 2, CompOp::eq);
        const auto pt = [&](double x) {
            return PolyZonotope(Zonotope(column({x, 0.0}), Tensor::zeros({2, 0})));
        };
        check(line.intersect(pt(1.0)).has_value(), b + ": a point on the set");
        check(!line.intersect(pt(2.0)).has_value(), b + ": a point beyond the set");
        // a contraction without a point: the circle misses the box
        const LevelSet circle(test::circle(0, 0, 1), 2, CompOp::eq);
        check(!circle.intersect(PolyZonotope(test::box({3, 3}, {4, 4}))).has_value(),
              b + ": a circle that misses the box");
        // a flat equation (x1^2 = 1 has no slope at the center): the box itself
        const LevelSet flat(pow(Expr::var(0), 2) - 1.0, 1, CompOp::eq);
        const Interval F = flat.intersect(PolyZonotope(test::box({-1}, {1})))->interval();
        check(close(F.inf(), column({-1.0})) && close(F.sup(), column({1.0})),
              b + ": a flat equation");
        intersect_property(b);
    });
    // the unsupported constraints and the dimension
    const LevelSet ineq(Expr::var(0), 1, CompOp::le);
    check(test::raises([&] { ineq.intersect(PolyZonotope(test::box({0}, {1}))); },
                       Error::not_supported),
          "intersect: inequalities");
    check(test::raises(
              [&] {
                  LevelSet(Expr::var(0) - 1.0, 1, CompOp::eq)
                      .intersect(PolyZonotope(test::box({0, 0}, {1, 1})));
              },
              Error::dimension_mismatch),
          "intersect: dimension");
    ChecksScope off(false);
    check(LevelSet(Expr::var(0) - 0.5, 1, CompOp::eq)
              .intersect(PolyZonotope(test::box({0}, {1})))
              .has_value(),
          "intersect: valid, checks off");
    return test::finish("level_set intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
