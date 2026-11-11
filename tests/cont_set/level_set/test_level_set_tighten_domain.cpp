// test_level_set_tighten_domain - LevelSet tighten_domain: the forward-backward contractor
//
// CORA: contSet/@levelSet/tightenDomain.m
// PROPERTY: contSet/@levelSet/tightenDomain.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
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

/// Stated property: every point of the circle (x - a)^2 + (y - b)^2 = r^2 in the box is in the
/// contracted box, for random circles and boxes.
void contractor_property(const std::string &name) {
    Rng rng(17);
    for (int k = 0; k < 20; ++k) {
        std::vector<double> p(5), q(2);
        rng.uniform(p.data(), 3, -1.0, 1.0);
        rng.uniform(p.data() + 3, 2, 0.2, 2.0);
        rng.uniform(q.data(), 2, -2.0, 0.0);
        const double a = p[0], bb = p[1], r = p[3];
        const Interval I = test::box({q[0], q[1]}, {q[0] + p[4] + 1, q[1] + p[3] + 1});
        const LevelSet ls(test::circle(a, bb, r), 2, CompOp::eq);
        const std::optional<Interval> J = ls.tighten_domain(I);
        const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
        // walk the circle and test the points that lie in the box
        for (int s = 0; s < 400; ++s) {
            const double x = a + r * std::cos(0.0157 * s), y = bb + r * std::sin(0.0157 * s);
            if (x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1]) continue;
            check(J.has_value() && J->contains(column({x, y}), 1e-12),
                  name + ": a solution stays in the box");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // x - 1 = 0 on [0, 3]: the box shrinks to x = 1
        const LevelSet line(Expr::var(0) - 1.0, 1, CompOp::eq);
        const auto J = line.tighten_domain(test::box({0}, {3}));
        check(J && close(J->inf(), {1.0}) && close(J->sup(), {1.0}), b + ": x - 1 = 0");
        // x^2 = 4 on [-3, 1]: x = -2 (a root beyond 1 is cut by the box)
        const LevelSet sq(pow(Expr::var(0), 2) - 4.0, 1, CompOp::eq);
        const auto S = sq.tighten_domain(test::box({-3}, {1}));
        check(S && close(S->inf(), {-2.0}, 1e-12) && close(S->sup(), {-2.0}, 1e-12),
              b + ": x^2 = 4 keeps the root in the box");
        // x^2 = 4 on [-3, 3] keeps both roots
        const auto B = sq.tighten_domain(test::box({-3}, {3}));
        check(B && close(B->inf(), {-2.0}, 1e-12) && close(B->sup(), {2.0}, 1e-12),
              b + ": x^2 = 4 keeps both roots");
        // no solution: x^2 = 4 on [-1, 1], and the circle far from the box
        check(!sq.tighten_domain(test::box({-1}, {1})), b + ": an empty contraction is none");
        check(!LevelSet(test::circle(0, 0, 1), 2, CompOp::eq)
                   .tighten_domain(test::box({3, 3}, {4, 4})),
              b + ": a circle that misses the box");
        // x y = 1 with x in [1, 2], y in [0, 4]: y in [0.5, 1]
        const LevelSet prod(Expr::var(0) * Expr::var(1) - 1.0, 2, CompOp::eq);
        const auto P = prod.tighten_domain(test::box({1, 0}, {2, 4}));
        check(P && close(P->inf(), {1.0, 0.5}, 1e-12) && close(P->sup(), {2.0, 1.0}, 1e-12),
              b + ": a product");
        // sin is left as it is: no narrowing, no unsound cut
        const LevelSet sn(sin(Expr::var(0)), 1, CompOp::eq);
        const auto N = sn.tighten_domain(test::box({-4}, {4}));
        check(N && close(N->inf(), {-4.0}) && close(N->sup(), {4.0}), b + ": sin does not narrow");
        // the stated property, on random circles
        contractor_property(b);
    });
    const LevelSet line(Expr::var(0) - 1.0, 1, CompOp::eq);
    check(test::raises([&] { line.tighten_domain(test::box({0}, {1}), "bogus"); },
                       Error::wrong_value),
          "tighten_domain: unknown type");
    // the unported contractors, and the inputs that make no sense
    check(test::raises([&] { line.tighten_domain(test::box({0}, {1}), "taylor"); },
                       Error::not_supported),
          "tighten_domain: an unported contractor");
    check(test::raises(
              [] { LevelSet(Expr::var(0), 1, CompOp::le).tighten_domain(test::box({0}, {1})); },
              Error::noops),
          "tighten_domain: inequalities");
    check(test::raises([&] { line.tighten_domain(test::box({0, 0}, {1, 1})); },
                       Error::dimension_mismatch),
          "tighten_domain: dimension");
    // with the checks off, a valid call still works
    ChecksScope off(false);
    check(line.tighten_domain(test::box({0}, {3})).has_value(),
          "tighten_domain: valid, checks off");
    return test::finish("level_set tighten_domain");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
