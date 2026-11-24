// test_level_set_tighten_domain - LevelSet tighten_domain: the forward-backward contractor
//
// CORA: contSet/@levelSet/tightenDomain.m
// CORA: specification/syntaxTree.m:backpropagation
// CORA: specification/syntaxTree.m:plus
// CORA: specification/syntaxTree.m:minus
// CORA: specification/syntaxTree.m:times
// CORA: specification/syntaxTree.m:power
// CORA: specification/syntaxTree.m:sqrt
// CORA: specification/syntaxTree.m:exp
// CORA: specification/syntaxTree.m:sin
// CORA: specification/syntaxTree.m:cos
// CORA: specification/syntaxTree.m:tan
// PROPERTY: contSet/@levelSet/tightenDomain.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "level_set_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::vec;

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
            check(J.has_value() && J->contains(vec({x, y}), 1e-12),
                  name + ": a solution stays in the box");
        }
    }
}

/// The contractor for f(x) = f(x0) with an elementary function f: x0 stays in the box, and the
/// box shrinks (on both branches of cosh, to the two mirrored roots).
void elementary_contractors(const std::string &name) {
    const Expr x = Expr::var(0);
    struct Case {
        const char *op;
        Expr f;
        double x0, lo, hi;
    };
    const std::vector<Case> cases = {
        {"log", log(x), 1.5, 0.5, 4.0},     {"asin", asin(x), 0.3, -0.9, 0.9},
        {"acos", acos(x), 0.3, -0.9, 0.9},  {"atan", atan(x), 1.2, -3.0, 5.0},
        {"sinh", sinh(x), 0.8, -2.0, 3.0},  {"cosh", cosh(x), 0.8, 0.1, 3.0},
        {"tanh", tanh(x), 0.8, -2.0, 3.0},  {"asinh", asinh(x), 0.8, -2.0, 3.0},
        {"acosh", acosh(x), 2.0, 1.1, 5.0}, {"atanh", atanh(x), 0.4, -0.9, 0.9}};
    for (const Case &c : cases) {
        const LevelSet ls(c.f - c.f.eval({c.x0}), 1, CompOp::eq);
        const auto J = ls.tighten_domain(test::box({c.lo}, {c.hi}));
        const std::string label = name + ": " + c.op;
        check(J && J->contains(vec({c.x0}), 1e-12), label + " keeps the solution");
        check(J && J->sup().data()[0] - J->inf().data()[0] < 1e-6, label + " narrows to the root");
    }
    // cosh x = cosh 0.8 on [-3, 3] keeps both mirrored roots, and misses a value below 1
    const double v = std::cosh(0.8);
    const auto B = LevelSet(cosh(x) - v, 1, CompOp::eq).tighten_domain(test::box({-3}, {3}));
    check(B && B->inf().data()[0] < -0.79 && B->sup().data()[0] > 0.79 &&
              B->sup().data()[0] - B->inf().data()[0] < 1.7,
          name + ": cosh keeps both roots");
    check(!LevelSet(cosh(x) - 0.5, 1, CompOp::eq).tighten_domain(test::box({-3}, {3})),
          name + ": cosh below 1 has no solution");
}


/// Stated property: every solution of sin, cos or tan of x = c in the box lies in the contracted
/// box.
void periodic_property(const std::string &name) {
    Rng rng(5);
    for (int k = 0; k < 30; ++k) {
        double u[3];
        rng.uniform(u, 3, -8.0, 8.0);
        const double c = std::sin(u[2]) * 0.9, lo = std::min(u[0], u[1]), hi = std::max(u[0], u[1]);
        const LevelSet ls(sin(Expr::var(0)) - c, 1, CompOp::eq);
        const auto J = ls.tighten_domain(test::box({lo}, {hi}));
        // the solutions of sin x = c on a fine grid of the box, where the sign of sin x - c flips
        for (double x = lo; x < hi; x += 0.01)
            if ((std::sin(x) - c) * (std::sin(x + 0.01) - c) <= 0)
                check(J.has_value() && J->contains(vec({x}), 1e-2),
                      name + ": a sine solution stays");
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
        // sin(x) = 1/2 on [-10, 10]: the solutions pi/6 + 2 pi k and 5 pi/6 + 2 pi k
        const LevelSet sn(sin(Expr::var(0)) - 0.5, 1, CompOp::eq);
        const auto N = sn.tighten_domain(test::box({-10}, {10}));
        check(N && close(N->inf(), {5 * M_PI / 6 - 4 * M_PI}, 1e-12) &&
                  close(N->sup(), {5 * M_PI / 6 + 2 * M_PI}, 1e-12),
              b + ": sin narrows to the first and last solution");
        // cos(x) = 2 has no solution; tan(x) = 5 keeps atan(5) + k pi, also above 1
        check(!LevelSet(cos(Expr::var(0)) - 2.0, 1, CompOp::eq).tighten_domain(test::box({0}, {9})),
              b + ": cos = 2");
        const auto T = LevelSet(tan(Expr::var(0)) - 5.0, 1, CompOp::eq)
                           .tighten_domain(test::box({0.1}, {1.4}));
        check(T && close(T->inf(), {std::atan(5.0)}, 1e-12) &&
                  close(T->sup(), {std::atan(5.0)}, 1e-12),
              b + ": tan keeps a value above 1");
        // x + y = 3, sqrt(x) = 2 and exp(x) = 1
        const auto S2 = LevelSet(Expr::var(0) + Expr::var(1) - 3.0, 2, CompOp::eq)
                            .tighten_domain(test::box({0, 0}, {1, 5}));
        check(S2 && close(S2->inf(), {0.0, 2.0}, 1e-12) && close(S2->sup(), {1.0, 3.0}, 1e-12),
              b + ": a sum");
        const auto R2 =
            LevelSet(sqrt(Expr::var(0)) - 2.0, 1, CompOp::eq).tighten_domain(test::box({0}, {10}));
        check(R2 && close(R2->inf(), {4.0}, 1e-12) && close(R2->sup(), {4.0}, 1e-12),
              b + ": a square root");
        const auto E2 =
            LevelSet(exp(Expr::var(0)) - 1.0, 1, CompOp::eq).tighten_domain(test::box({-2}, {2}));
        check(E2 && close(E2->inf(), {0.0}, 1e-12) && close(E2->sup(), {0.0}, 1e-12),
              b + ": an exponential");
        periodic_property(b);
        // the stated property, on random circles
        contractor_property(b);
        elementary_contractors(b);
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
