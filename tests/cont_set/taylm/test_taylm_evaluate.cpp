// test_taylm_evaluate - Taylm::evaluate: expressions on Taylor models
//
// CORA.cpp only: CORA applies the function handle of the dynamics to the taylm objects directly
// PROPERTY: contSet/@taylm/times.m
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the model of the Lotka-Volterra flow x1 (3 - 3 x2) over a box bounds the flow
/// at sampled points.
void flow_property(const std::string &name) {
    Rng rng(9);
    const Expr x1 = Expr::var(0), x2 = Expr::var(1);
    const Expr f = x1 * (Expr(3.0) - x2 * 3.0) + sin(x2) + x1 / (x2 * x2 + 1.0);
    for (int k = 0; k < 20; ++k) {
        std::vector<double> lo(2), w(2), u(2);
        rng.uniform(lo.data(), 2, -2.0, 2.0);
        rng.uniform(w.data(), 2, 0.1, 1.5);
        const Interval I(column(lo), column({lo[0] + w[0], lo[1] + w[1]}));
        const auto [bl, bu] = Taylm::evaluate(f, Taylm::from_interval(I)).bounds();
        for (int s = 0; s < 40; ++s) {
            rng.uniform(u.data(), 2, 0.0, 1.0);
            const double a = lo[0] + u[0] * w[0], c = lo[1] + u[1] * w[1];
            const double v = a * (3.0 - 3.0 * c) + std::sin(c) + a / (c * c + 1.0);
            check(bl - 1e-12 <= v && v <= bu + 1e-12, name + ": the model bounds the flow");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Expr x = Expr::var(0), y = Expr::var(1);
        const std::vector<Taylm> v =
            Taylm::from_interval(Interval(column({1.0, 2.0}), column({3.0, 2.0})));
        // (x - 1)^2 + y^2 on [1, 3] x {2}: x - 1 in [0, 2], so [4, 8]
        const Taylm g = Taylm::evaluate(pow(x - 1.0, 2) + pow(y, 2), v);
        check(g.bounds()[0] <= 4.0 + 1e-12 && g.bounds()[1] >= 8.0 - 1e-12, b + ": a polynomial");
        // 2x - x - x is 0 through the expression too
        check(close(Taylm::evaluate(x * 2.0 - x - x, v).bounds()[1], 0.0),
              b + ": the dependency stays");
        // a division by a constant is a product
        check(close(Taylm::evaluate(x / 2.0, v).bounds()[1], 1.5), b + ": division by a constant");
        // exp falls back to the range: e^1 .. e^3
        const auto e = Taylm::evaluate(exp(x), v).bounds();
        check(e[0] <= std::exp(1.0) + 1e-12 && e[1] >= std::exp(3.0) - 1e-12, b + ": exp");
        flow_property(b);
    });
    check(test::raises([] { Taylm::evaluate(Expr::var(0), {}); }, Error::wrong_value),
          "evaluate: no model");
    check(test::raises(
              [] {
                  Taylm::evaluate(Expr::var(0), {Taylm::constant(1.0, 1), Taylm::constant(1.0, 2)});
              },
              Error::dimension_mismatch),
          "evaluate: variables");
    ChecksScope off(false);
    check(close(Taylm::evaluate(Expr(2.0), {Taylm::constant(1.0, 1)}).bounds()[0], 2.0),
          "evaluate: valid with checks off");
    return test::finish("taylm evaluate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
