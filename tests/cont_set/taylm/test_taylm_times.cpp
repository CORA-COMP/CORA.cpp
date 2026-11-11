// test_taylm_times - Taylm times: products, truncation and the enclosure of sampled products
//
// CORA: contSet/@taylm/times.m
// CORA: contSet/@taylm/mtimes.m
// PROPERTY: contSet/@taylm/times.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the product of models bounds the product of the values at sampled points of
/// the box, with a low max_order so that the truncation is exercised.
void product_property(const std::string &name) {
    Rng rng(5);
    for (int k = 0; k < 20; ++k) {
        std::vector<double> lo(2), w(2), u(2);
        rng.uniform(lo.data(), 2, -2.0, 2.0);
        rng.uniform(w.data(), 2, 0.1, 2.0);
        const Interval I(column(lo), column({lo[0] + w[0], lo[1] + w[1]}));
        const std::vector<Taylm> x = Taylm::from_interval(I, 2);
        // (x0 x1 + x0)(x0 - x1): degree 3 > max_order 2
        const Taylm t = x[0].times(x[1]).plus(x[0]).times(x[0].minus(x[1]));
        const auto [bl, bu] = t.bounds();
        for (int s = 0; s < 50; ++s) {
            rng.uniform(u.data(), 2, 0.0, 1.0);
            const double a = lo[0] + u[0] * w[0], c = lo[1] + u[1] * w[1];
            const double v = (a * c + a) * (a - c);
            check(bl - 1e-12 <= v && v <= bu + 1e-12, name + ": the product bounds a sample");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const std::vector<Taylm> x = Taylm::from_interval(Interval(column({-1.0}), column({2.0})));
        // x^2 on [-1, 2]: a polynomial with the monomial a^2 in [0, 1]: 0.25 + 1.5 a + 2.25 a^2
        const Taylm sq = x[0].times(x[0]);
        check(close(sq.bounds()[1], 0.25 + 1.5 + 2.25) && sq.bounds()[0] <= 0.0 + 1e-12,
              b + ": the square");
        // a scalar factor scales the bounds, a negative one swaps them
        check(close(x[0].times(2.0).bounds()[1], 4.0) && close(x[0].times(-2.0).bounds()[0], -4.0),
              b + ": a number");
        // above max_order the terms go to the remainder: x^3 with max_order 2 still bounds [-1, 8]
        const std::vector<Taylm> y =
            Taylm::from_interval(Interval(column({-1.0}), column({2.0})), 2);
        const Taylm cube = y[0].times(y[0]).times(y[0]);
        check(cube.bounds()[0] <= -1.0 && cube.bounds()[1] >= 8.0, b + ": truncation encloses");
        product_property(b);
    });
    check(test::raises([] { Taylm::constant(1.0, 1).times(Taylm::constant(1.0, 2)); },
                       Error::dimension_mismatch),
          "times: variables");
    ChecksScope off(false);
    check(close(Taylm::constant(2.0, 1).times(Taylm::constant(3.0, 1)).bounds()[0], 6.0),
          "times: valid with checks off");
    return test::finish("taylm times");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
