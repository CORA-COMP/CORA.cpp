// test_taylm_taylm - Taylm: the models of a box, a zonotope and a constant
//
// CORA: contSet/@taylm/taylm.m
// CORA: contSet/@taylm/dim.m
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


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // the box [1, 3] x [-2, 2]: x_0 = 2 + a_0, x_1 = 2 a_1
        const std::vector<Taylm> x =
            Taylm::from_interval(Interval(column({1.0, -2.0}), column({3.0, 2.0})));
        check(x.size() == 2 && x[0].num_vars() == 2,
              b + ": one model and one variable per coordinate");
        check(close(x[0].bounds()[0], 1.0) && close(x[0].bounds()[1], 3.0) &&
                  close(x[1].bounds()[0], -2.0) && close(x[1].bounds()[1], 2.0),
              b + ": the bounds of a box are the box");
        // the zonotope (1, 0) + a (1, 1) + b (0, 2): x_1 = a + 2 b
        const Zonotope Z(column({1.0, 0.0}), Tensor::from_data({1, 0, 1, 2}, {2, 2}));
        const std::vector<Taylm> z = Taylm::from_zonotope(Z);
        check(close(z[1].bounds()[0], -3.0) && close(z[1].bounds()[1], 3.0),
              b + ": the bounds of a zonotope coordinate");
        // a constant and a range
        check(Taylm::constant(2.5, 3).bounds()[0] == 2.5, b + ": a constant");
        const Taylm r = Taylm::from_range(-1.0, 3.0, 1);
        check(close(r.bounds()[0], -1.0) && close(r.bounds()[1], 3.0) &&
                  r.coefficients().size() == 1,
              b + ": a range keeps the midpoint and a remainder");
        check(close(x[0].interval().inf(), {1.0}) && close(x[0].interval().sup(), {3.0}),
              b + ": interval()");
    });
    check(test::raises([] { Taylm::from_interval(Interval(column({0.0}), column({1e308 * 10}))); },
                       Error::wrong_value),
          "from_interval: unbounded");
    check(test::raises([] { Taylm::constant(1.0, 1, 0); }, Error::wrong_value), "constant: order");
    check(test::raises([] { Taylm::from_range(2.0, 1.0, 1); }, Error::wrong_value),
          "from_range: order");
    ChecksScope off(false);
    check(Taylm::constant(1.0, 2).num_vars() == 2, "constant: valid with checks off");
    return test::finish("taylm taylm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
