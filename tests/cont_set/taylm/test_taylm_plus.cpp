// test_taylm_plus - Taylm plus, minus, uminus: sums keep the dependency
//
// CORA: contSet/@taylm/plus.m
// CORA: contSet/@taylm/minus.m
// CORA: contSet/@taylm/uminus.m
// CORA: contSet/@taylm/uplus.m
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
        const std::vector<Taylm> x =
            Taylm::from_interval(Interval(column({1.0, 0.0}), column({3.0, 2.0})));
        // x - x is exactly 0, unlike interval arithmetic's [-2, 2]
        const Taylm d = x[0].minus(x[0]);
        check(close(d.bounds()[0], 0.0) && close(d.bounds()[1], 0.0), b + ": x - x = 0");
        // x_0 + x_1 over [1, 3] x [0, 2] is [1, 5]; plus a number shifts
        const Taylm s = x[0].plus(x[1]);
        check(close(s.bounds()[0], 1.0) && close(s.bounds()[1], 5.0), b + ": the sum");
        check(close(s.plus(1.0).bounds()[1], 6.0) && close(s.minus(1.0).bounds()[0], 0.0),
              b + ": a number");
        // -x_0 is [-3, -1], the remainder swaps its bounds
        const Taylm n = Taylm::from_range(1.0, 3.0, 2).uminus();
        check(close(n.bounds()[0], -3.0) && close(n.bounds()[1], -1.0), b + ": uminus");
        check(close(n.uplus().bounds()[0], -3.0), b + ": uplus");
    });
    check(test::raises([] { Taylm::constant(1.0, 1).plus(Taylm::constant(1.0, 2)); },
                       Error::dimension_mismatch),
          "plus: variables");
    check(test::raises([] { Taylm::constant(1.0, 1).minus(Taylm::constant(1.0, 2)); },
                       Error::dimension_mismatch),
          "minus: variables");
    ChecksScope off(false);
    check(close(Taylm::constant(1.0, 1).plus(Taylm::constant(2.0, 1)).bounds()[0], 3.0),
          "plus: valid with checks off");
    return test::finish("taylm plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
