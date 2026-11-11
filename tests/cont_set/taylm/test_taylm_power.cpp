// test_taylm_power - Taylm power: integer powers
//
// CORA: contSet/@taylm/power.m
// CORA: contSet/@taylm/mpower.m
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
        const std::vector<Taylm> x = Taylm::from_interval(Interval(column({-1.0}), column({2.0})));
        check(close(x[0].power(0).bounds()[0], 1.0) && close(x[0].power(0).bounds()[1], 1.0),
              b + ": x^0 = 1");
        // x^2 over [-1, 2] contains [0, 4]; x^3 contains [-1, 8]
        check(x[0].power(2).bounds()[0] <= 0.0 + 1e-12 && x[0].power(2).bounds()[1] >= 4.0 - 1e-12,
              b + ": x^2");
        check(x[0].power(3).bounds()[0] <= -1.0 && x[0].power(3).bounds()[1] >= 8.0 - 1e-12,
              b + ": x^3");
        // a power of a shifted variable keeps its dependency: (x - x)^2 = 0
        check(close(x[0].minus(x[0]).power(2).bounds()[1], 0.0), b + ": (x - x)^2 = 0");
    });
    check(test::raises([] { Taylm::constant(1.0, 1).power(-1); }, Error::wrong_value),
          "power: negative exponent");
    ChecksScope off(false);
    check(close(Taylm::constant(2.0, 1).power(3).bounds()[0], 8.0), "power: valid with checks off");
    return test::finish("taylm power");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
