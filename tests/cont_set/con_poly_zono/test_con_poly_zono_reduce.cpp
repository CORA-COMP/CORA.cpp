// test_con_poly_zono_reduce - the reduction holds x0 with at most order * n generators
//
// CORA: contSet/@conPolyZono/reduce.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_testing.h"

using namespace cora;
using namespace test::cpz;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &bk) {
        const ConPolyZono S = known();
        const std::vector<double> x0 = point(S, kA0, {kBeta0});
        const ConPolyZono R = S.reduce(1);
        check(in_box(R, x0) &&
                  R.generators().shape()[0] + R.independent_generators().shape()[0] <= 2,
              bk + ": reduce");
    });
    check(test::raises([] { (void)known().reduce(0); }, Error::wrong_value),
          "reduce: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().reduce(2);
    }
    return test::finish("con_poly_zono reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
