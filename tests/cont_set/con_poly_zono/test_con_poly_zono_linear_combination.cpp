// test_con_poly_zono_linear_combination - the linear combination with itself holds x0
//
// CORA: contSet/@conPolyZono/linComb.m
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
        check(in_box(S.linear_combination(S), x0), bk + ": linear_combination");
    });
    check(test::raises([] { (void)known().linear_combination(ConPolyZono::origin(3)); },
                       Error::dimension_mismatch),
          "linear_combination: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().linear_combination(known());
    }
    return test::finish("con_poly_zono linear_combination");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
