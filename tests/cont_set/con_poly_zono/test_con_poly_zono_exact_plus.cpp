// test_con_poly_zono_exact_plus - the sum over the same factors holds 2 x0
//
// CORA: contSet/@conPolyZono/exactPlus.m
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
        check(in_box(S.exact_plus(S), {2 * x0[0], 2 * x0[1]}), bk + ": exact_plus");
    });
    check(
        test::raises([] { (void)known().exact_plus(ConPolyZono::origin(2)); }, Error::wrong_value),
        "exact_plus: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().exact_plus(known());
    }
    return test::finish("con_poly_zono exact_plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
