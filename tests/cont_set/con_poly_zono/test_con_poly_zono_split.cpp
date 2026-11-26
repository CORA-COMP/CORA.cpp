// test_con_poly_zono_split - one of the halves holds x0
//
// CORA: contSet/@conPolyZono/split.m
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
        const auto [lo, hi] = S.split(0);
        check(in_box(lo, x0) || in_box(hi, x0), bk + ": split covers x0");
    });
    check(test::raises([] { (void)known().split(2); }, Error::wrong_value),
          "split: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().split(1);
    }
    return test::finish("con_poly_zono split");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
