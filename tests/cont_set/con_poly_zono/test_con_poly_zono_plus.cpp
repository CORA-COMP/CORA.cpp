// test_con_poly_zono_plus - the sums with a point, a zonotope and a set hold the sums of the points
//
// CORA: contSet/@conPolyZono/plus.m
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
        check(in_box(S + test::vec({1.0, 2.0}), {x0[0] + 1, x0[1] + 2}), bk + ": a point");
        const Zonotope Z(test::vec({1.0, 1.0}), Tensor::from_data({1, 0}, {1, 2}));
        check(in_box(S + Z, {x0[0] + 1, x0[1] + 1}), bk + ": a zonotope");
        check(in_box(S.plus(S), {2 * x0[0], 2 * x0[1]}), bk + ": the set itself");
    });
    check(test::raises([] { (void)known().plus(test::vec({1.0})); }, Error::dimension_mismatch),
          "plus: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().plus(test::vec({1.0, 1.0}));
    }
    return test::finish("con_poly_zono plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
