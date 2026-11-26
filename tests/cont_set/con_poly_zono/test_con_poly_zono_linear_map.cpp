// test_con_poly_zono_linear_map - the image M * S keeps the image of x0
//
// CORA: contSet/@conPolyZono/mtimes.m
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
        const Tensor M = Tensor::from_data({2, 1, -1, 0.5}, {2, 2});
        check(in_box(S.linear_map(M), {2 * x0[0] + x0[1], -x0[0] + 0.5 * x0[1]}) &&
                  in_box(M * S, {2 * x0[0] + x0[1], -x0[0] + 0.5 * x0[1]}),
              bk + ": linear_map");
    });
    check(test::raises([] { (void)known().linear_map(Tensor::ones({2, 3})); },
                       Error::dimension_mismatch),
          "linear_map: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().linear_map(Tensor::ones({1, 2}));
    }
    return test::finish("con_poly_zono linear_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
