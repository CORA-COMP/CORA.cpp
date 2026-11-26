// test_con_poly_zono_split_dep_factor - the upper half meets its row at the mapped factor point
//
// CORA: contSet/@conPolyZono/splitDepFactor.m
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
        const auto [s1, s2] = S.split_dep_factor(1);
        check(s1.dim() == 2 &&
                  s2.constraints().contains(Tensor::from_data({0.0, kA0[1]}, {1, 2}), 1e-12),
              bk + ": split_dep_factor");
    });
    check(test::raises([] { (void)known().split_dep_factor(7); }, Error::wrong_value),
          "split_dep_factor: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().split_dep_factor(2);
    }
    return test::finish("con_poly_zono split_dep_factor");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
