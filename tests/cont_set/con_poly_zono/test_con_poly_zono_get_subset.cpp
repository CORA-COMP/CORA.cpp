// test_con_poly_zono_get_subset - x0 at the substituted factor, and the new row met there
//
// CORA: contSet/@conPolyZono/getSubset.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_testing.h"

using namespace cora;
using namespace test::cpz;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &bk) {
        const ConPolyZono S = known();
        const std::vector<double> x0 = point(S, kA0, {kBeta0});
        const ConPolyZono sub = S.get_subset(1, Interval(test::vec({0.25}), test::vec({0.75})));
        check(close(Tensor::from_data(point(sub, {0.0, kA0[1]}, {kBeta0}), {2}), x0, 1e-12) &&
                  sub.constraints().contains(Tensor::from_data({0.0, kA0[1]}, {1, 2}), 1e-12),
              bk + ": get_subset");
    });
    check(test::raises(
              [] { (void)known().get_subset(7, Interval(test::vec({0.0}), test::vec({1.0}))); },
              Error::wrong_value),
          "get_subset: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().get_subset(2, Interval(test::vec({0.0}), test::vec({1.0})));
    }
    return test::finish("con_poly_zono get_subset");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
