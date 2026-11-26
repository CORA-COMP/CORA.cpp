// test_con_poly_zono_compact - compact keeps the point at every factor point
//
// CORA: contSet/@conPolyZono/compact_.m
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
        check(close(Tensor::from_data(point(S.compact(), kA0, {kBeta0}), {2}), x0, 1e-12),
              bk + ": compact keeps the points");
    });
    check(test::raises([] { (void)known().compact("states"); }, Error::no_specific_alg),
          "compact: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().compact();
    }
    return test::finish("con_poly_zono compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
