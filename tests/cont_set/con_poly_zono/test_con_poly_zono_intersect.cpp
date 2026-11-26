// test_con_poly_zono_intersect - the intersection with itself holds x0
//
// CORA: contSet/@conPolyZono/and_.m
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
        check(in_box(S.intersect(S), x0), bk + ": intersect");
        check(S.dependent().independent_generators().shape()[0] == 0, bk + ": dependent");
    });
    check(test::raises([] { (void)known().intersect(ConPolyZono::origin(3)); },
                       Error::dimension_mismatch),
          "intersect: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().intersect(known());
    }
    return test::finish("con_poly_zono intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
