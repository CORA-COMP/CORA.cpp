// test_con_poly_zono_convex_hull - the hull with itself holds x0
//
// CORA: contSet/@conPolyZono/convHull_.m
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
        check(in_box(S.convex_hull(S), x0), bk + ": convex_hull");
    });
    check(test::raises([] { (void)known().convex_hull(ConPolyZono::origin(3)); },
                       Error::dimension_mismatch),
          "convex_hull: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().convex_hull(known());
    }
    return test::finish("con_poly_zono convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
