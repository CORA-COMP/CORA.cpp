// test_con_poly_zono_lift - the permutation and the projection move x0 along
//
// CORA: contSet/@conPolyZono/lift_.m
// CORA: contSet/@conPolyZono/project.m
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
        check(in_box(S.lift(2, {1, 0}), {x0[1], x0[0]}), bk + ": lift");
        check(in_box(S.project({1}).as<ConPolyZono>(), {x0[1]}), bk + ": project");
    });
    check(test::raises([] { (void)known().lift(3, {0, 1, 2}); }, Error::not_supported),
          "lift: an invalid operand");
    {
        ChecksScope off(false);
        (void)known().lift(2, {0, 1});
    }
    return test::finish("con_poly_zono lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
