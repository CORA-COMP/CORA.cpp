// test_con_poly_zono_interval - the box, the zonotope and the polynomial part hold x0
//
// CORA: contSet/@conPolyZono/interval.m
// CORA: contSet/@conPolyZono/zonotope.m
// CORA: contSet/@conPolyZono/polyZonotope.m
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
        check(in_box(S, x0) && in_box(S.zonotope(), x0) && in_box(S.poly_zonotope(), x0),
              bk + ": the enclosures");
    });
    {
        ChecksScope off(false);
        (void)known().interval();
    }
    return test::finish("con_poly_zono interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
