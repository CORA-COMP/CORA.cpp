// test_lean_zonotope_radius - Zonotope::ball_radius against the CORALean oracle, bit for bit
//
// The oracle's `zonotope.radiusCora` takes the smaller of the sum of the generator lengths and the
// norm of the half-widths of the box, built as CORA builds it (`zonotope.radius` is another
// definition and is not compared). The request carries an error box, all zeros.
//
// ORACLE: contSet/@zonotope/radius.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.radiusCora", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 4, 6})
            for (const int64_t m : {1, 3, 8}) {
                const Zonotope Z = rnd.zonotope(n, m);
                oracle::check(Z.ball_radius(), lean::Zonotope(Z).radius_cora());
            }
    });
    return test::finish("lean zonotope radius");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
