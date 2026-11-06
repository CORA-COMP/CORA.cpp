// test_lean_support_function_bound - the support function of ZonoBundle and ConZonotope
//
// Against the CORALean oracle, as a bound.
//
// The oracle's value is an upper bound from a linear program (`boundDotLP`, `boundDot`), CORA.cpp's
// the optimum of a simplex: the optimum does not exceed the bound. No bit-exact test exists, so
// there is no ORACLE tag.
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/zono_bundle.h"
#include "oracle_testing.h"
#include <cmath>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonoBundle.supportFunc", [](oracle::Random &rnd) {
        for (const int n : {2, 3}) {
            const ZonoBundle a = rnd.zono_bundle(n, 2, 4);
            const Tensor d = rnd.direction(n);
            const double mine = a.support_function(d).item();
            const double theirs = lean::ZonoBundle(a).support_function(d);
            test::check(mine <= theirs + 1e-9 * (1 + std::abs(theirs)),
                        "zonoBundle.supportFunc: below the bound");
        }
    });
    // the same bound for a constrained zonotope
    oracle::run("conZonotope.supportFunc", [](oracle::Random &rnd) {
        for (const int n : {2, 3}) {
            const ConZonotope cz = rnd.con_zonotope(n, 5, 2);
            const Tensor d = rnd.direction(n);
            const double mine = cz.support_function(d).item();
            const double theirs = lean::ConZonotope(cz).support_function(d);
            test::check(mine <= theirs + 1e-9 * (1 + std::abs(theirs)),
                        "conZonotope.supportFunc: below the bound");
        }
    });
    return test::finish("lean support function bound");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
