// test_lean_con_zonotope_interval_tight - ConZonotope::interval_tight against the CORALean oracle
//
// The box of c + G J with J the factor bounds after k Fourier-Motzkin passes on the constraint: the
// same bits as CORALean's `ConZonotope.intervalTight`, for several k. CORA's interval solves linear
// programs and is not what is compared.
//
// CORA.cpp only: tests interval_tight, CORALean's float replacement of the programs of interval
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/oracle/con_zonotope.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("conZonotope.intervalTight", [](test::oracle::Random &rnd) {
        // k <= 3: the oracle's passes are not memoized, its cost grows like (columns of G)^k
        for (const int k : {0, 1, 2, 3}) {
            const ConZonotope cz = rnd.con_zonotope(3, 5, 2);
            const Interval mine = cz.interval_tight(k);
            test::oracle::check(mine, lean::ConZonotope(cz).interval_tight(k));
        }
        // no constraint: the box of the zonotope whatever k is
        const ConZonotope plain(ConZonotope(rnd.zonotope(4, 6)));
        test::oracle::check(plain.interval_tight(3), lean::ConZonotope(plain).interval_tight(3));
    });
    return test::finish("lean con zonotope interval tight");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
