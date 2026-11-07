// test_lean_zonotope_support_function - Zonotope::support_function against the CORALean oracle
//
// ORACLE: contSet/@zonotope/supportFunc_.m
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    // on three sizes
    test::oracle::run("zonotope.supportFunc", [](test::oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {20, 200}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor d = rnd.direction(n);
            test::oracle::check(Z.support_function(d), lean::Zonotope(Z).support_function(d));
        }
    });
    return test::finish("lean zonotope support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
