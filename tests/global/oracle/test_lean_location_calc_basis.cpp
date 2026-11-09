// test_lean_location_calc_basis - the box basis against the CORALean oracle
//
// The `box` basis is the identity of the state dimension, one per entry of `enclose`; the oracle
// has no float definition of `pca` and `flow`, which stay CORA's.
//
// ORACLE: hybridDynamics/@location/calcBasis.m
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/location.h"
#include "hybrid_dynamics/location/private/priv.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("location.calcBasis", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 5}) {
            const Location loc("l", Polytope::inf(n), {}, LinearSys(rnd.matrix(n, n)));
            const Polytope guard = Polytope::inf(n);
            const std::vector<Set> R = {rnd.zonotope(n, n)};
            const std::vector<Tensor> mine =
                priv_calc_basis(loc, R, guard, {alg::Enclose::box, alg::Enclose::box},
                                Zonotope(loc.c(), loc.c().zeros_like({n, 0})));
            const std::vector<Tensor> theirs = lean::Location::calc_basis(n, {"box", "box"});
            for (std::size_t k = 0; k < mine.size(); ++k) test::oracle::check(mine[k], theirs[k]);
        }
    });
    return test::finish("lean location calcBasis");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
