// test_lean_linear_reset_evaluate - LinearReset::evaluate against the CORALean oracle
//
// The image A X + B U + c is the oracle's zonotope.mtimes and zonotope.plus, in that order (as
// CORALean's Transition.jumpFloat, the reset W Rg plus the point w).
//
// ORACLE: hybridDynamics/@linearReset/evaluate.m
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/zonotope.h"
#include "cora/hybrid_dynamics/linear_reset/linear_reset.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    // the reset of a state set, then with an input set, on two sizes
    test::oracle::run("resetEvaluate", [](test::oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 12}}) {
            const Zonotope X = rnd.zonotope(n, m), U = rnd.zonotope(2, 2);
            const Tensor A = rnd.matrix(n, n), B = rnd.matrix(n, 2), c = rnd.vector(n);
            const lean::Zonotope point(c, c.zeros_like({n, 0}));
            const LinearReset plain(A, std::nullopt, c), with_input(A, B, c);
            test::oracle::check(plain.evaluate(X).as<Zonotope>(),
                                lean::Zonotope(X).linear_map(A).plus(point));
            test::oracle::check(
                with_input.evaluate(X, U).as<Zonotope>(),
                lean::Zonotope(X).linear_map(A).plus(lean::Zonotope(U).linear_map(B)).plus(point));
        }
    });
    return test::finish("lean linear_reset evaluate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
