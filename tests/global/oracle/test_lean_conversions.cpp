// test_lean_conversions - the conversions of sets that the CORALean oracle serves, bit for bit
//
// A zonotope and a box as a zonotope bundle of one member, a zonotope as a matrix zonotope of
// column matrices, the empty set and the full space as polytopes. The oracle builds the same
// object from the same operand (its centre, generators, bounds), so the stored data must agree.
//
// ORACLE: contSet/@zonotope/zonoBundle.m
// ORACLE: contSet/@interval/zonoBundle.m
// ORACLE: contSet/@zonotope/matZonotope.m
// ORACLE: contSet/@emptySet/polytope.m
// ORACLE: contSet/@fullspace/polytope.m
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zono_bundle.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.zonoBundle", [](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{1, 1}, {2, 3}, {5, 30}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(ZonoBundle::from(Z), lean::ZonoBundle::from_zonotope(lean::Zonotope(Z)));
        }
    });
    oracle::run("interval.zonoBundle", [](oracle::Random &rnd) {
        for (const int64_t n : {1, 2, 6}) {
            const Interval I = rnd.box(n);
            oracle::check(ZonoBundle::from(I), lean::ZonoBundle::from_interval(lean::Interval(I)));
        }
    });
    // the zonotope as a matrix zonotope of column matrices
    oracle::run("zonotope.matZonotope", [](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{1, 1}, {2, 3}, {5, 30}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(Z.mat_zonotope(), lean::MatZonotope::from_zonotope(lean::Zonotope(Z)));
        }
    });
    // the empty set and the full space as polytopes
    oracle::run("emptySet.polytope", [](oracle::Random &) {
        for (const int64_t n : {1, 2, 5})
            oracle::check(Polytope(EmptySet(n)), lean::Polytope::from_empty_set(n));
    });
    oracle::run("fullspace.polytope", [](oracle::Random &) {
        for (const int64_t n : {1, 2, 5})
            oracle::check(Polytope(Fullspace(n)), lean::Polytope::from_fullspace(n));
    });
    return test::finish("lean conversions");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
