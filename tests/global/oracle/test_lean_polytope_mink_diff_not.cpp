// test_lean_polytope_mink_diff_not - Polytope::mink_diff of a box and complement against
//
// the CORALean oracle
//
// minkDiff of a box takes the offsets lowered by the support function of the box, in the order of
// CORA's d'c + |d|'r (`polytope.minkDiffCora`); not shifts the opposite halfspace by eps, which is
// CORA.cpp's machine epsilon 2^-52 (`polytope.not`). Both are bit for bit at dtype nearest.
//
// ORACLE: contSet/@polytope/minkDiff.m
// ORACLE: contSet/@polytope/not.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/polytope.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// m random halfspaces of n dimensions with offsets around 2.
Polytope aux_halfspaces(oracle::Random &rnd, int64_t n, int64_t m) {
    const Tensor b = rnd.matrix(m, 1).squeeze(-1) * 2.0;
    return {rnd.matrix(m, n), b};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("polytope.minkDiffCora", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 4}) {
            const Polytope P = aux_halfspaces(rnd, n, n + 2);
            // a box around the origin: the radius is at most 0.1
            std::vector<double> r(n);
            for (double &x : r) x = rnd.uniform(0, 0.1);
            const Tensor rad = Tensor::from_data(r, {n});
            const Interval I(rad * -1.0, rad);
            oracle::check(P.mink_diff(I).as<Polytope>(), lean::Polytope(P).mink_diff_cora(I));
        }
    });
    oracle::run("polytope.not", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 4}) {
            const Polytope P = aux_halfspaces(rnd, n, 1);
            oracle::check(P.complement(), lean::Polytope(P).complement(std::ldexp(1.0, -52)));
        }
    });
    return test::finish("lean polytope minkDiff and not");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
