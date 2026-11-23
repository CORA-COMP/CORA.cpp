// test_lean_polytope_normalize_constraints - Polytope::normalize_constraints against the oracle
//
// Bit for bit. Type "b": an inequality row is divided by the magnitude of its offset, an equality
// row by its offset, a row of offset zero by 1 (CORA.cpp divides; `polytope.normalizeConstraints`
// multiplies by the reciprocal and is another set of bits). The oracle takes a box dom for the
// rounding of the rows; at nearest the rounding is none. The offsets are of both signs, and one is
// zero.
//
// ORACLE: contSet/@polytope/normalizeConstraints.m
// Authors:       Claude Sonnet 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/polytope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("polytope.normalizeConstraints", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 4}) {
            const int64_t m = n + 2;
            Tensor b = rnd.matrix(m, 1).squeeze(-1) * 3.0;
            std::vector<double> offsets = b.data();
            offsets[0] = 0.0;
            b = Tensor::from_data(offsets, {m});
            const Tensor be = Tensor::from_data({1.5, -2.5}, {2});
            const Polytope P(rnd.matrix(m, n), b, rnd.matrix(2, n), be);
            const Interval dom(Tensor::ones({n}) * -1.0, Tensor::ones({n}));
            oracle::check(P.normalize_constraints("b"),
                          lean::Polytope(P).normalize_constraints_cora(dom));
            // without equalities
            const Polytope Q(rnd.matrix(m, n), b);
            oracle::check(Q.normalize_constraints("b"),
                          lean::Polytope(Q).normalize_constraints_cora(dom));
        }
    });
    return test::finish("lean polytope normalize_constraints");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
