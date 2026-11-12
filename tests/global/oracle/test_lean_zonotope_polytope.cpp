// test_lean_zonotope_polytope - the halfspaces of a Zonotope against the CORALean oracle
//
// ORACLE: contSet/@zonotope/polytope.m
//
// Bit for bit: the facet normals of the generators in general position (full-dimensional, so none
// of CORA's degenerate branches runs) and the box of a zonotope without generators. The rows are
// [C; -C] in the oracle's order of the generator subsets, the offsets at the support values.
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/polytope.h"
#include "oracle_testing.h"

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    oracle::run("zonotope.polytope", [](oracle::Random &rnd) {
        // n dimensions with n - 1, n and a few more generators
        for (const auto &[n, m] : std::vector<std::pair<int, int>>{
                 {1, 1}, {2, 2}, {2, 5}, {3, 3}, {3, 6}, {4, 5}, {4, 7}, {5, 7}, {6, 7}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check("n = " + std::to_string(n) + ", m = " + std::to_string(m), Z.polytope(),
                          lean::Polytope::from_zonotope(lean::Zonotope(Z)));
        }
        // no generators: the box of a point; the oracle prints -0 as 0, so CORA.cpp's -0 rows too
        const Zonotope point(rnd.vector(3), Tensor::zeros({3, 0}));
        const Polytope box = point.polytope();
        oracle::check("a point", Polytope(box.A() + 0.0, box.b() + 0.0),
                      lean::Polytope::from_zonotope(lean::Zonotope(point)));
    });
    return test::finish("lean zonotope polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
