// test_lean_zonotope_quad_map_enclosure - Zonotope::quad_map inside the CORALean oracle's box
//
// CORA: contSet/@zonotope/quadMap.m
//
// CORALean maps the zonotope as a polynomial zonotope, an enclosure with (n + m + 1)^2 generators
// and the center as a generator, where CORA keeps the center and has 1 + m + m (m + 1) / 2
// generators; a bit-exact test needs a float version of its Zonotope.quadMapTri
// (docs/oracle-requests.md). CORA.cpp's box lies inside the oracle's.
//
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
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
    oracle::run("zonotope.quadMap", [](oracle::Random &rnd) {
        for (const auto &[n, m, k] : std::vector<std::tuple<int, int, int>>{
                 {1, 1, 1}, {2, 3, 2}, {3, 5, 3}, {4, 6, 1}, {6, 4, 6}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            std::vector<Tensor> Q;
            for (int i = 0; i < k; ++i) Q.push_back(rnd.matrix(n, n));
            const std::vector<lean::Tensor> theirs_Q(Q.begin(), Q.end());
            oracle::check_inside(Z.quad_map(Q).interval(),
                                 lean::Zonotope(Z).quad_map(theirs_Q).interval());
        }
    });
    return test::finish("lean zonotope quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
