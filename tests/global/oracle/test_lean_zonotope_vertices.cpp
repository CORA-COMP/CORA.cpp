// test_lean_zonotope_vertices - the corners of a planar Zonotope against the CORALean oracle
//
// ORACLE: contSet/@zonotope/vertices_.m
//
// CORALean's polygon lists the 2 (h + 1) corners of the sorted walk, the first and the last corner
// of its second half repeated (they are the last and the first of the first half); CORA.cpp lists
// each corner once. The 2 h corners are compared bit for bit, in order, with those repeats dropped.
// The generators here are in general position, so that no two are parallel (CORA.cpp merges them,
// CORALean keeps both) and none vanishes.
//
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

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The corners of CORA.cpp as a (2, k) tensor of the same layout as the oracle's V.
Tensor aux_columns(const Zonotope &Z, const Polygon &corners) {
    std::vector<double> data(2 * corners.size());
    for (std::size_t k = 0; k < corners.size(); ++k) {
        data[k] = corners[k][0];
        data[corners.size() + k] = corners[k][1];
    }
    return Tensor::like(Z.center(), data, {2, int64_t(corners.size())});
}

/// The oracle's corners without the two repeated ones: columns 1 .. h of the second half.
Tensor aux_without_repeats(const Tensor &V, int64_t h) {
    const Tensor first = V.slice(-1, 0, h + 1), second = V.slice(-1, h + 2, 2 * h + 1);
    return Tensor::concat({first, second}, -1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    oracle::run("zonotope.vertices", [](oracle::Random &rnd) {
        for (const int m : {1, 2, 3, 5, 12, 40}) {
            const Zonotope Z = rnd.zonotope(2, m);
            const std::vector<Polygon> mine = Z.vertices();
            const Tensor theirs = lean::Zonotope(Z).vertices().V.gather();
            oracle::check("m = " + std::to_string(m), aux_columns(Z, mine[0]),
                          aux_without_repeats(theirs, m));
        }
    });
    return test::finish("lean zonotope vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
