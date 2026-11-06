// test_lean_zonotope_filter_out - Zonotope::filter_out against the CORALean oracle
//
// The indices of the cells that no other cell contains: a smaller copy of a cell inside it is
// dropped, a far cell stays.
//
// ORACLE: contSet/@zonotope/filterOut.m
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

// the cells as Lean zonotopes
std::vector<lean::Zonotope> lifted(const std::vector<Zonotope> &cells) {
    std::vector<lean::Zonotope> out;
    for (const Zonotope &cell : cells) out.emplace_back(cell);
    return out;
}


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.filterOut", [](oracle::Random &rnd) {
        const Zonotope big = rnd.zonotope(3, 3);
        // a cell inside (small generators, the center moved a little) and a far one
        const Tensor inside_center = big.center() + big.generators().slice(-1, 0, 1) * 0.25;
        const Zonotope inside(inside_center, big.generators() * 0.3);
        const Zonotope far(big.center() + 100.0, rnd.matrix(3, 3));
        const std::vector<Zonotope> cells{inside, big, far};
        oracle::check(Zonotope::filter_out(cells), lean::Zonotope::filter_out(lifted(cells)));
        const std::vector<Zonotope> apart{big, far};
        oracle::check(Zonotope::filter_out(apart), lean::Zonotope::filter_out(lifted(apart)));
    });
    return test::finish("lean zonotope filter_out");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
