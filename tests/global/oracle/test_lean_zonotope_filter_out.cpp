// test_lean_zonotope_filter_out - Zonotope::filter_out against the CORALean oracle
//
// The indices of the cells that no other cell contains: a smaller copy of a cell inside it is
// dropped, a far cell stays.
//
// ORACLE: contSet/@zonotope/filterOut.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
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
    out.reserve(cells.size());
    for (const Zonotope &cell : cells) out.emplace_back(cell);
    return out;
}


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.filterOut", [](oracle::Random &rnd) {
        const Zonotope big = rnd.zonotope(3, 3);
        // a cell inside (small generators, the center moved a little) and a far one
        const Tensor inside_center = cora::detail::cora_center(big) +
                                     cora::detail::cora_generators(big).slice(-1, 0, 1) * 0.25;
        // layout: #163 step 2
        const Zonotope inside = cora::detail::zonotope_of_columns(
            inside_center, cora::detail::cora_generators(big) * 0.3);
        // layout: #163 step 2
        const Zonotope far = cora::detail::zonotope_of_columns(
            cora::detail::cora_center(big) + 100.0, rnd.matrix(3, 3).matrix_transpose());
        const std::vector<Zonotope> cells{inside, big, far};
        oracle::check(Zonotope::filter_out(cells), lean::Zonotope::filter_out(lifted(cells)));
        const std::vector<Zonotope> apart{big, far};
        oracle::check(Zonotope::filter_out(apart), lean::Zonotope::filter_out(lifted(apart)));
    });
    return test::finish("lean zonotope filter_out");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
