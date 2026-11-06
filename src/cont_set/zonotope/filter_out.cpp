// filter_out - the zonotopes of a list that no other contains, as CORA's zonotope.filterOut
//
// CORA keeps a cell that has a remainder after the set differences with all the others, which
// decides whether a union covers the cell. A union of polytopes is not decided here, only the easy
// cover: a cell is dropped when another single cell contains it (so two equal cells drop each
// other, as in CORA). The cells kept are a superset of CORA's. CORA lists the cells sorted by
// volume; the indices are in order.
//
// Syntax:   std::vector<int64_t> kept = Zonotope::filter_out(Zs);
// Inputs:   Zs - a nonempty list of zonotopes of one dimension (no batch)
// Outputs:  kept - the indices (0-based) of the cells that no other cell contains
// See also: Zonotope::contains
//
// CORA: contSet/@zonotope/filterOut.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<int64_t> Zonotope::filter_out(const std::vector<Zonotope> &Zs) {
    detail::Entry entry;
    CORA_CHECK(!Zs.empty(), Error::wrong_value,
               [] { return std::string("filter_out: the list of zonotopes must not be empty"); });
    for (const Zonotope &Z : Zs)
        CORA_CHECK(Z.dim() == Zs.front().dim() && Z.center().ndim() == 2, Error::wrong_value, [&] {
            return "filter_out: the zonotopes must be single sets of one dimension, got " +
                   shape_text(Z.center().shape());
        });
    std::vector<int64_t> kept;
    // a cell stays unless another one contains it
    for (std::size_t i = 0; i < Zs.size(); ++i) {
        bool covered = false;
        for (std::size_t j = 0; j < Zs.size() && !covered; ++j)
            covered = j != i && Zs[j].contains(Zs[i]);
        if (!covered) kept.push_back(int64_t(i));
    }
    return kept;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
