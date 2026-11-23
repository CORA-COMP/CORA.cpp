// rm_id - a polynomial zonotope without factors, as CORA's rmId
//
// The rows of the factors leave the exponent matrix, so the factors are fixed to 1 in every
// generator; generators that then have equal exponents are added. CORA ignores an identifier
// that the set lacks, here it is refused.
//
// Syntax:   S = pZ.rm_id(ids);
// Inputs:   ids - the identifiers to remove (of the set)
// Outputs:  S - the set over the remaining identifiers
// See also: restore_id, get_subset
//
// CORA: contSet/@polyZonotope/rmId.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::rm_id(const std::vector<int64_t> &ids) const {
    detail::Entry entry;
    CORA_CHECK(
        std::all_of(ids.begin(), ids.end(), [&](int64_t v) { return priv_find_id(id_, v) >= 0; }),
        Error::wrong_value,
        [] { return std::string("rm_id: ids must only contain identifiers of the set"); });
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < id_.size(); ++i)
        if (std::find(ids.begin(), ids.end(), id_[i]) == ids.end()) keep.push_back(i);
    // the remaining identifiers and, per generator, the remaining exponents
    std::vector<int64_t> id;
    id.reserve(keep.size());
    for (const std::size_t i : keep) id.push_back(id_[i]);
    Exponents E;
    E.reserve(E_.size());
    for (const std::vector<int> &col : E_) {
        std::vector<int> kept;
        kept.reserve(keep.size());
        for (const std::size_t i : keep) kept.push_back(col[i]);
        E.push_back(std::move(kept));
    }
    MergedColumns merged = priv_merge_columns(G_, E);
    return {c_, std::move(merged.G), GI_, std::move(merged.E), std::move(id), detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
