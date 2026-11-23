// restore_id - a polynomial zonotope with identifiers added, as CORA's restoreId
//
// The identifiers the set lacks get a row of zero exponents at the end, in the order given; the
// set does not change. CORA adds an identifier that is repeated in ids once per repetition, here
// repeated identifiers are refused.
//
// Syntax:   S = pZ.restore_id(ids);
// Inputs:   ids - the identifiers that the result must have, distinct
// Outputs:  S - the same set over the identifiers of pZ followed by the missing ones
// See also: rm_id, replace_id
//
// CORA: contSet/@polyZonotope/restoreId.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <set>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::restore_id(const std::vector<int64_t> &ids) const {
    detail::Entry entry;
    CORA_CHECK(std::set<int64_t>(ids.begin(), ids.end()).size() == ids.size(), Error::wrong_value,
               [] { return std::string("restore_id: ids must not contain duplicates"); });
    std::vector<int64_t> id = id_;
    for (const int64_t v : ids)
        if (priv_find_id(id_, v) < 0) id.push_back(v);
    // zero exponents for the new identifiers in every column
    Exponents E = E_;
    for (std::vector<int> &col : E) col.resize(id.size(), 0);
    return {c_, G_, GI_, std::move(E), std::move(id), detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
