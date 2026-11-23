// replace_id - a polynomial zonotope with renamed factors, as CORA's replaceId
//
// The identifiers old_ids become new_ids; factors that end up with the same identifier are one
// factor, whose exponents are the sums of the merged ones (in the order of first occurrence).
//
// Syntax:   S = pZ.replace_id(old_ids, new_ids);   S = pZ.replace_id(new_ids);
// Inputs:   old_ids - identifiers of the set (default: all of them, in order)
//           new_ids - the new identifiers, as many as old_ids
// Outputs:  S - the renamed set; the same points when the identifiers stay distinct
// See also: restore_id, rm_id
//
// CORA: contSet/@polyZonotope/replaceId.m
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

PolyZonotope PolyZonotope::replace_id(const std::vector<int64_t> &new_ids) const {
    detail::Entry entry;
    return replace_id(id_, new_ids);
}

PolyZonotope PolyZonotope::replace_id(const std::vector<int64_t> &old_ids,
                                      const std::vector<int64_t> &new_ids) const {
    detail::Entry entry;
    CORA_CHECK(old_ids.size() == new_ids.size() &&
                   std::all_of(old_ids.begin(), old_ids.end(),
                               [&](int64_t v) { return priv_find_id(id_, v) >= 0; }),
               Error::wrong_value, [] {
                   return std::string("replace_id: old_ids and new_ids must have the same length, "
                                      "and old_ids only identifiers of the set");
               });
    std::vector<int64_t> renamed = id_;
    for (std::size_t i = 0; i < old_ids.size(); ++i)
        renamed[std::size_t(priv_find_id(id_, old_ids[i]))] = new_ids[i];
    // the distinct identifiers in order of first occurrence, and the row each factor goes to
    std::vector<int64_t> id;
    std::vector<std::size_t> row(renamed.size());
    for (std::size_t i = 0; i < renamed.size(); ++i) {
        const int64_t at = priv_find_id(id, renamed[i]);
        row[i] = at < 0 ? id.size() : std::size_t(at);
        if (at < 0) id.push_back(renamed[i]);
    }
    Exponents E(E_.size(), std::vector<int>(id.size(), 0));
    for (std::size_t j = 0; j < E_.size(); ++j)
        for (std::size_t i = 0; i < renamed.size(); ++i) E[j][row[i]] += E_[j][i];
    return {c_, G_, GI_, std::move(E), std::move(id), detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
