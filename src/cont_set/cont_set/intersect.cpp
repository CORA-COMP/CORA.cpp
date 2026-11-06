// intersect - the intersection of two sets, as CORA's contSet.and
//
// Syntax:   S = intersect(S1, S2);   S = Set(S1) & Set(S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  S - a set that contains S1 ∩ S2 (exact where the pair allows)
// See also: registry, unite
//
// CORA: contSet/@contSet/and.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set intersect(const ContSet &S1, const ContSet &S2) {
    detail::Entry entry;
    const detail::Refusal r = detail::refusal(BinaryOp::intersect, S1, S2);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return detail::intersect_impl(S1, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
