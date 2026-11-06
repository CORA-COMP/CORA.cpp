// intersect - the intersection of the whole space with a set, as CORA's fullspace.and_
//
// Syntax:   Set R = F.intersect(S);   Set R = F & S;
// Inputs:   S - a set of the dimension of F (a point has no class here)
// Outputs:  R - S itself: the whole space meets every set in that set
// See also: Interval::intersect
//
// CORA: contSet/@fullspace/and_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Fullspace::intersect(const Set &S) const {
    detail::Entry entry;
    CORA_CHECK(S->dim() == n_, Error::dimension_mismatch, [&] {
        return "intersect: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S->dim());
    });
    return S;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
