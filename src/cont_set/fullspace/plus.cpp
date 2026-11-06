// plus - the Minkowski sum of the whole space and a set, as CORA's fullspace.plus
//
// S is empty if it is an EmptySet or an empty interval, where CORA returns an emptySet.
// Error::not_supported in R^0, as CORA, unless S is an EmptySet (CORA hands that sum to the empty
// set first).
//
// Syntax:   Set R = F.plus(S);
// Inputs:   S - a set of the dimension of F (a point p: the interval [p, p])
// Outputs:  R - the whole space, or the empty set of the dimension if S is empty
// See also: EmptySet::plus, Interval::plus
//
// CORA: contSet/@fullspace/plus.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"

#include "cont_set/fullspace/private/priv_represents_empty.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Fullspace::plus(const ContSet &S) const {
    detail::Entry entry;
    CORA_CHECK(n_ > 0 || S.class_id() == ClassId::empty_set, Error::not_supported,
               [] { return std::string("plus: the Minkowski sum of R^0 is not supported"); });
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "plus: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    if (priv_represents_empty(S)) return Set(EmptySet(n_));
    return Set(*this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
