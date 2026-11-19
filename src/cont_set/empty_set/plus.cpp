// plus - the Minkowski sum of the empty set with a set, as CORA's emptySet.plus
//
// Syntax:   EmptySet R = E.plus(S);   EmptySet R = E.plus(v);
// Inputs:   S - a set of the dimension of E
//           v - a point (..., n), a translation
// Outputs:  R - the empty set
// See also: intersect, Interval::plus
//
// CORA: contSet/@emptySet/plus.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::plus(const ContSet &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "plus: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    return *this;
}

EmptySet EmptySet::plus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(is_vector(v, n_), Error::dimension_mismatch, [&] {
        return "plus: the point must be a vector of the dimension " + std::to_string(n_) +

               " of the set, got the shape " + shape_text(v.shape());
    });
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
