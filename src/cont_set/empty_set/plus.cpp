// plus - the Minkowski sum of the empty set with a set, as CORA's emptySet.plus
//
// Syntax:   EmptySet R = E.plus(S);   EmptySet R = E.plus(v);
// Inputs:   S - a set of the dimension of E;  v - a column (..., n, 1), a translation
// Outputs:  R - the empty set
// See also: intersect, Interval::plus
//
// CORA: contSet/@emptySet/plus.m

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
    CORA_CHECK(is_column(v, n_), Error::dimension_mismatch, [&] {
        return "plus: the vector must be a column of the dimension " + std::to_string(n_) +
               " of the set, got the shape " + shape_text(v.shape());
    });
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
