// intersect - the intersection of the empty set with a set, as CORA's emptySet.and_
//
// Syntax:   EmptySet R = E.intersect(S);
// Inputs:   S - a set of the dimension of E
// Outputs:  R - the empty set
// See also: plus, is_intersecting
//
// CORA: contSet/@emptySet/and_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::intersect(const ContSet &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "intersect: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
