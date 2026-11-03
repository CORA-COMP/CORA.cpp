// reduce - a set with fewer generators that encloses it, as CORA's contSet.reduce
//
// Syntax:   R = S.reduce(order);
// Inputs:   S - set;  order - generators per dimension that remain (Girard's method), >= 1
// Outputs:  R - a set that contains S (a box stays a box)
// See also: Interval::reduce, Zonotope::reduce
//
// CORA: contSet/@contSet/reduce.m

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::reduce(int order) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the order must be at least 1, got " + std::to_string(order);
    });
    return reduce_impl(order);
}

Set ContSet::reduce_impl(int) const { priv_noops("reduce", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
