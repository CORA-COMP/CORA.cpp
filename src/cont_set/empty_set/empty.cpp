// empty - the empty set of n dimensions, as CORA's emptySet.empty
//
// Syntax:   EmptySet E = EmptySet::empty(n);
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  E - the empty set of n dimensions
// See also: EmptySet::EmptySet, Interval::empty
//
// CORA: contSet/@emptySet/empty.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    return EmptySet(n);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
