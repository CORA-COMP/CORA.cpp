// empty_set - the constructor of the empty set, checked, as CORA's emptySet constructor
//
// Syntax:   EmptySet E(n);
// Inputs:   n - the dimension, an integer >= 0
// Outputs:  E - the empty set of n dimensions
// See also: empty, Interval::Interval
//
// CORA: contSet/@emptySet/emptySet.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet::EmptySet(int64_t n) : ContSet(kClassId), n_(n) {
    detail::Entry entry;
    CORA_CHECK(n_ >= 0, Error::wrong_input_in_constructor, [&] {
        return "EmptySet: the dimension must be a nonnegative integer, got " + std::to_string(n_);
    });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
