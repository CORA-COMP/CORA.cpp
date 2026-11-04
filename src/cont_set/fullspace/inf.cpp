// inf - the whole space of n dimensions, as CORA's fullspace.Inf
//
// Syntax:   Fullspace F = Fullspace::inf(n);
// Inputs:   n - dimension, an integer >= 0
// Outputs:  F - R^n
// See also: Interval::unbounded, enclose_points
//
// CORA: contSet/@fullspace/Inf.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::inf(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value,
               [&] { return "inf: the dimension must be nonnegative, got " + std::to_string(n); });
    return Fullspace(n);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
