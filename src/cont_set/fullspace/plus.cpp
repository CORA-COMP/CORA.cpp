// plus - the Minkowski sum of the whole space and a set, as CORA's fullspace.plus
//
// Syntax:   Fullspace R = F.plus(S);
// Inputs:   S - a set of the dimension of F (a point p: the interval [p, p])
// Outputs:  R - the whole space. Error::not_supported in R^0, as CORA, and for an empty interval
//           S: the sum is empty, which CORA returns as an emptySet, a class CORA.cpp lacks
// See also: Interval::plus
//
// CORA: contSet/@fullspace/plus.m

#include "cora/cont_set/fullspace/fullspace.h"

#include "cont_set/fullspace/private/priv_represents_empty.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::plus(const ContSet &S) const {
    detail::Entry entry;
    CORA_CHECK(n_ > 0, Error::not_supported,
               [] { return std::string("plus: the Minkowski sum of R^0 is not supported"); });
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "plus: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    CORA_CHECK(!priv_represents_empty(S), Error::not_supported, [] {
        return std::string("plus: the sum with an empty set is empty; there is no EmptySet yet");
    });
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
