// complement - the complement of the empty set, as CORA's emptySet.not
//
// Syntax:   Fullspace F = E.complement();
// Inputs:   -
// Outputs:  F - the whole space of the dimension of E. CORA calls it `not`, a keyword of C++
// See also: Fullspace::complement
//
// CORA: contSet/@emptySet/not.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace EmptySet::complement() const {
    detail::Entry entry;
    return Fullspace(n_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
