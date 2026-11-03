// complement - the complement of the whole space, as CORA's fullspace.not
//
// Syntax:   EmptySet E = F.complement();
// Inputs:   -
// Outputs:  E - the empty set of the dimension of F. CORA calls it `not`, a keyword of C++
// See also: EmptySet::complement
//
// CORA: contSet/@fullspace/not.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet Fullspace::complement() const {
    detail::Entry entry;
    return EmptySet(n_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
