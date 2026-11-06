// complement - the complement of the empty set, as CORA's emptySet.not
//
// Syntax:   Fullspace F = E.complement();
// Inputs:   -
// Outputs:  F - the whole space of the dimension of E. CORA calls it `not`, a keyword of C++
// See also: Fullspace::complement
//
// CORA: contSet/@emptySet/not.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

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
