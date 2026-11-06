// is_bounded - whether the whole space is bounded, as CORA's fullspace.isBounded
//
// Syntax:   bool b = F.is_bounded();
// Inputs:   -
// Outputs:  b - true only in R^0
// See also: Interval::is_bounded
//
// CORA: contSet/@fullspace/isBounded.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Fullspace::is_bounded() const {
    detail::Entry entry;
    return n_ == 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
