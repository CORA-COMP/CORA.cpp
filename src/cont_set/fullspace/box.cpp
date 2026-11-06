// box - the smallest box around the whole space, as CORA's fullspace.box
//
// Syntax:   Fullspace B = F.box();
// Inputs:   -
// Outputs:  B - the set itself: the whole space is its own box (interval() has its bounds)
// See also: interval
//
// CORA: contSet/@fullspace/box.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::box() const {
    detail::Entry entry;
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
