// center - the center of a set, as CORA's contSet.center
//
// Syntax:   c = S.center();
// Inputs:   S - set
// Outputs:  c - the center (..., n); NaN for an empty member
// See also: interval, Zonotope::center_impl
//
// CORA: contSet/@contSet/center.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ContSet::center() const {
    detail::Entry entry;
    entry.release();
    return center_impl();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
