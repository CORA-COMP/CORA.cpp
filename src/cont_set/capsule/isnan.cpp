// isnan - whether a capsule holds NaN, as CORA's capsule.isnan
//
// CORA answers false for every capsule; the constructor refuses non-finite entries, so it is true.
//
// Syntax:   res = C.isnan();
// Inputs:   -
// Outputs:  res - false
// See also: is_bounded
//
// CORA: contSet/@capsule/isnan.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Capsule::isnan() const {
    detail::Entry entry;
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
