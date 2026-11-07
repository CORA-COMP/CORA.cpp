// isnan - whether an ellipsoid holds NaN, as CORA's ellipsoid.isnan
//
// CORA answers false for every ellipsoid; the constructor refuses non-finite entries, so it is
// true.
//
// Syntax:   res = E.isnan();
// Inputs:   -
// Outputs:  res - false
// See also: is_bounded
//
// CORA: contSet/@ellipsoid/isnan.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Ellipsoid::isnan() const {
    detail::Entry entry;
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
