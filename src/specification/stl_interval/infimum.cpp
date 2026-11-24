// infimum - the lower bound of the interval and whether it belongs to it
//
// Syntax:   StlEndpoint e = I.infimum();
// Inputs:   (none)
// Outputs:  e - the bound and whether it is attained
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/infimum.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlEndpoint StlInterval::infimum() const {
    detail::Entry entry;
    if (empty_) return {std::nan(""), true};
    return {lower_, lc_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
