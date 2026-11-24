// to_left - the times from 0 up to the start of the interval
//
// Syntax:   K = I.to_left();
// Inputs:   (none)
// Outputs:  K - the times before the interval
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/toLeft.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlInterval StlInterval::to_left() const {
    detail::Entry entry;
    CORA_CHECK(!empty_, Error::empty_set,
               [] { return std::string("to_left: the interval is empty"); });
    return {0, lower_, true, !lc_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
