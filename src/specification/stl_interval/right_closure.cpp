// right_closure - the time interval with its upper end closed
//
// Syntax:   K = I.right_closure();
// Inputs:   (none)
// Outputs:  K - the interval, upper end closed
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/rightClosure.m
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

StlInterval StlInterval::right_closure() const {
    detail::Entry entry;
    if (empty_ || std::isinf(upper_)) return *this;
    return {lower_, upper_, lc_, true};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
