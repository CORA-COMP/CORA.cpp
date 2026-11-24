// to_right - the times after the end of the interval
//
// Syntax:   K = I.to_right();
// Inputs:   (none)
// Outputs:  K - the times after the interval
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/toRight.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlInterval StlInterval::to_right() const {
    detail::Entry entry;
    CORA_CHECK(!empty_, Error::empty_set,
               [] { return std::string("to_right: the interval is empty"); });
    return {upper_, std::numeric_limits<double>::infinity(), !rc_, false};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
