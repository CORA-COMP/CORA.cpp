// left_closure - the time interval with its lower end closed
//
// Syntax:   K = I.left_closure();
// Inputs:   (none)
// Outputs:  K - the interval, lower end closed
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/leftClosure.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlInterval StlInterval::left_closure() const {
    detail::Entry entry;
    if (empty_) return *this;
    return {lower_, upper_, true, rc_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
