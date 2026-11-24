// empty - the empty interval, as CORA's stlInterval.empty(n)
//
// Syntax:   I = StlInterval::empty();
// Inputs:   n - the dimension, 1
// Outputs:  I - the empty interval
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/empty.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlInterval StlInterval::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n == 1, Error::wrong_value,
               [] { return std::string("empty: STL intervals are always one-dimensional"); });
    return {};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
