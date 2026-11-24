// interval - the closed Interval of the bounds of the interval
//
// Syntax:   Interval J = I.interval();
// Inputs:   (none)
// Outputs:  J - the closed one-dimensional Interval
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/interval.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The empty interval gives the empty Interval, as CORALean's [1, 0].
CORA_FROM_NUMBERS Interval StlInterval::interval() const {
    detail::Entry entry;
    if (empty_) return Interval::empty(1);
    return {Tensor::from_data({lower_}, {1}), Tensor::from_data({upper_}, {1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
