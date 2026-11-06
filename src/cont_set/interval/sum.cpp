// sum - the sum of the entries of an interval matrix along a dimension, as CORA's interval.sum
//
// The sums nest to the right, as CORALean's sum, so the roundings agree with it.
//
// Syntax:   Interval R = I.sum();   Interval R = I.sum(dim);
// Inputs:   dim - 1: down the rows, 2: along the columns; default: 1, or 2 for a single row
// Outputs:  R - the bounds summed
// See also: prod, plus
//
// CORA: contSet/@interval/sum.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::sum(std::optional<int> dim) const {
    detail::Entry entry;
    const int along = dim.value_or(inf_.size(-2) == 1 ? 2 : 1);
    CORA_CHECK(along == 1 || along == 2, Error::wrong_value,
               [] { return std::string("sum: the second argument must be either 1 or 2"); });
    const int axis = along == 1 ? -2 : -1;
    return {inf_.sum(axis, true), sup_.sum(axis, true)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
