// priv_empty - the empty interval of the shape of a tensor
//
// Syntax:   Interval E = priv_empty(like);
// Inputs:   like - a tensor of the shape of the bounds (..., n, k) and the backend of the result
// Outputs:  E - the interval of bounds (..., n, 0): no columns, as CORA's empty interval
// See also: intersect, mink_diff
//
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Interval priv_empty(const Interval &like);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
