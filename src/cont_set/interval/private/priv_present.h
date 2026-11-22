// priv_present - which members of a batch of intervals are not empty
//
// Syntax:   Tensor ok = priv_present(I);
// Inputs:   I - a batch of intervals
// Outputs:  ok - 1 for a member that is not empty, else 0, of the batch shape of I
// See also: priv_empty, mask
//
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Tensor priv_present(const Interval &I);

/// The flags t of the entries of the elements of I (..., n) or (..., r, c), reduced to one flag
/// per member: 1 where every entry of the member is 1.
Tensor priv_all_entries(const Tensor &t, const Interval &I);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
