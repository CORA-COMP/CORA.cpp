// priv_times - the entrywise product of two boxes
//
// A NaN corner (0 * Inf) is taken as 0, as CORA's times.
//
// Syntax:   auto [lo, hi] = priv_times(a_inf, a_sup, b_inf, b_sup);
// Inputs:   a_inf, a_sup, b_inf, b_sup - the bounds of the factors, broadcast against each other
// Outputs:  lo, hi - the hull of the four corner products
// See also: priv_matmul, prod, rdivide
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

std::pair<Tensor, Tensor> priv_times(const Tensor &a_inf, const Tensor &a_sup, const Tensor &b_inf,
                                     const Tensor &b_sup);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
