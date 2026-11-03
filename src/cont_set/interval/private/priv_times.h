// priv_times - the entrywise product of two boxes
//
// Syntax:   auto [lo, hi] = priv_times(a_inf, a_sup, b_inf, b_sup);
// Inputs:   a_inf, a_sup, b_inf, b_sup - the bounds of the factors, broadcast against each other
// Outputs:  lo, hi - the hull of the four corner products, a NaN corner (0 * Inf) taken as 0, as
//           CORA's times
// See also: priv_matmul, prod, rdivide

#pragma once

#include "cora/global/tensor/tensor.h"

#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

std::pair<Tensor, Tensor> priv_times(const Tensor &a_inf, const Tensor &a_sup, const Tensor &b_inf,
                                     const Tensor &b_sup);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
