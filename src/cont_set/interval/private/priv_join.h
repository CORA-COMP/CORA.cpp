// priv_join - boxes joined along a dimension: the work of cat and cart_prod
//
// Syntax:   bool ok = priv_joinable(A, B, 1);   Interval J = priv_join({I1, I2}, 1);
// Inputs:   A, B - bounds;  Is - intervals;  dim - 1: stacked (rows), 2: side by side (columns)
// Outputs:  priv_joinable - whether the shapes agree in every dimension but dim;
//           priv_join - the interval of the joined bounds (the shapes are not checked)
// See also: cat, cart_prod

#pragma once

#include "cora/cont_set/interval/interval.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_joinable(const Tensor &A, const Tensor &B, int dim);

Interval priv_join(const std::vector<Interval> &Is, int dim);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
