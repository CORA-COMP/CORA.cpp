// center - the center of a box, as CORA's interval.center
//
// The whole space gives the origin, an empty member NaN (CORA: zeros(n, 0)). ContSet::center()
// gives a box as a column (..., n, 1).
//
// Syntax:   Tensor c = I.center();
// Inputs:   -
// Outputs:  c - (inf + sup) / 2 elementwise, of the shape of the bounds, NaN where it is infinite
// See also: half_width, Zonotope::center
//
// CORA: contSet/@interval/center.m
// CORA: matrixSet/@intervalMatrix/center.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::center() const {
    detail::Entry entry;
    const Tensor c = (inf_ + sup_) * 0.5;
    const Tensor nan = c.full_like(c.shape(), std::numeric_limits<double>::quiet_NaN());
    if (is_empty()) return nan;
    const Tensor finite_or_nan = Tensor::where(c.isinf(), nan, c);
    if (elem_rank_ != 1) return finite_or_nan;
    // the whole space of a box has the origin as its center (per batch member)
    const double kInf = std::numeric_limits<double>::infinity();
    const Tensor all_below = inf_.equal(inf_.full_like(inf_.shape(), -kInf)).all(-1, true);
    const Tensor all_above = sup_.equal(sup_.full_like(sup_.shape(), kInf)).all(-1, true);
    return Tensor::where(all_below.multiply(all_above), c.zeros_like(), finite_or_nan);
}

// the generic center is a column, as every other set's
Tensor Interval::center_impl() const {
    return elem_rank_ == 1 ? center().expand_dims(-1) : center();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
