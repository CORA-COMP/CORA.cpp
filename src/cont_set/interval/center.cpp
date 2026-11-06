// center - the center of a box, as CORA's interval.center
//
// Syntax:   Tensor c = I.center();
// Inputs:   -
// Outputs:  c - (inf + sup) / 2 elementwise, NaN where it is infinite; the origin for the whole
//           space, and (n, 0) for an empty interval
// See also: rad, Zonotope::center
//
// CORA: contSet/@interval/center.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::center_impl() const {
    const Tensor c = (inf_ + sup_) * 0.5;
    const Tensor nan = c.full_like(c.shape(), std::numeric_limits<double>::quiet_NaN());
    const Tensor finite_or_nan = Tensor::where(c.isinf(), nan, c);
    if (is_empty() || inf_.shape().back() != 1) return finite_or_nan;
    // the whole space of a box has the origin as its center (per batch member)
    const double kInf = std::numeric_limits<double>::infinity();
    const Tensor all_below = inf_.equal(inf_.full_like(inf_.shape(), -kInf)).all(-2, true);
    const Tensor all_above = sup_.equal(sup_.full_like(sup_.shape(), kInf)).all(-2, true);
    return Tensor::where(all_below.multiply(all_above), c.zeros_like(), finite_or_nan);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
