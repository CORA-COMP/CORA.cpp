// max - the elementwise larger of two boxes, as CORA's interval.max
//
// Without Y, the upper bound sup. MATLAB's dim and nanflag options are not offered.
//
// Syntax:   Interval R = I.max(Y);   Tensor ub = I.max();
// Inputs:   Y - a set (its interval hull is used) or a number or array of the shape of the bounds
// Outputs:  R - [max(inf, inf_Y), max(sup, sup_Y)]: exactly the entrywise maxima of members
// See also: min, unite
//
// CORA: contSet/@interval/max.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::max(const ContSet &Y) const {
    detail::Entry entry;
    CORA_CHECK(Y.dim() == dim(), Error::dimension_mismatch, [&] {
        return "max: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Y.dim());
    });
    const Interval box = Y.interval();
    return {inf_.maximum(box.inf()), sup_.maximum(box.sup())};
}

Interval Interval::max(const Tensor &Y) const {
    detail::Entry entry;
    CORA_CHECK(Y.numel() == 1 || is_column(Y, dim()), Error::dimension_mismatch, [&] {
        return "max: the array must be a number or a column of the dimension " +
               std::to_string(dim()) + " of the interval, got the shape " + shape_text(Y.shape());
    });
    return {inf_.maximum(Y), sup_.maximum(Y)};
}

Tensor Interval::max() const {
    detail::Entry entry;
    return sup_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
