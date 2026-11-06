// min - the elementwise smaller of two boxes, as CORA's interval.min
//
// Without Y, the lower bound inf. MATLAB's dim and nanflag options are not offered.
//
// Syntax:   Interval R = I.min(Y);   Tensor lb = I.min();
// Inputs:   Y - a set (its interval hull is used) or a number or array of the shape of the bounds
// Outputs:  R - [min(inf, inf_Y), min(sup, sup_Y)]: exactly the entrywise minima of members
// See also: max, unite
//
// CORA: contSet/@interval/min.m
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

Interval Interval::min(const ContSet &Y) const {
    detail::Entry entry;
    CORA_CHECK(Y.dim() == dim(), Error::dimension_mismatch, [&] {
        return "min: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Y.dim());
    });
    const Interval box = Y.interval();
    return {inf_.minimum(box.inf()), sup_.minimum(box.sup())};
}

Interval Interval::min(const Tensor &Y) const {
    detail::Entry entry;
    CORA_CHECK(Y.numel() == 1 || is_column(Y, dim()), Error::dimension_mismatch, [&] {
        return "min: the array must be a number or a column of the dimension " +
               std::to_string(dim()) + " of the interval, got the shape " + shape_text(Y.shape());
    });
    return {inf_.minimum(Y), sup_.minimum(Y)};
}

Tensor Interval::min() const {
    detail::Entry entry;
    // the lower bound
    return inf_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
