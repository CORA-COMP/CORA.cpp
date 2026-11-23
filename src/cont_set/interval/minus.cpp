// minus - the difference of a box and a box or a point, as CORA's interval.minus
//
// Syntax:   Interval R = I.minus(I2);   Interval R = I.minus(v);
//           Interval R = Interval::minus(v, I);   I - I2;   I - v;   v - I;
// Inputs:   I2 - interval of the shape of I
//           v - a number, or an array of the shape of the bounds
// Outputs:  R - [inf - sup2, sup - inf2]; [inf - v, sup - v]; [v - sup, v - inf]
// See also: plus, mink_diff
//
// CORA: contSet/@interval/minus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_broadcastable.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether v is a number or has the shape of the elements of I (batches broadcast).
bool aux_fits(const Tensor &v, const Interval &I) {
    return v.numel() == 1 || priv_fits_elements(v, I);
}

/// What the check of v says.
std::string aux_message(const Tensor &v) {
    return "minus: the subtrahend must be a number or of the shape of the elements, got " +
           shape_text(v.shape());
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::minus(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "minus: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    return {inf_ - I2.sup(), sup_ - I2.inf(), std::max(elem_rank_, I2.elem_rank_),
            empty_.either(I2.empty_)};
}

Interval Interval::minus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(aux_fits(v, *this), Error::dimension_mismatch, [&] { return aux_message(v); });
    return with(inf_ - v, sup_ - v);
}

Interval Interval::minus(const Tensor &v, const Interval &I) {
    detail::Entry entry;
    CORA_CHECK(aux_fits(v, I), Error::dimension_mismatch, [&] { return aux_message(v); });
    return I.with(v - I.sup(), v - I.inf());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
