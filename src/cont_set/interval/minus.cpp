// minus - the difference of a box and a box or a point, as CORA's interval.minus
//
// Syntax:   Interval R = I.minus(I2);   Interval R = I.minus(v);
//           Interval R = Interval::minus(v, I);   I - I2;   I - v;   v - I;
// Inputs:   I2 - interval of the shape of I;  v - a number, or an array of the shape of the bounds
// Outputs:  R - [inf - sup2, sup - inf2]; [inf - v, sup - v]; [v - sup, v - inf]
// See also: plus, minkowski_difference
//
// CORA: contSet/@interval/minus.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether v is a number or has the matrix shape (n, k) of the bounds (batches broadcast).
bool aux_fits(const Tensor &v, const Tensor &bounds) {
    return v.numel() == 1 || (v.ndim() >= 2 && v.size(-2) == bounds.size(-2) &&
                              v.size(-1) == bounds.size(-1));
}

/// What the check of v says.
std::string aux_message(const Tensor &v) {
    return "minus: the subtrahend must be a number or of the shape of the bounds, got " +
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
    return {inf_ - I2.sup(), sup_ - I2.inf()};
}

Interval Interval::minus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(aux_fits(v, inf_), Error::dimension_mismatch, [&] { return aux_message(v); });
    return {inf_ - v, sup_ - v};
}

Interval Interval::minus(const Tensor &v, const Interval &I) {
    detail::Entry entry;
    CORA_CHECK(aux_fits(v, I.inf()), Error::dimension_mismatch, [&] { return aux_message(v); });
    return {v - I.sup(), v - I.inf()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
