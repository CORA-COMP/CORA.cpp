// enlarge - a box scaled about its center, as CORA's interval.enlarge
//
// Syntax:   Interval R = I.enlarge(factor);   Interval R = I.enlarge(f);
// Inputs:   factor - a number, or a column (..., n, 1) of one factor per dimension (a matrix of
//           the shape of the bounds for an interval matrix); nonnegative
// Outputs:  R - [c - r f, c + r f] with the center c and the radius r. Where r is Inf: f > 1
//           gives (-Inf, Inf), f = 1 keeps the bounds (CORA gives NaN where the center is NaN),
//           and f < 1 is refused (Error::not_supported)
// See also: center, rad
//
// CORA: contSet/@interval/enlarge.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether a factor fits the bounds: one number, or the shape of the last two dimensions of them.
bool aux_fits(const Tensor &factor, const Tensor &bounds) {
    if (factor.ndim() < 2) return false;
    const bool number = factor.size(-1) == 1 && factor.size(-2) == 1;
    return number || (factor.size(-1) == bounds.size(-1) && factor.size(-2) == bounds.size(-2));
}

/// The box scaled by the factors (checked): CORA's formula, with the unbounded half-widths set.
Interval aux_enlarge(const Interval &I, const Tensor &factor) {
    const Tensor c = I.center(), r = I.rad();
    const double kInf = std::numeric_limits<double>::infinity();
    const Tensor unbounded = r.equal(r.full_like(r.shape(), kInf));
    if (unbounded.multiply(factor.less(factor.ones_like())).max().item() != 0)
        CORA_FAIL(Error::not_supported,
                  "enlarge: an unbounded interval needs a factor of at least 1");
    const Tensor scaled = r.multiply(factor);
    const Tensor grows = unbounded.multiply(factor.greater(factor.ones_like()));
    const Tensor all = r.full_like(r.shape(), kInf);
    // scaling an unbounded half-width by 1 keeps the bound (CORA's NaN center would spoil it)
    const Tensor keeps = unbounded.multiply(factor.equal(factor.ones_like()));
    const Tensor lower = Tensor::where(keeps, I.inf(), c - scaled);
    const Tensor upper = Tensor::where(keeps, I.sup(), c + scaled);
    return {Tensor::where(grows, -all, lower), Tensor::where(grows, all, upper)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::enlarge(const Tensor &factor) const {
    detail::Entry entry;
    CORA_CHECK(aux_fits(factor, inf_), Error::dimension_mismatch, [&] {
        return "enlarge: the factor must be a number or of the shape of the bounds " +
               shape_text(inf_.shape()) + ", got " + shape_text(factor.shape());
    });
    CORA_CHECK(!(factor.min().item() < 0), Error::wrong_value,
               [] { return std::string("enlarge: the factor must be nonnegative"); });
    return aux_enlarge(*this, factor);
}

Interval Interval::enlarge(double factor) const {
    detail::Entry entry;
    CORA_CHECK(!(factor < 0), Error::wrong_value, [&] {
        return "enlarge: the factor must be nonnegative, got " + std::to_string(factor);
    });
    return aux_enlarge(*this, inf_.full_like({1, 1}, factor));
}

Set Interval::enlarge_impl(const Tensor &factor) const { return enlarge(factor); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
