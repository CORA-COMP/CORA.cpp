// enlarge - a box scaled about its center, as CORA's interval.enlarge
//
// For an interval matrix factor is a matrix of the shape of the bounds. Where r is Inf: f > 1
// gives (-Inf, Inf), f = 1 keeps the bounds (CORA gives NaN where the center is NaN), and f < 1
// is refused (Error::not_supported).
//
// Syntax:   Interval R = I.enlarge(factor);   Interval R = I.enlarge(f);
// Inputs:   factor - a number, or a column (..., n, 1) of one factor per dimension; nonnegative
// Outputs:  R - [c - r f, c + r f] with the center c and the radius r
// See also: center, rad
//
// CORA: contSet/@interval/enlarge.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_broadcastable.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <limits>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether a factor fits the bounds: one number, or of the shape of the elements.
bool aux_fits(const Tensor &factor, const Interval &I) {
    return factor.numel() == 1 || priv_fits_elements(factor, I);
}

/// The bounds of the box scaled by the factors (checked): CORA's formula, with the unbounded
/// half-widths set.
std::pair<Tensor, Tensor> aux_enlarge(const Interval &I, const Tensor &factor) {
    const Tensor c = I.center(), r = I.half_width();
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
    CORA_CHECK(aux_fits(factor, *this), Error::dimension_mismatch, [&] {
        return "enlarge: the factor must be a number or of the shape of the elements " +
               shape_text(elem_shape()) + ", got " + shape_text(factor.shape());
    });
    CORA_CHECK(!(factor.min().item() < 0), Error::wrong_value,
               [] { return std::string("enlarge: the factor must be nonnegative"); });
    auto [lo, hi] = aux_enlarge(*this, factor);
    return with(std::move(lo), std::move(hi));
}

Interval Interval::enlarge(double factor) const {
    detail::Entry entry;
    CORA_CHECK(!(factor < 0), Error::wrong_value, [&] {
        return "enlarge: the factor must be nonnegative, got " + std::to_string(factor);
    });
    auto [lo, hi] = aux_enlarge(*this, inf_.full_like({1}, factor));
    return with(std::move(lo), std::move(hi));
}

Set Interval::enlarge_impl(const Tensor &factor) const { return enlarge(factor); }


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
