// contains - point containment in a box, as CORA's interval.contains
//
// Syntax:   res = I.contains(p);   res = I.contains(p, tol);   res = I.contains(p, method, tol);
// Inputs:   p - point (..., n), or a matrix (..., r, c) of an interval matrix
//           method - 'exact' or 'approx' (CORA's; a box answers both exactly)
//           tol - how far p may lie outside, per coordinate; default: the repository tolerance
// Outputs:  res - true if inf - tol <= p <= sup + tol, false for an empty member (a mask)
// See also: Specification::check
//
// CORA: contSet/@interval/contains_.m
// CORA: matrixSet/@intervalMatrix/contains.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_broadcastable.h"
#include "cont_set/interval/private/priv_present.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// 1 where the lower bound lo is below x + tol, or lo and x agree up to tol.
Tensor aux_above(const Tensor &lo, const Tensor &x, double tol) {
    return lo.less(x + tol).maximum(within_tol(lo, x, tol));
}

/// 1 where the upper bound hi is above x - tol, or hi and x agree up to tol.
Tensor aux_under(const Tensor &hi, const Tensor &x, double tol) {
    return (x - tol).less(hi).maximum(within_tol(hi, x, tol));
}

} // namespace


// ===========================================  MAIN  =========================================== //

Mask Interval::contains(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "contains");
    // a point of a box, or a matrix of an interval matrix
    CORA_CHECK(priv_fits_elements(p, *this), Error::dimension_mismatch, [&] {
        return "contains: the point must have the shape " + shape_text(elem_shape()) +
               " of the elements, got " + shape_text(p.shape());
    });
    // the bounds and the points broadcast to one shape (a batch of boxes, or of points)
    const Tensor zero = p.zeros_like() + inf_.zeros_like();
    const Tensor x = p + zero;
    // coordinate by coordinate as CORA's interval.contains_: inf - tol <= x <= sup + tol
    const Tensor inside = aux_above(inf_ + zero, x, tol).minimum(aux_under(sup_ + zero, x, tol));
    return Mask(priv_all_entries(inside, *this).minimum(priv_present(*this)));
}

Mask Interval::contains(const Tensor &p, const std::string &method, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(method == "exact" || method == "approx", Error::wrong_value, [&] {
        return "contains: the method must be 'exact' or 'approx', got '" + method + "'";
    });
    return contains(p, resolve_tol(entry, tol, inf_.dtype(), "contains"));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
