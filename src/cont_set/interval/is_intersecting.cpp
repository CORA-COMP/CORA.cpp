// is_intersecting - whether two boxes meet, as CORA's interval.isIntersecting_
//
// Syntax:   bool b = I.is_intersecting(I2);   bool b = I.is_intersecting(I2, tol);
//           bool b = I.is_intersecting(p, tol);
// Inputs:   I2 - interval of the dimension of I;  p - a point, a column (..., n, 1)
//           tol - how far apart the bounds of a coordinate may be, as within_tol; default: the
//                 repository tolerance (docs/architecture.md)
// Outputs:  b - true if in every coordinate the bounds overlap up to tol; false if one is empty;
//           for a point, whether it is in the box (up to tol); for a batch, true if all members do.
//           Every entry of an interval matrix counts (CORA tests only the first max(rows, columns))
// See also: contains, intersect
//
// CORA: contSet/@interval/isIntersecting_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// 1 where a <= b, or a and b agree up to tol.
Tensor aux_below(const Tensor &a, const Tensor &b, double tol) {
    return a.less_equal(b).maximum(within_tol(a, b, tol));
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Interval::is_intersecting(const Interval &I2, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "is_intersecting");
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "is_intersecting: the intervals differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(I2.dim());
    });
    if (is_empty() || I2.is_empty()) return false;
    // the lower bound of the box that starts first lies below the upper bound of the other
    const Tensor first = aux_below(inf_, I2.inf(), tol);
    const Tensor meet =
        Tensor::where(first, aux_below(I2.inf(), sup_, tol), aux_below(inf_, I2.sup(), tol));
    return meet.min().item() != 0;
}

bool Interval::is_intersecting(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "is_intersecting");
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        return "is_intersecting: the point must be a column of the dimension " +
               std::to_string(dim()) + " of the interval, got the shape " + shape_text(p.shape());
    });
    return !is_empty() && contains(p, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
