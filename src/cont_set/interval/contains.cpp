// contains - point containment in a box, as CORA's interval.contains
//
// Syntax:   res = I.contains(p);   res = I.contains(p, tol);
// Inputs:   p - point (..., n, 1)
//           tol - how far p may lie outside, per coordinate, absolutely or relative to the bound
//                 (as within_tol); default: the
//                 repository tolerance (docs/architecture.md)
// Outputs:  res - true if inf - tol <= p <= sup + tol; for a batch, if all points are
// See also: Specification::check
//
// CORA: contSet/@interval/contains_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether every point is inside, coordinate by coordinate as CORA's interval.contains_.
bool aux_all_inside(const std::vector<double> &lo, const std::vector<double> &x,
                    const std::vector<double> &hi, double tol) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const bool above = lo[i] < x[i] + tol || within_tol(lo[i], x[i], tol);
        const bool below = hi[i] > x[i] - tol || within_tol(hi[i], x[i], tol);
        if (!above || !below) return false;
    }
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Interval::contains(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "contains");
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        return "contains: the point must be a column of the dimension " + std::to_string(dim()) +
               " of the interval, got the shape " + shape_text(p.shape());
    });
    // the bounds and the points broadcast to one shape (a batch of boxes, or of points)
    const Tensor zero = p.zeros_like() + inf_.zeros_like();
    return aux_all_inside((inf_ + zero).data(), (p + zero).data(), (sup_ + zero).data(), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
