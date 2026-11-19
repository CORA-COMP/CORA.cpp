// contains - point containment in a box, as CORA's interval.contains
//
// Syntax:   res = I.contains(p);   res = I.contains(p, tol);   res = I.contains(p, type, tol);
// Inputs:   p - point (..., n), or a matrix (..., r, c) of an interval matrix
//           type - 'exact' or 'approx' (CORA's; a box answers both exactly)
//           tol - how far p may lie outside, per coordinate; default: the repository tolerance
// Outputs:  res - true if inf - tol <= p <= sup + tol; for a batch, if all points are
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
    // a point of a box, or a matrix of an interval matrix
    CORA_CHECK(priv_fits_elements(p, *this), Error::dimension_mismatch, [&] {
        return "contains: the point must have the shape " + shape_text(elem_shape()) +
               " of the elements, got " + shape_text(p.shape());
    });
    if (is_empty()) return false;
    // the bounds and the points broadcast to one shape (a batch of boxes, or of points)
    const Tensor zero = p.zeros_like() + inf_.zeros_like();
    return aux_all_inside((inf_ + zero).data(), (p + zero).data(), (sup_ + zero).data(), tol);
}

bool Interval::contains(const Tensor &p, const std::string &type, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(type == "exact" || type == "approx", Error::wrong_value, [&] {
        return "contains: the type must be 'exact' or 'approx', got '" + type + "'";
    });
    return contains(p, resolve_tol(entry, tol, inf_.dtype(), "contains"));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
