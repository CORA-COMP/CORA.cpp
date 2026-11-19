// isequal - whether two boxes are equal, as CORA's interval.isequal
//
// False for boxes of different shape; for a batch, true if every member is equal.
//
// Syntax:   bool b = I.isequal(I2);   bool b = I.isequal(I2, tol);   bool b = I.isequal(p, tol);
// Inputs:   I2 - interval
//           p - a point, the box [p, p]
//           tol - how far finite bounds may differ (within_tol); default: the repository one
// Outputs:  b - true if the bounds agree up to tol and the infinite ones are the same
// See also: is_intersecting, less_equal
//
// CORA: contSet/@interval/isequal.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the bounds a and b are equal: the same infinities, finite entries up to tol.
bool aux_same(const Tensor &a, const Tensor &b, double tol) {
    const Tensor infinite = a.isinf().maximum(b.isinf());
    return Tensor::where(infinite, a.equal(b), within_tol(a, b, tol)).min().item() != 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Interval::isequal(const Interval &I2, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "isequal");
    // CORA answers false for a different dimension, it does not raise
    if (inf_.shape() != I2.inf().shape() || elem_rank_ != I2.elem_rank_) return false;
    if (!(empty_ == I2.empty_)) return false;
    return is_empty() || inf_.numel() == 0 ||
           (aux_same(inf_, I2.inf(), tol) && aux_same(sup_, I2.sup(), tol));
}

bool Interval::isequal(const Tensor &p, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= elem_rank_, Error::dimension_mismatch, [&] {
        return "isequal: the point must be a vector or a matrix, got the shape " +
               shape_text(p.shape());
    });
    return isequal(Interval(p, p, elem_rank_, {}), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
