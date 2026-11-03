// isequal - whether two boxes are equal, as CORA's interval.isequal
//
// Syntax:   bool b = I.isequal(I2);   bool b = I.isequal(I2, tol);   bool b = I.isequal(p, tol);
// Inputs:   I2 - interval;  p - a point, the box [p, p]
//           tol - how far finite bounds may differ, as within_tol; CORA's default eps
// Outputs:  b - true if the bounds agree up to tol and the infinite ones are the same; false for
//           boxes of different shape; for a batch, true if every member is equal
// See also: is_intersecting, less_equal
//
// CORA: contSet/@interval/isequal.m

#include "cont_set/interval/private/priv_within_tol.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the bounds a and b are equal: the same infinities, finite entries up to tol.
bool aux_same(const Tensor &a, const Tensor &b, double tol) {
    const Tensor infinite = a.isinf().maximum(b.isinf());
    return Tensor::where(infinite, a.equal(b), priv_within_tol(a, b, tol)).min().item() != 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Interval::isequal(const Interval &I2, double tol) const {
    detail::Entry entry;
    CORA_CHECK(tol >= 0, Error::wrong_value,
               [&] { return "isequal: tol must be nonnegative, got " + std::to_string(tol); });
    // CORA answers false for a different dimension, it does not raise
    if (inf_.shape() != I2.inf().shape()) return false;
    return inf_.numel() == 0 || (aux_same(inf_, I2.inf(), tol) && aux_same(sup_, I2.sup(), tol));
}

bool Interval::isequal(const Tensor &p, double tol) const {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= 2, Error::dimension_mismatch, [&] {
        return "isequal: the point must be a column or a matrix, got the shape " +
               shape_text(p.shape());
    });
    return isequal(Interval(p, p), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
