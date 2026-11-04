// ellipsoid_norm - the norm whose unit ball is an ellipsoid, as CORA's ellipsoidNorm
//
// Syntax:   Tensor res = E.ellipsoid_norm(p);
// Inputs:   p - vectors (..., n, 1)
// Outputs:  res - sqrt(|p' Q^+ p|) (..., 1, 1): from Q \ p for a full-rank Q; else from the
//           singular value decomposition Q = U S V', with singular values and entries of U' p
//           below 1e-10 set to zero, so that a direction outside the range of Q has norm Inf
// See also: generators, contains
//
// CORA: contSet/@ellipsoid/ellipsoidNorm.m

#include "cont_set/ellipsoid/private/priv.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x where the mask is set, else y; the mask is (..., 1, 1).
Tensor aux_select(const Tensor &mask, const Tensor &x, const Tensor &y) {
    return Tensor::where(mask, x, y);
}

/// p' Q^+ p for a degenerate Q through its singular value decomposition, entries below the
/// threshold cut to zero as in CORA; a division by a zero singular value gives Inf, 0 / 0 gives 0.
Tensor aux_degenerate(const Tensor &Q, const Tensor &p) {
    const double zero = numerics::kEllipsoidNormZero;
    const std::array<Tensor, 3> usv = Q.svd();
    const Tensor s = aux_select(usv[1].less(usv[1].full_like(usv[1].shape(), zero)),
                                usv[1].zeros_like(), usv[1]);
    const Tensor rotated = usv[0].matrix_transpose().matmul(p);
    const Tensor cut = aux_select(rotated.abs().less(rotated.full_like(rotated.shape(), zero)),
                                  rotated.zeros_like(), rotated);
    const Tensor scaled = (1.0 / s).multiply(cut);
    const Tensor q = aux_select(scaled.isnan(), scaled.zeros_like(), scaled);
    return cut.matrix_transpose().matmul(q);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::ellipsoid_norm(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(!priv_is_empty(*this), Error::empty_set,
               [] { return std::string("ellipsoid_norm: the ellipsoid is empty"); });
    CORA_CHECK(p.defined() && is_column(p, dim()), Error::wrong_value, [&] {
        return "ellipsoid_norm: p must be a column (..., " + std::to_string(dim()) +
               ", 1), got the shape " + (p.defined() ? shape_text(p.shape()) : "()");
    });
    // a full-rank member is solved, the others use the decomposition; the solve sees the
    // identity for a degenerate member
    const Tensor full = priv_rank(Q_, tol_).equal(Q_.ones_like({1, 1}) * double(dim()));
    const Tensor solved = aux_select(full, Q_, Q_.eye_like(dim())).solve(p);
    const Tensor quadratic =
        aux_select(full, p.matrix_transpose().matmul(solved), aux_degenerate(Q_, p));
    const Tensor res = quadratic.abs().sqrt();
    // NaN (Inf - Inf in the decomposition) is Inf
    return aux_select(res.isnan(),
                      res.full_like(res.shape(), std::numeric_limits<double>::infinity()), res);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
