// priv_ellipsoid_norm - the ellipsoid norm of columns, the kernel of Ellipsoid::ellipsoid_norm
//
// It is computed from Q \ p for a full-rank Q; else from the singular value decomposition Q = U S
// V', with singular values and entries of U' p below 1e-10 set to zero, so that a direction
// outside the range of Q has norm Inf. CORA's layout: the points are columns.
//
// Syntax:   Tensor res = priv_ellipsoid_norm(E, p);
// Inputs:   E - a nonempty ellipsoid
//           p - columns (..., n, 1)
// Outputs:  res - sqrt(|p' Q^+ p|) (..., 1, 1)
// See also: priv.h, Ellipsoid::ellipsoid_norm, Ellipsoid::contains
//
// CORA: contSet/@ellipsoid/ellipsoidNorm.m
// CORA authors: Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"

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

Tensor priv_ellipsoid_norm(const Ellipsoid &E, const Tensor &p) {
    const Tensor &Q = E.Q();
    const int64_t n = E.dim();
    // a full-rank member is solved, others use the decomposition; a degenerate one the identity
    const Tensor full = priv_rank(Q, E.tol()).equal(Q.ones_like({1, 1}) * double(n));
    const Tensor solved = aux_select(full, Q, Q.eye_like(n)).solve(p);
    const Tensor quadratic =
        aux_select(full, p.matrix_transpose().matmul(solved), aux_degenerate(Q, p));
    const Tensor res = quadratic.abs().sqrt();
    // NaN (Inf - Inf in the decomposition) is Inf
    return aux_select(res.isnan(),
                      res.full_like(res.shape(), std::numeric_limits<double>::infinity()), res);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
