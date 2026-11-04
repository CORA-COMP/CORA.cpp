// is_bigger - whether one centered ellipsoid contains another, as CORA's ellipsoid.isBigger
//
// Syntax:   bool res = E1.is_bigger(E2);
// Inputs:   E2 - an ellipsoid of the dimension of E1; a single one, not empty
// Outputs:  res - true if E2 - q2 lies in E1 - q1: the largest generalized eigenvalue of Q2
//           against Q1 is below 1 + tol or equal to it up to tol, tol the least tolerance of the
//           two. If E1 is not full-dimensional the comparison is made in the subspace that Q1 + Q2
//           spans, and false if E2 is full-dimensional. CORA returns true for every pair of
//           dimension 1 without comparing (docs/cora-differences.md); here it compares
// See also: is_bad_dir, ellipsoid_norm
//
// CORA: contSet/@ellipsoid/isBigger.m

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/global/within_tol.h"
#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The leading r x r block of the shape matrix Q seen in the basis U (columns): the part of
/// the ellipsoid in the first r directions of the subspace.
Tensor aux_subspace_shape(const Tensor &U, const Tensor &Q, int64_t r) {
    return priv_congruence(U.matrix_transpose(), Q).slice(0, 0, r).slice(1, 0, r);
}

/// Whether the shape Q2 lies in Q1 (both unbatched, the first positive definite): the largest
/// generalized eigenvalue is at most 1 + tol up to tol.
bool aux_loewner(const Tensor &Q1, const Tensor &Q2, double tol) {
    const double ratio = priv_simdiag(Q1, Q2, tol).second.diagonal().max().item();
    return ratio < 1.0 + tol || within_tol(ratio, 1.0 + tol, tol);
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Ellipsoid::is_bigger(const Ellipsoid &E2) const {
    detail::Entry entry;
    const detail::Refusal refusal = priv_pair_refusal(*this, E2, "is_bigger");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const double tol = std::min(tol_, E2.tol());
    if (is_full_dim()) return aux_loewner(Q_, E2.Q(), tol);
    if (E2.is_full_dim()) return false;
    // both degenerate: compare in the subspace that both span, seen in the basis U of Q1 + Q2
    const Tensor sum = (Q_ + E2.Q()) * 0.5;
    const int64_t r = sum.matrix_rank();
    if (r == dim()) return false;
    if (r == 0) return true;
    const Tensor U = sum.svd()[0];
    const Tensor origin = Q_.zeros_like({r, 1});
    return Ellipsoid(aux_subspace_shape(U, Q_, r), origin, tol_)
        .is_bigger(Ellipsoid(aux_subspace_shape(U, E2.Q(), r), origin, E2.tol()));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
