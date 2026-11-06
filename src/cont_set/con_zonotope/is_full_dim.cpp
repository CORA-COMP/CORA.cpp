// is_full_dim - whether a constrained zonotope has a nonempty interior, and the subspace it spans
//
// Syntax:   res = cz.is_full_dim();   res = cz.is_full_dim(tol);
//           [res, subspace] = cz.is_full_dim_subspace(tol);
// Inputs:   tol - singular values of the generators up to tol count as zero; default: the
//           repository tolerance (docs/architecture.md)
// Outputs:  res - true if the generators, restricted to the directions that the equalities and
//           the unit box leave to the factors, have rank n; subspace - the identity if so, else
//           an orthonormal basis (n, r) of their span. CORA tests rank(G null(A)) = n, which
//           holds for a set that the box pins to a point and for an empty one; the factors that
//           the programs fix are left out here (docs/cora-differences.md). Not a batch
// See also: is_bounded, Zonotope::is_full_dim
//
// CORA: contSet/@conZonotope/isFullDim.m
// CORA authors: Niklas Kochdumper, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the columns of Gf (n, k) span R^n up to tol, and the basis of their span.
std::pair<bool, Tensor> aux_span(const Tensor &Gf, int64_t n, double tol) {
    if (Gf.shape().back() == 0) return {n == 0, Gf.zeros_like({n, 0})};
    const int64_t rank = Gf.matrix_rank(tol);
    if (rank == n) return {true, Gf.eye_like(n)};
    return {false, Gf.svd()[0].slice(-1, 0, rank)};
}

/// The generators G N of the directions in which the set can move: the factors that are not fixed
/// by the constraints and the box, restricted to the null space of their constraint columns.
std::optional<Tensor> aux_free_generators(const ConZonotope &cz) {
    const HostConZonotope H = priv_host_con_zonotope(cz);
    if (priv_is_empty(H)) return std::nullopt;
    const auto [lo, hi] = priv_factor_ranges(H);
    std::vector<int64_t> free;
    for (int64_t j = 0; j < H.m; ++j)
        if (hi[j] - lo[j] > numerics::kFixedFactor) free.push_back(j);
    const Tensor G = priv_select_columns(cz.generators(), free);
    if (free.empty()) return G;
    return G.matmul(priv_null_space(priv_select_columns(cz.A(), free)));
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<bool, Tensor> ConZonotope::is_full_dim_subspace(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_full_dim");
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("is_full_dim: a rank per set; it supports a single set, not a batch");
    });
    const int64_t n = dim();
    if (A_.shape()[0] == 0) return aux_span(G_, n, tol);
    const std::optional<Tensor> free = aux_free_generators(*this);
    // an empty set has no interior
    if (!free) return {false, G_.zeros_like({n, 0})};
    return aux_span(*free, n, tol);
}

bool ConZonotope::is_full_dim(Tol tol) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol, c_.dtype(), "is_full_dim"));
    return is_full_dim_subspace(tol).first;
}

bool ConZonotope::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
