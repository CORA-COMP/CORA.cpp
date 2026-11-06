// is_full_dim - whether a zonotope has a nonempty interior, and the subspace it spans
//
// Syntax:   res = Z.is_full_dim();   res = Z.is_full_dim(tol);
//           [res, subspace] = Z.is_full_dim_subspace(tol);
// Inputs:   tol - singular values of the generators up to tol count as zero; default: the
//                 repository tolerance (docs/architecture.md)
// Outputs:  res - true if the generators span the space, i.e. their rank is the dimension;
//           subspace - the identity if so, else an orthonormal basis (n, r) of the span of the
//           generators (the leading left singular vectors)
// See also: is_bounded, compact
//
// CORA: contSet/@zonotope/isFullDim.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the call is refused, if it is: a batch (a rank per zonotope).
Refusal aux_refusal(const Tensor &G) {
    if (G.ndim() != 2)
        return std::pair{Error::not_supported, std::string("a rank per zonotope: it supports a "
                                                           "single zonotope, not a batch")};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Zonotope::is_full_dim(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, G_.dtype(), "is_full_dim");
    const Refusal refusal = aux_refusal(G_);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_full_dim: " + refusal->second; });
    return G_.shape().back() > 0 && G_.matrix_rank(tol) == dim();
}

std::pair<bool, Tensor> Zonotope::is_full_dim_subspace(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, G_.dtype(), "is_full_dim");
    const Refusal refusal = aux_refusal(G_);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_full_dim: " + refusal->second; });
    const int64_t n = dim();
    if (G_.shape().back() == 0) return {n == 0, G_.zeros_like({n, 0})};
    const int64_t rank = G_.matrix_rank(tol);
    if (rank == n) return {true, G_.eye_like(n)};
    return {false, G_.svd()[0].slice(-1, 0, rank)};
}

bool Zonotope::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
