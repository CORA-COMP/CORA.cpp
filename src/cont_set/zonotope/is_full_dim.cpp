// is_full_dim - whether a zonotope has a nonempty interior, and the subspace it spans
//
// Syntax:   res = Z.is_full_dim();   res = Z.is_full_dim(tol);
//           [res, subspace] = Z.is_full_dim_subspace(tol);
// Inputs:   tol - singular values of the generators up to tol count as zero (CORA's default 1e-6)
// Outputs:  res - true if the generators span the space, i.e. their rank is the dimension;
//           subspace - the identity if so, else an orthonormal basis (n, r) of the span of the
//           generators (the leading left singular vectors)
// See also: is_bounded, compact
//
// CORA: contSet/@zonotope/isFullDim.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the call is refused, if it is: a batch (a rank per zonotope), a negative tol.
Refusal aux_refusal(const Tensor &G, double tol) {
    if (G.ndim() != 2)
        return std::pair{Error::not_supported, std::string("a rank per zonotope: it supports a "
                                                           "single zonotope, not a batch")};
    if (!(tol >= 0))
        return std::pair{Error::wrong_value, "tol must be nonnegative, got " + std::to_string(tol)};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Zonotope::is_full_dim(double tol) const {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(G_, tol);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_full_dim: " + refusal->second; });
    return G_.shape().back() > 0 && G_.matrix_rank(tol) == dim();
}

std::pair<bool, Tensor> Zonotope::is_full_dim_subspace(double tol) const {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(G_, tol);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_full_dim: " + refusal->second; });
    const int64_t n = dim();
    if (G_.shape().back() == 0) return {n == 0, G_.zeros_like({n, 0})};
    const int64_t rank = G_.matrix_rank(tol);
    if (rank == n) return {true, G_.eye_like(n)};
    return {false, G_.svd()[0].slice(-1, 0, rank)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
