// project_high_dim - a polytope embedded in a larger space, as CORA's polytope.projectHighDim
//
// Coordinate j of the set becomes coordinate dims[j] of the N-dimensional space and the others are
// zero: the halfspaces are lifted and get the equalities x_k = 0 for the other coordinates, the
// vertices are padded with zeros. CORA's version for N = n permutes the columns of the rows by
// dims, the inverse of the placement it gives the vertices and the larger spaces; here both place
// coordinate j at dims[j] (docs/cora-differences.md). A repeated position is refused.
//
// Syntax:   Pres = P.project_high_dim(N, dims);
// Inputs:   N - the dimension of the space, N >= n
//           dims - the position of each coordinate (0-based, n distinct entries below N)
// Outputs:  Pres - the set in N dimensions, in the forms P holds
// See also: Polytope::lift, Polytope::project
//
// CORA: contSet/@polytope/projectHighDim.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows x_k = 0 of the coordinates that dims leaves out, as (rows, N) with the batch of like.
Tensor aux_zero_rows(const Tensor &like, int64_t N, const std::vector<int64_t> &dims) {
    std::vector<int64_t> missing;
    for (int64_t k = 0; k < N; ++k)
        if (std::find(dims.begin(), dims.end(), k) == dims.end()) missing.push_back(k);
    return priv_identity(like, N).take(missing, -2);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::project_high_dim(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::Refusal refusal = detail::project_high_dim_refusal(dim(), N, dims);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    std::vector<int64_t> sorted = dims;
    std::sort(sorted.begin(), sorted.end());
    CORA_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), Error::wrong_value,
               [] { return std::string("project_high_dim: a position is used twice"); });
    Parts parts;
    if (is_h_rep()) {
        // lifted rows, and an equality for every coordinate that is new
        const Tensor zeros = aux_zero_rows(A_, N, dims);
        parts.A = priv_place(A_, N, dims, -1);
        parts.b = b_;
        parts.Ae = Tensor::concat({priv_place(Ae_, N, dims, -1), zeros}, -2);
        parts.be = Tensor::concat({be_, priv_zeros(be_, zeros.shape()[zeros.ndim() - 2], 1)}, -2);
    }
    if (is_v_rep()) parts.V = priv_place(V_, N, dims, -2);
    return Polytope(std::move(parts));
}

Set Polytope::project_high_dim_impl(int64_t N, const std::vector<int64_t> &proj) const {
    return project_high_dim(N, proj);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
