// lift - a polytope in a larger space, as CORA's polytope.lift_
//
// Coordinate j of the set is coordinate dims[j] of the N-dimensional space and the others are
// free: the columns of the rows move there, the other columns are zero. The vertices of a lifted
// set would be unbounded, so only the halfspace form is returned, except for N = n, where the
// vertices are permuted. CORA moves the vertices of N = n to V(dims, :), the inverse permutation of
// its rows (docs/cora-differences.md); a repeated position is refused.
//
// Syntax:   Pres = P.lift(N, dims);
// Inputs:   N - the dimension of the space, N >= n
//           dims - the position of each coordinate (0-based, n distinct entries below N)
// Outputs:  Pres - the set in N dimensions
// See also: Polytope::project_high_dim, Polytope::project
//
// CORA: contSet/@polytope/lift_.m
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

/// Why the lift to N dimensions with these positions is refused, if it is.
detail::Refusal aux_refusal(int64_t n, int64_t N, const std::vector<int64_t> &dims,
                            const char *operation) {
    if (const detail::Refusal r = detail::project_high_dim_refusal(n, N, dims, operation)) return r;
    std::vector<int64_t> sorted = dims;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        return std::pair{Error::wrong_value, std::string(operation) + ": a position is used twice"};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(dim(), N, dims, "lift");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // the vertices of a set that does not grow are permuted
    if (is_v_rep() && N == dim()) return Polytope(priv_place(V_, N, dims, -2));
    const Constraints C = constraints();
    return {priv_place(C.A, N, dims, -1), C.b, priv_place(C.Ae, N, dims, -1), C.be};
}

Set Polytope::lift_impl(int64_t N, const std::vector<int64_t> &proj) const { return lift(N, proj); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
