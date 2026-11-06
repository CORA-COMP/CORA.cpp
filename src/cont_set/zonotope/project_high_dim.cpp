// project_high_dim - a zonotope placed into a larger space
//
// A repeated entry of dims keeps the last state.
//
// Syntax:   Zonotope R = Z.project_high_dim(N, dims);
// Inputs:   N - dimension of the space, N >= dim
//           dims - the position (0-based) in it of each state of Z, in the order of Z
// Outputs:  R - the zonotope whose state dims[i] is state i of Z and whose other states are 0
// See also: lift, project
//
// CORA: contSet/@zonotope/projectHighDim_.m
// CORA: contSet/@contSet/projectHighDim.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows of [c G] (..., n, k) at the positions dims of N rows, zero rows elsewhere.
Tensor aux_embed(const Tensor &rows, int64_t N, const std::vector<int64_t> &dims) {
    const int64_t n = rows.shape()[rows.ndim() - 2];
    Shape zero_shape = rows.shape();
    zero_shape[rows.ndim() - 2] = 1;
    // the zero row is row n of the padded matrix; a position nobody maps to selects it
    std::vector<int64_t> source(N, n);
    for (int64_t i = 0; i < n; ++i) source[dims[i]] = i;
    return Tensor::concat({rows, rows.zeros_like(zero_shape)}, -2).take(source, -2);
}

/// What is wrong with N and the positions dims for a zonotope of dimension n, or "" if nothing.
std::string aux_problem(int64_t N, const std::vector<int64_t> &dims, int64_t n) {
    if (N < n)
        return "the space of " + std::to_string(N) +
               " dimensions must not be smaller than the "
               "zonotope of " +
               std::to_string(n);
    if (int64_t(dims.size()) != n)
        return "one position per state is needed, got " + std::to_string(dims.size()) + " for " +
               std::to_string(n);
    for (const int64_t d : dims)
        if (d < 0 || d >= N)
            return "the position " + std::to_string(d) + " is not in [0, " + std::to_string(N) +
                   ")";
    return "";
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::project_high_dim(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const std::string problem = aux_problem(N, dims, dim());
    CORA_CHECK(problem.empty(), Error::wrong_value, [&] { return "project_high_dim: " + problem; });
    return {aux_embed(c_, N, dims), aux_embed(G_, N, dims)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
