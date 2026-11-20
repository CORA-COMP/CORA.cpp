// corners - the corners of a zonotope: c + G s for every sign vector s
//
// The corners contain the vertices, so a convex set holds Z exactly if it holds them; the
// containment tests of other sets ask for them. Not every corner is a vertex (CORA's vertices
// removes the inner ones); at most 30 generators (2^30 corners).
//
// Syntax:   Tensor P = Z.corners();
// Inputs:   -
// Outputs:  P - c + G s for the 2^m sign vectors s in {-1, 1}^m, as rows (..., 2^m, n)
// See also: vertices, Capsule::contains, ConZonotope::contains
//
// CORA.cpp only: containment tests need only the corners, which MATLAB takes from vertices
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::corners() const {
    detail::Entry entry;
    const int64_t m = nr_generators();
    CORA_CHECK(m <= 30, Error::not_supported, [&] {
        return "corners: " + std::to_string(m) + " generators give 2^" + std::to_string(m) +
               " corners, at most 30 generators are supported";
    });
    // sign j of corner k is bit j of k; the corners as rows s' G + c' (..., 2^m, n)
    const int64_t count = int64_t{1} << m;
    std::vector<double> signs(std::size_t(m * count));
    for (int64_t k = 0; k < count; ++k)
        for (int64_t j = 0; j < m; ++j) signs[k * m + j] = ((k >> j) & 1) ? 1.0 : -1.0;
    const Tensor rows = Tensor::like(c_, signs, {count, m}).matmul(G_);
    return c_.expand_dims(-2) + rows;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
