// priv_quad_map_rows - one output of CORA's quadratic map of a zonotope, from its matrix
//
// The generators are the N halved squares, then the strictly lower triangle of M + M' in MATLAB's
// order.
//
// Syntax:   QuadRow row = priv_quad_map_single_row(M, N);   Shape s = priv_batch_shape(like, 1, 3);
// Inputs:   M - [c G]' Q [c G] (..., N+1, N+1) for N generators
// Outputs:  row - the center (..., 1, 1) and the generators (..., 1, N + N(N+1)/2) of the output
// See also: Zonotope::quad_map, ConZonotope::quad_map
//
// CORA: contSet/@zonotope/quadMap.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Shape priv_batch_shape(const Tensor &like, int64_t rows, int64_t cols) {
    Shape shape;
    for (int64_t i = 0; i + 2 < like.ndim(); ++i) shape.push_back(like.shape()[i]);
    shape.push_back(rows);
    shape.push_back(cols);
    return shape;
}

QuadRow priv_quad_map_single_row(const Tensor &M, int64_t N) {
    const Tensor squares = M.slice(-2, 1, N + 1).slice(-1, 1, N + 1).diagonal() * 0.5;
    const Tensor center = M.slice(-2, 0, 1).slice(-1, 0, 1) + squares.sum(-2, true);
    std::vector<int64_t> lower;
    for (int64_t j = 0; j <= N; ++j)
        for (int64_t r = j + 1; r <= N; ++r) lower.push_back(r * (N + 1) + j);
    const Tensor mixed = lower.empty() ? M.zeros_like(priv_batch_shape(M, 1, 0))
                                       : (M + M.matrix_transpose())
                                             .reshape(priv_batch_shape(M, 1, (N + 1) * (N + 1)))
                                             .take(lower, -1);
    return {center, Tensor::concat({squares.matrix_transpose(), mixed}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
