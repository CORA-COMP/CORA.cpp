// vertices - the vertices of a matrix zonotope as matrices, as CORA's matZonotope.vertices
//
// The corner matrices C +- G_1 +- ... +- G_h, without repetition and in ascending lexicographic
// order of the stacked matrices (CORA's unique). CORA returns the extreme points of the polytope;
// the corners are the same polytope's generating points, and all of them when the generators are in
// general position. At most 20 generators.
//
// Syntax:   std::vector<Tensor> V = M.vertices();
// Inputs:   -
// Outputs:  V - per batch member the matrices (K, rows, cols) at the corners
// See also: IntervalMatrix::vertices, zonotope
//
// CORA: matrixSet/@matZonotope/vertices.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"
#include "matrix_set/private/priv_vertex_matrices.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> MatZonotope::vertices() const {
    detail::Entry entry;
    const int64_t h = num_generators();
    CORA_CHECK(h <= kMaxCornerCoordinates, Error::not_supported, [&] {
        return "vertices: " + std::to_string(h) + " generators give 2^" + std::to_string(h) +
               " corners, at most " + std::to_string(kMaxCornerCoordinates) + " are supported";
    });
    // row k of the signs has +1 where bit h - 1 - j of k is set, else -1: the corner k
    const int64_t K = int64_t(1) << h;
    std::vector<double> signs(static_cast<std::size_t>(K * h));
    for (int64_t k = 0; k < K; ++k)
        for (int64_t j = 0; j < h; ++j)
            signs[std::size_t(k * h + j)] = ((k >> (h - 1 - j)) & 1) ? 1.0 : -1.0;
    const Tensor S = Tensor::like(C_, signs, {K, h});
    // the signed generators are added from the last to the first, then the center (as CORALean)
    const Tensor Gt = priv_stack_generators(G_).matrix_transpose();
    const std::vector<double> zeros(static_cast<std::size_t>(K * rows() * cols()));
    Tensor sum = Tensor::like(C_, zeros, {K, rows() * cols()});
    for (int64_t j = h - 1; j >= 0; --j)
        sum = sum + S.slice(-1, j, j + 1).multiply(Gt.slice(-2, j, j + 1));
    const Tensor corners = priv_stack_columns(C_).matrix_transpose() + sum;
    return priv_vertex_matrices(corners, rows(), cols());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
