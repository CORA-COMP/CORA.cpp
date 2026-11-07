// priv_vertex_matrices - the distinct matrices among corner points
//
// Syntax:   std::vector<Tensor> V = priv_vertex_matrices(corners, rows, cols);
// Inputs:   corners - stacked matrices as rows (..., K, rows * cols), column by column
//           rows, cols - the shape of a matrix
// Outputs:  V - per batch member the distinct rows in ascending order as matrices (K', rows, cols)
// See also: IntervalMatrix::vertices, MatZonotope::vertices
//
// CORA.cpp only: CORA's vertices of the matrix sets share MATLAB's unique and vec2mat
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The most coordinates (entries or generators) whose 2^n corners are listed.
inline constexpr int64_t kMaxCornerCoordinates = 20;

std::vector<Tensor> priv_vertex_matrices(const Tensor &corners, int64_t rows, int64_t cols);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
