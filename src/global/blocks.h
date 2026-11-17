// blocks - the block matrices that the set classes share
//
// Zeros, constants and identities of a batch shape, the selection of columns and the null space.
// The sets build their block matrices from these (constrained zonotopes, polytopes, ellipsoids
// and bundles), each like an operand so that the backend, the device and the batch shape follow.
//
// Syntax:   Tensor Z = zeros_block(like, rows, cols);   Tensor N = null_space(A);
// See also: cora/global/tensor/tensor.h (Tensor::block_diag)
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The constant matrix (..., rows, cols) of one value, with the batch shape and the data of `like`.
inline Tensor full_block(const Tensor &like, int64_t rows, int64_t cols, double value) {
    Shape shape = like.shape();
    shape[shape.size() - 2] = rows;
    shape.back() = cols;
    return like.full_like(shape, value);
}

/// Zeros (..., rows, cols) with the batch shape and the data of `like`.
inline Tensor zeros_block(const Tensor &like, int64_t rows, int64_t cols) {
    return full_block(like, rows, cols, 0.0);
}

/// The identity (..., k, k) with the batch shape of `like`.
inline Tensor identity_block(const Tensor &like, int64_t k) {
    Shape shape = like.shape();
    shape[shape.size() - 2] = k;
    shape.back() = k;
    return like.eye_like(k).broadcast_to(shape);
}

/// The columns of A (..., p, m) at the positions of `columns` kept, the others dropped.
inline Tensor select_columns(const Tensor &A, const std::vector<int64_t> &columns) {
    if (columns.empty()) return zeros_block(A, A.shape()[A.ndim() - 2], 0);
    return A.take(columns, -1);
}

/// An orthonormal basis (m, k) of the null space of A (p, m) (MATLAB's null): the right singular
/// vectors of the singular values up to max(p, m) eps times the largest.
Tensor null_space(const Tensor &A);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
