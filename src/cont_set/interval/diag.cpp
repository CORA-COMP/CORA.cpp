// diag - the diagonal matrix of a vector, or the diagonal of a matrix, as CORA's interval.diag
//
// Syntax:   Interval R = I.diag();   Interval R = I.diag(k);
// Inputs:   k - the diagonal, 0 the main one (default), k > 0 above it, k < 0 below it
// Outputs:  R - for a column or a row of bounds, the square matrix with it on the k-th diagonal
//           and [0, 0] elsewhere; for a matrix, the column of its k-th diagonal (an error if
//           that diagonal is empty); nothing rounds
// See also: ctranspose
//
// CORA: contSet/@interval/diag.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Zeros of the shape of `like` with its last two dimensions replaced.
Tensor aux_zeros(const Tensor &like, int64_t rows, int64_t cols) {
    std::vector<int64_t> shape = like.shape();
    shape[shape.size() - 2] = rows;
    shape.back() = cols;
    return like.zeros_like(shape);
}

/// The vector (a column, or a row) on the k-th diagonal of a square matrix.
Tensor aux_diagonal_matrix(const Tensor &v, int64_t k) {
    const Tensor column = v.size(-1) == 1 ? v : v.matrix_transpose();
    const Tensor D = column.diag();
    const int64_t n = D.size(-1), m = k < 0 ? -k : k;
    if (k == 0) return D;
    if (k > 0) {
        const Tensor wide = Tensor::concat({aux_zeros(D, n, m), D}, -1);
        return Tensor::concat({wide, aux_zeros(D, m, n + m)}, -2);
    }
    const Tensor tall = Tensor::concat({aux_zeros(D, m, n), D}, -2);
    return Tensor::concat({tall, aux_zeros(D, n + m, m)}, -1);
}

/// The k-th diagonal of a matrix as a column.
Tensor aux_diagonal_column(const Tensor &A, int64_t k) {
    if (k > 0) return A.slice(-1, k, A.size(-1)).diagonal();
    if (k < 0) return A.slice(-2, -k, A.size(-2)).diagonal();
    return A.diagonal();
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::diag(int64_t k) const {
    detail::Entry entry;
    const int64_t rows = inf_.size(-2), cols = inf_.size(-1);
    const bool vector = rows == 1 || cols == 1;
    CORA_CHECK(vector || (k < cols && -k < rows), Error::not_supported, [&] {
        return "diag: the " + std::to_string(k) + "-th diagonal of the bounds " +
               shape_text(inf_.shape()) + " is empty";
    });
    if (vector) return {aux_diagonal_matrix(inf_, k), aux_diagonal_matrix(sup_, k)};
    return {aux_diagonal_column(inf_, k), aux_diagonal_column(sup_, k)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
