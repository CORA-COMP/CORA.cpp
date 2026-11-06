// priv_vertex_matrices - the distinct matrices among corner points, in lexicographic order
//
// The corners are rows (..., K, n) of stacked matrices (column by column); each batch member keeps
// its distinct rows in ascending lexicographic order (MATLAB's unique) and reads them as rows x
// cols matrices.
//
// Syntax:   std::vector<Tensor> V = priv_vertex_matrices(corners, rows, cols);
// Inputs:   corners - (..., K, rows * cols)
//           rows, cols - the shape of a matrix
// Outputs:  V - one tensor (K', rows, cols) per batch member (the batch flattened)
// See also: IntervalMatrix::vertices, MatZonotope::vertices
//
// CORA.cpp only: CORA's vertices of the matrix sets share MATLAB's unique and vec2mat
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "matrix_set/private/priv_vertex_matrices.h"
#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows of the (K, n) host values without repetition, in ascending lexicographic order: the
/// index of the first of each.
std::vector<int64_t> aux_unique_rows(const double *values, int64_t K, int64_t n) {
    std::vector<int64_t> order(K);
    std::iota(order.begin(), order.end(), 0);
    const auto row_less = [&](int64_t a, int64_t b) {
        return std::lexicographical_compare(values + a * n, values + (a + 1) * n, values + b * n,
                                            values + (b + 1) * n);
    };
    std::stable_sort(order.begin(), order.end(), row_less);
    std::vector<int64_t> keep;
    for (const int64_t c : order)
        if (keep.empty() || row_less(keep.back(), c)) keep.push_back(c);
    return keep;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> priv_vertex_matrices(const Tensor &corners, int64_t rows, int64_t cols) {
    const int64_t K = corners.size(-2), n = rows * cols;
    int64_t members = 1;
    for (int64_t d = 0; d + 2 < corners.ndim(); ++d) members *= corners.shape()[std::size_t(d)];
    const Tensor flat = corners.reshape({members, K, n});
    const std::vector<double> values = flat.data();
    std::vector<Tensor> out;
    for (int64_t b = 0; b < members; ++b) {
        const std::vector<int64_t> keep = aux_unique_rows(values.data() + b * K * n, K, n);
        const Tensor rows_kept = flat.slice(0, b, b + 1).squeeze(0).take(keep, 0); // (K', n)
        // each row is a matrix read column by column
        out.push_back(rows_kept.reshape({int64_t(keep.size()), cols, rows}).matrix_transpose());
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
