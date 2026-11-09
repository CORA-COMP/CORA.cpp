// sparse_matrix - the compressed rows of a sparse matrix, its products and its dense form
//
// The coordinates are sorted by row, then column (a stable sort, so repeated coordinates are
// summed in their given order); a sum of zero stays stored, where MATLAB drops it, which changes
// no product. matvec runs row by row, each row's entries in column order.
//
// Syntax:   SparseMatrix A(rows, cols, i, j, v);   A.matvec(x, y);
// Inputs:   rows, cols - the size
//           i, j, v - the coordinates (0-based) and the values of the entries
// Outputs:  A - the matrix
// See also: LinearSys
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/sparse_matrix.h"
#include "cora/global/error.h"
#include "cora/global/settings.h"

#include <algorithm>
#include <numeric>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// Construction ------------------------------------------------------------------------------------

SparseMatrix::SparseMatrix(int64_t rows, int64_t cols, const std::vector<int64_t> &i,
                           const std::vector<int64_t> &j, const std::vector<double> &v)
    : rows_(rows), cols_(cols) {
    detail::Entry entry;
    CORA_CHECK(rows >= 0 && cols >= 0 && i.size() == j.size() && i.size() == v.size(),
               Error::wrong_input_in_constructor, [&] {
                   return "SparseMatrix: the size must be nonnegative and i, j, v of one length, "
                          "got " +
                          std::to_string(i.size()) + ", " + std::to_string(j.size()) + ", " +
                          std::to_string(v.size());
               });
    for (size_t k = 0; k < i.size(); ++k)
        CORA_CHECK(i[k] >= 0 && i[k] < rows && j[k] >= 0 && j[k] < cols,
                   Error::wrong_input_in_constructor, [&] {
                       return "SparseMatrix: the entry " + std::to_string(k) + " at (" +
                              std::to_string(i[k]) + ", " + std::to_string(j[k]) +
                              ") is outside the size (" + std::to_string(rows) + ", " +
                              std::to_string(cols) + ")";
                   });
    // the entries by row, then column; equal coordinates keep their order and are summed
    std::vector<size_t> order(i.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return i[a] != i[b] ? i[a] < i[b] : j[a] < j[b];
    });
    row_start_.assign(size_t(rows) + 1, 0);
    for (size_t k = 0; k < order.size(); ++k) {
        const size_t e = order[k];
        if (k > 0 && i[e] == i[order[k - 1]] && j[e] == j[order[k - 1]]) {
            values_.back() += v[e];
            continue;
        }
        columns_.push_back(j[e]);
        values_.push_back(v[e]);
        ++row_start_[size_t(i[e]) + 1];
    }
    std::partial_sum(row_start_.begin(), row_start_.end(), row_start_.begin());
}

SparseMatrix SparseMatrix::from_dense(const Tensor &M) {
    detail::Entry entry;
    // a single matrix, read on the host
    CORA_CHECK(M.ndim() == 2, Error::dimension_mismatch, [&] {
        return "SparseMatrix::from_dense: a single matrix is needed, got " +
               std::to_string(M.ndim()) + " dimensions";
    });
    const int64_t rows = M.shape()[0], cols = M.shape()[1];
    const std::vector<double> data = M.data();
    std::vector<int64_t> i, j;
    std::vector<double> v;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            if (data[size_t(r * cols + c)] != 0.0) {
                i.push_back(r);
                j.push_back(c);
                v.push_back(data[size_t(r * cols + c)]);
            }
    return {rows, cols, i, j, v};
}

// Products and forms ------------------------------------------------------------------------------

void SparseMatrix::matvec(const double *x, double *y) const {
    for (int64_t r = 0; r < rows_; ++r) {
        double sum = 0.0;
        for (int64_t k = row_start_[size_t(r)]; k < row_start_[size_t(r) + 1]; ++k)
            sum += values_[size_t(k)] * x[columns_[size_t(k)]];
        y[r] = sum;
    }
}

SparseMatrix SparseMatrix::transpose() const {
    std::vector<int64_t> i, j;
    i.reserve(values_.size());
    j.reserve(values_.size());
    // each entry (r, c) becomes (c, r)
    for (int64_t r = 0; r < rows_; ++r)
        for (int64_t k = row_start_[size_t(r)]; k < row_start_[size_t(r) + 1]; ++k) {
            i.push_back(columns_[size_t(k)]);
            j.push_back(r);
        }
    return {cols_, rows_, i, j, values_};
}

std::vector<double> SparseMatrix::host_dense() const {
    std::vector<double> data(size_t(rows_ * cols_), 0.0);
    for (int64_t r = 0; r < rows_; ++r)
        for (int64_t k = row_start_[size_t(r)]; k < row_start_[size_t(r) + 1]; ++k)
            data[size_t(r * cols_ + columns_[size_t(k)])] = values_[size_t(k)];
    return data;
}

CORA_FROM_NUMBERS Tensor SparseMatrix::dense() const {
    return Tensor::from_data(host_dense(), {rows_, cols_});
}

Tensor SparseMatrix::dense_like(const Tensor &like) const {
    return Tensor::like(like, host_dense(), {rows_, cols_});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
