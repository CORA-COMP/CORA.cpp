// sparse_matrix - a sparse matrix of doubles on the host (compressed rows), for A x products
//
// The state matrix of a large system (a discretized partial differential equation, as heat3d)
// has a few entries per row; the Krylov subspace of LinearSys::reach needs only its products with
// vectors. Built from coordinates (MATLAB's sparse(i, j, v), 0-based; repeated entries are
// summed), or from a dense matrix.
//
// Syntax:     SparseMatrix A(rows, cols, i, j, v);   A.matvec(x, y);   Tensor D = A.dense();
// Operations: matvec, transpose, dense (sparse_matrix.cpp)
// See also:   LinearSys, benchmark_data.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class SparseMatrix {
  public:
    /// The matrix (rows, cols) with the entries v[k] at (i[k], j[k]), 0-based; repeated
    /// coordinates are summed in their order, as MATLAB's sparse.
    SparseMatrix(int64_t rows, int64_t cols, const std::vector<int64_t> &i,
                 const std::vector<int64_t> &j, const std::vector<double> &v);

    /// The nonzero entries of the single matrix M (rows, cols), read on the host.
    static SparseMatrix from_dense(const Tensor &M);

    int64_t rows() const noexcept { return rows_; }
    int64_t cols() const noexcept { return cols_; }
    /// The number of stored entries.
    int64_t nnz() const noexcept { return int64_t(values_.size()); }

    /// y = M x for host vectors x (cols) and y (rows); y must not overlap x.
    void matvec(const double *x, double *y) const;

    /// The transpose M'.
    SparseMatrix transpose() const;

    /// The dense matrix (rows, cols) like `like` (its backend, device and dtype), else float64 on
    /// the default backend.
    Tensor dense() const;
    Tensor dense_like(const Tensor &like) const;

  private:
    int64_t rows_ = 0, cols_ = 0;
    std::vector<int64_t> row_start_; ///< rows + 1 offsets into columns_ and values_
    std::vector<int64_t> columns_;   ///< the column of each entry, ascending within a row
    std::vector<double> values_;

    std::vector<double> host_dense() const; ///< row-major (rows, cols)
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
