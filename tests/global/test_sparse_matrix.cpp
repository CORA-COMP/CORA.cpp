// test_sparse_matrix - SparseMatrix: coordinates, products, transpose, dense form, checks
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/sparse_matrix.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// [[1 0 2] [0 0 3]] from unsorted coordinates, the 2 given as 1.5 + 0.5.
SparseMatrix example() { return {2, 3, {1, 0, 0, 0}, {2, 2, 0, 2}, {3.0, 1.5, 1.0, 0.5}}; }

/// The coordinates, the sum of repeated ones, the product and the dense form.
void products() {
    const SparseMatrix S = example();
    check(S.rows() == 2 && S.cols() == 3 && S.nnz() == 3, "sparse: size and entries");
    const std::vector<double> x = {1.0, 10.0, 100.0};
    std::vector<double> y(2);
    S.matvec(x.data(), y.data());
    check(y[0] == 201.0 && y[1] == 300.0, "sparse: matvec");
    const std::vector<double> d = S.dense().data();
    check(d == std::vector<double>{1, 0, 2, 0, 0, 3}, "sparse: dense, repeated entries summed");
    check(S.dense_like(Tensor({{0.0}})).shape() == Shape{2, 3}, "sparse: dense_like");
}

/// The transpose and the round trip through a dense matrix.
void transpose_and_dense() {
    const SparseMatrix T = example().transpose();
    check(T.rows() == 3 && T.cols() == 2, "sparse: transpose size");
    check(T.dense().data() == std::vector<double>{1, 0, 0, 0, 2, 3}, "sparse: transpose");
    const Tensor M({{0.0, -1.0}, {2.0, 0.0}});
    const SparseMatrix S = SparseMatrix::from_dense(M);
    check(S.nnz() == 2 && S.dense().data() == M.data(), "sparse: from_dense round trip");
    const SparseMatrix empty(3, 3, {}, {}, {});
    std::vector<double> y(3, 7.0);
    const std::vector<double> x(3, 1.0);
    empty.matvec(x.data(), y.data());
    check(y == std::vector<double>(3, 0.0), "sparse: no entries, a zero product");
}

/// Entries outside the size and lists of different lengths are refused, unless checks are off.
void checks_and_their_switch() {
    check(test::raises([] { SparseMatrix(2, 2, {2}, {0}, {1.0}); },
                       Error::wrong_input_in_constructor),
          "sparse: row outside");
    check(test::raises([] { SparseMatrix(2, 2, {0, 1}, {0}, {1.0}); },
                       Error::wrong_input_in_constructor),
          "sparse: lengths differ");
    check(test::raises([] { SparseMatrix::from_dense(Tensor::zeros({2, 2, 2})); },
                       Error::dimension_mismatch),
          "sparse: from_dense of a batch");
    ChecksScope off(false);
    // the same matrix with the checks off
    check(example().dense().data() == std::vector<double>{1, 0, 2, 0, 0, 3},
          "sparse: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    products();
    transpose_and_dense();
    checks_and_their_switch();
    return test::finish("sparse matrix");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
