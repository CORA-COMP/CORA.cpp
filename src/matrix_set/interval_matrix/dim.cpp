// dim - the number of rows and columns of an interval matrix, as CORA's intervalMatrix.dim
//
// Syntax:   std::array<int64_t, 2> n = A.dim();   int64_t rows = A.dim(1);
// Inputs:   rc - 1 for the rows, 2 for the columns (CORA's second argument)
// Outputs:  n - {rows, cols} of each matrix; a batch has the same size in every member
// See also: rows, cols
//
// CORA: matrixSet/@intervalMatrix/dim.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::array<int64_t, 2> IntervalMatrix::dim() const {
    detail::Entry entry;
    return {rows(), cols()};
}

int64_t IntervalMatrix::dim(int rc) const {
    detail::Entry entry;
    CORA_CHECK(rc == 1 || rc == 2, Error::wrong_value, [&] {
        return "dim: rc must be 1 (rows) or 2 (columns), got " + std::to_string(rc);
    });
    return rc == 1 ? rows() : cols();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
