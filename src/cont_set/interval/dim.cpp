// dim - the number of rows or columns of an interval matrix, as CORA's intervalMatrix.dim(rc)
//
// Syntax:   int64_t rows = A.dim(1);   int64_t cols = A.dim(2);
// Inputs:   rc - 1 for the rows, 2 for the columns (CORA's second argument)
// Outputs:  n - the size of each matrix; a batch has the same size in every member
// See also: ContSet::dim (the rows, as CORA's dim(A, 1))
//
// CORA: matrixSet/@intervalMatrix/dim.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

int64_t Interval::dim(int rc) const {
    detail::Entry entry;
    CORA_CHECK(rc == 1 || rc == 2, Error::wrong_value, [&] {
        return "dim: rc must be 1 (rows) or 2 (columns), got " + std::to_string(rc);
    });
    // a box is a column
    if (elem_rank_ == 1) return rc == 1 ? inf_.size(-1) : 1;
    return inf_.size(rc == 1 ? -2 : -1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
