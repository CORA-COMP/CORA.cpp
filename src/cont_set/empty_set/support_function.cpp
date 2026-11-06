// support_function - the support function of the empty set, as CORA's emptySet.supportFunc_
//
// Syntax:   rho = E.support_function(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - -Inf, the supremum over no point, shape (..., 1, 1) (CORA's type 'upper')
// See also: Interval::support_function
//
// CORA: contSet/@emptySet/supportFunc_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor EmptySet::support_function_impl(const Tensor &d) const {
    const Tensor shape_of = d.matrix_transpose().matmul(d); // (..., 1, 1), also for n = 0
    return shape_of.full_like(shape_of.shape(), -std::numeric_limits<double>::infinity());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
