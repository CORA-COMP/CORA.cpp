// support_function - the support function of the empty set, as CORA's emptySet.supportFunc_
//
// Syntax:   rho = E.support_function(d);
// Inputs:   d - direction (..., n)
// Outputs:  rho - -Inf, the supremum over no point, shape (...) (CORA's type 'upper')
// See also: Interval::support_function
//
// CORA: contSet/@emptySet/supportFunc_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor EmptySet::support_function_impl(const Tensor &d) const {
    Shape shape = d.shape();
    shape.pop_back(); // (...), also for n = 0
    return d.full_like(shape, -std::numeric_limits<double>::infinity());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
