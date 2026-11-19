// support_function - the support function of the whole space, as CORA's fullspace.supportFunc_
//
// CORA's type 'upper'. CORA answers Inf along the zero direction too, where the supremum of 0'x
// is 0; CORA.cpp follows CORALean (docs/cora-differences.md).
//
// Syntax:   rho = F.support_function(d);
// Inputs:   d - direction (..., n)
// Outputs:  rho - Inf along a nonzero direction and 0 along the zero one, shape (...)
// See also: EmptySet::support_function, Interval::support_function
//
// CORA: contSet/@fullspace/supportFunc_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Fullspace::support_function_impl(const Tensor &d) const {
    // max d'x over R^n is Inf unless d = 0; the largest |d_i| tells which
    const Tensor largest = d.abs().max(-1);

    const double kInf = std::numeric_limits<double>::infinity();
    return Tensor::where(largest.equal(largest.zeros_like()), largest,
                         largest.full_like(largest.shape(), kInf));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
